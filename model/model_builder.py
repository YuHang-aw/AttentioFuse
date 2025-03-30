# model/model_builder.py

import torch
import torch.nn as nn
import pandas as pd # Added import
#from dataprocess import index_cleaning # The file no longer exists here, so this call breaks.
# Moved the function definition to /dataprocess/data_cleaning.py

def process_layers_and_build_masks(layer, mtx):
    """
    Processes each layer, generates relation DataFrames, and builds mask matrices
    using build_mask_from_config, from the last layer to the first. Cleans input
    features.

    Args:
        layers (list): A list of dictionaries, where each dictionary represents a layer.
                       Each dictionary maps child nodes (outputs) to a list of parent nodes (inputs).
        mtx (pd.DataFrame): DataFrame representing the input feature matrix, used to clean features.

    Returns:
        tuple: A tuple containing two lists:
               - masks (list): A list of mask matrices for each layer.
               - mappings (list): A list of (input_mapping, output_mapping) tuples for each layer.
    """
    from dataprocess.data_cleaning import index_cleaning
    # Generate relation DataFrames for each layer
    # layer_relation_dfs = generate_layer_relation_df(layers)
    layer_relation_dfs = layer # No longer generating

    masks = []
    mappings = []
    prev_output_features = None  # Output features from the previous layer

    # Process from the last layer to the first
    for i in range(len(layer_relation_dfs) - 1, -1, -1):
        df = layer_relation_dfs[i]

        if prev_output_features is None:
            # On the first iteration, clean the initial mtx
            mtx_cleaned, relation_sorted = index_cleaning(mtx, df, 'input_features')
        else:
            # For subsequent layers, ensure the input features match the previous layer's output
            input_features = df['input_features'].isin(prev_output_features)
            relation_sorted = df[input_features]
            mtx_cleaned = None  # No need to clean mtx again

        # Build the mask matrix using the cleaned DataFrame
        mask, input_mapping, output_mapping = build_mask_from_config(relation_sorted)

        # Update previous layer's output features to the current layer's output features
        prev_output_features = output_mapping.values()

        # Store the mask and mappings
        masks.append(mask)
        mappings.append((input_mapping, output_mapping))

        print(f"Layer {len(layer_relation_dfs) - i}: Mask shape {mask.shape}")

    # Return the reversed lists (because we processed from last to first)
    return masks[::-1], mappings[::-1]  # Reverses the list because we are traversing from last to first

def create_mlp_layer(input_size, output_size, layer_number, mask=None):
    """
    Creates an MLP layer with an option to apply a mask to initialize weights.

    Args:
        input_size (int): Number of input nodes.
        output_size (int): Number of output nodes.
        layer_number (int): Layer number for naming purposes.
        mask (torch.Tensor, optional): Mask matrix to apply to the layer's weights.
                                        If None, no mask is applied. Defaults to None.

    Returns:
        nn.Linear: An initialized MLP layer (linear transformation).
    """
    print(f"Creating MLP Layer {layer_number}: {input_size} -> {output_size}")
    layer = nn.Linear(input_size, output_size) # Create a linear layer

    if mask is not None: # Apply mask if provided
        with torch.no_grad():  # Disable gradient calculation for direct weight modification
            layer.weight *= mask  # Apply the mask

    return layer # Return the created layer

class MaskedMLP(nn.Module):
    """
    A Masked Multi-Layer Perceptron (MLP) class that allows for applying masks
    to the weights of the hidden layers during the forward pass.
    """
    def __init__(self, input_size, hidden_sizes, output_size, masks, device=None):
        """
        Initializes the MaskedMLP model.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list): A list of integers defining the size of each hidden layer.
            output_size (int): The number of output classes.
            masks (list): A list of mask matrices (torch.Tensor) for each hidden layer.
                          Must be the same length as hidden_sizes.
            device (torch.device): The device to run the model on ('cuda' or 'cpu').
        """
        super(MaskedMLP, self).__init__()

        # Store the initialization parameters
        self.init_params = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'masks': masks,
            'device': device
        }

        self.device = device

        # Convert masks to the specified device (if not None)
        self.masks = [mask.to(self.device) if mask is not None else None for mask in masks]

        # Define layers and activation functions
        self.layers = nn.ModuleList()
        last_size = input_size # Keep track of the last layer size

        for i, hidden_size in enumerate(hidden_sizes):
            #Linear transformation
            linear = nn.Linear(last_size, hidden_size)
            #Activation Function
            activation = nn.LeakyReLU(negative_slope=0.01)
            self.layers.append(nn.Sequential(linear, activation))
            last_size = hidden_size #Update layer size

        # Output layer
        self.output_layer = nn.Linear(last_size, output_size)
        #Output activation function depending on output shape
        self.output_activation = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)

        # Used to store each layers output for debugging and checking weights, bias, etc
        self.layer_outputs = []

    def forward(self, x):
        """
        Performs the forward pass through the MaskedMLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        self.layer_outputs = []  # Clear the layer outputs for each forward pass

        for i, layer in enumerate(self.layers):
            linear = layer[0]  # Get the linear layer
            activation = layer[1]  # Get the activation function

            # Apply the mask (if any)
            if self.masks[i] is not None:
                with torch.no_grad(): # Disable gradient calculation
                    linear.weight *= self.masks[i]  # Apply mask directly to weights

            x = linear(x)  # Linear Transformation
            x = activation(x)  # Activation function to give network non-linearity
            self.layer_outputs.append(x)  # Store the output

        x = self.output_layer(x) # Linearly transform last layer
        x = self.output_activation(x)  # Calculate output using activation function

        self.layer_outputs.append(x)  # Store the final output

        return x  # Return only the final output for loss calculation and evaluation

def build_mask_from_config(relation_sorted):
    """
    Builds a mask matrix, an input mapping, and an output mapping from a
    sorted relation DataFrame.

    Args:
        relation_sorted (pd.DataFrame): Sorted DataFrame containing 'input_features' and
                                         'output_nodes' columns representing the relationships
                                         between input and output nodes.

    Returns:
        tuple: A tuple containing:
               - mask (torch.Tensor): A mask matrix representing the connections
                 between input and output nodes.
               - input_mapping (dict): A dictionary mapping column names to input indices.
               - output_mapping (dict): A dictionary mapping output nodes to column names.
    """
    import torch

    # Extract unique input and output feature names from the relation DataFrame
    input_features = relation_sorted['input_features'].unique()
    output_nodes = relation_sorted['output_nodes'].unique()

    # Create mappings from feature names to indices
    input_mapping = {col: i for i, col in enumerate(input_features)}
    output_mapping = {node: col for col, node in enumerate(output_nodes)}

    # Initialize the mask matrix with zeros
    mask = torch.zeros(len(output_nodes), len(input_features))

    # Populate the mask based on the relations
    for _, row in relation_sorted.iterrows():
        input_idx = input_mapping[row['input_features']]
        output_idx = output_mapping[row['output_nodes']]
        mask[output_idx, input_idx] = 1  # Set corresponding entry to 1 if relation exists

    return mask, input_mapping, output_mapping

if __name__ == "__main__":
    # Example Usage / Self-Test
    print("Running model_builder self-test...")

    # --- Create Dummy Data ---
    import numpy as np
    # Dummy expression matrix
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'feature3': [11, 12, 13, 14, 15]}
    mtx = pd.DataFrame(data)

    # Dummy layer information (list of DataFrames)
    layer1_data = {'input_features': ['feature1', 'feature2', 'feature3'],
                    'output_nodes': ['node1', 'node1', 'node2']}
    layer1 = [pd.DataFrame(layer1_data)]

    # --- Test the Process Layers Function ---
    try:
        masks, mappings = process_layers_and_build_masks(layer1, mtx)
        print("\nprocess_layers_and_build_masks test passed.")
        print(f"Mask shape: {masks[0].shape}")
    except Exception as e:
        print(f"\nError in process_layers_and_build_masks test: {e}")

    # --- Test MaskedMLP ---
    if 'masks' in locals() and masks: #Checks if the local masks variable are loaded and populated
        try:
            input_size = mtx.shape[1]
            hidden_sizes = [4] # Example hidden layer size
            output_size = 2 # Example output size
            mlp = MaskedMLP(input_size, hidden_sizes, output_size, masks)
            print("\nMaskedMLP test passed. Shape = ", mlp.init_params['masks'][0].shape) #Prints first mask dimensions
        except Exception as e:
            print(f"\nError in MaskedMLP test: {e}")
    else:
        print("Skipping MaskedMLP creation test. No masks")

    print("\nSelf-test complete.")