# model/reactome_utils.py

import pandas as pd
import itertools
import numpy as np
try:
    from GMT import * # Check and import GMT library if available
except ImportError:
    print("Warning: GMT library not found. Install it to use Reactome functions.")
    # Define dummy classes to prevent errors if GMT is not installed
    class Reactome:
        def __init__(self):
            self.pathway_names = None
            self.hierarchy = None
            self.pathway_genes = None
    class ReactomeNetwork:
        def get_layers(self, n_levels=4):
            return [] # Return an empty list so there's nothing to generate

    reactome = Reactome() # instantiate dummy reactome object
    reactome_net = ReactomeNetwork() # instantiate dummy reactome_net object
else:
    reactome = Reactome() # If GMT is importable use it
    reactome_net = ReactomeNetwork()

    names_df = reactome.pathway_names
    hierarchy_df = reactome.hierarchy
    genes_df = reactome.pathway_genes


def generate_layer_relation_df(layers):
    """
    Generates a DataFrame representing the input-output relationships for each layer.

    Args:
        layers (list): A list of dictionaries, where each dictionary represents a layer.
                       Each dictionary maps child nodes (outputs) to a list of parent nodes (inputs).

    Returns:
        list: A list of pandas DataFrames, where each DataFrame represents the relationships
              between input and output nodes for a single layer.
    """
    layer_relation_dfs = []

    for i, layer in enumerate(layers):
        # Extract parent and child nodes for the current layer
        parents = list(itertools.chain.from_iterable(layer.values()))  # All parent nodes
        children = list(layer.keys())  # All child nodes

        # Remove duplicates from parent and child lists
        parents = list(np.unique(parents))
        children = list(np.unique(children))

        # Build the relationship DataFrame
        relations = []
        for child, parent_list in layer.items():
            for parent in parent_list:
                relations.append([parent, child])

        # Create pandas DataFrame
        df = pd.DataFrame(relations, columns=['input_features', 'output_nodes'])
        layer_relation_dfs.append(df)

        print(f"Layer {i + 1}: Generated relation dataframe with shape: {df.shape}")

    return layer_relation_dfs

if __name__ == "__main__":
    # Example Usage / Self-Test (requires GMT library)
    print("Running reactome_utils self-test...")

    # Check if the dummy object has been created
    if reactome.pathway_names is None:
        print("\nSkipping Reactome test because GMT library is not installed.")
    else:

        # This will only run if GMT is properly installed
        # Create dummy layers for testing
        # Assumes `reactome_net` is already defined and the library is installed.

        try:
            layers = reactome_net.get_layers(n_levels=2) # 2 for the example

            # Generate layer relation DataFrames
            layer_relation_dfs = generate_layer_relation_df(layers)

            # Print the first few rows of each DataFrame
            for i, df in enumerate(layer_relation_dfs):
                print(f"\nLayer {i + 1} Relation DataFrame:")
                print(df.head())
        except Exception as e:
            print(f"\nError during Reactome processing: {e}")

    print("\nSelf-test complete.")