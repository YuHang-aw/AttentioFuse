# model/model_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path # Added for path handling
from collections import defaultdict # Added for pathway analysis


def clean_and_split_data(data, labels, test_size=0.2, random_state=42):
    """
    Cleans labels (combining if necessary) and splits the data into
    training and testing sets, converting them to PyTorch tensors.

    Args:
        data (pd.DataFrame): Input data (expression matrix).
        labels (pd.DataFrame or pd.Series): Labels for the data.
                                             If a DataFrame with multiple
                                             columns, labels are combined.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the training and testing data as
              PyTorch tensors, along with the fitted LabelEncoder:
              {
                  'train_input': torch.Tensor,
                  'test_input': torch.Tensor,
                  'train_label': torch.Tensor,
                  'test_label': torch.Tensor,
                  'label_encoder': LabelEncoder
              }
    """
    # Combine labels if labels is a DataFrame with multiple columns
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] > 1:
            # Combine labels using an underscore as separator
            labels_combined = labels.apply(lambda row: '_'.join(row.astype(str)), axis=1)
        else:
            labels_combined = labels.iloc[:, 0] # Use the first column as labels
    else:
        # If labels is a Series, use it directly
        labels_combined = labels

    # Encode the labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels_combined)

    # Split the data into training and testing sets, stratifying by the encoded labels
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Convert data to PyTorch tensors
    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32),
        'test_input': torch.tensor(X_test, dtype=torch.float32),
        'train_label': torch.tensor(y_train, dtype=torch.long),
        'test_label': torch.tensor(y_test, dtype=torch.long),
        'label_encoder': encoder  # Save the encoder for later use
    }

    return dataset


# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    """
    Custom Dataset class for loading data into PyTorch models.
    """
    def __init__(self, inputs, labels):
        """
        Initializes the dataset.

        Args:
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Corresponding labels.
        """
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves the input data and label at the given index.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: A tuple containing the input data and label.
        """
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return input_data, label


def init_weights(m):
    """
    Initializes the weights of a PyTorch linear layer using Kaiming He initialization.
    """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0) # Initialize bias to zero


def train_model(model, dataset, epochs, batch_size, optimizer,
               early_stopping_patience=10, device=None, print_every=5,
               init_model=True):
    """
    Improved training function for PyTorch models, providing training history,
    early stopping, learning rate scheduling, and optional initialization.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataset (dict): A dictionary containing the training and testing data
                        as PyTorch tensors (output of clean_and_split_data).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        early_stopping_patience (int): Patience for early stopping.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        print_every (int): Print training progress every this many epochs.
        init_model (bool): Whether to initialize the model weights.

    Returns:
        nn.Module: The trained PyTorch model.
    """
    # Initialize training history dictionary
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    # Determine device (CPU or GPU)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model weights
    if init_model:
        model.apply(init_weights)

    # Move model to the device
    model.to(device)

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(CustomDataset(dataset['train_input'], dataset['train_label']),
                             batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CustomDataset(dataset['test_input'], dataset['test_label']),
                           batch_size=batch_size, shuffle=False, num_workers=0)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Learning rate scheduler with ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move batch to the device

            optimizer.zero_grad() # Zero gradients
            outputs = model(inputs) # Forward pass
            loss = loss_fn(outputs, targets) # Calculate the loss
            loss.backward() # Backward pass (calculate gradients)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients

            optimizer.step() # Update weights
            total_loss += loss.item() # Accumulate the loss

        avg_train_loss = total_loss / len(train_loader) # Average training loss
        avg_val_loss, val_report = evaluate(model, val_loader, loss_fn, device,
                                           label_encoder=dataset.get('label_encoder'))

        # Record training metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_report.loc['accuracy', 'f1-score'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print training progress
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1:03d} | '
                  f'LR: {history["lr"][-1]:.1e} | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Val Acc: {history["val_acc"][-1]:.2%}')

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}!')
            break

    # Final evaluation and visualization
    # Re-evaluate the model on the validation set and print the report
    _, val_report = evaluate(
        model, val_loader, loss_fn, device,
        label_encoder=dataset.get('label_encoder'),
        print_report=True
    )

    print("\n=== Best Model Validation Report ===")
    print(val_report.to_string())

    # Plotting the training history
    plt.figure(figsize=(12, 4))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    return model


def evaluate(model, data_loader, loss_fn, device=None, label_encoder=None, print_report=False, return_preds=False):
    """
    Evaluates the PyTorch model on a given dataset, providing loss and
    classification report.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        loss_fn (nn.Module): Loss function to use (e.g., CrossEntropyLoss).
        device (torch.device): Device to use for evaluation ('cuda' or 'cpu').
        label_encoder (LabelEncoder): The LabelEncoder used to encode the labels.
        print_report (bool): Whether to print the classification report.
        return_preds (bool): Whether to return the true labels and predictions.

    Returns:
        tuple: A tuple containing the average loss and the classification report
               (as a pandas DataFrame). If return_preds is True, also returns
               the true labels and predictions.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_labels = []
    all_preds = []
    total_loss = 0
    num_batches = 0

    model.eval() # Set the model to evaluation mode

    with torch.no_grad(): # Disable gradient calculation for efficiency
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device) # Move batch to device
            outputs = model(data) # Forward pass
            loss = loss_fn(outputs, labels)  # Calculate loss
            total_loss += loss.item()  # Accumulate loss
            num_batches += 1

            _, preds = torch.max(outputs, 1) # Get predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            all_preds.extend(preds.cpu().numpy())  # Collect predictions

    # Generate classification report
    if label_encoder is not None:
        target_names = label_encoder.classes_  # Use original class names
    else:
        target_names = None # If no encoder, keep it numerical

    # Use zero_division=1 to avoid UndefinedMetricWarning
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=1)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    # Ensure "support" column is integer type
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].astype(int)

    if print_report:
        print(report_df.to_string())  # Print the report in a nice format

    avg_loss = total_loss / num_batches  # Average loss

    if return_preds:
        return avg_loss, {'true': all_labels, 'pred': all_preds, 'report': report_df}
    else:
        return avg_loss, report_df



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# The other functions (clean_and_split_data, CustomDataset, init_weights, train_model, evaluate) from the previous steps go here.

def evaluate_models(dataset):
    """
    Evaluates multiple classical machine learning models.

    Args:
        dataset (dict): A dictionary containing the training and testing data
                      as PyTorch tensors (output of clean_and_split_data).
                      The 'train_input', 'train_label', 'test_input', and
                      'test_label' keys are expected.

    Returns:
        tuple: A tuple containing:
               - results_df (pd.DataFrame): A DataFrame containing the evaluation
                 metrics for each model.
               - predictions (dict): A dictionary containing the predictions
                 (y_pred) and predicted probabilities (y_prob) for each model.
    """
    # Prepare data
    X_train = dataset['train_input'].numpy()
    y_train = dataset['train_label'].numpy()
    X_test = dataset['test_input'].numpy()
    y_test = dataset['test_label'].numpy()

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }

    results = []
    predictions = {}

    for name, model in models.items():
        try: # Add try-except
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            # Handle cases where predict_proba might not be available or fails for binary
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc_roc = roc_auc_score(y_test, y_prob)
            except (AttributeError, ValueError):
                 print(f"Warning: Could not calculate AUC-ROC for {name}.")
                 y_prob = np.zeros(len(y_test)) # Placeholder probability
                 auc_roc = np.nan # Assign NaN if AUC cannot be calculated

            predictions[name] = {'pred': y_pred, 'prob': y_prob}

            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0), # Handle zero division
                'Recall': recall_score(y_test, y_pred, zero_division=0), # Handle zero division
                'F1 Score': f1_score(y_test, y_pred, zero_division=0), # Handle zero division
                'AUC-ROC': auc_roc
            }
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating model {name}: {e}") # Print error if model fails
            # Optionally append placeholder results or skip
            results.append({'Model': name, 'Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'F1 Score': np.nan, 'AUC-ROC': np.nan})


    results_df = pd.DataFrame(results)

    # Visualize results
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    for metric in metrics_to_plot:
        # Filter out NaN values for plotting
        plot_df = results_df.dropna(subset=[metric])
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Model', y=metric, data=plot_df) # Plot only non-NaN
            plt.xticks(rotation=45)
            plt.title(f'{metric} Comparison') # English title
            plt.tight_layout()
            plt.show()
        else:
            print(f"Skipping plot for {metric} as no valid data is available.")


    return results_df, predictions

# --- NEW Multi-Omics Utilities ---

class MultiOmicsDataset(Dataset):
    """
    Custom PyTorch Dataset for handling multiple omics input modalities.
    """
    def __init__(self, omics_inputs, labels):
        """
        Args:
            omics_inputs (dict): Dictionary where keys are omics names (str)
                                 and values are the corresponding input data tensors.
                                 e.g., {'mrna': tensor, 'cnv': tensor}
            labels (torch.Tensor): Tensor containing the labels.
        """
        self.omics_inputs = omics_inputs
        self.labels = labels
        # Check that all omics data tensors have the same first dimension (number of samples)
        self._validate_inputs()

    def _validate_inputs(self):
        """Checks if all input tensors have the same number of samples."""
        first_key = next(iter(self.omics_inputs))
        self.num_samples = len(self.omics_inputs[first_key])
        if not all(len(data) == self.num_samples for data in self.omics_inputs.values()):
            raise ValueError("All omics input tensors must have the same number of samples.")
        if len(self.labels) != self.num_samples:
             raise ValueError("Number of labels must match the number of samples in omics inputs.")

    def __len__(self):
        """Returns the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves the sample data for all omics types and the corresponding label
        at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                   - sample (dict): Dictionary {'omics_name': data_tensor_for_sample_idx}.
                   - label (torch.Tensor): The label for the sample.
        """
        # Retrieve data for the specific index from each omics tensor
        sample = {name: data[idx] for name, data in self.omics_inputs.items()}
        label = self.labels[idx]
        return sample, label


def prepare_multi_omics_data(omics_data_dict, labels, test_size=0.2, random_state=42):
    """
    Prepares and splits multi-omics data for training and testing.

    Args:
        omics_data_dict (dict): Dictionary {'omics_name': pd.DataFrame or np.ndarray}.
                                Assumes data is aligned by sample index.
        labels (pd.Series or np.ndarray): Labels corresponding to the samples.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        dict: A dictionary containing the split data:
              {
                  'train_input': {'omics_name': train_tensor},
                  'test_input': {'omics_name': test_tensor},
                  'train_label': train_label_tensor,
                  'test_label': test_label_tensor,
                  'label_encoder': fitted LabelEncoder instance
              }
    """
    # Prepare labels
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] > 1:
            labels_combined = labels.apply(lambda row: '_'.join(row.astype(str)), axis=1)
        else:
            labels_combined = labels.iloc[:, 0]
    elif isinstance(labels, pd.Series):
         labels_combined = labels
    else: # Assume numpy array or list
        labels_combined = labels

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels_combined)

    # Get indices for splitting (assuming all omics have the same index length)
    num_samples = len(y_encoded)
    indices = np.arange(num_samples)

    # Split indices into training and testing sets
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded # Ensure stratification based on labels
    )

    # Create dictionaries to hold split data tensors for each omics type
    train_data = {}
    test_data = {}

    for omics_name, omics_data in omics_data_dict.items():
        # Convert to numpy array if it's a DataFrame
        if isinstance(omics_data, pd.DataFrame):
            omics_values = omics_data.values
        else:
            omics_values = omics_data # Assume numpy array

        # Split data based on indices and convert to tensors
        train_data[omics_name] = torch.tensor(omics_values[train_idx], dtype=torch.float32)
        test_data[omics_name] = torch.tensor(omics_values[test_idx], dtype=torch.float32)

    # Prepare label tensors
    train_labels = torch.tensor(y_encoded[train_idx], dtype=torch.long)
    test_labels = torch.tensor(y_encoded[test_idx], dtype=torch.long)

    return {
        'train_input': train_data,
        'test_input': test_data,
        'train_label': train_labels,
        'test_label': test_labels,
        'label_encoder': encoder
    }


def train_multi_omics_model(model, dataset, epochs, batch_size, optimizer,
                           early_stopping_patience=10, device=None, print_every=5,
                           init_model=True):
    """
    Training function for multi-omics models, consistent with the single-omics version.
    Includes history tracking, early stopping, LR scheduling, and visualization.

    Args:
        model (nn.Module): The multi-omics PyTorch model to train.
        dataset (dict): Dataset dictionary from prepare_multi_omics_data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        optimizer (torch.optim.Optimizer): Optimizer.
        early_stopping_patience (int): Patience for early stopping.
        device (torch.device): Device ('cuda' or 'cpu').
        print_every (int): Frequency of printing progress.
        init_model (bool): Whether to initialize model weights.

    Returns:
        nn.Module: The trained model (best weights loaded).
    """
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []
    }

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if init_model:
        # Assuming init_weights is defined elsewhere in model_utils
        model.apply(init_weights)

    model.to(device)

    # Create multi-omics DataLoaders
    train_loader = DataLoader(
        MultiOmicsDataset(dataset['train_input'], dataset['train_label']),
        batch_size=batch_size, shuffle=True, num_workers=0 # Set num_workers=0 for simplicity/debugging
    )
    val_loader = DataLoader(
        MultiOmicsDataset(dataset['test_input'], dataset['test_label']),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            # Move all input omics tensors and labels to the device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) # Pass the dictionary of inputs
            loss = loss_fn(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        # Use the corresponding multi-omics evaluation function
        avg_val_loss, val_report = evaluate_multi_omics(
            model, val_loader, loss_fn, device,
            label_encoder=dataset.get('label_encoder')
        )

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        # Ensure 'accuracy' row exists and use its f1-score value
        if 'accuracy' in val_report.index:
             history['val_acc'].append(val_report.loc['accuracy', 'f1-score'])
        else:
             # Handle case where accuracy might not be directly calculated (e.g., multi-label)
             # Use macro avg f1-score as a fallback or choose another appropriate metric
             history['val_acc'].append(val_report.loc['macro avg', 'f1-score'])

        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
             print(f'Epoch {epoch+1:03d} | LR: {history["lr"][-1]:.1e} | '
                   f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | '
                   f'Val Acc: {history["val_acc"][-1]:.2%}') # Use last recorded accuracy


        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()} # Save best weights
        else:
            epochs_no_improve += 1

        scheduler.step(avg_val_loss) # LR scheduler step

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}!')
            break

    # Load best model weights before final evaluation
    if best_weights:
        print("\nTraining completed. Loading best model weights...")
        model.load_state_dict(best_weights)
    else:
        print("\nTraining completed without improvement. Using final model weights.")


    # Final evaluation
    _, final_val_report = evaluate_multi_omics(
        model, val_loader, loss_fn, device,
        label_encoder=dataset.get('label_encoder'),
        print_report=True
    )

    print("\n=== Best Model Validation Report ===")
    print(final_val_report.to_string())

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model


def evaluate_multi_omics(model, data_loader, loss_fn, device=None,
                    label_encoder=None, print_report=False, return_preds=False):
    """
    Evaluation function for multi-omics models.

    Args:
        model (nn.Module): The multi-omics model to evaluate.
        data_loader (DataLoader): DataLoader (using MultiOmicsDataset).
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device.
        label_encoder (LabelEncoder): Fitted LabelEncoder.
        print_report (bool): Whether to print the classification report.
        return_preds (bool): Whether to return predictions along with metrics.

    Returns:
        tuple: Average loss and classification report (pd.DataFrame).
               If return_preds is True, returns (avg_loss, preds_dict).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_labels = []
    all_preds = []
    total_loss = 0
    num_batches = 0

    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move all input omics tensors and labels to the device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(inputs) # Get model predictions
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            _, preds = torch.max(outputs, 1) # Get predicted class index
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Get target names from label encoder
    if label_encoder is not None:
        try:
            target_names = label_encoder.classes_
        except AttributeError:
             print("Warning: LabelEncoder does not have classes_ attribute.")
             target_names = None # Fallback if classes_ not found
    else:
        target_names = None

    # Generate classification report
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0 # Set zero_division to 0 or 1
    )

    # Handle the 'accuracy' key properly for DataFrame conversion
    if 'accuracy' in report_dict:
        accuracy_value = report_dict.pop('accuracy') # Remove scalar accuracy
        # Create a structure similar to other rows for DataFrame consistency
        report_dict['accuracy'] = {
            'precision': np.nan, # Accuracy doesn't have precision/recall
            'recall': np.nan,
            'f1-score': accuracy_value, # Store accuracy value here
            'support': report_dict['macro avg']['support'] # Use total support
        }


    report_df = pd.DataFrame(report_dict).transpose().round(2)

    # Ensure support column is integer
    if "support" in report_df.columns:
        # Handle potential NaN introduced for the accuracy row
        report_df["support"] = report_df["support"].fillna(0).astype(int)


    if print_report:
        print("\n--- Multi-Omics Evaluation Report ---")
        print(report_df.to_string())

    if return_preds:
        return avg_loss, {'true': all_labels, 'pred': all_preds, 'report': report_df}
    else:
        return avg_loss, report_df





# Global color configuration for consistency
OMICS_COLORS = {
    'mrna': '#EF767A',  # Reddish
    'cnv': '#456990',   # Bluish
    'snv': '#48C0AA',   # Greenish
    'fusion': '#FFA500' # Orange
}
# Rename keys to match common usage if needed
OMICS_COLORS['CNV'] = OMICS_COLORS['cnv']
OMICS_COLORS['SNV'] = OMICS_COLORS['snv']


def analyze_fusion_layer(fusion_model, omics_outputs, mappings_dict=None, save_path=None):
    """
    Analyzes contributions within the fusion layers of EnhancedAttentionFusion.

    Args:
        fusion_model (EnhancedAttentionFusion): The trained fusion model.
        omics_outputs (dict): Dictionary of outputs from each omics sub-network
                              (e.g., {'mrna': tensor, 'cnv': tensor}).
        mappings_dict (dict, optional): Dictionary containing mappings (not directly used
                                       in this specific function but kept for signature consistency).
        save_path (str or Path, optional): Path to save the contribution plot.

    Returns:
        dict: Dictionary containing contribution scores for each omics type.
    """
    print("Analyzing fusion layer contributions...")
    # Ensure model is on CPU for numpy conversion, handle potential errors
    try:
        fusion_model.cpu() # Move model to CPU temporarily for analysis

        # Get weights - check if layers exist before accessing weights
        feature_attn_weights = None
        if hasattr(fusion_model, 'feature_attention') and len(fusion_model.feature_attention) > 3 and isinstance(fusion_model.feature_attention[3], nn.Linear):
            feature_attn_weights = fusion_model.feature_attention[3].weight.data.numpy()

        cross_attn_weights = None
        if hasattr(fusion_model, 'cross_attn') and hasattr(fusion_model.cross_attn, 'in_proj_weight'):
             # MultiheadAttention combines query, key, value weights. We might need to split them.
             # For simplicity, using the whole in_proj_weight as an indicator.
             # Shape is typically (3 * embed_dim, embed_dim). We need to be careful interpreting this.
             cross_attn_weights = fusion_model.cross_attn.in_proj_weight.data.numpy()


        fusion_weights = None
        if hasattr(fusion_model, 'fusion') and len(fusion_model.fusion) > 0 and isinstance(fusion_model.fusion[0], nn.Linear):
            fusion_weights = fusion_model.fusion[0].weight.data.numpy() # Weight of the first linear layer in fusion block

    except Exception as e:
        print(f"Error accessing model weights for fusion analysis: {e}")
        return {} # Return empty if weights can't be accessed

    contribution_scores = {}
    num_omics = len(omics_outputs)
    embed_dim = next(iter(omics_outputs.values())).shape[-1] # Get embed_dim from output tensor

    # Calculate contribution scores (handle cases where weights might be None)
    for omics_idx, omics_name in enumerate(omics_outputs.keys()):
        start = omics_idx * embed_dim
        end = start + embed_dim

        # Use np.nanmean/nansum to ignore NaNs if they occur
        with np.errstate(invalid='ignore', divide='ignore'): # Ignore warnings for mean/sum of empty/NaN slices
            feat_contrib = np.nanmean(np.abs(feature_attn_weights[:, start:end])) if feature_attn_weights is not None else np.nan
            # Cross-attention contribution is harder to interpret directly from in_proj_weight.
            # A simplified approach: take the mean of the relevant slice.
            # Note: This is a simplification. Proper analysis might involve attention scores.
            cross_contrib = np.nanmean(np.abs(cross_attn_weights[:embed_dim, start:end])) if cross_attn_weights is not None else np.nan # Only query part?
            fusion_contrib = np.nansum(np.abs(fusion_weights[:, start:end])) if fusion_weights is not None else np.nan # Input contribution to first fusion layer


        # Combine scores (handle NaNs in weighting)
        weights = {'feature': 0.4, 'cross': 0.3, 'fusion': 0.3}
        total_contrib = (weights['feature'] * (feat_contrib if not np.isnan(feat_contrib) else 0) +
                         weights['cross'] * (cross_contrib if not np.isnan(cross_contrib) else 0) +
                         weights['fusion'] * (fusion_contrib if not np.isnan(fusion_contrib) else 0))
        # Normalize total weight if some components were NaN
        total_weight = (weights['feature'] * (not np.isnan(feat_contrib)) +
                        weights['cross'] * (not np.isnan(cross_contrib)) +
                        weights['fusion'] * (not np.isnan(fusion_contrib)))
        total_contrib = total_contrib / total_weight if total_weight > 0 else np.nan


        contribution_scores[omics_name] = {
            'feature_attention': feat_contrib,
            'cross_attention': cross_contrib, # Simplified interpretation
            'fusion_layer': fusion_contrib,
            'total': total_contrib
        }

    # --- Visualization ---
    if not contribution_scores:
         print("No contribution scores calculated.")
         return contribution_scores

    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False) # Don't share Y axis initially
        metrics_to_plot = ['feature_attention', 'cross_attention', 'fusion_layer']
        titles = ['Feature Attention Contribution', 'Cross Attention Contribution (Simplified)', 'Fusion Layer Input Contribution']
        plot_successful = False

        for ax, metric, title in zip(axes, metrics_to_plot, titles):
            omics_names = list(contribution_scores.keys())
            values = [contribution_scores[name].get(metric, np.nan) for name in omics_names] # Use .get for safety
            colors = [OMICS_COLORS.get(name, '#808080') for name in omics_names] # Use default gray if key missing

            # Filter out NaN values for plotting this metric
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            if not valid_indices:
                ax.set_title(f"{title}\n(No data)", fontsize=12)
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue # Skip plotting if no valid data

            plot_names = [omics_names[i] for i in valid_indices]
            plot_values = [values[i] for i in valid_indices]
            plot_colors = [colors[i] for i in valid_indices]


            bars = ax.bar(plot_names, plot_values, color=plot_colors)
            ax.set_title(title, fontsize=12)

            # Set Y limits based on valid data
            max_val = max(plot_values) if plot_values else 0
            ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1)
            ax.tick_params(axis='x', rotation=45)


            # Add numerical labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', # More precision
                        ha='center', va='bottom', fontsize=9)
            plot_successful = True # Mark that at least one plot was made

        if plot_successful:
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

             # Save results
             if save_path:
                 save_path = Path(save_path)
                 save_path.parent.mkdir(parents=True, exist_ok=True)
                 plt.savefig(save_path, bbox_inches='tight')
                 print(f"Saved fusion contribution plot to {save_path}")
             plt.show() # Display the plot
        else:
             print("No valid contribution data to plot.")

        plt.close(fig) # Close the figure explicitly


    except Exception as e:
        print(f"Error during visualization of fusion contributions: {e}")
        if 'fig' in locals(): plt.close(fig) # Ensure figure is closed on error

    return contribution_scores


def calculate_integrated_gradients(model, input_data_dict, target_class_idx, baseline_mode='zero', steps=50, target_output_idx=None):
    """
    Calculates Integrated Gradients for multi-omics models.

    Args:
        model (nn.Module): The trained PyTorch model.
        input_data_dict (dict): Dictionary {'omics_name': sample_tensor} for ONE sample.
                                Tensors should be on the correct device.
        target_class_idx (int): The index of the target class for attribution.
        baseline_mode (str): 'zero' for zero baseline, 'gaussian' for Gaussian noise baseline.
        steps (int): Number of steps in the integration approximation.
        target_output_idx (int, optional): If the model has multiple outputs per class,
                                          specify which one to target. Defaults to None.

    Returns:
        dict: Dictionary {'omics_name': integrated_gradients_array}.
    """
    print("Calculating Integrated Gradients...")
    device = next(model.parameters()).device
    model.eval() # Set model to evaluation mode

    # --- Baseline Definition ---
    baseline_dict = {}
    if baseline_mode == 'zero':
        baseline_dict = {k: torch.zeros_like(v) for k, v in input_data_dict.items()}
    elif baseline_mode == 'gaussian':
         baseline_dict = {k: torch.randn_like(v) for k, v in input_data_dict.items()}
    # Add more baseline modes if needed (e.g., mean, random sample)
    else:
        raise ValueError(f"Unsupported baseline mode: {baseline_mode}")

    # --- Interpolation ---
    # input - baseline difference
    input_baseline_diff = {k: input_data_dict[k] - baseline_dict[k] for k in input_data_dict}

    # Store gradients for each step
    step_gradients = {k: [] for k in input_data_dict}

    for i in range(steps + 1):
        alpha = i / steps
        # Calculate interpolated input
        interpolated_input = {
            k: (baseline_dict[k] + alpha * input_baseline_diff[k]).requires_grad_(True)
            for k in input_data_dict
        }

        # --- Forward Pass ---
        output = model(interpolated_input) # Get model output for the interpolated input

        # --- Gradient Calculation ---
        model.zero_grad() # Clear previous gradients

        # Select the target output for gradient calculation
        target_output = output[:, target_class_idx]
        if target_output_idx is not None and target_output.ndim > 1:
             target_output = target_output[:, target_output_idx]

        # Calculate gradients w.r.t. the target output
        target_output.sum().backward() # Sum is needed for multi-sample batch (even if batch=1)

        # Store gradients for each input modality
        for k in input_data_dict:
            if interpolated_input[k].grad is not None:
                 step_gradients[k].append(interpolated_input[k].grad.detach().clone())
            else:
                 # Handle cases where gradient might be None (if input not used for target output)
                 step_gradients[k].append(torch.zeros_like(interpolated_input[k]))


    # --- Integration (Trapezoidal Rule Approximation) ---
    integrated_grads = {}
    for k in input_data_dict:
        # Stack gradients along a new dimension (steps, batch, *features)
        grads_tensor = torch.stack(step_gradients[k], dim=0)
        # Average gradients across steps (approximates integral) - using trapezoidal rule is better
        # Trapezoidal approx: (grad_0 + 2*grad_1 + ... + 2*grad_{n-1} + grad_n) / (2*steps)
        trapz_grads = (grads_tensor[1:] + grads_tensor[:-1]) / 2.0
        avg_trapz_grad = torch.mean(trapz_grads, dim=0) # Average over steps dimension

        # Integrated Gradients = (input - baseline) * avg_gradient
        ig = input_baseline_diff[k] * avg_trapz_grad
        integrated_grads[k] = ig.cpu().numpy() # Move to CPU and convert to numpy

    print("Integrated Gradients calculation complete.")
    return integrated_grads


def analyze_pathway_contributions(relation_dfs, feature_importances, aggregation='sum', save_path=None):
    """
    Analyzes pathway contributions based on feature importances.

    Args:
        relation_dfs (dict): Dictionary {'omics_name': pd.DataFrame}, where each DataFrame
                             maps features ('input_features') to pathways ('output_nodes').
        feature_importances (dict): Dictionary {'omics_name': np.array} containing feature
                                     importance scores (e.g., from Integrated Gradients).
                                     The order must match the features in the original data matrix.
        aggregation (str): Method to aggregate feature importances ('sum' or 'mean').
        save_path (str or Path, optional): Path to save the heatmap plot.

    Returns:
        dict: Dictionary {'omics_name': {'pathway_name': contribution_score}}.
    """
    print("Analyzing pathway contributions...")
    pathway_contrib_all_omics = {}

    for omics_name, importance_scores in feature_importances.items():
        if omics_name not in relation_dfs:
            print(f"Warning: No relation DataFrame found for {omics_name}. Skipping pathway analysis.")
            continue

        relation_df = relation_dfs[omics_name]
        if relation_df.empty:
             print(f"Warning: Relation DataFrame for {omics_name} is empty. Skipping.")
             continue

        # Create a mapping from input feature name to its importance score
        # This assumes importance_scores correspond to the columns of the *original* matrix
        # used to generate the relation_dfs (e.g., the matrix before index_cleaning for Reactome masks)
        # If feature_importances comes from IG calculated on aligned/resampled data, mapping needs care.
        # Let's assume feature_importances keys *match* the 'input_features' in relation_df for now.
        # --> This needs careful verification based on how feature_importances are generated!
        if isinstance(importance_scores, dict): # If importance is already a feature->score map
             feature_score_map = importance_scores
        else: # Assume it's an array corresponding to features in relation_df['input_features'].unique() ?
             # This mapping is ambiguous if importance scores don't have names.
             # Safest assumption: importance_scores is an array matching the *order* of unique features
             # that were *actually used* as input to the model.
             # For pathway analysis, we need mapping back to the 'input_features' listed in relation_df.
             # --> Let's refine: Assume feature_importances is a dict: {feature_name: score}
             if not isinstance(importance_scores, dict):
                  print(f"Warning: Feature importances for {omics_name} are not a dictionary. Cannot reliably map to pathways. Skipping.")
                  continue
             feature_score_map = importance_scores


        # Aggregate scores per pathway
        pathway_aggregation = defaultdict(list)
        for _, row in relation_df.iterrows():
            feature_name = row['input_features']
            pathway_name = row['output_nodes']
            # Get importance score for this feature, default to 0 if not found
            score = feature_score_map.get(feature_name, 0.0)
            pathway_aggregation[pathway_name].append(score)

        # Calculate final contribution based on aggregation method
        pathway_contrib_this_omics = {}
        if aggregation == 'sum':
            pathway_contrib_this_omics = {name: np.sum(scores) for name, scores in pathway_aggregation.items()}
        elif aggregation == 'mean':
            pathway_contrib_this_omics = {name: np.mean(scores) if scores else 0 for name, scores in pathway_aggregation.items()}
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        # Normalize contributions within this omics type (optional, but good for comparison)
        total_contribution = sum(abs(v) for v in pathway_contrib_this_omics.values()) # Sum of absolute values
        if total_contribution > 0:
             pathway_contrib_all_omics[omics_name] = {
                 name: contrib / total_contribution
                 for name, contrib in pathway_contrib_this_omics.items()
             }
        else:
             pathway_contrib_all_omics[omics_name] = pathway_contrib_this_omics # Keep as is if total is zero

    # --- Visualization ---
    if not pathway_contrib_all_omics:
        print("No pathway contributions calculated.")
        return pathway_contrib_all_omics

    try:
        # Create DataFrame for heatmap (handle potential missing omics types)
        plot_df = pd.DataFrame(pathway_contrib_all_omics).fillna(0).T # Transpose for heatmap

        if plot_df.empty:
             print("No data available for pathway contribution heatmap.")
             return pathway_contrib_all_omics


        plt.figure(figsize=(min(20, plot_df.shape[1] * 0.5), max(8, plot_df.shape[0] * 0.5))) # Dynamic figsize
        sns.heatmap(plot_df,
                    annot=True, fmt=".2f", # Annotate with scores
                    cmap=sns.light_palette(OMICS_COLORS.get('snv', '#48C0AA'), as_cmap=True), # Use a default color
                    linewidths=.5, cbar=True, # Add color bar
                    annot_kws={"size": 8}) # Adjust annotation font size
        plt.title("Relative Pathway Contributions per Omics Type", fontsize=14)
        plt.xlabel("Pathways", fontsize=10)
        plt.ylabel("Omics Type", fontsize=10)
        plt.xticks(rotation=90, fontsize=8) # Rotate pathway names if many
        plt.yticks(rotation=0, fontsize=8)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved pathway contribution heatmap to {save_path}")
        plt.show() # Display the plot
        plt.close() # Close the figure

    except Exception as e:
        print(f"Error during visualization of pathway contributions: {e}")
        plt.close() # Ensure figure is closed on error

    print("Pathway contribution analysis complete.")
    return pathway_contrib_all_omics

# Note: The original `enhanced_explain` function was complex and made many assumptions
# about mappings and intermediate outputs. It's often better to call the specific
# analysis functions (`analyze_fusion_layer`, `calculate_integrated_gradients`,
# `analyze_pathway_contributions`) separately with the correct inputs after training.
# We will keep it here but recommend calling the individual functions from analyze.py

def enhanced_explain(fusion_model, multi_omics_data_dict, relation_dfs, mappings_dict, output_dir="./results", sample_idx=0, target_class_idx=0):
    """
    Enhanced explanation workflow for mid-fusion models.
    Calls fusion layer analysis, Integrated Gradients, and pathway analysis.

    Args:
        fusion_model (nn.Module): Trained mid-fusion model (e.g., EnhancedAttentionFusion).
        multi_omics_data_dict (dict): Dictionary {'omics_name': ALL_SAMPLES_DATAFRAME_OR_ARRAY}.
        relation_dfs (dict): Dictionary {'omics_name': relation_df} mapping features to pathways.
                             Features should match original data used for mask/relation generation.
        mappings_dict (dict): Dictionary possibly containing original Reactome mappings.
                             NOTE: Interpretation for sub-networks might be complex.
        output_dir (str): Directory to save analysis results.
        sample_idx (int): Index of the sample to use for Integrated Gradients.
        target_class_idx (int): Target class index for Integrated Gradients.

    Returns:
        dict: Dictionary containing analysis results.
    """
    print("\n--- Starting Enhanced Explanation Workflow ---")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = next(fusion_model.parameters()).device
    fusion_model.eval() # Ensure model is in eval mode

    # --- 1. Fusion Layer Analysis ---
    # Need sub-network outputs for the *entire dataset* or a representative batch
    print("Calculating sub-network outputs for fusion analysis...")
    omics_outputs_all = {}
    with torch.no_grad():
        for omics_name, data in multi_omics_data_dict.items():
            # Convert full dataset to tensor for this step
            if isinstance(data, pd.DataFrame):
                data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
            else:
                data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

            if omics_name in fusion_model.omics_networks:
                 omics_outputs_all[omics_name] = fusion_model.omics_networks[omics_name](data_tensor)
            else:
                 print(f"Warning: No sub-network found for {omics_name} in fusion model.")


    fusion_contribution = {}
    if omics_outputs_all:
         fusion_contribution = analyze_fusion_layer(
             fusion_model,
             omics_outputs_all,
             mappings_dict, # Pass mappings, though not used directly in current analyze_fusion_layer
             save_path=output_path / "fusion_layer_contributions.pdf"
         )
    else:
         print("Skipping fusion layer analysis as no sub-network outputs were generated.")


    # --- 2. Integrated Gradients for Feature Importance ---
    print(f"\nCalculating Integrated Gradients for sample index {sample_idx}...")
    # Select the specific sample for IG
    sample_data_dict = {}
    feature_names_dict = {} # Store feature names for mapping later
    valid_sample = True
    for omics_name, data in multi_omics_data_dict.items():
         if omics_name in fusion_model.omics_networks: # Only process omics used by the model
            if isinstance(data, pd.DataFrame):
                if sample_idx < len(data):
                     sample_data_dict[omics_name] = torch.tensor(data.iloc[[sample_idx]].values, dtype=torch.float32).to(device)
                     feature_names_dict[omics_name] = data.columns.tolist() # Get feature names
                else:
                     print(f"Error: sample_idx {sample_idx} out of bounds for {omics_name} data (size {len(data)}).")
                     valid_sample = False
                     break
            else: # Handle numpy arrays
                if sample_idx < len(data):
                     sample_data_dict[omics_name] = torch.tensor(data[[sample_idx]], dtype=torch.float32).to(device)
                     # We lose feature names if input was numpy array, pathway mapping might fail
                     feature_names_dict[omics_name] = [f"{omics_name}_feat_{j}" for j in range(data.shape[1])]
                     print(f"Warning: Input for {omics_name} is numpy array, using generic feature names.")
                else:
                     print(f"Error: sample_idx {sample_idx} out of bounds for {omics_name} data (size {len(data)}).")
                     valid_sample = False
                     break
         else:
             print(f"Skipping {omics_name} for IG as it's not in the fusion model's networks.")


    integrated_grads = {}
    feature_importance_dict = {} # For pathway analysis {omics: {feature_name: score}}
    if valid_sample and sample_data_dict:
         try:
            integrated_grads = calculate_integrated_gradients(
                fusion_model,
                sample_data_dict,
                target_class_idx=target_class_idx
            )
            print("Integrated Gradients calculated.")

            # Convert IG results into feature_name -> score dictionary
            for omics_name, ig_array in integrated_grads.items():
                 if omics_name in feature_names_dict:
                      if ig_array.shape[1] == len(feature_names_dict[omics_name]):
                           # Sum IG across the sample dimension (if batch was > 1, though usually 1 for IG)
                           # Take absolute value or keep sign? Usually absolute for importance ranking.
                           feature_importance_dict[omics_name] = dict(zip(feature_names_dict[omics_name], np.abs(ig_array.sum(axis=0))))
                      else:
                           print(f"Warning: Mismatch between IG shape ({ig_array.shape}) and feature name count ({len(feature_names_dict[omics_name])}) for {omics_name}.")
                 else:
                      print(f"Warning: Could not find feature names for {omics_name} to create importance dictionary.")

         except Exception as e:
            print(f"Error calculating Integrated Gradients: {e}")
    else:
         print("Skipping Integrated Gradients calculation due to invalid sample or missing data.")


    # --- 3. Pathway Contribution Analysis ---
    pathway_contrib = {}
    if feature_importance_dict: # Check if we have importance scores to analyze
         # Ensure relation_dfs keys match feature_importance_dict keys
         valid_relation_dfs = {k: v for k, v in relation_dfs.items() if k in feature_importance_dict}
         if valid_relation_dfs:
              pathway_contrib = analyze_pathway_contributions(
                  relation_dfs=valid_relation_dfs,
                  feature_importances=feature_importance_dict, # Pass the dict {feature: score}
                  aggregation='sum', # or 'mean'
                  save_path=output_path / "pathway_contributions_ig.pdf"
              )
         else:
              print("Skipping pathway analysis: No matching relation DataFrames found for omics with importance scores.")

    else:
         print("Skipping pathway analysis as no feature importances were generated.")

    # --- 4. Consolidate Results ---
    # The original `enhanced_explain` also included sub-network node importance.
    # This is complex to extract reliably without modifying the models or using hooks.
    # We focus on fusion layer, IG, and pathway analysis based on IG here.
    results = {
        'fusion_contribution': fusion_contribution,
        'integrated_gradients': integrated_grads, # Raw IG arrays
        'feature_importance_ig': feature_importance_dict, # Mapped feature importances
        'pathway_contribution_ig': pathway_contrib
    }

    print("--- Enhanced Explanation Workflow Finished ---")
    return results