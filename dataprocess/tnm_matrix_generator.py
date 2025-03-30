# dataprocess/tnm_matrix_generator.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fill_missing_with_dbscan(data, eps=0.5, min_samples=5):
    """
    Fills missing values in a matrix using DBSCAN clustering to impute values
    within clusters, preserving the original index and column names.

    Args:
        data (pd.DataFrame): Input matrix (DataFrame), with rows as samples,
                             columns as features.  Allows missing values (np.nan).
                             The index *must* be preserved.
        eps (float): The epsilon parameter for DBSCAN (neighborhood distance).
        min_samples (int): The min_samples parameter for DBSCAN (minimum number of samples in a cluster).

    Returns:
        pd.DataFrame: The filled matrix (DataFrame) with missing values imputed.
                      The original index and column names are preserved.
    """
    # Preserve original index and column names
    original_index = data.index
    original_columns = data.columns

    # Convert to numpy array for efficient processing
    data_values = data.values

    # --- Step 1: Initial Imputation (Mean Imputation) ---
    # Use SimpleImputer for initial imputation to allow scaling and DBSCAN.
    # This avoids issues with NaN values during clustering.
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(data_values) # shape is preserved

    # --- Step 2: Data Scaling ---
    # Standardize data to have zero mean and unit variance for DBSCAN
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed) # shape is preserved

    # --- Step 3: DBSCAN Clustering ---
    # Apply DBSCAN to identify clusters of similar samples.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    clusters = dbscan.fit_predict(data_scaled) # Assign each sample to a cluster (or -1 for noise)

    # --- Step 4: Cluster-Based Imputation ---
    # Impute missing values within each cluster using the mean of the non-missing
    # values in that cluster.
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Skip noise points (assigned to no cluster)
            continue

        # Get indices of samples belonging to the current cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = data_values[cluster_indices]

        # Calculate the mean of each feature within the cluster, ignoring NaNs
        # If all values in a cluster are NaN for a particular feature, use the
        # global mean of that feature.
        cluster_mean = np.nanmean(cluster_data, axis=0)
        cluster_mean = np.where(np.isnan(cluster_mean), np.nanmean(data_values, axis=0), cluster_mean) # Fallback to global if all NaN

        # Impute missing values in the current cluster
        for i in cluster_indices:
            nan_indices = np.isnan(data_values[i]) # Find NaN indices in the sample
            data_values[i, nan_indices] = cluster_mean[nan_indices] # Fill with cluster mean

    # --- Step 5: Handle Unimputed Samples (Noise) ---
    # Impute any remaining missing values (samples that were identified as noise by DBSCAN)
    # using the global mean of each feature.  This is a fallback for samples
    # that didn't fall into any cluster.
    nan_rows = np.isnan(data_values).any(axis=1)  # Identify rows with any remaining NaNs

    if nan_rows.any():
        global_mean = np.nanmean(data_values, axis=0) # Calculate the global mean for each feature
        for i in np.where(nan_rows)[0]: # Iterate over rows with NaNs
            nan_indices = np.isnan(data_values[i]) # Identify NaN indices in the sample
            data_values[i, nan_indices] = global_mean[nan_indices]  # Impute using the global mean

    # --- Step 6: Create DataFrame with Imputed Values ---
    # Convert the filled numpy array back to a DataFrame, preserving
    # the original index and column names.
    filled_data = pd.DataFrame(data_values, index=original_index, columns=original_columns)
    return filled_data


def process_tnm_matrices(mtx, df_label):
    """
    Generates three stage-specific matrices (T, N, M) and corresponding labels
    based on the filled expression matrix and the label DataFrame.

    Args:
        mtx (pd.DataFrame): The filled expression matrix (from fill_missing_with_dbscan),
                             containing sample indices and molecular features.
        df_label (pd.DataFrame): Label DataFrame containing 'short_id' and
                                 'T_label', 'N_label', 'M_label' columns.

    Returns:
        dict: A dictionary containing the three stage-specific matrices and labels:
              {
                  'mtx_t': mtx_t, 'label_t': label_t,
                  'mtx_n': mtx_n, 'label_n': label_n,
                  'mtx_m': mtx_m, 'label_m': label_m
              }
    """
    results = {}

    # --- T Stage Matrix and Labels ---
    valid_t = df_label['T_label'].isin(['TL', 'TH'])  # Boolean mask for valid T stages
    mtx_t = mtx.loc[valid_t]  # Select rows from mtx where T stage is valid
    label_t = df_label.loc[valid_t, 'T_label']  # Select T labels for valid samples
    results['mtx_t'] = mtx_t
    results['label_t'] = label_t

    # --- N Stage Matrix and Labels ---
    valid_n = df_label['N_label'].isin(['NL', 'NH'])
    mtx_n = mtx.loc[valid_n]
    label_n = df_label.loc[valid_n, 'N_label']
    results['mtx_n'] = mtx_n
    results['label_n'] = label_n

    # --- M Stage Matrix and Labels ---
    valid_m = df_label['M_label'].isin(['ML', 'MH'])
    mtx_m = mtx.loc[valid_m]
    label_m = df_label.loc[valid_m, 'M_label']
    results['mtx_m'] = mtx_m
    results['label_m'] = label_m

    return results


def fill_and_generate_tnm_matrices(mtx, df_label, eps=0.5, min_samples=5):
    """
    Main function: Fills missing values in the expression matrix and generates
    T, N, and M stage-specific matrices and corresponding labels.

    Args:
        mtx (pd.DataFrame): Expression matrix (DataFrame) containing missing values.
        df_label (pd.DataFrame): Label DataFrame containing 'short_id' and
                                 'T_label', 'N_label', 'M_label' columns.
        eps (float): The epsilon parameter for DBSCAN (neighborhood distance).
        min_samples (int): The min_samples parameter for DBSCAN (minimum number of samples in a cluster).

    Returns:
        dict: A dictionary containing the three stage-specific matrices and labels:
              {
                  'mtx_t': mtx_t, 'label_t': label_t,
                  'mtx_n': mtx_n, 'label_n': label_n,
                  'mtx_m': mtx_m, 'label_m': label_m
              }
    """
    # Fill missing values using DBSCAN imputation
    filled_mtx = fill_missing_with_dbscan(mtx, eps=eps, min_samples=min_samples)

    # Check for any remaining missing values after imputation
    if filled_mtx.isnull().any().any():
        print("Warning: Missing values still exist in the filled matrix!")
        print("Rows with missing values:")
        print(filled_mtx[filled_mtx.isnull().any(axis=1)])  # Print rows with NaN values
    else:
        print("No missing values in the filled matrix.")

    # Generate T, N, and M stage-specific matrices and labels
    tnm_results = process_tnm_matrices(filled_mtx, df_label)

    return tnm_results


if __name__ == "__main__":
    # Example Usage / Self-Test
    print("Running fill_and_generate_tnm_matrices self-test...")

    # --- Create Dummy Data ---
    # Expression Matrix with Missing Values
    data = {'feature1': [1, 2, np.nan, 4, 5, np.nan],
            'feature2': [6, np.nan, 8, 9, np.nan, 11],
            'feature3': [12, 13, 14, np.nan, 16, 17]}
    index = ['TCGA-A', 'TCGA-B', 'TCGA-C', 'TCGA-D', 'TCGA-E', 'TCGA-F'] # Simplified sample names
    mtx = pd.DataFrame(data, index=index)

    # Label DataFrame (must include short_id)
    label_data = {'short_id': ['TCGA-A', 'TCGA-B', 'TCGA-C', 'TCGA-D', 'TCGA-E', 'TCGA-F'],
                  'T_label': ['TL', 'TH', 'TL', None, 'TH', 'TL'],
                  'N_label': ['NL', 'NH', 'NL', 'NL', None, 'NH'],
                  'M_label': [None, 'ML', 'MH', 'ML', 'MH', 'ML']}
    df_label = pd.DataFrame(label_data)

    print("\nOriginal Expression Matrix (mtx):")
    print(mtx)
    print("\nLabel DataFrame (df_label):")
    print(df_label)

    # --- Run the Full Pipeline ---
    tnm_matrices = fill_and_generate_tnm_matrices(mtx, df_label)

    print("\nGenerated TNM Matrices:")
    for stage, data in tnm_matrices.items():
        print(f"\n{stage}:")
        print(data) # Show content of each matrix/label set

    print("\nSelf-test complete.")