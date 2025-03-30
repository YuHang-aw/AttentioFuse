# dataprocess/label_processor.py

import pandas as pd
import re # Regular expression operations, used implicitly by str.extract

def extract_labels(mtx, clin):
    """
    Generates a label DataFrame based on the expression matrix and clinical data.

    Args:
        mtx (pd.DataFrame): Matrix with sample identifiers as index
                            (e.g., TCGA-XX-XXXX-01A) and molecular features as columns.
        clin (pd.DataFrame): DataFrame containing patient clinical data.
                             Must include 'submitter_id' (e.g., TCGA-XX-XXXX)
                             and the staging columns: 'ajcc_pathologic_t',
                             'ajcc_pathologic_n', 'ajcc_pathologic_m'.

    Returns:
        pd.DataFrame: A DataFrame with the same index as mtx, containing
                      the patient ID ('short_id') and categorized stage labels
                      ('T_label', 'N_label', 'M_label'). Returns None for labels
                      if staging info is missing or doesn't match categories.
    """
    # Create a new label DataFrame, preserving the index from mtx
    df_Label = pd.DataFrame(index=mtx.index)

    # Helper function to extract the patient ID part (e.g., TCGA-XX-XXXX)
    def extract_patient_id(identifier):
        if isinstance(identifier, str):
            # Extract patient ID matching the 'TCGA-XX-XXXX' format
            match_parts = identifier.split('-')
            if len(match_parts) >= 3:
                return '-'.join(match_parts[:3])
        return identifier # Return original if not string or format mismatch

    df_Label['short_id'] = df_Label.index.map(extract_patient_id)

    # --- Prepare clinical data for mapping ---
    # Make a copy to avoid modifying the original DataFrame potentially
    clin_proc = clin.copy()

    # Extract the standard TCGA patient ID format (TCGA-XX-XXXX) to use as index key.
    # Handle potential NaNs or non-matches in submitter_id gracefully.
    # Using regex ensures we capture the specific format.
    clin_proc['patient_key'] = clin_proc['submitter_id'].str.extract(r'(TCGA-\w{2}-\w{4})', expand=False)

    # Drop rows where the patient key couldn't be extracted
    clin_proc = clin_proc.dropna(subset=['patient_key'])

    # If there are duplicate patient IDs, keep only the first occurrence.
    # You might need a different strategy depending on your data (e.g., check for consistency).
    clin_proc = clin_proc.drop_duplicates(subset=['patient_key'], keep='first')

    # Convert clinical data columns to dictionaries for faster mapping using the extracted key
    # Using .to_dict() on a Series after set_index is efficient.
    t_dict = clin_proc.set_index('patient_key')['ajcc_pathologic_t'].to_dict()
    n_dict = clin_proc.set_index('patient_key')['ajcc_pathologic_n'].to_dict()
    m_dict = clin_proc.set_index('patient_key')['ajcc_pathologic_m'].to_dict()

    # Map the original T, N, M stage values using the extracted short_id
    df_Label['T'] = df_Label['short_id'].map(t_dict)
    df_Label['N'] = df_Label['short_id'].map(n_dict)
    df_Label['M'] = df_Label['short_id'].map(m_dict)

    # --- Helper functions to categorize T, N, M stages ---
    # These functions handle potential NaN values and variations in notation.

    def categorize_t(value):
        if pd.isna(value):
            return None # Use None for missing/unmappable values
        value_str = str(value).strip().upper() # Convert to uppercase string and remove whitespace
        if 'TIS' in value_str or value_str.startswith('T1') or value_str.startswith('T2'):
            return 'TL' # T Low stage
        elif value_str.startswith('T3') or value_str.startswith('T4'):
            return 'TH' # T High stage
        # Decide how to handle ambiguous values like TX, T0, etc. Currently returns None.
        return None # Return None if not matching known categories

    def categorize_n(value):
        if pd.isna(value):
            return None
        value_str = str(value).strip().upper()
        if value_str.startswith('N0'):
            return 'NL' # N Low stage (node negative)
        elif value_str.startswith('N1') or value_str.startswith('N2') or value_str.startswith('N3'):
            return 'NH' # N High stage (node positive)
        # Handle NX?
        return None

    def categorize_m(value):
        if pd.isna(value):
            return None
        value_str = str(value).strip().upper()
        # Handle variations like cM0, pM0
        if value_str.startswith('M0') or 'M0' in value_str: # More robust check for M0 variants
             return 'ML' # M Low stage (no distant metastasis)
        elif value_str.startswith('M1'):
             return 'MH' # M High stage (distant metastasis)
        # Handle MX?
        return None

    # Apply categorization functions to create the final label columns
    df_Label['T_label'] = df_Label['T'].apply(categorize_t)
    df_Label['N_label'] = df_Label['N'].apply(categorize_n)
    df_Label['M_label'] = df_Label['M'].apply(categorize_m)

    # Select and return only the patient ID and the final label columns
    df_Label_final = df_Label[['short_id', 'T_label', 'N_label', 'M_label']]

    return df_Label_final

# You can add an optional block for testing the script directly
if __name__ == "__main__":
    # This block executes only when the script is run directly (e.g., python dataprocess/label_processor.py)
    # It's useful for basic testing.
    print("Testing extract_labels function...")

    # Create dummy dataframes for testing
    dummy_mtx_index = ['TCGA-A1-A0SD-01A-11R-A085-07', 'TCGA-A2-A0CM-01A-11R-A085-07', 'TCGA-A2-A0D2-06A-12R-A034-07']
    dummy_mtx = pd.DataFrame({'gene1': [1, 2, 3], 'gene2': [4, 5, 6]}, index=dummy_mtx_index)

    dummy_clin_data = {
        'submitter_id': ['TCGA-A1-A0SD', 'TCGA-A2-A0CM', 'TCGA-B3-B0XY'],
        'ajcc_pathologic_t': ['T2a', 'T4', 'T1'],
        'ajcc_pathologic_n': ['N0 (i+)', 'N1', 'NX'],
        'ajcc_pathologic_m': ['M0', 'M1a', None]
    }
    dummy_clin = pd.DataFrame(dummy_clin_data)

    print("\nDummy Input Matrix (mtx):")
    print(dummy_mtx)
    print("\nDummy Input Clinical Data (clin):")
    print(dummy_clin)

    # Call the function
    extracted_labels = extract_labels(dummy_mtx, dummy_clin)

    print("\nGenerated Labels:")
    print(extracted_labels)