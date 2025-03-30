# dataprocess/data_loader.py

import pandas as pd
import numpy as np
import os
from dataprocess.data_cleaning import index_cleaning
from dataprocess.label_processor import extract_labels
from dataprocess.tnm_matrix_generator import fill_and_generate_tnm_matrices, process_tnm_matrices # Import the sub-function too
import config

def load_and_preprocess_data(use_precomputed=False):
    """
    Loads, preprocesses multi-omics data, returning ALL TNM stages.

    Args:
        use_precomputed (bool): If True, load precomputed data files.

    Returns:
        dict: {'mrna': {'T': {'data': df, 'labels': series}, 'N': ..., 'M': ...}, 'cnv': ..., 'snv': ...}
    """
    results = {}
    data_path = config.data_path
    dataset_name = config.dataset # Renamed for clarity

    print(f"--- Loading data for {dataset_name} ---")
    clinical_file = os.path.join(data_path, config.clinical_file)
    clin = pd.read_csv(clinical_file, sep=',', index_col=0)

    # --- Process each omics type ---
    for omics_type in ['mrna', 'cnv', 'snv']:
        print(f"\nProcessing {omics_type.upper()} data...")
        results[omics_type] = {} # Initialize dict for this omics type

        # Determine file paths based on config
        if omics_type == 'mrna':
            raw_file = os.path.join(data_path, config.mrna_file)
            preprocessed_file = os.path.join(data_path, f"preprocessed_mrna_{dataset_name}.csv")
        elif omics_type == 'cnv':
            raw_file = os.path.join(data_path, config.cnv_file)
            preprocessed_file = os.path.join(data_path, f"preprocessed_cnv_{dataset_name}.csv")
        elif omics_type == 'snv':
            raw_file = os.path.join(data_path, config.snv_file)
            preprocessed_file = os.path.join(data_path, f"preprocessed_snv_{dataset_name}.csv")
        else:
            continue # Should not happen

        # --- Load or Preprocess Raw Data ---
        if use_precomputed and os.path.exists(preprocessed_file):
            print(f"Loading preprocessed {omics_type} data...")
            omics_processed = pd.read_csv(preprocessed_file, index_col=0)
        else:
            print(f"Processing raw {omics_type} data...")
            # --- mRNA Specific Preprocessing ---
            if omics_type == 'mrna':
                mrna = pd.read_csv(raw_file, sep=',', index_col=0)
                mrna = mrna.T
                mt_genes = [col for col in mrna.columns if col.startswith('MT-')]
                mrna_cleaned = mrna.drop(columns=mt_genes)
                omics_processed = np.log1p(mrna_cleaned) # Log transform

            # --- CNV Specific Preprocessing ---
            elif omics_type == 'cnv':
                 # CNV needs different separators and processing based on dataset
                 sep = ',' if dataset_name == 'LUAD' else '\t' # Adjust separator
                 cnv = pd.read_csv(raw_file, sep=sep, index_col=0)
                 if dataset_name == 'LUSC':
                     cnv = cnv.T
                     cnv = cnv[2:] # Skip header rows for LUSC format
                 cnv.index = cnv.index.str[:12] # Extract patient ID
                 cnv_grouped = cnv.groupby(cnv.index).mean()
                 cnv_grouped.index = cnv_grouped.index.str[:12] # Ensure index format
                 omics_processed = cnv_grouped.apply(pd.to_numeric, errors='coerce')

            # --- SNV Specific Preprocessing ---
            elif omics_type == 'snv':
                 # SNV processing might differ slightly based on input format if needed
                 snv = pd.read_csv(raw_file, index_col=0) # Assuming index_col=0 works for both
                 exclude = ['Silent', 'Intron', "3\'UTR", "5\'UTR", 'RNA', 'IGR']
                 filtered = snv[~snv['Variant_Classification'].isin(exclude)]
                 # Quality filtering might need adjustment if columns differ
                 if 't_alt_count' in filtered.columns:
                     filtered = filtered[filtered['t_alt_count'] > 5]
                 else:
                      print(f"Warning: 't_alt_count' column not found in {omics_type} data for {dataset_name}. Skipping quality filter.")

                 # Sample barcode column might differ
                 sample_col = 'Tumor_Sample_Barcode' # Assume this standard name
                 if sample_col not in filtered.columns:
                      # Try alternative common names or raise error
                      potential_cols = ['sample', 'Sample_ID'] # Add others if needed
                      found = False
                      for col in potential_cols:
                           if col in filtered.columns:
                                sample_col = col
                                found = True
                                break
                      if not found:
                           raise ValueError(f"Cannot find sample identifier column in {omics_type} data for {dataset_name}.")


                 filtered = filtered.drop_duplicates(subset=[sample_col, 'Hugo_Symbol'])
                 omics_processed = filtered.pivot_table(
                     index=sample_col,
                     columns='Hugo_Symbol',
                     values='Variant_Classification',
                     aggfunc=lambda x: 1 if len(x) > 0 else 0
                 ).fillna(0)

            # Save the processed data
            omics_processed.to_csv(preprocessed_file)
            print(f"Saved processed {omics_type} data.")

        # --- Generate Labels and TNM Splits for this omics type ---
        print(f"Generating labels and TNM splits for {omics_type}...")
        # Impute missing values *before* splitting into TNM stages
        omics_imputed = fill_missing_with_dbscan(omics_processed.dropna(axis=1, how='all')) # Drop fully NaN columns first

        # Extract labels using the imputed data's index
        df_label_omics = extract_labels(omics_imputed, clin)

        # Generate T, N, M matrices and labels *from the imputed data*
        # We use process_tnm_matrices directly to get the dictionary structure
        tnm_splits_omics = process_tnm_matrices(omics_imputed, df_label_omics)

        # Store results for each stage
        results[omics_type]['T'] = {'data': tnm_splits_omics.get('mtx_t', pd.DataFrame()), 'labels': tnm_splits_omics.get('label_t', pd.Series(dtype='object'))}
        results[omics_type]['N'] = {'data': tnm_splits_omics.get('mtx_n', pd.DataFrame()), 'labels': tnm_splits_omics.get('label_n', pd.Series(dtype='object'))}
        results[omics_type]['M'] = {'data': tnm_splits_omics.get('mtx_m', pd.DataFrame()), 'labels': tnm_splits_omics.get('label_m', pd.Series(dtype='object'))}


    print("\n--- All Data Loading and Initial Processing Complete ---")
    return results

# Keep the __main__ block for basic testing
if __name__ == "__main__":
    print("Running data_loader self-test...")
    # Load data for the dataset specified in config.py
    data = load_and_preprocess_data(use_precomputed=False) # Force processing for test
    print("\nLoaded data structure keys:")
    for omics, stages in data.items():
        print(f"- {omics}: {list(stages.keys())}")
        if 'T' in stages:
            print(f"  - T data shape: {stages['T']['data'].shape}")
            print(f"  - T labels head: \n{stages['T']['labels'].head()}")
    print("\nSelf-test complete.")