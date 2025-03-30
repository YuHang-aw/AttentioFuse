# config.py

# Dataset selection
dataset = 'LUSC'  # Choose either 'LUSC' or 'LUAD'

# Base data directory (assuming LUAD and LUSC are subdirectories)
data_path = '../data/'

# --- Dataset-Specific Settings ---
if dataset == 'LUSC':
    clinical_file = 'LUSC/LUSC_clinical_SE.csv'
    mrna_file = 'LUSC/TCGA-LUSC_mrna_expr_tpm.csv'
    cnv_file = 'LUSC/all_data_by_genes.txt'
    snv_file = 'LUSC/LUSC_snv.csv'

elif dataset == 'LUAD':
    clinical_file = 'LUAD/clinIndexData.csv' # Fixed name
    mrna_file = 'LUAD/TCGA-LUAD_mrna_expr_tpm.csv'
    cnv_file = 'LUAD/cnv_grouped_by_patient.csv'
    snv_file = 'LUAD/LUAD_snv_mtx.csv'
else:
    raise ValueError("Invalid dataset selection. Choose 'LUSC' or 'LUAD'.")

# --- Other Configuration ---
use_cuda = True  # Whether to use CUDA if available
test_size = 0.2  # Test set size
random_state = 42  # Random seed for reproducibility