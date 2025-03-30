# model/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

# Data loading and initial processing
from dataprocess.data_loader import load_and_preprocess_data
# Data cleaning
from dataprocess.data_cleaning import index_cleaning
# Label processing and TNM generation
from dataprocess.label_processor import extract_labels
from dataprocess.tnm_matrix_generator import fill_and_generate_tnm_matrices, process_tnm_matrices
# Workflow processing (Early Fusion - might not be used if WORKFLOW='mid_fusion')
# from dataprocess.workflow_processor import process_early_fusion_workflow
# Model building utilities (includes mid-fusion models)
from model.model_builder import process_layers_and_build_masks, MaskedMLP, OmicsSubNetwork, EnhancedAttentionFusion, DirectFusionNetwork, MaskedDirectFusionNetwork # Added Masked version back
# Model training and evaluation utilities (includes multi-omics versions)
from model.model_utils import clean_and_split_data, train_model, evaluate, evaluate_models, MultiOmicsDataset, prepare_multi_omics_data, train_multi_omics_model, evaluate_multi_omics
# Reactome utilities
from model.reactome_utils import reactome_net, generate_layer_relation_df
# Resampling tools
from imblearn.over_sampling import BorderlineSMOTE
# Weight initialization utility
from model.model_utils import init_weights
# Configuration
import config
from pathlib import Path # For saving results


# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
# Ensure WORKFLOW is set if needed elsewhere, but this script now focuses on Mid-Fusion based on user code
TARGET_STAGE_KEY = 'T' # Choose 'T', 'N', or 'M'
TARGET_DATA_KEY = 'data'
TARGET_LABEL_KEY = 'labels'
REACTOME_CLEANING_LAYER_INDEX = 4 # Which Reactome layer DF to use for index_cleaning (0-based index)

DATA_TYPE_FOR_REACTOME = 'mrna' # Assumed base for Reactome structure generation
N_REACTOME_LEVELS = 4 # Number of *hidden* layers derived from Reactome (implies N+1 original layers needed)
OMICS_TYPES = ['mrna', 'cnv', 'snv'] # Omics to use in mid-fusion

# Mid-fusion specific
EMBED_DIM = 29
ATTN_HEADS = 1

# Training Hyperparameters
EPOCHS = 200 # From user code
BATCH_SIZE = 128 # From user code
LR = 0.01 # From user code
EARLY_STOPPING = 20 # From user code
RANDOM_STATE = config.random_state
RESULTS_DIR = Path("./results") # Base directory for saving models/results
# --- End Configuration ---

def main():
    print(f"--- Starting Mid-Fusion Training Pipeline ---")
    print(f"Selected Dataset: {config.dataset}")
    print(f"Target Stage: {TARGET_STAGE_KEY}")
    print(f"Device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Create results dir

    # --- 1. Load Initial Data (Raw/Minimally Processed) ---
    # We need data before TNM splitting for this workflow
    print("Loading initial data (forcing raw processing)...")
    initial_data_full = load_and_preprocess_data(use_precomputed=False) # Returns dict {omics: {stage: {data: df, labels: series}}}

    # --- 2. Generate Reactome Layers ---
    print("\nGenerating Reactome layers...")
    # Need N_REACTOME_LEVELS + 1 layers for N hidden masks + input mask
    original_reactome_layers = reactome_net.get_layers(n_levels=N_REACTOME_LEVELS + 1)
    original_reactome_dfs = generate_layer_relation_df(original_reactome_layers)
    if len(original_reactome_dfs) <= REACTOME_CLEANING_LAYER_INDEX:
        raise ValueError(f"Not enough Reactome layers generated ({len(original_reactome_dfs)}) to use layer index {REACTOME_CLEANING_LAYER_INDEX} for cleaning.")
    reactome_cleaning_df = original_reactome_dfs[REACTOME_CLEANING_LAYER_INDEX]
    print(f"Using Reactome layer {REACTOME_CLEANING_LAYER_INDEX+1} for feature cleaning.")

    # --- 3. Mid-Fusion Preprocessing Loop (Per Omics) ---
    print("\n--- Starting Per-Omics Preprocessing for Mid-Fusion ---")
    processed_omics_data = {} # Store cleaned data {'omics': df}
    omics_masks_dict = {}     # Store masks {'omics': [mask1, mask2,...]}
    omics_mappings_dict = {}  # Store mappings {'omics': [(map_in, map_out),...]}
    omics_relation_dfs_dict = {} # Store relation dfs after cleaning {'omics': df}
    aligned_labels_dict = {} # Store aligned labels per stage {'T': series, 'N': series, 'M': series}

    # Need clinical data separately for label extraction if done here
    clin_data = pd.read_csv(os.path.join(config.data_path, config.clinical_file), sep=',', index_col=0)

    primary_labels_extracted = False
    for stage_key_loop in ['T', 'N', 'M']: # Process labels for all stages initially
        print(f"\nProcessing Labels for Stage: {stage_key_loop}")
        # Use a reference omics type (e.g., mRNA) to get labels for this stage
        ref_omics_data = initial_data_full[DATA_TYPE_FOR_REACTOME][stage_key_loop][TARGET_DATA_KEY]
        if ref_omics_data.empty:
            print(f"Warning: Reference omics data for label extraction is empty for stage {stage_key_loop}. Skipping label generation.")
            aligned_labels_dict[stage_key_loop] = pd.Series(dtype='object')
            continue
        # Generate labels based on the reference omics data index for this stage
        stage_labels = extract_labels(ref_omics_data, clin_data) # Use raw clin data
        aligned_labels_dict[stage_key_loop] = stage_labels[f'{stage_key_loop}_label'] # Extract the specific label column

        if stage_key_loop == TARGET_STAGE_KEY:
             primary_labels_extracted = True

    if not primary_labels_extracted or aligned_labels_dict[TARGET_STAGE_KEY].empty:
         raise ValueError(f"Failed to extract labels for the target stage {TARGET_STAGE_KEY}.")

    # Now process data for the TARGET stage
    print(f"\n--- Processing Omics Data for Target Stage: {TARGET_STAGE_KEY} ---")
    target_labels = aligned_labels_dict[TARGET_STAGE_KEY]

    for omics_name in OMICS_TYPES:
        print(f"\nProcessing {omics_name} for stage {TARGET_STAGE_KEY}...")

        # Get the data matrix for the target stage
        X_omics_stage = initial_data_full[omics_name][TARGET_STAGE_KEY][TARGET_DATA_KEY]
        if X_omics_stage.empty:
            print(f"  Skipping {omics_name} - data is empty for stage {TARGET_STAGE_KEY}.")
            continue

        # a) Clean features using Reactome layer
        print(f"  Cleaning {omics_name} features...")
        X_omics_cleaned, omics_relation_df = index_cleaning(X_omics_stage, reactome_cleaning_df.copy(), 'input_features') # Use copy of cleaning df
        if X_omics_cleaned.empty:
             print(f"  Skipping {omics_name} - data is empty after cleaning for stage {TARGET_STAGE_KEY}.")
             continue
        omics_relation_dfs_dict[omics_name] = omics_relation_df # Store relation df after cleaning

        # b) Generate Reactome masks based on *cleaned* data
        print(f"  Generating masks for {omics_name}...")
        # process_layers_and_build_masks expects a list of layer DFs
        # We need to pass the *original* reactome DFs, but use the *cleaned* matrix for initial alignment
        omics_masks, omics_mappings = process_layers_and_build_masks(original_reactome_dfs, X_omics_cleaned)
        omics_masks_dict[omics_name] = omics_masks
        omics_mappings_dict[omics_name] = omics_mappings
        print(f"  Generated {len(omics_masks)} masks for {omics_name}.")

        # c) Resampling (using BorderlineSMOTE) - Align labels first!
        print(f"  Resampling {omics_name} data...")
        common_idx_resample = X_omics_cleaned.index.intersection(target_labels.index)
        if common_idx_resample.empty:
            print(f"  Skipping resampling for {omics_name} - no common samples with labels.")
            processed_omics_data[omics_name] = X_omics_cleaned # Store un-resampled cleaned data
            continue

        X_omics_aligned = X_omics_cleaned.loc[common_idx_resample]
        y_labels_aligned = target_labels.loc[common_idx_resample]

        # Check for sufficient samples for SMOTE
        if len(y_labels_aligned.unique()) < 2 or y_labels_aligned.value_counts().min() < 2:
             print(f"  Skipping resampling for {omics_name} - insufficient samples or classes for SMOTE.")
             processed_omics_data[omics_name] = X_omics_aligned
             continue # Store aligned but not resampled


        smote = BorderlineSMOTE(random_state=RANDOM_STATE)
        try:
            X_resampled, y_resampled = smote.fit_resample(X_omics_aligned, y_labels_aligned)
            # Convert back to DataFrame, preserving column order and using resampled index
            processed_omics_data[omics_name] = pd.DataFrame(X_resampled, index=y_resampled.index, columns=X_omics_aligned.columns)
            # Update target labels only once with the first successful resampling
            if TARGET_STAGE_KEY in processed_omics_data: # Check if first omics was processed
                 target_labels = y_resampled # Update target labels to the resampled ones
            print(f"  Resampled {omics_name} shape: {processed_omics_data[omics_name].shape}")
        except Exception as e:
            print(f"  Error during SMOTE for {omics_name}: {e}. Using data before resampling.")
            processed_omics_data[omics_name] = X_omics_aligned # Store aligned but not resampled

    # Filter out omics types that failed processing
    final_omics_data_for_training = {
         name: data for name, data in processed_omics_data.items() if not data.empty
    }
    if not final_omics_data_for_training:
        raise SystemExit("Error: No omics data available after processing. Exiting.")

    print("\n--- Per-Omics Preprocessing Complete ---")

    # --- 4. Instantiate Sub-Networks ---
    print("\nInstantiating Omics Sub-Networks...")
    omics_networks = {}
    missing_masks = False
    for omics_name, X_data in final_omics_data_for_training.items():
        input_size = X_data.shape[1]
        if omics_name not in omics_masks_dict or len(omics_masks_dict[omics_name]) <= N_REACTOME_LEVELS:
            print(f"Error: Insufficient masks generated for {omics_name} (needed {N_REACTOME_LEVELS+1}, got {len(omics_masks_dict.get(omics_name,[]))}). Cannot create sub-network.")
            missing_masks = True
            continue # Skip this omics type

        # Extract relevant masks based on N_REACTOME_LEVELS
        # Assuming masks[0] is input->L1, masks[1] L1->L2 ... masks[N] LN->LN+1
        # Hidden layers are L1 to LN. We need N masks for N hidden layers.
        # Input size is defined by the *data* (X_data.shape[1])
        # Hidden sizes are defined by the *output* dimension of preceding mask/layer
        # Example: L1 size = masks[0].shape[0], L2 size = masks[1].shape[0], ... LN size = masks[N-1].shape[0]
        subnet_masks = omics_masks_dict[omics_name][:N_REACTOME_LEVELS] # Masks for the hidden layers
        subnet_hidden_sizes = [m.shape[0] for m in subnet_masks] # Output size of each layer

        # Verify input dimension of first mask matches data dimension
        # The first mask (omics_masks[0]) connects original features to the first hidden layer.
        # Its input dim (shape[1]) should conceptually match input_size.
        # However, process_layers_and_build_masks cleans features *before* building mask[0].
        # Let's trust the input_size derived from the *final* data being fed in.
        # The *masks passed* should correspond to the layers *within* the subnet.

        # Masks passed to OmicsSubNetwork should correspond to transitions BETWEEN layers
        # Layer 1 (input -> hidden1) needs mask 0
        # Layer 2 (hidden1 -> hidden2) needs mask 1
        # ...
        # Layer N (hiddenN-1 -> hiddenN) needs mask N-1
        # So, pass subnet_masks = omics_masks_dict[omics_name][:N_REACTOME_LEVELS]

        print(f"  Creating subnet for {omics_name}: Input={input_size}, Hidden={subnet_hidden_sizes}, Output={EMBED_DIM}")
        omics_networks[omics_name] = OmicsSubNetwork(
            input_size=input_size,
            hidden_sizes=subnet_hidden_sizes,
            masks=subnet_masks, # Pass the N masks for the N hidden layers
            output_size=EMBED_DIM,
            device=DEVICE
        )

    if missing_masks or not omics_networks:
        raise SystemExit("Error: Could not create sub-networks due to missing masks or data. Exiting.")

    # --- 5. Instantiate Fusion Models ---
    print("\nInstantiating Fusion Models...")
    # Ensure target_labels is a Series for unique()
    if isinstance(target_labels, np.ndarray):
        target_labels_series = pd.Series(target_labels)
    else:
        target_labels_series = target_labels

    num_classes = len(target_labels_series.unique())
    if num_classes < 2:
        raise ValueError(f"Number of unique classes in target labels is {num_classes}. Need at least 2.")
    print(f"Detected {num_classes} classes for output layer.")

    # Model 1: Enhanced Attention Fusion
    fusion_att_model = EnhancedAttentionFusion(
        omics_networks=omics_networks,
        output_size=num_classes,
        embed_dim=EMBED_DIM,
        nhead=ATTN_HEADS,
        device=DEVICE
    ).to(DEVICE)
    print("  - Instantiated EnhancedAttentionFusion")

    # Model 2: Masked Direct Fusion
    # Create the connection mask (example: diagonal connection)
    num_present_omics = len(final_omics_data_for_training)
    expected_mask_shape = (EMBED_DIM, num_present_omics * EMBED_DIM)
    connection_mask = torch.zeros(expected_mask_shape)
    print(f"  Creating connection mask with shape: {connection_mask.shape}")
    # Simple diagonal connection example (each omics i maps to shared node i)
    # This assumes EMBED_DIM is large enough and might not be biologically meaningful
    for i in range(min(EMBED_DIM, num_present_omics * EMBED_DIM)): # Iterate up to mask dimensions
         # Example: Connect omics block i's feature j to shared node j
         # This needs refinement based on desired connectivity.
         # Let's make a block diagonal mask: connect omics i features only to shared block i
         # This doesn't make sense if shared layer output is EMBED_DIM.
         # Let's use the user's simple mask: connect omics i feature j to shared feature j
        for omics_idx in range(num_present_omics):
            for feature_idx in range(EMBED_DIM):
                 input_col_idx = omics_idx * EMBED_DIM + feature_idx
                 output_row_idx = feature_idx
                 if output_row_idx < EMBED_DIM and input_col_idx < expected_mask_shape[1]:
                      connection_mask[output_row_idx, input_col_idx] = 1.0
    print(f"  Generated connection mask. Sparsity: {(connection_mask == 0).sum() / connection_mask.numel():.2f}")


    masked_direct_fusion_model = MaskedDirectFusionNetwork(
        omics_networks=omics_networks,
        output_size=num_classes,
        connection_mask=connection_mask, # Pass the created mask
        embed_dim=EMBED_DIM,
        device=DEVICE
    ).to(DEVICE)
    print("  - Instantiated MaskedDirectFusionNetwork")

    # --- 6. Prepare Final Dataset for Training ---
    print("\nPreparing final multi-omics dataset for DataLoader...")
    multi_omics_dataset_final = prepare_multi_omics_data(
        omics_data_dict=final_omics_data_for_training, # Use the final processed dict
        labels=target_labels, # Use the potentially resampled labels
        test_size=config.test_size,
        random_state=RANDOM_STATE
    )

    # --- 7. Train Models ---
    models_to_train = {
         'AttentionFusion': fusion_att_model,
         'MaskedDirectFusion': masked_direct_fusion_model
    }
    trained_models = {}

    for model_name, model_instance in models_to_train.items():
        print(f"\n=== Training {model_name} ===")
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=LR)
        trained_model = train_multi_omics_model(
            model=model_instance,
            dataset=multi_omics_dataset_final,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            early_stopping_patience=EARLY_STOPPING,
            optimizer=optimizer,
            device=DEVICE,
            init_model=True # Initialize weights
        )
        # Save the trained model
        model_save_path = RESULTS_DIR / f'{model_name}_{config.dataset}_{TARGET_STAGE_KEY}.pth'
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Saved trained {model_name} to {model_save_path}")
        trained_models[model_name] = trained_model # Store for potential analysis call

    # --- 8. Analysis (Optional - Call analyze.py separately) ---
    # If you want to run analysis immediately after training the first model:
    if 'AttentionFusion' in trained_models:
        print("\n--- Running Analysis on Trained Attention Fusion Model ---")
        try:
            # Prepare inputs for enhanced_explain
            # Use the *non-resampled* but cleaned & aligned data for interpretation
            # Re-align original cleaned data
            aligned_original_data_dict = {}
            original_target_labels = aligned_labels_dict[TARGET_STAGE_KEY] # Get original labels for the stage

            for omics_name in final_omics_data_for_training.keys(): # Use omics that were actually trained
                 original_cleaned_data = initial_data_full[omics_name][TARGET_STAGE_KEY][TARGET_DATA_KEY]
                 # Re-clean (or use stored cleaned version before resampling)
                 original_cleaned_data_reactome, _ = index_cleaning(original_cleaned_data, reactome_cleaning_df.copy(), 'input_features')
                 # Align to original labels
                 common_idx_orig = original_cleaned_data_reactome.index.intersection(original_target_labels.index)
                 aligned_original_data_dict[omics_name] = original_cleaned_data_reactome.loc[common_idx_orig]

            # Construct relation_dfs dict for analysis (using stored relation dfs after cleaning)
            analysis_relation_dfs = {
                 name: df for name, df in omics_relation_dfs_dict.items() if name in final_omics_data_for_training
            }
            # Construct mappings_dict for analysis (using stored mappings)
            analysis_mappings_dict = {
                 name: maps for name, maps in omics_mappings_dict.items() if name in final_omics_data_for_training
            }


            analysis_output_dir = RESULTS_DIR / f"analysis_{config.dataset}_{TARGET_STAGE_KEY}_AttentionFusion"

            results_analysis = enhanced_explain(
                fusion_model=trained_models['AttentionFusion'], # Use the trained model
                multi_omics_data_dict=aligned_original_data_dict, # Use NON-resampled data for explain
                relation_dfs=analysis_relation_dfs,
                mappings_dict=analysis_mappings_dict,
                output_dir=str(analysis_output_dir), # Convert Path to string
                sample_idx=0, # Analyze first sample
                target_class_idx=1 # Analyze contribution to class 1
            )
            print("Analysis complete. Results stored in:", analysis_output_dir)
            # Print some summary from results_analysis if desired
            print("\nFusion Contributions (from Analysis):")
            print(pd.DataFrame(results_analysis.get('fusion_contribution', {})).T)

        except Exception as e:
            print(f"\nError during post-training analysis: {e}")

    print("\n--- Mid-Fusion Training Pipeline Finished ---")


if __name__ == "__main__":
    main()