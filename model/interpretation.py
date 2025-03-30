# analyze.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse # For command-line arguments

# Import necessary components from your project structure
import config # Your configuration
from dataprocess.data_loader import load_and_preprocess_data
from dataprocess.workflow_processor import align_omics_data # If needed separately
from model.model_builder import EnhancedAttentionFusion, OmicsSubNetwork # Import necessary model classes
from model.reactome_utils import reactome_net, generate_layer_relation_df # For relation_dfs
from model.model_utils import enhanced_explain # The main analysis function
# Add any other necessary imports (e.g., process_layers_and_build_masks if mappings_dict needs regen)


def load_model(model_class, model_path, device, **model_kwargs):
    """Loads a trained PyTorch model state dictionary."""
    print(f"Loading model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Instantiate model architecture - Requires necessary args like omics_networks for fusion
    # This part is tricky - we need to know the architecture args used during training.
    # Option 1: Save architecture args alongside model weights.
    # Option 2: Reconstruct args based on config used for training.
    # Let's assume reconstruction based on config for now.

    # Example for EnhancedAttentionFusion - requires reconstructing omics_networks dict
    # This reconstruction needs access to input sizes derived from the data *used for training*.
    # This highlights the need to potentially save data stats or reconstruct carefully.
    # For simplicity here, we might skip full reconstruction if enhance_explain doesn't need the full model instance internals beyond weights.
    # However, IG *does* need the model instance.

    # --- Placeholder for reconstructing omics_networks ---
    # This needs actual data loading to get input sizes, similar to train.py
    print("Warning: Model instantiation in analyze.py requires careful reconstruction based on training config/data.")
    # You would need to load data, determine input sizes for each omics,
    # instantiate OmicsSubNetwork for each, and pass the dict to EnhancedAttentionFusion.
    # For demonstration, let's assume a simplified instantiation is possible or we load a full model object if saved that way.
    # ----
    try:
        # If model_kwargs are provided (e.g., from a saved config)
        model = model_class(**model_kwargs, device=device)
        # If instantiation requires complex steps, handle them here
    except Exception as e:
        print(f"Error instantiating model {model_class.__name__}: {e}")
        print("Please ensure model arguments can be reconstructed or load the full model object.")
        # As a fallback for analysis functions that only need weights (like analyze_fusion_layer):
        # model = model_class(...) # Try with placeholder args if possible
        # Or handle differently depending on which analysis is run.
        # For IG, a working model instance is essential.
        # --> Let's raise the error for now, requiring proper reconstruction.
        raise

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def main(args):
    """Main function to run analysis."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Running Analysis ---")
    print(f"Model Path: {args.model_path}")
    print(f"Dataset: {config.dataset}") # Assuming config reflects training dataset
    print(f"Target Stage: {args.target_stage}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # --- 1. Load Data (Use same config as training) ---
    # We need the *full* dataset potentially, or at least the test set structure
    # Load with use_precomputed=True if preprocessing is done
    print("Loading data...")
    initial_data = load_and_preprocess_data(use_precomputed=True)

    # Select and align data for the target stage
    omics_data_stage = {
        omics: initial_data[omics][args.target_stage] for omics in config.OMICS_TYPES # Use config for omics
    }
    aligned_omics_data, _ = align_omics_data(omics_data_stage) # We only need the data dict here

    # --- 2. Load/Generate Reactome Info ---
    print("Loading/Generating Reactome Info...")
    # This assumes relation_dfs structure matches what pathway analysis expects
    # And that mappings_dict can be reconstructed if needed by enhanced_explain
    # Regeneration based on the data used for initial mask generation:
    data_for_masks = initial_data[config.DATA_TYPE_FOR_REACTOME][args.target_stage][config.TARGET_DATA_KEY] # Use config constants
    if data_for_masks.empty:
        raise ValueError("Cannot generate Reactome info: Data for mask generation is empty.")

    original_reactome_layers = reactome_net.get_layers(n_levels=config.N_REACTOME_LEVELS + 1)
    original_reactome_dfs_list = generate_layer_relation_df(original_reactome_layers)
    # The pathway analysis function expects a dict {'omics_name': relation_df}
    # This mapping might be complex. Simplification: Assume the *same* Reactome structure
    # applies conceptually to features from all omics for pathway mapping?
    # Or only analyze pathways for the omics type used for mask generation?
    # Let's assume we analyze pathways based on the Reactome structure derived from one omics type (e.g., mRNA)
    # but use feature importances calculated for each omics type.
    # The relation_dfs dict should map omics names to the *same* relevant Reactome layer df?
    # Example: Using the *last* layer's relation df for pathway mapping for all omics
    # This needs domain knowledge - which Reactome layer represents the "pathways"? Usually the higher levels.
    pathway_relation_layer_index = 0 # Example: Use the first layer (most granular pathways?) - ADJUST AS NEEDED
    relation_dfs_dict = {
        omics: original_reactome_dfs_list[pathway_relation_layer_index]
        for omics in config.OMICS_TYPES if omics in aligned_omics_data and not aligned_omics_data[omics].empty
    }


    # Reconstruct mappings_dict if needed by enhance_explain (might be complex)
    # Simplification: Pass an empty dict or None if not strictly required by the parts of enhance_explain being used.
    mappings_dict = {} # Placeholder

    # --- 3. Load Trained Model ---
    # This requires reconstructing the model architecture arguments correctly!
    # We need input sizes, hidden sizes etc., used during training.
    # Example for EnhancedAttentionFusion (NEEDS ACCURATE RECONSTRUCTION)
    omics_networks_args = {}
    subnetwork_hidden_sizes = [64, 32] # Example - MUST MATCH TRAINING
    for omics_name, data in aligned_omics_data.items():
        if not data.empty:
            omics_networks_args[omics_name] = OmicsSubNetwork(
                input_size=data.shape[1],
                hidden_sizes=subnetwork_hidden_sizes, # Must match training
                masks=[], # Masks usually aren't saved/reloaded easily unless part of state_dict
                output_size=config.EMBED_DIM, # Use config
                device=DEVICE
            )

    if not omics_networks_args:
         raise ValueError("Could not reconstruct any omics sub-networks. Cannot load fusion model.")


    # Choose the correct model class based on the saved file/args
    # Assuming EnhancedAttentionFusion was saved
    model_class = EnhancedAttentionFusion
    model_kwargs = {
        'omics_networks': omics_networks_args,
        'output_size': len(pd.unique(initial_data[config.DATA_TYPE_FOR_REACTOME][args.target_stage][config.TARGET_LABEL_KEY])), # Get num classes from data
        'embed_dim': config.EMBED_DIM,
        'nhead': config.ATTN_HEADS,
        # device is passed separately
    }

    trained_model = load_model(model_class, args.model_path, DEVICE, **model_kwargs)


    # --- 4. Run Enhanced Explanation ---
    print("\nRunning Enhanced Explanation...")
    # Select a sample index (e.g., the first one) for IG
    sample_idx_for_ig = 0
    # Determine target class index (e.g., class 1) for IG - make this an argument?
    target_class_idx_for_ig = 1 # Example: Explain contribution to class 1

    analysis_results = enhanced_explain(
        fusion_model=trained_model,
        multi_omics_data_dict=aligned_omics_data, # Use the aligned data for the target stage
        relation_dfs=relation_dfs_dict, # Pass the constructed dict
        mappings_dict=mappings_dict, # Pass placeholder or reconstructed mappings
        output_dir=str(OUTPUT_DIR), # Ensure output dir is string
        sample_idx=sample_idx_for_ig,
        target_class_idx=target_class_idx_for_ig
    )

    # --- 5. Save/Print Analysis Results ---
    print("\n--- Analysis Complete ---")
    print("\nFusion Layer Contributions:")
    print(pd.DataFrame(analysis_results.get('fusion_contribution', {})).T) # Transpose for better view

    print("\nPathway Contributions (Relative):")
    # Find the first non-empty pathway contribution dict
    pathway_df = None
    for omics_contrib in analysis_results.get('pathway_contribution_ig', {}).values():
        if omics_contrib: # Check if the dictionary is not empty
             pathway_df = pd.DataFrame.from_dict(omics_contrib, orient='index', columns=['Contribution'])
             print(pathway_df.sort_values('Contribution', ascending=False).head(10)) # Print top 10 pathways
             break # Only print for the first omics type with results for brevity
    if pathway_df is None:
        print("No pathway contributions calculated.")


    # Optionally save the full results dictionary (e.g., using pickle)
    # import pickle
    # with open(OUTPUT_DIR / 'full_analysis_results.pkl', 'wb') as f:
    #     pickle.dump(analysis_results, f)
    # print(f"Full analysis results saved to {OUTPUT_DIR / 'full_analysis_results.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on trained multi-omics models.")
    parser.add_argument("model_path", help="Path to the trained model (.pth file)")
    parser.add_argument("-s", "--target_stage", default="T", choices=["T", "N", "M"],
                        help="Target TNM stage used for training the model (default: T)")
    parser.add_argument("-o", "--output_dir", default="./analysis_results",
                        help="Directory to save analysis plots and results (default: ./analysis_results)")
    # Add other arguments if needed (e.g., specific sample index, target class for IG)

    args = parser.parse_args()

    # Use config for dataset selection and other settings implicitly
    # Make sure config.py reflects the settings used for the training run of the model being analyzed
    main(args)