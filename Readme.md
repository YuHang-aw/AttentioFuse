# AttentioFuse: Multi-Omics Cancer Stage Prediction

A Python project exploring multi-omics data fusion techniques, primarily using attention mechanisms, for predicting cancer staging (TNM) based on TCGA LUSC and LUAD datasets. This project includes implementations for both early and mid-fusion strategies and compares them with classical machine learning models. It also incorporates biological pathway information from Reactome using masked MLP layers.

![Project Architecture Diagram](attfusion/figure1-2_02.png "My Project Flow")

## Introduction

Predicting cancer stage accurately is crucial for treatment planning and prognosis. Multi-omics data (e.g., mRNA expression, CNV, SNV) provides a comprehensive view of the tumor biology. This project investigates how to effectively fuse these different data types to improve TNM stage prediction. We explore:

* **Early Fusion:** Concatenating features from different omics types before feeding them into a model.
* **Mid-Fusion:** Using separate sub-networks for each omics type and then fusing their learned representations, including attention-based methods.
* **Reactome Integration:** Incorporating pathway knowledge through masked connections in neural network layers.
* **Classical ML Comparison:** Benchmarking fusion models against standard machine learning algorithms.

The primary datasets used are TCGA-LUSC and TCGA-LUAD.

## Features

* Data loading and preprocessing pipelines for TCGA multi-omics data (mRNA, CNV, SNV).
* Implementation of TNM stage label generation and data splitting.
* Missing value imputation using DBSCAN-based methods.
* Data normalization using StandardScaler.
* Alignment of multi-omics data based on patient identifiers.
* **Early Fusion Workflow:** Concatenates cleaned features and trains models (Classical ML, MaskedMLP, Sequential NN).
* **Mid-Fusion Workflow:**
  * Omics-specific sub-networks (optionally masked).
  * `EnhancedAttentionFusion` model.
  * `DirectFusionNetwork` model (masked and unmasked versions).
* Integration with Reactome pathway data for building masked MLP layers.
* Training and evaluation utilities for both single-input and multi-input PyTorch models.
* Model interpretation and analysis utilities (Fusion layer analysis, Integrated Gradients, Pathway contribution).
* Configurable dataset selection (LUSC/LUAD) via `config.py`.

## Installation

1. **Clone the repository:**

   git clone [<GitHub URL>](https://github.com/YuHang-aw/AttentioFuse.git)
   cd attentiofuse
2. **Create and activate a virtual environment (Recommended):**

   ```bash
   python -m venv venv
   # On Windows:
   # venv\Scripts\activate
   # On Linux/macOS:
   # source venv/bin/activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   * **Note on PyTorch:** `requirements.txt` lists `torch`. You might need to install a specific version compatible with your system (CPU/GPU CUDA version). Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command.
   * **Note on GMT:** Ensure the `gmt-python` package (or the correct name for your GMT library) in `requirements.txt` is installable via pip or provide separate installation instructions if needed.
4. **(Optional) Install in editable mode for development:**

   ```bash
   pip install -e .
   ```

## Data Setup

**Data is NOT included in this repository.**

1. Download the required TCGA LUSC and LUAD datasets.
2. Create the `./data/` directory in the project root if it doesn't exist.
3. Organize the data files within `./data/` according to the paths expected by `config.py` and `dataprocess/data_loader.py`. The expected structure and filenames (based on the code) are:
   ```
   data/
   ├── LUSC/
   │   ├── LUSC_clinical_SE.csv
   │   ├── TCGA-LUSC_mrna_expr_tpm.csv
   │   ├── all_data_by_genes.txt
   │   └── LUSC_snv.csv
   └── LUAD/
       ├── clinIndexData.csv
       ├── TCGA-LUAD_mrna_expr_tpm.csv
       ├── cnv_grouped_by_patient.csv
       └── LUAD_snv_mtx.csv
   ```
4. It is strongly recommended to add the `data/` directory to your `.gitignore` file, especially if the data is large or subject to access restrictions.

## Configuration

Project settings are managed in `config.py`. Key settings include:

* `dataset`: Set to `'LUSC'` or `'LUAD'` to select the dataset for processing and training.
* `data_path`: Path to the base data directory.
* Other parameters related to data processing and model training can also be added here.

## Usage

### Training

The main training script is `model/train.py`. It orchestrates data loading, preprocessing (based on selected workflow), model building, and training.

1. **Configure:** Edit `config.py` to select the desired `dataset` ('LUSC' or 'LUAD').
2. **Configure `model/train.py`:**

   * Set the `WORKFLOW` variable (e.g., `'mid_fusion'` or `'early_fusion'`) if you re-introduce workflow switching logic. (The last provided code focused on mid-fusion).
   * Set the `TARGET_STAGE_KEY` (e.g., `'T'`, `'N'`, `'M'`) to specify which TNM stage to train models for.
   * Adjust other hyperparameters (epochs, batch size, learning rate) as needed.
3. **Run Training:** Execute from the project root directory:

   ```bash
   python model/train.py
   ```

   Trained models will be saved (by default) in the project root or potentially in the `./results/` directory, depending on the final code modifications.

### Analysis

After training and saving a model (e.g., `attention_fusion_LUSC_T.pth`), you can run the analysis script:

1. **Run Analysis:** Execute `analyze.py` from the project root, providing the path to the saved model, the target stage it was trained for, and an output directory for analysis results:

   ```bash
   python analyze.py <path/to/saved_model.pth> -s <TNM_STAGE> -o <output_directory_path>

   # Example:
   python analyze.py ./attn_fusion_model_LUSC_T.pth -s T -o ./results/LUSC_T_attn_analysis
   ```

   This will perform analyses like fusion layer contribution, Integrated Gradients, and pathway contribution, saving plots and potentially results to the specified output directory.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Open a Pull Request.

Please open an issue first to discuss major changes.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details, or state it here.
