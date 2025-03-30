# AttentionFusion: Multi-Omics Cancer Stage Prediction

A Python project exploring multi-omics data fusion techniques, primarily using attention mechanisms, for predicting cancer staging (TNM) based on TCGA LUSC and LUAD datasets. This project includes implementations for both early and mid-fusion strategies and compares them with classical machine learning models. It also incorporates biological pathway information from Reactome using masked MLP layers.

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

## Project Structure
