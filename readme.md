# DF-HCFN: Deep Fusion Hierarchical Collaborative Feature Network

## Project Overview

DF-HCFN is a deep learning framework designed for multimodal medical image segmentation, specifically targeting brain tumor segmentation tasks (BraTS challenge dataset). The project implements an innovative feature fusion network architecture that effectively processes multimodal medical imaging data and maintains good segmentation performance even when some modalities are missing.

## Features

- **Multimodal Fusion**: Supports feature fusion of multiple MRI modalities (T1, T1ce, T2, FLAIR)
- **Missing Modality Handling**: Capable of handling cases with missing modalities, improving model robustness
- **Hierarchical Feature Extraction**: Employs a hierarchical feature extraction and fusion strategy
- **Cross-Modal Attention Mechanism**: Uses cross-modal attention mechanisms to enhance feature representation
- **CLIP-Based Feature Enhancement**: Integrates CLIP model for feature enhancement

## Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA support (recommended)
- Other dependencies:
  - numpy
  - medpy
  - timm
  - monai
  - transformers
  - einops

## Installation

```bash
# Clone the repository
git clone 
cd DF-HCFN

# Install dependencies
pip install -r requirements.txt  # Note: You need to create this file yourself
```

## Data Preparation

1. Download the BraTS challenge dataset (this project uses BraTS 2018)
2. Process the raw data using the `process.py` script:

```bash
# Modify src_path and tar_path in process.py to your data paths
python process.py
```

## Usage

### Training the Model

```bash
# Use the job.sh script for training
bash job.sh

# Or directly use the Python command
python train.py --batch_size=1 --datapath BRATS18/BRATS2018_Training_none_npy --savepath output --num_epochs 1000 --dataname BRATS2018
```

### Parameter Description

- `--batch_size`: Batch size
- `--datapath`: Dataset path
- `--dataname`: Dataset name
- `--savepath`: Model save path
- `--resume`: Path to model for resuming training
- `--pretrain`: Path to pretrained model
- `--lr`: Learning rate
- `--weight_decay`: Weight decay
- `--num_epochs`: Number of training epochs

### Prediction/Evaluation

```bash
# Evaluate model performance
python train.py --batch_size=1 --datapath BRATS18/BRATS2018_Training_none_npy --savepath output --num_epochs 0 --dataname BRATS2018 --resume output/model_last.pth
```

## Model Architecture

The DF-HCFN model consists of the following components:

1. **Multimodal Feature Extraction Module**: Extracts features from different MRI modalities (T1, T1ce, T2, FLAIR)
2. **Hierarchical Feature Fusion**: Enhances model representation capability through multi-level feature fusion
3. **Cross-Modal Attention Mechanism**: Implements information interaction between different modalities using the CrossAttention module
4. **CLIP Feature Enhancement**: Integrates the CLIP model to provide additional semantic information
5. **Segmentation Head**: Final segmentation prediction layer

## Evaluation Metrics

The model uses the following metrics to evaluate segmentation performance:

- Dice coefficient: Evaluates the overlap between segmentation results and ground truth labels
- Specific evaluation metrics for BraTS tasks:
  - Whole Tumor (WT) Dice coefficient
  - Tumor Core (TC) Dice coefficient
  - Enhancing Tumor (ET) Dice coefficient

