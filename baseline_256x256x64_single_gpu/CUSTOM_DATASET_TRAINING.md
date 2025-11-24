# Training VQGAN on Custom Dataset - Setup Guide

## Overview
This guide helps you train VQGAN on your custom dataset structured as:
```
data_new/
├── patient_001/
│   ├── patient_001.nii.gz
│   ├── patient_001_lat_drr.png
│   ├── patient_001_pa_drr.png
│   └── patient_001_swapped_bmd.nii.gz  (will be excluded)
├── patient_002/
│   ├── patient_002.nii.gz
│   ├── patient_002_lat_drr.png
│   ├── patient_002_pa_drr.png
│   └── patient_002_swapped_bmd.nii.gz  (will be excluded)
└── ...
```

## Key Changes Made

### 1. Fixed `dataset/default.py`
- **Fixed typo**: `endsiwth` → `endswith`
- **Recursive file scanning**: Now searches through all patient subfolders
- **Smart filtering**: Automatically excludes files containing 'swapped' in the filename
- **Correct output shape**: Returns data in (C, T, H, W) format for VQGAN

### 2. Created `config/dataset/custom_data.yaml`
Configuration file pointing to your data_new folder with proper settings

### 3. Updated `train/get_dataset.py`
Added support for CUSTOM_DATA dataset type in the training pipeline

### 4. Created `train/scripts/train_vqgan_custom.sh`
Pre-configured bash script for training on your custom dataset

## Dataset Requirements

Your data must be in NIfTI format (.nii.gz) with the following characteristics:

### Preprocessing (Applied Automatically)
- **Intensity Rescaling**: Values normalized to [-1, 1]
- **Cropping/Padding**: All volumes padded/cropped to (256, 256, 32)
  - If your volumes are larger: will be cropped from center
  - If smaller: will be zero-padded
  - Depth is standardized to 32 slices

### Augmentation (Training Only)
- **Random Flip**: 50% chance of flipping along axis 1 during training

## Step-by-Step Training Instructions

### Step 1: Prepare Your Dataset
```bash
# Ensure your data_new folder is at:
# ../datasets/data_new/
# 
# Or update the path in config/dataset/custom_data.yaml
# name: CUSTOM_DATA
# root_dir: /path/to/your/data_new/
```

### Step 2: Verify Configuration
Edit `config/dataset/custom_data.yaml` if needed:
```yaml
name: CUSTOM_DATA
root_dir: ../datasets/data_new/  # Update this path
image_channels: 1
```

### Step 3: Run Training (Option A - Using Bash Script)
```bash
cd /path/to/X-ray2CTPA
chmod +x train/scripts/train_vqgan_custom.sh
bash train/scripts/train_vqgan_custom.sh
```

### Step 3: Run Training (Option B - Direct Command)
```bash
cd /path/to/X-ray2CTPA
export PYTHONPATH=$PWD

python train/train_vqgan.py \
    dataset=custom_data \
    model=vq_gan_3d \
    model.gpus=1 \
    model.precision=16 \
    model.embedding_dim=8 \
    model.n_hiddens=16 \
    model.downsample=[2,2,2] \
    model.num_workers=4 \
    model.batch_size=2 \
    model.lr=3e-4 \
    model.default_root_dir_postfix='custom_data'
```

### Step 3: Run Training (Option C - Python Script)
Create a file `train_custom.py`:
```python
import os
import subprocess
from pathlib import Path

os.chdir('/path/to/X-ray2CTPA')
os.environ['PYTHONPATH'] = os.getcwd()

cmd = [
    'python', 'train/train_vqgan.py',
    'dataset=custom_data',
    'model=vq_gan_3d',
    'model.gpus=1',
    'model.precision=16',
    'model.batch_size=2',
    'model.lr=3e-4'
]

subprocess.run(cmd)
```

## Important Parameters to Adjust

### Memory-Related (if getting CUDA OOM)
- `model.batch_size`: Reduce from 2 to 1
- `model.num_workers`: Reduce from 4 to 0-2
- `model.precision`: Change from 16 (half-precision) to 32 (full precision) if needed
- `model.downsample`: Increase values [2,2,2] → [4,4,4] for more compression

### Data-Related
- `model.embedding_dim`: Latent space dimension (8, 16, 32...)
- `model.n_codes`: Codebook size (8192, 16384, 32768...)

### Training-Related
- `model.lr`: Learning rate (default 3e-4)
- `model.discriminator_iter_start`: Steps before discriminator trains (default 10000)
- `model.perceptual_weight`: Weight of perceptual loss (default 4)

## Expected File Handling

### Files That Will Be Used
✅ `patient_001.nii.gz`
✅ `patient_001_lat_drr.png`
✅ `patient_001_pa_drr.png`
✅ `patient_002.nii.gz`
etc.

### Files That Will Be Skipped
❌ `patient_001_swapped_bmd.nii.gz` (contains "swapped")
❌ `patient_001_lat_drr.npy` (not .nii.gz)
❌ `patient_001_pa_drr.nii` (incomplete extension)

## Troubleshooting

### Issue: "Dataset is not available"
**Solution**: Make sure `dataset=custom_data` matches the config file name (without .yaml)

### Issue: "No such file or directory"
**Solution**: Verify the `root_dir` in `config/dataset/custom_data.yaml` points to correct path

### Issue: "No files found"
**Solution**: Ensure your .nii.gz files follow the naming pattern and are in patient subfolders

### Issue: CUDA Out of Memory
**Solutions**:
1. Reduce `model.batch_size` to 1
2. Reduce `model.num_workers` to 0
3. Use `model.precision=32` instead of 16
4. Increase `model.downsample` values

### Issue: Training too slow
**Solution**: Increase `model.num_workers` if you have spare CPU cores and RAM

## Output

Training will create:
```
lightning_logs/
└── version_0/
    ├── checkpoints/
    │   ├── latest_checkpoint.ckpt
    │   └── epoch-step-recon_loss.ckpt
    └── logs/
```

## Next Steps After VQGAN Training

Once VQGAN is trained, you can:
1. Use it to encode your dataset to latent space
2. Train diffusion model (DDPM) on the latent representations
3. Use the trained models for inference

For more details, see the main README.md
