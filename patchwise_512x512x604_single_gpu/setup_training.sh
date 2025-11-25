#!/bin/bash

# Automated Setup Script for Patch-Wise VQ-GAN Training
# Run this on RunPod or your training environment

set -e  # Exit on any error

echo "================================================"
echo "Patch-Wise VQ-GAN Training Setup"
echo "================================================"

# Step 1: Check GPU
echo ""
echo "[Step 1] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ GPU not detected. Make sure you're on a GPU node."
    exit 1
}

# Step 2: Create directories
echo ""
echo "[Step 2] Creating necessary directories..."
mkdir -p outputs/checkpoints
mkdir -p outputs/tensorboard_logs
mkdir -p outputs/logs
mkdir -p ../datasets
mkdir -p ../datasets_nifti
echo "✓ Directories created"

# Step 3: Install dependencies
echo ""
echo "[Step 3] Installing dependencies (this may take 5-10 minutes)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Step 4: Verify imports
echo ""
echo "[Step 4] Verifying Python imports..."
python -c "
import torch
import pytorch_lightning
import nibabel
import omegaconf
import hydra
import tensorboard
import einops
print('✓ All critical imports successful')
print(f'  PyTorch version: {torch.__version__}')
print(f'  GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" || {
    echo "❌ Import verification failed"
    exit 1
}

# Step 5: Display next steps
echo ""
echo "================================================"
echo "✓ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Preprocess DICOM files:"
echo "   - python ../preprocess/dicom_to_nifti.py --input ../datasets --output ../datasets_nifti"
echo ""
echo "2. Update config file:"
echo "   - Edit config/dataset/full_resolution_ctpa.yaml"
echo "   - Set data_dir to your preprocessed dataset path"
echo ""
echo "3. Launch training (choose one):"
echo "   - Option A: ./launch_distributed_training.sh"
echo "   - Option B: python train/train_vqgan_distributed.py"
echo ""
echo "4. Monitor training:"
echo "   - tensorboard --logdir outputs/tensorboard_logs"
echo ""
echo "================================================"
echo ""
