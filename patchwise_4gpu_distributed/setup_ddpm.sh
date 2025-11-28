#!/bin/bash

# DDPM Setup & Test Guide for RunPod
# Run this after git pull to verify everything is ready

echo "=================================================="
echo "DDPM TRAINING SETUP & VERIFICATION"
echo "=================================================="
echo ""

# Step 1: Git pull
echo "Step 1: Pulling latest changes..."
git pull origin main

if [ $? -ne 0 ]; then
    echo "ERROR: Git pull failed!"
    exit 1
fi

echo ""
echo "Step 2: Checking VQ-GAN checkpoint..."
VQGAN_CKPT="./outputs/vqgan_patches_4gpu/checkpoints/last.ckpt"

if [ ! -f "$VQGAN_CKPT" ]; then
    echo "WARNING: VQ-GAN checkpoint not found: $VQGAN_CKPT"
    echo "Available checkpoints:"
    ls -lh ./outputs/vqgan_patches_4gpu/checkpoints/ 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "If you stopped training early, you may have epoch checkpoints instead."
    echo "Update config/model/ddpm_4gpu.yaml to point to the correct checkpoint."
else
    echo "✓ VQ-GAN checkpoint found"
    ls -lh "$VQGAN_CKPT"
fi

echo ""
echo "Step 3: Installing required dependencies..."
pip install SimpleITK pillow > /dev/null 2>&1
echo "✓ Dependencies installed"

echo ""
echo "Step 4: Testing dataset loading..."
cd patchwise_4gpu_distributed || cd .
python test_dataset.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Dataset test failed!"
    echo "Please check:"
    echo "  1. X-ray PNG files exist alongside CTPA .nii.gz files"
    echo "  2. X-ray naming pattern matches: *_pa_drr.png"
    echo "  3. Dataset path is correct in config/dataset/xray_ctpa_ddpm.yaml"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ SETUP VERIFIED!"
echo "=================================================="
echo ""
echo "Ready to start DDPM training!"
echo ""
echo "To launch 4-GPU DDPM training, run:"
echo "  cd patchwise_4gpu_distributed"
echo "  chmod +x launch_ddpm_4gpu.sh"
echo "  ./launch_ddpm_4gpu.sh"
echo ""
echo "=================================================="
