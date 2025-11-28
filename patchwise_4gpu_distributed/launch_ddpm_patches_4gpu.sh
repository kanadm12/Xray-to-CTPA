#!/bin/bash

# Launch Patchwise DDPM Training on 4 GPUs
# Uses torchrun for distributed data parallel training
# Processes 128×128×128 patches from full 512×512×604 volumes

set -e

echo "=========================================="
echo "Patchwise DDPM Training (4×H200 GPUs)"
echo "X-ray → CTPA Generation with Patches"
echo "=========================================="

# Configuration
GPUS=4
MASTER_PORT=29501

# Check VQ-GAN checkpoint exists
VQGAN_CKPT="./outputs/vqgan_patches_4gpu/checkpoints/last.ckpt"
if [ ! -f "$VQGAN_CKPT" ]; then
    echo "ERROR: VQ-GAN checkpoint not found at $VQGAN_CKPT"
    echo "Train VQ-GAN first using: ./launch_4gpu_vqgan_disc.sh"
    exit 1
fi

echo "VQ-GAN checkpoint found: $VQGAN_CKPT"
echo "Using $GPUS GPUs with master port $MASTER_PORT"
echo ""

# Create output directory
mkdir -p checkpoints/ddpm_4gpu_patches

# Launch distributed training
echo "Launching patchwise DDPM training with torchrun..."
torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    train_ddpm_patches_4gpu.py \
    model=ddpm_4gpu_patches

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints/ddpm_4gpu_patches/"
echo "=========================================="
