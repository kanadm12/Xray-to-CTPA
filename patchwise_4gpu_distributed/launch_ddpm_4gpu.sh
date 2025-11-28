#!/bin/bash

# DDPM Training Launch Script (4 GPUs)
# Trains diffusion model for X-ray → CTPA generation in VQ-GAN latent space

echo "=================================================="
echo "4-GPU DDPM TRAINING - X-ray to CTPA Generation"
echo "=================================================="

# Environment setup
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
MASTER_PORT=29501

echo "Number of GPUs: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"

# Verify VQ-GAN checkpoint exists
VQGAN_CKPT="./outputs/vqgan_patches_4gpu/checkpoints/last.ckpt"
if [ ! -f "$VQGAN_CKPT" ]; then
    echo "ERROR: VQ-GAN checkpoint not found: $VQGAN_CKPT"
    echo "Train VQ-GAN first using: ./launch_4gpu_training.sh"
    exit 1
fi

echo "Using VQ-GAN checkpoint: $VQGAN_CKPT"
echo ""

# Create output directory
mkdir -p checkpoints/ddpm_4gpu

# Launch distributed training
echo "Starting DDPM training with torchrun..."
echo "=================================================="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_ddpm_4gpu.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints/ddpm_4gpu/"
echo "=================================================="
