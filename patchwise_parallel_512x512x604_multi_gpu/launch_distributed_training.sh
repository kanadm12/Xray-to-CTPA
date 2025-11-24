#!/bin/bash

# Launch distributed training on 4 H200 GPUs
# Run this script on the RunPod instance

echo "=================================================="
echo "PATCH-WISE PARALLEL VQ-GAN TRAINING"
echo "Full Resolution: 512×512×604"
echo "GPUs: 4× H200 (144GB each)"
echo "=================================================="

# Set up environment
export PYTHONPATH=/workspace/Xray-2CTPA_spartis/patchwise_parallel_512x512x604_multi_gpu:$PYTHONPATH
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# Verify GPUs
echo ""
echo "Available GPUs:"
nvidia-smi --list-gpus

# Navigate to training directory
cd /workspace/Xray-2CTPA_spartis/patchwise_parallel_512x512x604_multi_gpu

# Launch distributed training using torchrun
echo ""
echo "Launching distributed training..."
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    train/train_vqgan_distributed.py \
    dataset=full_resolution_ctpa \
    model=vq_gan_3d_patches

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
