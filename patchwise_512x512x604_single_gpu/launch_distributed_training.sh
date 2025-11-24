#!/bin/bash

# Launch patch-wise training on single GPU
# Trains on full 512×512×604 resolution using patch extraction

echo "=================================================="
echo "PATCH-WISE VQ-GAN TRAINING (Single GPU)"
echo "Full Resolution: 512×512×604"
echo "Patch Size: 256×256×128"
echo "=================================================="

# Set up environment
export PYTHONPATH=/workspace/Xray-2CTPA_spartis/patchwise_parallel_512x512x604_multi_gpu:$PYTHONPATH

# Verify GPU
echo ""
echo "Available GPU:"
nvidia-smi --list-gpus

# Navigate to training directory
cd /workspace/Xray-2CTPA_spartis/patchwise_parallel_512x512x604_multi_gpu

# Launch training
echo ""
echo "Launching training..."
python train/train_vqgan_distributed.py \
    dataset=full_resolution_ctpa \
    model=vq_gan_3d_patches

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
