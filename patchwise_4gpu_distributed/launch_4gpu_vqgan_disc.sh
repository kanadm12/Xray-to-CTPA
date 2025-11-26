#!/bin/bash

##############################################################################
# VQ-GAN 4-GPU Training with Discriminator
##############################################################################
#
# Enables adversarial training for sharper, more realistic reconstructions
# PSNR target: 30-32 dB (vs 28-30 dB without discriminator)
#
##############################################################################

echo "=========================================="
echo "VQ-GAN 4-GPU with Discriminator Training"
echo "=========================================="
echo ""

# Check GPU availability
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "ERROR: Need 4 GPUs but found only $NUM_GPUS"
    exit 1
fi

echo "✓ Found $NUM_GPUS GPUs"
echo ""

# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
MASTER_PORT=29501  # Different port to avoid conflicts

echo "Configuration:"
echo "  Model: VQ-GAN with Discriminator"
echo "  GPUs: 4"
echo "  Batch size: 4 (1 per GPU)"
echo "  Discriminator start: Step 10000 (~5 epochs)"
echo "  Losses: L1 + GAN + Perceptual"
echo ""

# Launch training
echo "Starting training..."
echo "Logs: training_4gpu_disc.log"
echo ""

nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    train_vqgan_4gpu.py \
    --config-path config \
    --config-name base_cfg \
    dataset=ctpa_4gpu \
    model=vqgan_4gpu_disc \
    dataset.max_patients=null \
    model.max_epochs=50 \
    model.discriminator_iter_start=10000 \
    model.image_gan_weight=1.0 \
    model.video_gan_weight=0.5 \
    model.perceptual_weight=0.1 \
    model.gan_feat_weight=4.0 \
    > training_4gpu_disc.log 2>&1 &

PID=$!
echo "Training launched with PID: $PID"
echo ""

sleep 5
if ps -p $PID > /dev/null; then
    echo "✓ Training successfully started!"
else
    echo "✗ Training failed to start. Check training_4gpu_disc.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Monitoring:"
echo "=========================================="
echo "  Logs:      tail -f training_4gpu_disc.log"
echo "  GPUs:      nvidia-smi"
echo "  Kill:      kill $PID"
echo ""
echo "Expected:"
echo "  - Warmup phase (0-10k steps): L1 loss only"
echo "  - Adversarial phase (10k+ steps): GAN losses appear"
echo "  - Final PSNR: 30-32 dB"
echo "  - Training time: ~6-7 hours for 50 epochs"
echo "=========================================="
