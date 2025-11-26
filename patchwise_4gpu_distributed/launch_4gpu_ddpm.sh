#!/bin/bash

##############################################################################
# DDPM 4-GPU Distributed Training
##############################################################################
#
# Trains diffusion model for X-ray → CTPA generation in latent space
# Requires: Trained VQ-GAN checkpoint from previous step
#
##############################################################################

echo "=========================================="
echo "DDPM 4-GPU Training - X-ray → CTPA"
echo "=========================================="
echo ""

# Check VQ-GAN checkpoint exists
VQGAN_CKPT="./outputs/vqgan_patches_4gpu_disc/checkpoints/last.ckpt"
if [ ! -f "$VQGAN_CKPT" ]; then
    echo "ERROR: VQ-GAN checkpoint not found: $VQGAN_CKPT"
    echo ""
    echo "Please train VQ-GAN first:"
    echo "  ./launch_4gpu_vqgan_disc.sh"
    echo ""
    exit 1
fi

echo "✓ Found VQ-GAN checkpoint: $VQGAN_CKPT"
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
MASTER_PORT=29502  # Different port from VQ-GAN

echo "Configuration:"
echo "  Model: DDPM (Diffusion Model)"
echo "  Task: X-ray → CTPA generation"
echo "  Latent space: VQ-GAN encoded"
echo "  GPUs: 4"
echo "  Batch size: 4 (1 per GPU)"
echo "  Training steps: 100,000"
echo "  Estimated time: ~24-48 hours"
echo ""

# Launch training
echo "Starting DDPM training..."
echo "Logs: training_ddpm_4gpu.log"
echo ""

nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    train_ddpm_4gpu.py \
    --config-path config \
    --config-name base_cfg_ddpm \
    dataset=xray_ctpa_ddpm \
    model=ddpm_4gpu \
    dataset.max_patients=null \
    model.vqgan_ckpt=$VQGAN_CKPT \
    model.train_num_steps=100000 \
    model.batch_size=1 \
    model.num_workers=4 \
    > training_ddpm_4gpu.log 2>&1 &

PID=$!
echo "Training launched with PID: $PID"
echo ""

sleep 5
if ps -p $PID > /dev/null; then
    echo "✓ DDPM training successfully started!"
else
    echo "✗ Training failed to start. Check training_ddpm_4gpu.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Monitoring:"
echo "=========================================="
echo "  Logs:      tail -f training_ddpm_4gpu.log"
echo "  GPUs:      nvidia-smi"
echo "  Kill:      kill $PID"
echo ""
echo "Checkpoints saved to:"
echo "  ./checkpoints/ddpm_4gpu/"
echo ""
echo "Expected:"
echo "  - First 10k steps: Warmup, learning noise schedule"
echo "  - 10k-50k steps: Main training, loss decreasing"
echo "  - 50k+ steps: Fine-tuning, high quality samples"
echo "  - Samples generated every 1000 steps"
echo "=========================================="
