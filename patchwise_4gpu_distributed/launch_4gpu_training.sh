#!/bin/bash

##############################################################################
# VQ-GAN 4-GPU Distributed Training Script
##############################################################################
#
# This script trains VQ-GAN across 4 H200 GPUs using PyTorch DDP.
# Each GPU processes 1 patient volume simultaneously (effective batch=4).
#
# Hardware: 4× H200 GPUs (140GB each = 560GB total VRAM)
# Training: Patch-wise processing on 512×512×604 volumes
# Strategy: DistributedDataParallel (DDP)
#
##############################################################################

echo "=========================================="
echo "VQ-GAN 4-GPU Distributed Training"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Count available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 4 ]; then
    echo "ERROR: Need 4 GPUs but found only $NUM_GPUS"
    exit 1
fi

# Set environment variables for optimal performance
export NCCL_DEBUG=INFO  # Enable NCCL logging for debugging
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # GPU Direct RDMA
export OMP_NUM_THREADS=8  # OpenMP threads per process
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use first 4 GPUs

# Training configuration
MASTER_ADDR=localhost
MASTER_PORT=29500

echo "Configuration:"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Effective Batch Size: 1 × 4 = 4 volumes"
echo ""

# Launch training with torchrun (recommended for PyTorch 1.9+)
echo "Starting training..."
echo "Logs: training_4gpu.log"
echo ""

nohup torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_vqgan_4gpu.py \
    > training_4gpu.log 2>&1 &

# Get process ID
PID=$!
echo "Training launched with PID: $PID"
echo ""

# Wait a few seconds and check if process is running
sleep 5
if ps -p $PID > /dev/null; then
    echo "✓ Training successfully started!"
else
    echo "✗ Training failed to start. Check training_4gpu.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Monitoring Commands:"
echo "=========================================="
echo "  Watch logs:       tail -f training_4gpu.log"
echo "  GPU usage:        watch -n 1 nvidia-smi"
echo "  Kill training:    kill $PID"
echo "  Check process:    ps aux | grep train_vqgan_4gpu"
echo ""
echo "TensorBoard:        tensorboard --logdir=outputs/vqgan_patches_4gpu/lightning_logs"
echo ""
echo "=========================================="
