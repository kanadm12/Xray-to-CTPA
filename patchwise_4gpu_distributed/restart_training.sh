#!/bin/bash

# Clean restart script with all optimizations

echo "=========================================="
echo "CLEAN RESTART WITH OPTIMIZATIONS"
echo "=========================================="

# Step 1: Kill any existing training processes
echo "1. Stopping any running training..."
pkill -9 -f train_ddpm_4gpu.py
sleep 2

# Step 2: Reset local changes and sync with remote
echo "2. Syncing with GitHub..."
cd /workspace/Xray-to-CTPA
git fetch origin
git reset --hard origin/main
git clean -fd

# Step 3: Verify configuration
echo "3. Configuration check:"
echo "   batch_size: $(grep "^batch_size:" patchwise_4gpu_distributed/config/model/ddpm_4gpu.yaml)"
echo "   resume_ckpt: $(grep "^resume_ckpt:" patchwise_4gpu_distributed/config/model/ddpm_4gpu.yaml)"

# Step 4: Launch training
echo "4. Starting optimized training..."
cd patchwise_4gpu_distributed
bash launch_ddpm_4gpu.sh

