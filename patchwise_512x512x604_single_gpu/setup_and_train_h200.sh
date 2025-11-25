#!/bin/bash

##############################################################################
# H200 GPU Pod Setup and Training Script
# Complete end-to-end setup for X-ray2CTPA patchwise VQ-GAN training
##############################################################################

set -e  # Exit on any error

echo "================================================================================"
echo "X-ray2CTPA: Patchwise VQ-GAN Training on H200 GPU"
echo "================================================================================"
echo ""

# ============================================================================
# PHASE 1: Initial Setup
# ============================================================================

echo "[PHASE 1] Environment Setup"
echo "--------"

# Navigate to workspace
cd /workspace
echo "✓ Working directory: $(pwd)"

# Clone repository if not present
if [ ! -d "Xray-to-CTPA" ]; then
    echo "Cloning repository..."
    git clone https://github.com/kanadm12/Xray-to-CTPA.git
    cd Xray-to-CTPA
else
    echo "✓ Repository already exists"
    cd Xray-to-CTPA
    git pull origin main
fi

echo ""

# ============================================================================
# PHASE 2: Verify GPU
# ============================================================================

echo "[PHASE 2] GPU Verification"
echo "--------"

# Check GPU
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$GPU_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))")
    echo "✓ GPU Detected: $GPU_NAME"
    echo "✓ GPU Memory: ${GPU_MEMORY} GB"
else
    echo "✗ GPU not detected!"
    echo "Make sure you're on a GPU pod. Run 'nvidia-smi' to verify."
    exit 1
fi

echo ""

# ============================================================================
# PHASE 3: Navigate to Patchwise Directory
# ============================================================================

echo "[PHASE 3] Navigate to Patchwise Implementation"
echo "--------"

cd /workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu
echo "✓ Current directory: $(pwd)"

echo ""

# ============================================================================
# PHASE 4: Install Dependencies
# ============================================================================

echo "[PHASE 4] Installing Dependencies"
echo "--------"

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "(Skipping venv - using system Python)"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install requirements
echo "Installing PyTorch, Lightning, and dependencies..."
pip install --quiet -r requirements.txt

echo "✓ Dependencies installed successfully"

echo ""

# ============================================================================
# PHASE 5: Verify Dataset
# ============================================================================

echo "[PHASE 5] Dataset Verification"
echo "--------"

if [ ! -d "/workspace/datasets" ]; then
    echo "✗ Dataset not found at /workspace/datasets"
    echo "Please download the dataset first:"
    echo "  azcopy copy 'https://ctbigdata.blob.core.windows.net/ct-big-data/rsna/rsna_extracted/train' '/workspace/datasets/' --recursive"
    exit 1
fi

NIFTI_COUNT=$(find /workspace/datasets -name "*.nii.gz" 2>/dev/null | wc -l)
if [ "$NIFTI_COUNT" -eq 0 ]; then
    echo "✗ No .nii.gz files found in /workspace/datasets"
    exit 1
fi

echo "✓ Dataset found: $NIFTI_COUNT NIfTI files"

# Verify sample volume
python -c "
import nibabel as nib
import glob
files = glob.glob('/workspace/datasets/*.nii.gz')
if files:
    img = nib.load(files[0])
    print(f'✓ Sample volume shape: {img.shape}')
    print(f'✓ Data type: {img.get_fdata().dtype}')
else:
    print('✗ No files found')
" || exit 1

echo ""

# ============================================================================
# PHASE 6: Verify Configuration
# ============================================================================

echo "[PHASE 6] Configuration Verification"
echo "--------"

python -c "
from hydra import compose, initialize_config_dir
import os

config_dir = os.path.join(os.getcwd(), 'config')
initialize_config_dir(config_dir=config_dir, version_base='1.1')
cfg = compose(config_name='base_cfg')
print(f'✓ Dataset: {cfg.dataset.name}')
print(f'✓ Batch size: {cfg.model.batch_size}')
print(f'✓ Epochs: {cfg.model.max_epochs}')
print(f'✓ Learning rate: {cfg.model.learning_rate}')
" || {
    echo "✗ Configuration loading failed"
    exit 1
}

echo ""

# ============================================================================
# PHASE 7: Create Output Directories
# ============================================================================

echo "[PHASE 7] Creating Output Directories"
echo "--------"

mkdir -p outputs/vqgan_patches_distributed/checkpoints
mkdir -p outputs/vqgan_patches_distributed/tensorboard_logs
mkdir -p outputs/vqgan_patches_distributed/logs

echo "✓ Output directories created"

echo ""

# ============================================================================
# PHASE 8: Ready for Training
# ============================================================================

echo "================================================================================"
echo "✓ SETUP COMPLETE - Ready for Training!"
echo "================================================================================"
echo ""
echo "Next Steps:"
echo "--------"
echo ""
echo "Option 1: Start Training (Foreground)"
echo "  python train/train_vqgan_distributed.py"
echo ""
echo "Option 2: Start Training (Detachable - RunPod)"
echo "  screen -S training_patchwise"
echo "  python train/train_vqgan_distributed.py"
echo "  (Press Ctrl+A, then D to detach)"
echo "  (Later: screen -r training_patchwise)"
echo ""
echo "Option 3: Use Launch Script"
echo "  chmod +x launch_distributed_training.sh"
echo "  ./launch_distributed_training.sh"
echo ""
echo "Monitoring:"
echo "--------"
echo "  TensorBoard:"
echo "    tensorboard --logdir outputs/vqgan_patches_distributed/tensorboard_logs --port 6006"
echo ""
echo "  Training Logs:"
echo "    tail -f outputs/vqgan_patches_distributed/logs/training.log"
echo ""
echo "  GPU Usage:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "================================================================================"
echo "Expected Training Time: 6-8 hours (30 epochs on H200)"
echo "Expected GPU Memory: ~15-20 GB / 141 GB"
echo "Expected Batch Time: 15-25 seconds per batch"
echo "================================================================================"
