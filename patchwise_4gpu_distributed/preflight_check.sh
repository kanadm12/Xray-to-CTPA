#!/bin/bash

# Pre-Training Checklist for 4-GPU Setup
# Run this before starting training to ensure everything is ready

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        4-GPU VQ-GAN Training - Pre-Flight Checklist         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((pass_count++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((fail_count++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "═══════════════════════════════════════════════════════════════"
echo "  HARDWARE CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# GPU Count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ge 4 ]; then
    check_pass "Found $NUM_GPUS GPUs (need 4)"
else
    check_fail "Only $NUM_GPUS GPUs found (need 4)"
fi

# GPU Memory
echo ""
echo "GPU Memory Status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader 2>/dev/null | head -4

# CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
    check_pass "CUDA toolkit installed ($CUDA_VERSION)"
else
    check_warn "CUDA toolkit not in PATH (may be okay if PyTorch has CUDA)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SOFTWARE CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
check_pass "Python $PYTHON_VERSION"

# PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    check_pass "PyTorch $TORCH_VERSION"
    
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        check_pass "PyTorch CUDA support enabled"
    else
        check_fail "PyTorch CUDA support NOT available"
    fi
else
    check_fail "PyTorch not installed"
fi

# PyTorch Lightning
if python -c "import pytorch_lightning" 2>/dev/null; then
    PL_VERSION=$(python -c "import pytorch_lightning; print(pytorch_lightning.__version__)")
    check_pass "PyTorch Lightning $PL_VERSION"
else
    check_fail "PyTorch Lightning not installed"
fi

# Other dependencies
for pkg in "hydra" "omegaconf" "nibabel" "numpy"; do
    if python -c "import $pkg" 2>/dev/null; then
        check_pass "$pkg installed"
    else
        check_fail "$pkg not installed"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DATA CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# Dataset directory
DATASET_DIR="/workspace/Xray-to-CTPA/datasets/"
if [ -d "$DATASET_DIR" ]; then
    NUM_PATIENTS=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    NUM_FILES=$(find "$DATASET_DIR" -name "*.nii.gz" | wc -l)
    
    check_pass "Dataset directory found: $DATASET_DIR"
    check_pass "Patient folders: $NUM_PATIENTS"
    check_pass "Total .nii.gz files: $NUM_FILES"
    
    if [ "$NUM_FILES" -lt 10 ]; then
        check_warn "Only $NUM_FILES files found (expected 30-50)"
    fi
else
    check_fail "Dataset directory not found: $DATASET_DIR"
    check_warn "Update path in config/dataset/ctpa_4gpu.yaml"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  CONFIGURATION CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# Config files
for config_file in "config/base_cfg.yaml" "config/dataset/ctpa_4gpu.yaml" "config/model/vqgan_4gpu.yaml"; do
    if [ -f "$config_file" ]; then
        check_pass "$config_file exists"
    else
        check_fail "$config_file missing"
    fi
done

# Training script
if [ -f "train_vqgan_4gpu.py" ]; then
    check_pass "train_vqgan_4gpu.py exists"
else
    check_fail "train_vqgan_4gpu.py missing"
fi

# Launch script
if [ -f "launch_4gpu_training.sh" ]; then
    check_pass "launch_4gpu_training.sh exists"
    
    if [ -x "launch_4gpu_training.sh" ]; then
        check_pass "launch_4gpu_training.sh is executable"
    else
        check_warn "launch_4gpu_training.sh not executable (run: chmod +x launch_4gpu_training.sh)"
    fi
else
    check_fail "launch_4gpu_training.sh missing"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  MODEL IMPORTS CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# VQ-GAN imports
if python -c "import sys; sys.path.insert(0, '..'); from vq_gan_3d.model.vqgan_patches import VQGANPatches" 2>/dev/null; then
    check_pass "VQGANPatches import successful"
else
    check_fail "Cannot import VQGANPatches (check vq_gan_3d/ directory)"
fi

# Dataset imports
if python -c "import sys; sys.path.insert(0, '..'); from dataset.patch_dataset import PatchDataset" 2>/dev/null; then
    check_pass "PatchDataset import successful"
else
    check_fail "Cannot import PatchDataset (check dataset/ directory)"
fi

# Utils imports
if python -c "import sys; sys.path.insert(0, '..'); from utils.patch_utils import extract_patches_3d" 2>/dev/null; then
    check_pass "patch_utils import successful"
else
    check_fail "Cannot import patch_utils (check utils/ directory)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  NETWORK CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# NCCL
if python -c "import torch.distributed as dist" 2>/dev/null; then
    check_pass "torch.distributed available"
else
    check_fail "torch.distributed not available"
fi

# Port availability
MASTER_PORT=29500
if ! lsof -Pi :$MASTER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    check_pass "Port $MASTER_PORT available"
else
    check_warn "Port $MASTER_PORT in use (will use different port)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DISK SPACE CHECKS"
echo "═══════════════════════════════════════════════════════════════"

# Check available disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -gt 50 ]; then
    check_pass "Disk space: ${AVAILABLE_SPACE}GB available (need ~10GB for checkpoints)"
else
    check_warn "Only ${AVAILABLE_SPACE}GB available (may need more space)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo -e "Checks passed: ${GREEN}$pass_count${NC}"
echo -e "Checks failed: ${RED}$fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ ALL CHECKS PASSED - READY TO LAUNCH TRAINING!          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review configuration: cat config/model/vqgan_4gpu.yaml"
    echo "  2. Launch training:      ./launch_4gpu_training.sh"
    echo "  3. Monitor logs:         tail -f training_4gpu.log"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ✗ SOME CHECKS FAILED - PLEASE FIX BEFORE TRAINING        ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please fix the failed checks above before launching training."
    echo "For help, see README.md or QUICKSTART.md"
    echo ""
    exit 1
fi
