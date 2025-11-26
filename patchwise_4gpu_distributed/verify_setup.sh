#!/bin/bash

##############################################################################
# Pre-training verification script
# Checks all requirements before launching 4-GPU training
##############################################################################

echo "=========================================="
echo "VQ-GAN 4-GPU Setup Verification"
echo "=========================================="
echo ""

EXIT_CODE=0

# Check 1: GPU availability
echo "[1/7] Checking GPUs..."
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ge 4 ]; then
    echo "  ✓ Found $NUM_GPUS GPUs"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -4
else
    echo "  ✗ ERROR: Found only $NUM_GPUS GPUs, need 4"
    EXIT_CODE=1
fi
echo ""

# Check 2: PyTorch and CUDA
echo "[2/7] Checking PyTorch..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "  ✗ ERROR: PyTorch not found"
    EXIT_CODE=1
}
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
python -c "import torch; print(f'  ✓ CUDA version: {torch.version.cuda}')" 2>/dev/null
echo ""

# Check 3: PyTorch Lightning
echo "[3/7] Checking PyTorch Lightning..."
python -c "import pytorch_lightning as pl; print(f'  ✓ Lightning {pl.__version__}')" 2>/dev/null || {
    echo "  ✗ ERROR: PyTorch Lightning not found"
    EXIT_CODE=1
}
echo ""

# Check 4: Dataset path
echo "[4/7] Checking dataset..."
DATASET_PATH="/workspace/Xray-to-CTPA/datasets/"
if [ -d "$DATASET_PATH" ]; then
    NUM_PATIENTS=$(ls -d $DATASET_PATH/*/ 2>/dev/null | wc -l)
    echo "  ✓ Dataset found: $DATASET_PATH"
    echo "  ✓ Patient folders: $NUM_PATIENTS"
    
    # Count total .nii.gz files
    NUM_FILES=$(find $DATASET_PATH -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "  ✓ Total .nii.gz files: $NUM_FILES"
else
    echo "  ✗ ERROR: Dataset not found at $DATASET_PATH"
    echo "    Please update path in config/dataset/ctpa_4gpu.yaml"
    EXIT_CODE=1
fi
echo ""

# Check 5: Required Python packages
echo "[5/7] Checking Python dependencies..."
REQUIRED_PACKAGES="torch torchvision pytorch_lightning hydra-core omegaconf nibabel numpy"
for pkg in $REQUIRED_PACKAGES; do
    python -c "import ${pkg//-/_}" 2>/dev/null && echo "  ✓ $pkg" || {
        echo "  ✗ Missing: $pkg"
        EXIT_CODE=1
    }
done
echo ""

# Check 6: Model imports
echo "[6/7] Checking model imports..."
python -c "import sys; sys.path.insert(0, '..'); from vq_gan_3d.model.vqgan_patches import VQGANPatches; print('  ✓ VQGANPatches')" 2>/dev/null || {
    echo "  ✗ ERROR: Cannot import VQGANPatches"
    echo "    Check that vq_gan_3d/ directory exists in parent folder"
    EXIT_CODE=1
}
python -c "import sys; sys.path.insert(0, '..'); from dataset.patch_dataset import PatchDataset; print('  ✓ PatchDataset')" 2>/dev/null || {
    echo "  ✗ ERROR: Cannot import PatchDataset"
    echo "    Check that dataset/ directory exists in parent folder"
    EXIT_CODE=1
}
echo ""

# Check 7: Configuration files
echo "[7/7] Checking configuration files..."
CONFIG_FILES=(
    "config/base_cfg.yaml"
    "config/dataset/ctpa_4gpu.yaml"
    "config/model/vqgan_4gpu.yaml"
    "train_vqgan_4gpu.py"
    "launch_4gpu_training.sh"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ Missing: $file"
        EXIT_CODE=1
    fi
done
echo ""

# Summary
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "Ready to launch training:"
    echo "  chmod +x launch_4gpu_training.sh"
    echo "  ./launch_4gpu_training.sh"
else
    echo "✗ Some checks failed"
    echo ""
    echo "Please fix the errors above before training"
fi
echo "=========================================="

exit $EXIT_CODE
