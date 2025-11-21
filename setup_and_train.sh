#!/bin/bash

# ============================================
# Quick Start Script for RunPod Training
# Usage: bash setup_and_train.sh
# ============================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔════════════════════════════════════════╗"
echo "║   X-ray2CTPA RunPod Training Setup     ║"
echo "╚════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/Xray-2CTPA_spartis"
DATASET_DIR="$WORKSPACE/datasets"
DATASET_NAME="data_new"

# ============================================
# Step 1: Clone Repository
# ============================================
echo -e "${YELLOW}[1/7] Cloning repository...${NC}"
if [ -d "$REPO_DIR" ]; then
    echo -e "${GREEN}✓${NC} Repository already cloned, updating..."
    cd $REPO_DIR
    git pull
else
    cd $WORKSPACE
    git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
    echo -e "${GREEN}✓${NC} Repository cloned successfully"
fi

# ============================================
# Step 2: Install Dependencies
# ============================================
echo -e "${YELLOW}[2/7] Installing Python dependencies...${NC}"
cd $REPO_DIR

# Upgrade pip, setuptools, and wheel FIRST - critical for Python 3.12
echo -e "${YELLOW}Upgrading build tools for Python 3.12...${NC}"
pip install --upgrade pip setuptools wheel -q

# Install PyTorch from NVIDIA index first (this handles CUDA correctly)
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 -q

# ALWAYS use RunPod-compatible requirements (Python 3.11/3.12 compatible)
pip install -q -r requirements-runpod.txt --no-deps
echo -e "${GREEN}✓${NC} Dependencies installed"

# ============================================
# Step 3: Verify Dataset Exists
# ============================================
echo -e "${YELLOW}[3/7] Verifying dataset...${NC}"
if [ ! -d "$DATASET_DIR/$DATASET_NAME" ]; then
    echo -e "${RED}✗${NC} Dataset not found at: $DATASET_DIR/$DATASET_NAME"
    echo ""
    echo -e "${YELLOW}Please upload your dataset using one of these methods:${NC}"
    echo "1. RunPod Files: Upload 'data_new' folder to /workspace/datasets/"
    echo "2. Google Drive:"
    echo "   pip install gdown"
    echo "   gdown --id YOUR_FILE_ID -O /workspace/dataset.tar.gz"
    echo "   mkdir -p /workspace/datasets && tar -xzf /workspace/dataset.tar.gz -C /workspace/datasets/"
    echo "3. Hugging Face:"
    echo "   huggingface-cli download-cache --repo-type dataset YOUR_USERNAME/YOUR_DATASET --local-dir /workspace/datasets/data_new"
    exit 1
fi

# Count files
FILE_COUNT=$(find $DATASET_DIR/$DATASET_NAME -name "*.nii.gz" ! -name "*swapped*" | wc -l)
TOTAL_FILES=$(find $DATASET_DIR/$DATASET_NAME -name "*.nii.gz" | wc -l)
SWAPPED_FILES=$((TOTAL_FILES - FILE_COUNT))

echo -e "${GREEN}✓${NC} Dataset verified"
echo "  - Valid files: $FILE_COUNT"
echo "  - Excluded (swapped): $SWAPPED_FILES"
echo "  - Total files: $TOTAL_FILES"

# ============================================
# Step 4: Update Configuration
# ============================================
echo -e "${YELLOW}[4/7] Updating configuration...${NC}"
CONFIG_FILE="$REPO_DIR/config/dataset/custom_data.yaml"
sed -i "s|root_dir: .*|root_dir: $DATASET_DIR/$DATASET_NAME/|" $CONFIG_FILE
echo -e "${GREEN}✓${NC} Configuration updated"
echo "  Config: $CONFIG_FILE"

# ============================================
# Step 5: Verify CUDA & GPU
# ============================================
echo -e "${YELLOW}[5/7] Verifying CUDA setup...${NC}"
CUDA_INFO=$(python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')")
echo -e "${GREEN}✓${NC} $CUDA_INFO"

# Verify GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1}')
echo "  - GPU VRAM: ${GPU_MEMORY}MB"

# ============================================
# Step 6: Display Training Info
# ============================================
echo -e "${YELLOW}[6/7] Training configuration...${NC}"
echo "  - Repository: $REPO_DIR"
echo "  - Dataset: $DATASET_DIR/$DATASET_NAME/"
echo "  - Output: $REPO_DIR/lightning_logs/"
echo ""
echo -e "${YELLOW}Training will use:${NC}"
echo "  - Dataset: CUSTOM_DATA"
echo "  - Model: VQGAN (3D)"
echo "  - Batch Size: 2"
echo "  - Precision: fp16"
echo "  - Learning Rate: 3e-4"

# ============================================
# Step 7: Start Training
# ============================================
echo -e "${YELLOW}[7/7] Starting training...${NC}"
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}Training started! Monitor with:${NC}"
echo -e "${GREEN}  nvidia-smi -l 1${NC}"
echo -e "${GREEN}  tail -f lightning_logs/version_*/checkpoints/*.log${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""

export PYTHONPATH=$REPO_DIR
cd $REPO_DIR

# Run training with better error handling
if bash train/scripts/train_vqgan_custom.sh; then
    echo -e "${GREEN}"
    echo "════════════════════════════════════════"
    echo "Training completed successfully!"
    echo "════════════════════════════════════════"
    echo -e "${NC}"
else
    echo -e "${RED}"
    echo "════════════════════════════════════════"
    echo "Training encountered an error!"
    echo "════════════════════════════════════════"
    echo -e "${NC}"
    exit 1
fi
