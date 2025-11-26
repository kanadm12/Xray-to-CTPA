#!/bin/bash

##############################################################################
# Complete X-ray → CTPA Training Pipeline (4-GPU)
##############################################################################
#
# This script runs the full training pipeline:
# 1. VQ-GAN training with discriminator (compression + reconstruction)
# 2. DDPM training (X-ray → CTPA generation in latent space)
#
# Total time: ~30-36 hours on 4× H200 GPUs
#
##############################################################################

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    Complete X-ray → CTPA Training Pipeline (4-GPU DDP)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check prerequisites
echo "Checking prerequisites..."
echo ""

# GPU check
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -lt 4 ]; then
    echo -e "${RED}✗ ERROR: Need 4 GPUs but found only $NUM_GPUS${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Found $NUM_GPUS GPUs"

# Dataset check
DATASET_DIR="/workspace/Xray-to-CTPA/datasets/"
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}✗ ERROR: Dataset directory not found: $DATASET_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Dataset directory found"

# X-ray data check
XRAY_DIR="/workspace/Xray-to-CTPA/xray_data/"
if [ ! -d "$XRAY_DIR" ]; then
    echo -e "${YELLOW}⚠${NC} Warning: X-ray directory not found: $XRAY_DIR"
    echo "  This is needed for DDPM training (step 2)"
    echo "  You can skip step 2 if you only want VQ-GAN"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TRAINING PLAN"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Step 1: VQ-GAN with Discriminator (~6-7 hours)"
echo "  - Learns to compress and reconstruct CT volumes"
echo "  - Adversarial training for sharp, realistic outputs"
echo "  - Target PSNR: 30-32 dB"
echo "  - Target SSIM: 0.95+"
echo ""
echo "Step 2: DDPM Training (~24-48 hours)"
echo "  - Learns X-ray → CTPA mapping in latent space"
echo "  - Uses frozen VQ-GAN encoder/decoder"
echo "  - Generates CTPA from X-ray images"
echo ""
echo "Total estimated time: ~30-36 hours"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Ask user what to run
echo "What would you like to do?"
echo "  1) Run complete pipeline (VQ-GAN + DDPM)"
echo "  2) Run VQ-GAN only"
echo "  3) Run DDPM only (requires trained VQ-GAN)"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting complete pipeline..."
        TRAIN_VQGAN=true
        TRAIN_DDPM=true
        ;;
    2)
        echo ""
        echo "Starting VQ-GAN training only..."
        TRAIN_VQGAN=true
        TRAIN_DDPM=false
        ;;
    3)
        echo ""
        echo "Starting DDPM training only..."
        
        # Check if VQ-GAN checkpoint exists
        VQGAN_CKPT="./outputs/vqgan_patches_4gpu_disc/checkpoints/last.ckpt"
        if [ ! -f "$VQGAN_CKPT" ]; then
            echo -e "${RED}✗ ERROR: VQ-GAN checkpoint not found: $VQGAN_CKPT${NC}"
            echo "Please train VQ-GAN first (option 2)"
            exit 1
        fi
        
        TRAIN_VQGAN=false
        TRAIN_DDPM=true
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# STEP 1: VQ-GAN Training
if [ "$TRAIN_VQGAN" = true ]; then
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  STEP 1: VQ-GAN Training with Discriminator             ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    
    chmod +x launch_4gpu_vqgan_disc.sh
    ./launch_4gpu_vqgan_disc.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ VQ-GAN training failed to start${NC}"
        exit 1
    fi
    
    VQGAN_PID=$!
    echo ""
    echo -e "${GREEN}✓ VQ-GAN training started${NC}"
    echo "  PID: $VQGAN_PID"
    echo "  Logs: training_4gpu_disc.log"
    echo ""
    
    if [ "$TRAIN_DDPM" = true ]; then
        echo "Waiting for VQ-GAN training to complete..."
        echo "This will take approximately 6-7 hours."
        echo ""
        echo "Monitor progress:"
        echo "  tail -f training_4gpu_disc.log"
        echo ""
        
        # Wait for VQ-GAN to finish
        wait $VQGAN_PID
        
        # Check if checkpoint was created
        VQGAN_CKPT="./outputs/vqgan_patches_4gpu_disc/checkpoints/last.ckpt"
        if [ ! -f "$VQGAN_CKPT" ]; then
            echo -e "${RED}✗ VQ-GAN checkpoint not found after training${NC}"
            echo "Check training_4gpu_disc.log for errors"
            exit 1
        fi
        
        echo -e "${GREEN}✓ VQ-GAN training completed successfully${NC}"
        echo ""
    fi
fi

# STEP 2: DDPM Training
if [ "$TRAIN_DDPM" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  STEP 2: DDPM Training (X-ray → CTPA)                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    
    chmod +x launch_4gpu_ddpm.sh
    ./launch_4gpu_ddpm.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ DDPM training failed to start${NC}"
        exit 1
    fi
    
    DDPM_PID=$!
    echo ""
    echo -e "${GREEN}✓ DDPM training started${NC}"
    echo "  PID: $DDPM_PID"
    echo "  Logs: training_ddpm_4gpu.log"
    echo ""
    echo "This will take approximately 24-48 hours."
    echo ""
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TRAINING STATUS"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ "$TRAIN_VQGAN" = true ]; then
    echo -e "${GREEN}✓${NC} VQ-GAN training: Running or completed"
    echo "  Monitor: tail -f training_4gpu_disc.log"
    echo "  Checkpoints: ./outputs/vqgan_patches_4gpu_disc/checkpoints/"
    echo ""
fi

if [ "$TRAIN_DDPM" = true ]; then
    echo -e "${GREEN}✓${NC} DDPM training: Running"
    echo "  Monitor: tail -f training_ddpm_4gpu.log"
    echo "  Checkpoints: ./checkpoints/ddpm_4gpu/"
    echo ""
fi

echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To stop training:"
if [ "$TRAIN_VQGAN" = true ]; then
    echo "  pkill -f train_vqgan_4gpu"
fi
if [ "$TRAIN_DDPM" = true ]; then
    echo "  pkill -f train_ddpm_4gpu"
fi
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Pipeline launched successfully!${NC}"
echo ""
