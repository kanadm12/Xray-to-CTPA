#!/bin/bash
# Download checkpoints from Google Drive to RunPod using rclone
# This script sets up rclone and downloads VQ-GAN and DDPM checkpoints

set -e  # Exit on error

echo "=========================================="
echo "Google Drive Checkpoint Downloader"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Target directories on RunPod
VQGAN_CHECKPOINT_DIR="/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints"
DDPM_CHECKPOINT_DIR="/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches"

# Step 1: Check if rclone is installed
echo -e "${BLUE}[1/5] Checking rclone installation...${NC}"
if ! command -v rclone &> /dev/null; then
    echo -e "${YELLOW}rclone not found. Installing...${NC}"
    curl https://rclone.org/install.sh | sudo bash
    echo -e "${GREEN}✓ rclone installed${NC}"
else
    echo -e "${GREEN}✓ rclone already installed ($(rclone --version | head -n 1))${NC}"
fi
echo ""

# Step 2: Configure rclone for Google Drive
echo -e "${BLUE}[2/5] Configuring rclone for Google Drive...${NC}"
if rclone listremotes | grep -q "gdrive:"; then
    echo -e "${GREEN}✓ Google Drive remote 'gdrive' already configured${NC}"
    echo -e "${YELLOW}To reconfigure, run: rclone config reconnect gdrive:${NC}"
else
    echo -e "${YELLOW}Google Drive remote not found. Starting configuration...${NC}"
    echo ""
    echo -e "${YELLOW}Follow these steps:${NC}"
    echo "1. Choose: n (New remote)"
    echo "2. Name: gdrive"
    echo "3. Storage: drive (Google Drive)"
    echo "4. Client ID: (press Enter to skip)"
    echo "5. Client Secret: (press Enter to skip)"
    echo "6. Scope: 1 (Full access)"
    echo "7. Root folder: (press Enter to skip)"
    echo "8. Service account: (press Enter to skip)"
    echo "9. Auto config: n (No - we'll use the provided link)"
    echo "10. Copy the URL and open in browser to authenticate"
    echo "11. Paste the verification code back"
    echo ""
    rclone config
fi
echo ""

# Step 3: List available folders (optional)
echo -e "${BLUE}[3/5] Listing your Google Drive folders...${NC}"
echo -e "${YELLOW}Looking for checkpoint directories...${NC}"
echo ""
rclone lsd gdrive: | grep -i checkpoint || echo -e "${YELLOW}No 'checkpoint' folders found in root. You'll need to specify the full path.${NC}"
echo ""

# Step 4: Download VQ-GAN checkpoints
echo -e "${BLUE}[4/5] Downloading VQ-GAN checkpoints...${NC}"
echo -e "${YELLOW}Enter the Google Drive path to your VQ-GAN checkpoints folder:${NC}"
echo "Example: checkpoints/vqgan_patches_4gpu or My Drive/X-ray2CTPA/vqgan_checkpoints"
read -p "Path: " GDRIVE_VQGAN_PATH

if [ -n "$GDRIVE_VQGAN_PATH" ]; then
    echo -e "${BLUE}Creating local directory: ${VQGAN_CHECKPOINT_DIR}${NC}"
    mkdir -p "$VQGAN_CHECKPOINT_DIR"
    
    echo -e "${BLUE}Downloading from gdrive:${GDRIVE_VQGAN_PATH}/ to ${VQGAN_CHECKPOINT_DIR}/${NC}"
    rclone copy "gdrive:${GDRIVE_VQGAN_PATH}" "$VQGAN_CHECKPOINT_DIR" \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 5s \
        --include "*.ckpt" \
        --include "*.pt" \
        --include "*.pth" \
        --include "*.yaml" \
        --include "*.json"
    
    echo -e "${GREEN}✓ VQ-GAN checkpoints downloaded${NC}"
    echo "Files in $VQGAN_CHECKPOINT_DIR:"
    ls -lh "$VQGAN_CHECKPOINT_DIR"
else
    echo -e "${YELLOW}Skipping VQ-GAN download${NC}"
fi
echo ""

# Step 5: Download DDPM checkpoints
echo -e "${BLUE}[5/5] Downloading DDPM checkpoints...${NC}"
echo -e "${YELLOW}Enter the Google Drive path to your DDPM checkpoints folder:${NC}"
echo "Example: checkpoints/ddpm_4gpu_patches or My Drive/X-ray2CTPA/ddpm_checkpoints"
read -p "Path: " GDRIVE_DDPM_PATH

if [ -n "$GDRIVE_DDPM_PATH" ]; then
    echo -e "${BLUE}Creating local directory: ${DDPM_CHECKPOINT_DIR}${NC}"
    mkdir -p "$DDPM_CHECKPOINT_DIR"
    
    echo -e "${BLUE}Downloading from gdrive:${GDRIVE_DDPM_PATH}/ to ${DDPM_CHECKPOINT_DIR}/${NC}"
    rclone copy "gdrive:${GDRIVE_DDPM_PATH}" "$DDPM_CHECKPOINT_DIR" \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 5s \
        --include "*.ckpt" \
        --include "*.pt" \
        --include "*.pth" \
        --include "*.yaml" \
        --include "*.json"
    
    echo -e "${GREEN}✓ DDPM checkpoints downloaded${NC}"
    echo "Files in $DDPM_CHECKPOINT_DIR:"
    ls -lh "$DDPM_CHECKPOINT_DIR"
else
    echo -e "${YELLOW}Skipping DDPM download${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}=========================================="
echo -e "✓ Download Complete!"
echo -e "==========================================${NC}"
echo ""
echo "Checkpoint locations:"
echo "  VQ-GAN: $VQGAN_CHECKPOINT_DIR"
echo "  DDPM:   $DDPM_CHECKPOINT_DIR"
echo ""
echo "To verify downloads:"
echo "  ls -lh $VQGAN_CHECKPOINT_DIR"
echo "  ls -lh $DDPM_CHECKPOINT_DIR"
echo ""
echo "To sync again (if files were added):"
echo "  rclone sync gdrive:YOUR_PATH $VQGAN_CHECKPOINT_DIR"
