# RunPod Training Setup Guide - X-ray2CTPA

Complete step-by-step instructions to train VQGAN on your custom dataset using RunPod.

## Prerequisites
- RunPod account with GPU pod running (recommended: RTX4090, A100, or H100)
- Your dataset uploaded to cloud storage (Google Drive, AWS S3, or Hugging Face)
- GitHub repository cloned

---

## Phase 1: Initial Setup (First Time Only)

### Step 1: Launch RunPod Pod
1. Go to [RunPod.io](https://www.runpod.io/)
2. Select a GPU (RTX4090 recommended for faster training)
3. Choose a template with **PyTorch + CUDA 11.8+** or **NVIDIA CUDA 12.0+**
4. Click "Connect" and open JupyterLab or SSH into the pod

### Step 2: Clone Repository
```bash
cd /workspace
git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
cd Xray-2CTPA_spartis
```

### Step 3: Install Dependencies
```bash
# Update system packages
apt-get update && apt-get install -y git wget

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4: Download Your Dataset
Choose one method below:

#### Option A: From Google Drive
```bash
pip install gdown

# Download from Google Drive link
# Replace FILE_ID with your actual Google Drive file ID
gdown --id FILE_ID -O /workspace/dataset.tar.gz

# Extract
mkdir -p /workspace/datasets
tar -xzf /workspace/dataset.tar.gz -C /workspace/datasets/
```

#### Option B: From Hugging Face Hub
```bash
pip install huggingface-hub

# Download dataset
huggingface-cli download-cache --repo-type dataset your-username/your-dataset --local-dir /workspace/datasets/data_new
```

#### Option C: From AWS S3
```bash
pip install awscli

# Configure AWS credentials
aws configure

# Sync dataset
aws s3 sync s3://your-bucket/data_new /workspace/datasets/data_new/
```

#### Option D: Direct Upload via RunPod
1. Click "Files" in RunPod interface
2. Create folder: `/workspace/datasets/`
3. Upload your `data_new` folder

### Step 5: Verify Dataset Structure
```bash
# Check dataset is in correct location
ls -la /workspace/datasets/data_new/

# Count total .nii.gz files (excluding swapped)
find /workspace/datasets/data_new -name "*.nii.gz" ! -name "*swapped*" | wc -l

# Check directory structure
tree /workspace/datasets/data_new/ -L 2 | head -50
# (or use: ls -la /workspace/datasets/data_new/patient_001/)
```

---

## Phase 2: Configuration Setup

### Step 6: Update Dataset Path in Config
```bash
# Edit the custom_data config file
nano /workspace/Xray-2CTPA_spartis/config/dataset/custom_data.yaml
```

Update the `root_dir`:
```yaml
name: CUSTOM_DATA
root_dir: /workspace/datasets/data_new/  # Update this path
image_channels: 1
```

Save and exit (Ctrl+X, then Y, then Enter)

### Step 7: Optional - Adjust Training Parameters
Edit the training script if needed:
```bash
nano /workspace/Xray-2CTPA_spartis/train/scripts/train_vqgan_custom.sh
```

Key parameters you might adjust:
- `BATCH_SIZE`: 2 (or 1 if OOM), higher = faster training but more VRAM
- `NUM_WORKERS`: 4 (adjust based on pod specs)
- `CUDA_DEVICE`: 0 (or match your pod's GPU index)
- `EMBEDDING_DIM`: 8 (latent space dimension)
- `DOWNSAMPLE`: [2,2,2] (compression ratio)

---

## Phase 3: Start Training

### Option A: Using Bash Script (Recommended)
```bash
cd /workspace/Xray-2CTPA_spartis
export PYTHONPATH=$PWD

# Make script executable
chmod +x train/scripts/train_vqgan_custom.sh

# Run training
bash train/scripts/train_vqgan_custom.sh
```

### Option B: Direct Python Command
```bash
cd /workspace/Xray-2CTPA_spartis
export PYTHONPATH=$PWD

python train/train_vqgan.py \
    dataset=custom_data \
    model=vq_gan_3d \
    model.gpus=1 \
    model.precision=16 \
    model.embedding_dim=8 \
    model.n_hiddens=16 \
    model.downsample=[2,2,2] \
    model.num_workers=4 \
    model.batch_size=2 \
    model.lr=3e-4 \
    model.default_root_dir_postfix='custom_data'
```

### Option C: Using Python Script
Create `/workspace/train.py`:
```python
#!/usr/bin/env python3

import os
import subprocess
import sys

os.chdir('/workspace/Xray-2CTPA_spartis')
os.environ['PYTHONPATH'] = os.getcwd()

cmd = [
    'python', 'train/train_vqgan.py',
    'dataset=custom_data',
    'model=vq_gan_3d',
    'model.gpus=1',
    'model.precision=16',
    'model.batch_size=2',
    'model.num_workers=4',
    'model.lr=3e-4',
    'model.default_root_dir_postfix=custom_data'
]

result = subprocess.run(cmd)
sys.exit(result.returncode)
```

Then run:
```bash
python /workspace/train.py
```

---

## Phase 4: Monitoring Training

### Check Training Progress
```bash
# Option 1: Watch real-time logs
tail -f /workspace/Xray-2CTPA_spartis/lightning_logs/version_*/checkpoints/*.log

# Option 2: Monitor GPU usage
nvidia-smi -l 1  # Updates every 1 second

# Option 3: Monitor system resources
watch -n 1 'nvidia-smi && echo "---" && free -h'
```

### TensorBoard Monitoring (Optional)
```bash
# Start TensorBoard in background
tensorboard --logdir=/workspace/Xray-2CTPA_spartis/lightning_logs &

# Access via: http://your-pod-url:6006
```

### Expected Training Output
```
Epoch 0, Batch 10/100 - Loss: 0.523, Recon Loss: 0.412, GAN Loss: 0.111
Epoch 1, Batch 10/100 - Loss: 0.498, Recon Loss: 0.391, GAN Loss: 0.107
...
Checkpoint saved: latest_checkpoint.ckpt
```

---

## Phase 5: Handling Issues

### Issue: CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
model.batch_size=1

# Solution 2: Reduce num_workers
model.num_workers=0

# Solution 3: Use full precision instead of half
model.precision=32

# Solution 4: Increase compression
model.downsample=[4,4,4]
```

### Issue: Dataset files not found
```bash
# Verify dataset path
ls -la /workspace/datasets/data_new/

# Check file count
find /workspace/datasets/data_new -name "*.nii.gz" -type f | head -5

# Check for correct extensions
find /workspace/datasets/data_new -type f -name "*swapped*"
```

### Issue: "Dataset is not available"
```bash
# Verify config file exists
ls -la config/dataset/custom_data.yaml

# Check config content
cat config/dataset/custom_data.yaml

# Verify dataset name matches
# Config file should have: name: CUSTOM_DATA
```

### Issue: Training too slow
```bash
# Increase num_workers (if CPU/RAM allows)
model.num_workers=8

# Use higher precision (trades speed for VRAM)
model.precision=16  # or 32

# Check GPU utilization
nvidia-smi -l 1
```

### Issue: Pod disconnecting during training
```bash
# Use tmux or screen to keep session alive
tmux new-session -d -s training 'bash train/scripts/train_vqgan_custom.sh'

# Reconnect later with:
tmux attach-session -t training

# Or use nohup
nohup bash train/scripts/train_vqgan_custom.sh > training.log 2>&1 &
```

---

## Phase 6: Saving Checkpoints & Data

### Download Trained Model
```bash
# Zip the checkpoints
cd /workspace/Xray-2CTPA_spartis
tar -czf lightning_logs.tar.gz lightning_logs/

# Option 1: Download via RunPod Files
# Click Files → Find lightning_logs.tar.gz → Download

# Option 2: Upload to cloud
pip install rclone
rclone sync lightning_logs/ gdrive:my-folder/lightning_logs/
```

### Save to Cloud Storage During Training
```bash
# Create a backup script - save as backup.sh
#!/bin/bash
while true; do
    tar -czf /tmp/checkpoint_backup.tar.gz /workspace/Xray-2CTPA_spartis/lightning_logs/
    aws s3 cp /tmp/checkpoint_backup.tar.gz s3://your-bucket/backups/
    # Or with Google Drive:
    # rclone copy lightning_logs/ gdrive:backups/
    sleep 3600  # Backup every hour
done

# Run in background
bash backup.sh &
```

---

## Phase 7: Complete Automation Script

Save this as `/workspace/setup_and_train.sh`:

```bash
#!/bin/bash

set -e  # Exit on error

echo "======================================"
echo "X-ray2CTPA RunPod Training Setup"
echo "======================================"

# Variables
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/Xray-2CTPA_spartis"
DATASET_DIR="$WORKSPACE/datasets"

# Step 1: Clone repo
echo "[1/6] Cloning repository..."
if [ ! -d "$REPO_DIR" ]; then
    cd $WORKSPACE
    git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
else
    echo "Repository already cloned"
fi

# Step 2: Install dependencies
echo "[2/6] Installing dependencies..."
cd $REPO_DIR
pip install -q -r requirements.txt

# Step 3: Verify dataset
echo "[3/6] Verifying dataset..."
if [ ! -d "$DATASET_DIR/data_new" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR/data_new"
    echo "Please upload your dataset first"
    exit 1
fi

FILE_COUNT=$(find $DATASET_DIR/data_new -name "*.nii.gz" ! -name "*swapped*" | wc -l)
echo "Found $FILE_COUNT valid .nii.gz files"

# Step 4: Update config
echo "[4/6] Updating configuration..."
sed -i "s|root_dir: .*|root_dir: $DATASET_DIR/data_new/|" $REPO_DIR/config/dataset/custom_data.yaml

# Step 5: Verify CUDA
echo "[5/6] Verifying CUDA setup..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Step 6: Start training
echo "[6/6] Starting training..."
export PYTHONPATH=$REPO_DIR
cd $REPO_DIR
bash train/scripts/train_vqgan_custom.sh

echo "======================================"
echo "Training started!"
echo "Logs available at: $REPO_DIR/lightning_logs/"
echo "======================================"
```

Run it:
```bash
chmod +x /workspace/setup_and_train.sh
bash /workspace/setup_and_train.sh
```

---

## Phase 8: Quick Reference - Commands Cheat Sheet

```bash
# View GPU stats
nvidia-smi

# Check running processes
ps aux | grep python

# Kill training process
pkill -f train_vqgan.py

# View training logs
tail -f lightning_logs/version_0/checkpoints/*.log

# Check dataset files
find /workspace/datasets/data_new -name "*.nii.gz" | head -10

# View available disk space
df -h /workspace

# SSH into pod (from local machine)
ssh user@runpod-ip-address
```

---

## Final Checklist

Before starting training:
- [ ] GPU pod is running
- [ ] Repository cloned to `/workspace/`
- [ ] Dependencies installed successfully
- [ ] Dataset uploaded to `/workspace/datasets/data_new/`
- [ ] Config file updated with correct path
- [ ] Dataset files verified (at least 5+ patients)
- [ ] CUDA is available (`nvidia-smi` works)
- [ ] Enough disk space available (`df -h`)

After training starts:
- [ ] Monitor GPU usage (should be 80-100%)
- [ ] Check memory usage (should stabilize after warm-up)
- [ ] Verify checkpoints being saved
- [ ] Download/backup checkpoints periodically

---

## Next Steps After Training

Once VQGAN training is complete:
1. Download trained model: `lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt`
2. Train DDPM (diffusion model) on latent space
3. Combine VQGAN + DDPM for full pipeline
4. Run inference on test images

For more details, see `CUSTOM_DATASET_TRAINING.md`
