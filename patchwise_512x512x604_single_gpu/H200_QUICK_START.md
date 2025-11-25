# H200 Pod: Quick Start Commands

## One-Line Setup & Training (Recommended)

```bash
cd /workspace && git clone https://github.com/kanadm12/Xray-to-CTPA.git && cd Xray-to-CTPA/patchwise_512x512x604_single_gpu && chmod +x setup_and_train_h200.sh && ./setup_and_train_h200.sh
```

## Step-by-Step Commands

### 1. Clone Repository
```bash
cd /workspace
git clone https://github.com/kanadm12/Xray-to-CTPA.git
cd Xray-to-CTPA/patchwise_512x512x604_single_gpu
```

### 2. Run Automated Setup
```bash
chmod +x setup_and_train_h200.sh
./setup_and_train_h200.sh
```

This will:
- ✅ Verify GPU (H200)
- ✅ Install all dependencies
- ✅ Verify dataset at `/workspace/datasets/`
- ✅ Verify configuration
- ✅ Create output directories
- ✅ Show next steps

### 3. Start Training

**Option A: Direct (Foreground)**
```bash
python train/train_vqgan_distributed.py
```

**Option B: Detachable Session (RunPod Recommended)**
```bash
screen -S training_patchwise
python train/train_vqgan_distributed.py
# Press Ctrl+A, then D to detach
# Later to reattach: screen -r training_patchwise
```

**Option C: Using Launch Script**
```bash
chmod +x launch_distributed_training.sh
./launch_distributed_training.sh
```

---

## Monitoring During Training

### Terminal 1: TensorBoard
```bash
tensorboard --logdir outputs/vqgan_patches_distributed/tensorboard_logs --port 6006
# Access at: http://localhost:6006
```

### Terminal 2: GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Terminal 3: Training Logs
```bash
tail -f outputs/vqgan_patches_distributed/logs/training.log
```

---

## If Dataset Not Downloaded Yet

### Option 1: azcopy (Fastest - ~1-2 hours)
```bash
# Install azcopy first if needed
mkdir -p /workspace/datasets
azcopy copy "https://ctbigdata.blob.core.windows.net/ct-big-data/rsna/rsna_extracted/train" "/workspace/datasets/" --recursive
```

### Option 2: Python Script
```bash
cd /workspace/Xray-to-CTPA
python download_azure_dataset.py
```

---

## Performance Expectations

- **Training Time**: 6-8 hours for 30 epochs
- **GPU Memory Used**: ~15-20 GB / 141 GB
- **GPU Utilization**: 70-85%
- **Batch Time**: 15-25 seconds per batch
- **Expected PSNR**: >37 dB (vs baseline 35.3 dB)

---

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check if NVIDIA drivers work
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
```bash
# Edit config/model/vq_gan_3d_patches.yaml
batch_size: 2  # Reduce from 4
num_workers: 4  # Reduce from 8
```

### Training Slow
```bash
# Edit config/model/vq_gan_3d_patches.yaml
batch_size: 8          # Increase from 4
precision: 16-mixed    # Use mixed precision instead of 32
```

### Dataset Not Found
```bash
ls -la /workspace/datasets/  # Check path
find /workspace/datasets -name "*.nii.gz" | head -10  # Verify files exist
```

---

## Resume Training (if interrupted)

```bash
python train/train_vqgan_distributed.py \
  ckpt_path=outputs/vqgan_patches_distributed/checkpoints/epoch_15_step_0300.ckpt
```

---

## After Training Complete

### View Results
```bash
ls -la outputs/vqgan_patches_distributed/checkpoints/
# final.ckpt is your trained model
```

### Download Results Locally
```bash
# From your local machine:
scp -r user@h200_pod:/workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu/outputs ./results_h200
```

### Validate Model
```bash
python train/validate_patches.py \
  --checkpoint outputs/vqgan_patches_distributed/checkpoints/final.ckpt \
  --data_dir /workspace/datasets
```

---

## Key Metrics to Monitor (TensorBoard)

Watch these in real-time:
- **train/recon_loss**: Should decrease smoothly
- **train/perplexity**: Should stabilize ~256-512
- **val/psnr**: Should improve toward >37 dB
- **val/codebook_usage_%**: Target >85%

---

## Environment Variables (Optional)

```bash
# Reduce verbosity
export HYDRA_FULL_ERROR=0

# Use specific GPU (usually not needed on pod with 1 GPU)
export CUDA_VISIBLE_DEVICES=0

# Enable cuDNN benchmark for speed
export CUDNN_BENCHMARK=1
```

---

**Created**: November 2025
**Target**: Single H200 GPU Pod (141GB VRAM)
**Expected Outcome**: Production-quality VQ-GAN model (PSNR >37dB)
