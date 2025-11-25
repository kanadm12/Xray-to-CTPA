# Patch-Wise VQ-GAN Training Guide (Single GPU)

Complete step-by-step instructions to train patch-wise VQGAN on your dataset using RunPod.

This guide follows the same structure as the baseline training in `baseline_256x256x64_single_gpu/` but uses full resolution (512×512×604) with patch-based processing.

---

## Quick Overview

| Aspect | Baseline | Patch-Wise |
|--------|----------|-----------|
| Resolution | 256×256×64 | 512×512×604 |
| Patch Size | N/A | 256×256×128 |
| Memory/GPU | ~12 GB | ~24 GB (H100/RTX6000) |
| Training Time (30 epochs) | 8 hours | 5-7 days |
| Expected PSNR | 35.3 dB | >37 dB |
| Codebook Usage | 87% | >85% |

---

## Phase 1: Environment Setup

### Step 1: Navigate to Patch-Wise Directory

```bash
cd /workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Create venv (same as baseline approach)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements (same packages as baseline)
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Installation Time**: 5-10 minutes

---

## Phase 2: Data Preparation

### Step 4: Verify NIfTI Dataset Location

Your dataset should already be at `/workspace/datasets/` as NIfTI files (converted from DICOMs).

```bash
# Check dataset location
ls -lh /workspace/datasets/ | head -10

# Verify NIfTI files exist
find /workspace/datasets -name "*.nii.gz" | head -5

# Count total files
find /workspace/datasets -name "*.nii.gz" | wc -l

# Inspect sample volume
python -c "
import nibabel as nib
import glob

files = glob.glob('/workspace/datasets/*.nii.gz')
if files:
    img = nib.load(files[0])
    print(f'✓ Sample: {files[0].split(\"/\")[-1]}')
    print(f'  Shape: {img.shape}')
    print(f'  Data type: {img.get_fdata().dtype}')
    print(f'  Min/Max: {img.get_fdata().min():.1f} / {img.get_fdata().max():.1f}')
    print(f'  Total files: {len(files)}')
else:
    print('✗ No NIfTI files found')
"
```

Expected output:
```
✓ Sample: patient_001.nii.gz
  Shape: (512, 512, 604)
  Data type: float32
  Min/Max: 0.0 / 1.0
  Total files: 42
```

---

## Phase 3: Configuration Setup

### Step 5: Update Dataset Configuration

The baseline has a `custom_data.yaml` - create an equivalent for patchwise:

```bash
# Edit dataset config
nano config/dataset/full_resolution_ctpa.yaml
```

Add or update to:

```yaml
name: FULL_RESOLUTION_CTPA
root_dir: /workspace/datasets
image_channels: 1

train:
  split_ratio: 0.8
  
val:
  split_ratio: 0.2
```

### Step 6: Customize Training Hyperparameters

Edit the base configuration:

```bash
nano config/base_cfg.yaml
```

Ensure it looks like:

```yaml
defaults:
  - dataset: full_resolution_ctpa
  - model: vq_gan_3d_patches

training:
  batch_size: 4  # Adjust based on GPU VRAM
  num_epochs: 30
  learning_rate: 4.5e-6
  num_workers: 4
  
hardware:
  device: cuda
  mixed_precision: true
```

**GPU Memory Guidelines** (for full resolution):
- RTX 6000 / H100 (24GB): `batch_size: 4` ✓
- A100 (40GB): `batch_size: 6-8`
- L40S (48GB): `batch_size: 8-10`

### Step 7: Verify Configuration Loads Correctly

```bash
python -c "
from hydra import compose, initialize_config_dir
import os

config_dir = os.path.join(os.getcwd(), 'config')
initialize_config_dir(config_dir=config_dir, version_base='1.1')
cfg = compose(config_name='base_cfg')

print('✓ Configuration loaded successfully')
print(f'  Dataset: {cfg.dataset.name}')
print(f'  Root Dir: {cfg.dataset.root_dir}')
print(f'  Batch Size: {cfg.training.batch_size}')
print(f'  Epochs: {cfg.training.num_epochs}')
print(f'  Learning Rate: {cfg.training.learning_rate}')
"
```

---

## Phase 4: Training Execution

### Step 8: Launch Training

**Option 1: Using Launch Script (Recommended for RunPod)**

```bash
# Make script executable
chmod +x launch_distributed_training.sh

# Run in detachable screen session (survives SSH disconnect)
screen -S training_patchwise
./launch_distributed_training.sh

# Press Ctrl+A, then D to detach
# Later: screen -r training_patchwise  (to reattach)
```

**Option 2: Direct Python Command**

```bash
export PYTHONPATH=/workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu:$PYTHONPATH
python train/train_vqgan_distributed.py
```

**Option 3: Custom Parameters**

```bash
python train/train_vqgan_distributed.py \
  dataset.root_dir=/workspace/datasets \
  training.num_epochs=30 \
  training.batch_size=4 \
  hydra.run.dir=outputs/custom_run
```

### Step 9: Monitor Training in Real-Time

**Open separate terminals for each**:

```bash
# Terminal 1: GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Training logs
tail -f outputs/vqgan_patches_distributed/logs/training.log

# Terminal 3: TensorBoard (access at http://localhost:6006)
tensorboard --logdir outputs/vqgan_patches_distributed/tensorboard_logs --port 6006
```

**Key Metrics to Watch**:

| Metric | What to Look For | Concern |
|--------|-----------------|---------|
| **Loss** | Steady decrease across epochs | Flat = learning rate too low |
| **Codebook Usage** | >85% | <50% = codebook collapse |
| **Perplexity** | Stabilize 256-512 | Wildly fluctuating = instability |
| **PSNR** | >37 dB (baseline: 35.3) | Decreasing after epoch 5 = overfitting |

---

## Phase 5: Handling Common Issues

### Out of Memory (OOM)

```
Error: CUDA out of memory
```

**Solutions** (in order of effectiveness):
1. Reduce batch size: `training.batch_size=2`
2. Reduce patch size: `model.patch_size=192` (instead of 256)
3. Disable mixed precision: `hardware.mixed_precision=false`
4. Check for other GPU processes: `nvidia-smi`

### Training Loss Not Decreasing

```
Cause: Learning rate mismatch or data issue
```

**Solutions**:
1. Verify data is normalized to [0, 1] range
2. Try higher learning rate: `training.learning_rate=9.0e-6`
3. Check data dimensions match expectations in logs
4. Reduce batch size to see if it's a batch effect

### GPU Not Detected

```
Error: CUDA available: False
```

**Solutions**:
1. Check `nvidia-smi` works in terminal
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify CUDA 11.8+ installed: `nvcc --version`

---

## Phase 6: Checkpointing and Recovery

### Automatic Checkpoints

Training saves checkpoints automatically to:

```
outputs/vqgan_patches_distributed/
├── checkpoints/
│   ├── epoch_00_step_0000.ckpt
│   ├── epoch_05_step_0100.ckpt
│   └── epoch_30_step_0600.ckpt  (final)
├── tensorboard_logs/
├── logs/
│   ├── training.log
│   └── config.yaml
└── validation_results/
```

### Resume from Checkpoint

If training was interrupted:

```bash
python train/train_vqgan_distributed.py \
  ckpt_path=outputs/vqgan_patches_distributed/checkpoints/epoch_15_step_0300.ckpt
```

### Save Final Model

```bash
# Best model is saved at:
cp outputs/vqgan_patches_distributed/checkpoints/epoch_30_step_0600.ckpt \
   outputs/final_model_vqgan_patches.ckpt

# Download locally (from your local machine)
scp -r user@runpod_ip:/workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu/outputs ./results_patchwise
```

---

## Phase 7: Post-Training Validation

### Step 10: Run Validation Script

After training completes:

```bash
# Validate on full dataset
python train/validate_patches.py \
  --checkpoint outputs/vqgan_patches_distributed/checkpoints/epoch_30_step_0600.ckpt \
  --data_dir /workspace/datasets \
  --output_dir validation_results
```

### Step 11: Generate Sample Reconstructions

```bash
# Generate 10 sample reconstructions
python train/generate_samples.py \
  --checkpoint outputs/vqgan_patches_distributed/checkpoints/epoch_30_step_0600.ckpt \
  --num_samples 10 \
  --output_dir sample_reconstructions
```

---

## Architecture Details

### Why Patch-Wise Processing?

Full 512×512×604 volumes (~160M voxels) exceed GPU VRAM limits. Solution: Process as patches.

**Patch Configuration**:
- Patch Size: 256 × 256 × 128 voxels
- Stride: 192 × 192 × 96 voxels
- Overlap: 64 voxels (25%) per dimension
- Patches per Volume: ~24 (3×3×3 grid)

**Overlap Blending**:
- Overlapping regions merged with linear/Gaussian weighting
- Eliminates visible seams in reconstructed volumes
- Ensures smooth gradients at boundaries

### Memory Breakdown

```
Patch: 256×256×128 = 8.4M voxels
Per Patch: 32 MB (float32)
Batch 4: 128 MB input + 256 MB activations = ~400 MB
Model: ~36M parameters = ~144 MB
Total: ~600 MB per patch batch (well under 24GB)
```

### Training Timeline

```
Epoch 1:     10-15 min  (loss ~0.5)
Epoch 5:     10-15 min  (loss ~0.2)
Epoch 10:    10-15 min  (loss ~0.15)
Epoch 30:    10-15 min  (loss ~0.10)
─────────────────────────────
Total:       5-7 days   (for 30 epochs)
```

---

## Comparison with Baseline

Your patch-wise training vs baseline:

| Stage | Baseline (256³) | Patch-Wise (512×512×604) |
|-------|-----------------|-------------------------|
| **Training** | 8 hours / 30 epochs | 5-7 days / 30 epochs |
| **PSNR** | 35.3 dB | Expected >37 dB |
| **Codebook Usage** | 87% | Target >85% |
| **GPU Memory** | 12 GB | 24 GB |
| **Output Quality** | Lower res | Full clinical resolution |

---

## Next Steps After Training

1. **Compare Quality**: PSNR, SSIM with baseline
2. **Analyze Boundaries**: Check for patch seams
3. **Latent Analysis**: Extract encoded representations for DDPM
4. **Fine-tune if Needed**: Adjust hyperparameters based on results
5. **Deploy**: Export checkpoint for inference

---

## Troubleshooting Checklist

- [ ] GPU is available (`nvidia-smi` shows device)
- [ ] Dataset exists at `/workspace/datasets/`
- [ ] Config points to correct data directory
- [ ] Batch size fits in GPU memory (start with 4)
- [ ] First epoch starts without errors
- [ ] TensorBoard shows loss decreasing
- [ ] Codebook usage >50% by epoch 2

---

## Quick Reference Commands

```bash
# Setup and start
cd /workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu
pip install -r requirements.txt
./launch_distributed_training.sh

# Monitor
tensorboard --logdir outputs/vqgan_patches_distributed/tensorboard_logs

# Resume if interrupted
python train/train_vqgan_distributed.py ckpt_path=outputs/vqgan_patches_distributed/checkpoints/epoch_15_step_0300.ckpt

# Validate
python train/validate_patches.py --checkpoint outputs/vqgan_patches_distributed/checkpoints/epoch_30_step_0600.ckpt

# Archive results
tar -czf vqgan_patchwise_results.tar.gz outputs/vqgan_patches_distributed/
```

---

For more details, see:
- `README.md` - Architecture overview
- `TRAINING_GUIDE.md` - Detailed technical guide
- `VALIDATION_GUIDE.md` - Validation procedures
- `../baseline_256x256x64_single_gpu/RUNPOD_TRAINING_GUIDE.md` - Baseline reference
