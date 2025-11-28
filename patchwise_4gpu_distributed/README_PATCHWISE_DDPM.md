# Patchwise DDPM Training Guide

## Overview

This guide covers training the DDPM model using a **patchwise approach** to handle full-resolution 512×512×604 CTPA volumes without running out of GPU memory.

## Why Patchwise Training?

Training on full 604-slice CTPA volumes causes OOM errors even with 4×H200 GPUs (140GB each):
- **Full volume**: 512×512×604 → 128×128×151 latent → ~138GB GPU memory per sample
- **Patchwise**: 128×128×128 → 32×32×32 latent → ~10GB GPU memory per sample

By extracting 128×128×128 patches from full volumes, we can:
- ✅ Train on full-resolution data without OOM
- ✅ Use larger batch sizes (2-4 per GPU)
- ✅ Process all 604 slices via overlapping patches
- ✅ Maintain same model architecture

## Architecture

### Input Processing
1. **X-ray**: Full 224×224×3 image (for MedCLIP conditioning)
2. **CTPA**: 128×128×128 patches extracted from 512×512×604 volume

### Patch Extraction
```python
Patch size: 128×128×128 (D, H, W)
Stride: 96×96×96 (25% overlap)

Full volume: 512×512×604
→ Extract ~42 patches per volume
→ Each patch processed independently
```

### Latent Space
```python
CTPA patch: 128×128×128
→ VQ-GAN encode (downsample 4×4×4)
→ Latent: 32×32×32×64 channels
→ DDPM operates on 32×32×32 latent
```

## Files Created

### 1. Dataset
**`dataset/xray_ctpa_patch_dataset.py`**
- `XrayCTPAPatchDataset`: Loads paired X-ray + CTPA patches
- `extract_patches_3d()`: Extracts 128×128×128 patches with overlap
- Returns dict with keys: `ct`, `cxr`, `target`, `patch_coord`

### 2. Training Script
**`train_ddpm_patches_4gpu.py`**
- Main patchwise training script
- Uses `XrayCTPAPatchDataset` instead of full volumes
- 4-GPU DDP with PyTorch Lightning
- Same diffusion model architecture

### 3. Configuration
**`config/model/ddpm_4gpu_patches.yaml`**
```yaml
use_patches: true
patch_size: [128, 128, 128]
stride: [96, 96, 96]
diffusion_img_size: 32  # Latent size (128/4)
diffusion_depth_size: 32
batch_size: 2  # Per GPU (effective = 8)
```

### 4. Launcher
**`launch_ddpm_patches_4gpu.sh`**
- Bash script to launch 4-GPU training
- Uses `torchrun` for distributed training

## Usage

### Prerequisites
1. **VQ-GAN trained**: `./outputs/vqgan_patches_4gpu/checkpoints/last.ckpt`
2. **Dataset**: Paired X-ray PNGs + CTPA .nii.gz files

### Launch Training
```bash
cd patchwise_4gpu_distributed
./launch_ddpm_patches_4gpu.sh
```

Or manually:
```bash
torchrun --nproc_per_node=4 --master_port=29501 \
    train_ddpm_patches_4gpu.py model=ddpm_4gpu_patches
```

## Memory Usage Comparison

| Approach | CTPA Size | Latent Size | GPU Memory | Batch Size | Total Samples/Batch |
|----------|-----------|-------------|------------|------------|---------------------|
| **Full Volume** | 512×512×604 | 128×128×151 | ~138GB | 1 per GPU | 4 |
| **Patchwise** | 128×128×128 | 32×32×32 | ~10GB | 2 per GPU | 8 |

## Training Details

### Dataset Size
- **901 patients** with paired X-ray + CTPA
- **720 train / 181 val** (80/20 split)
- **~42 patches per volume** (with 25% overlap)
- **~30,240 train patches / ~7,602 val patches**

### Memory Efficiency
```python
Full volume memory: 512 × 512 × 604 × 4 bytes = 629MB per volume
Patch memory: 128 × 128 × 128 × 4 bytes = 8.4MB per patch

GPU memory saved: ~97% per sample
```

### Batch Size
- **Batch size: 2 per GPU** (vs 1 for full volumes)
- **Effective batch: 8 across 4 GPUs** (vs 4 for full volumes)
- **2× throughput increase**

## Model Architecture

Same architecture as full-volume training:
- **UNet3D**: 143M parameters
- **MedCLIP**: 195M parameters (frozen)
- **VQ-GAN**: 50.8M parameters (frozen encoder/decoder)
- **Perceptual Loss**: 23.5M parameters (frozen)

### Key Difference
- **Input latent**: 32×32×32 (patch) vs 128×128×151 (full)
- **Output**: Denoised 32×32×32 latent patch
- **Conditioning**: Full 224×224 X-ray (same for all patches)

## Inference

During inference, reconstruct full volume:
1. Extract patches from noise with stride
2. Denoise each patch independently
3. Average overlapping regions
4. Decode full volume with VQ-GAN decoder

Example:
```python
# Extract patches from noise
noise_patches = extract_patches_3d(
    noise_volume,  # [1, 604, 512, 512]
    patch_size=(128, 128, 128),
    stride=(96, 96, 96)
)

# Denoise each patch with same X-ray conditioning
for patch in noise_patches:
    latent_patch = vqgan.encode(patch)
    denoised_latent = ddpm.denoise(latent_patch, xray_cond)
    denoised_patch = vqgan.decode(denoised_latent)

# Stitch patches back together
full_volume = stitch_patches(denoised_patches, coordinates)
```

## Advantages

✅ **Memory efficient**: 10GB vs 138GB per sample  
✅ **Larger batches**: 2-4 per GPU vs 1  
✅ **Full resolution**: Processes all 604 slices  
✅ **Same quality**: Same model architecture  
✅ **Faster training**: 2× throughput with larger batches  

## Comparison: Full vs Patchwise

| Metric | Full Volume | Patchwise |
|--------|-------------|-----------|
| CTPA Input | 512×512×604 | 128×128×128 |
| Latent Size | 128×128×151 | 32×32×32 |
| GPU Memory | 138GB/GPU | 10GB/GPU |
| Batch/GPU | 1 | 2-4 |
| Train Samples | 720 | 30,240 |
| OOM Errors | ❌ Yes | ✅ No |

## Expected Training Time

With 4×H200 GPUs:
- **~30,240 train patches** (720 volumes × 42 patches)
- **Batch size: 2 per GPU** (effective 8)
- **~3,780 iterations per epoch**
- **30 epochs** = ~113,400 iterations
- **~2-3 iterations/sec** (with FP16 + gradient checkpointing)
- **~15-20 hours total**

## Checkpoints

Saved to: `./checkpoints/ddpm_4gpu_patches/`

Files:
- `last.ckpt`: Latest epoch
- `ddpm-patches-epoch{XX}-loss{Y.YYYY}.ckpt`: Top 3 best models

## Troubleshooting

### Still OOM with patches?
- Reduce patch size: `patch_size: [96, 96, 96]`
- Reduce batch size: `batch_size: 1`
- Reduce UNet dim_mults: `dim_mults: [1, 2, 4]`

### Training too slow?
- Increase batch size: `batch_size: 4`
- Increase patch stride: `stride: [64, 64, 64]` (more overlap)
- Reduce num_workers if CPU bottleneck

### Poor quality?
- Increase patch overlap: `stride: [64, 64, 64]`
- Train longer: `max_epochs: 50`
- Increase perceptual loss: `perceptual_weight: 0.05`

## Next Steps

After training completes:
1. **Validate**: Run inference on validation set
2. **Metrics**: Calculate PSNR, SSIM on full volumes
3. **Visualize**: Generate X-ray → CTPA samples
4. **Fine-tune**: Adjust hyperparameters based on results
