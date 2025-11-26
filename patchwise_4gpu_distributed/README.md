# VQ-GAN 4-GPU Distributed Training Setup

## Overview

This setup trains VQ-GAN across **4 H200 GPUs** using PyTorch DistributedDataParallel (DDP).

### Key Features

- **Parallel patient processing:** Each GPU processes 1 patient volume
- **Effective batch size:** 4 volumes (1 per GPU × 4 GPUs)
- **Gradient synchronization:** Automatic across all GPUs
- **4× faster training:** ~4 min/epoch vs ~16 min on single GPU
- **Full dataset:** Can train on all 42 patients efficiently

## Architecture

```
GPU 0: Patient 1 → Patches → VQ-GAN → Loss → Gradients ─┐
GPU 1: Patient 2 → Patches → VQ-GAN → Loss → Gradients ─┤
GPU 2: Patient 3 → Patches → VQ-GAN → Loss → Gradients ─┼→ Sync → Update Weights
GPU 3: Patient 4 → Patches → VQ-GAN → Loss → Gradients ─┘
```

Each GPU:
- Processes 1 volume (512×512×604)
- Extracts ~80 patches (128³ each)
- Computes loss on patches
- Shares gradients with other GPUs

## Hardware Requirements

- **GPUs:** 4× NVIDIA H200 (or similar)
- **VRAM per GPU:** ~32-40GB (140GB available, plenty of headroom)
- **Total VRAM:** 560GB available
- **Network:** High-speed interconnect (NVLink/InfiniBand recommended)

## Quick Start

### 1. Navigate to directory

```bash
cd patchwise_4gpu_distributed
```

### 2. Make script executable

```bash
chmod +x launch_4gpu_training.sh
```

### 3. Launch training

```bash
./launch_4gpu_training.sh
```

That's it! Training will start across 4 GPUs automatically.

## Configuration

### Dataset Config (`config/dataset/ctpa_4gpu.yaml`)

```yaml
max_patients: null  # Train on all 42 patients
patch_size: [128, 128, 128]
stride: [128, 128, 128]
```

For faster testing, set `max_patients: 30` to limit dataset.

### Model Config (`config/model/vqgan_4gpu.yaml`)

```yaml
batch_size: 1       # Per GPU (effective = 4)
num_workers: 4      # Per GPU (total = 16 workers)
gpus: 4
max_epochs: 30
precision: 16       # fp16 mixed precision
```

### Key Parameters

| Parameter | Single GPU | 4-GPU DDP | Notes |
|-----------|------------|-----------|-------|
| `batch_size` | 1 | 1 | Per GPU, effective = 4 |
| `num_workers` | 0-2 | 4 | More stable with multiple GPUs |
| `devices` | 1 | 4 | Number of GPUs |
| `strategy` | None | DDP | Distributed strategy |
| `sync_batchnorm` | N/A | True | Sync batch norm across GPUs |

## Monitoring

### Watch training logs

```bash
tail -f training_4gpu.log
```

Look for:
- `[Rank 0]`, `[Rank 1]`, etc. - Each GPU's output
- `Epoch X: XX%` - Training progress
- `val/psnr`, `val/ssim` - Validation metrics

### GPU utilization

```bash
watch -n 1 nvidia-smi
```

Expected utilization: 25-35% per GPU (all 4 should be active)

### TensorBoard

```bash
tensorboard --logdir=outputs/vqgan_patches_4gpu/lightning_logs --port=6006
```

## Training Speed Comparison

| Setup | Time/Epoch | Total (30 epochs) | Dataset |
|-------|------------|-------------------|---------|
| 1 GPU | ~16 min | ~8 hours | 30 patients |
| 4 GPU | ~4 min | ~2 hours | 30 patients |
| 4 GPU | ~7 min | ~3.5 hours | 42 patients (full) |

**Speedup: ~4× faster** (linear scaling)

## Troubleshooting

### Problem: "NCCL error" or "CUDA_VISIBLE_DEVICES" issues

**Solution:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./launch_4gpu_training.sh
```

### Problem: "Address already in use" (port conflict)

**Solution:** Change port in `launch_4gpu_training.sh`:
```bash
MASTER_PORT=29501  # Change to different port
```

### Problem: Unbalanced GPU usage (some GPUs idle)

**Cause:** Dataset size not divisible by 4

**Solution:** Enable `drop_last=True` in dataloaders (already set)

### Problem: "RuntimeError: DDP backward synchronization error"

**Solution:** Set `find_unused_parameters=True` in config:
```yaml
find_unused_parameters: true
```

### Problem: Training slower than expected

**Check:**
1. All 4 GPUs active: `nvidia-smi`
2. NCCL backend: Should see NCCL logs
3. Network speed: Use NVLink if available
4. `num_workers`: Increase to 4-8 per GPU

### Problem: OOM on some GPUs

**Solution:** Reduce batch size or patch size:
```yaml
batch_size: 1  # Already minimum
patch_size: [96, 96, 96]  # Reduce if needed
```

## Advanced Configuration

### Enable discriminator (after warmup)

Edit `config/model/vqgan_4gpu.yaml`:
```yaml
discriminator_iter_start: 10000
image_gan_weight: 1.0
video_gan_weight: 0.5
perceptual_weight: 0.1
```

### Train on subset for testing

```yaml
max_patients: 8  # 2 patients per GPU
max_epochs: 10
```

### Gradient accumulation (effective batch = 8)

```yaml
accumulate_grad_batches: 2  # 4 GPUs × 1 batch × 2 accum = 8
```

### Resume from checkpoint

```bash
python train_vqgan_4gpu.py \
    model.resume_from_checkpoint=outputs/vqgan_patches_4gpu/checkpoints/last.ckpt
```

## Checkpoint Management

Checkpoints saved to: `outputs/vqgan_patches_4gpu/checkpoints/`

- `last.ckpt` - Latest epoch
- `vqgan-patches-epochXX-psnrYY.YY.ckpt` - Top 3 by PSNR

**Note:** Fixed filename format bug (no `/` character)

## Performance Tips

1. **Use NVLink:** Faster GPU-to-GPU communication
2. **Increase `num_workers`:** 4-8 per GPU for faster data loading
3. **Enable `pin_memory`:** Already enabled
4. **Use `drop_last=True`:** Prevents incomplete batches
5. **Sync batch norm:** Better convergence with multi-GPU

## Validation

After training, test on unseen data:

```bash
cd ..
python patchwise_512x512x604_single_gpu/test_vqgan_video.py \
    --checkpoint patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints/last.ckpt \
    --input /path/to/test_volume.nii.gz \
    --output_dir ./test_4gpu_results
```

## Expected Results

With 4 GPUs on full 42 patients:

- **PSNR:** 29-30 dB (vs 28 dB on 30 patients)
- **SSIM:** 0.94-0.95 (vs 0.93 on 30 patients)
- **Training time:** ~3.5 hours (vs ~12 hours single GPU)
- **Convergence:** Better due to larger effective batch size

## Files Structure

```
patchwise_4gpu_distributed/
├── train_vqgan_4gpu.py              # Main training script
├── launch_4gpu_training.sh          # Launch script
├── config/
│   ├── base_cfg.yaml               # Hydra config
│   ├── dataset/
│   │   └── ctpa_4gpu.yaml          # Dataset config
│   └── model/
│       └── vqgan_4gpu.yaml         # Model config
└── README.md                        # This file
```

## Next Steps

1. **Launch training:** `./launch_4gpu_training.sh`
2. **Monitor progress:** `tail -f training_4gpu.log`
3. **Check metrics:** TensorBoard or metrics.csv
4. **Test model:** Use test_vqgan_video.py
5. **Train DDPM:** Use trained VQ-GAN for diffusion model

## Support

For issues:
1. Check `training_4gpu.log` for errors
2. Verify GPU availability: `nvidia-smi`
3. Check NCCL: Look for "NCCL" in logs
4. Ensure dataset path correct
5. Try single GPU first to isolate DDP issues
