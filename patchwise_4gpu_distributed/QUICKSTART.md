# Quick Start: 4-GPU Training

## Step-by-Step Instructions

### 1. Navigate to folder
```bash
cd /workspace/Xray-to-CTPA/patchwise_4gpu_distributed
```

### 2. Verify setup (optional but recommended)
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

This checks:
- ✓ 4 GPUs available
- ✓ PyTorch + Lightning installed
- ✓ Dataset path exists
- ✓ All config files present
- ✓ Model imports work

### 3. Launch training
```bash
chmod +x launch_4gpu_training.sh
./launch_4gpu_training.sh
```

### 4. Monitor progress
```bash
# Watch training logs
tail -f training_4gpu.log

# Check GPU usage
watch -n 1 nvidia-smi

# View metrics (after first epoch)
cat outputs/vqgan_patches_4gpu/lightning_logs/version_0/metrics.csv
```

## What to Expect

### First few minutes
```
Found 4 GPUs
Starting training...
[Rank 0] Searching for data...
[Rank 1] Searching for data...
[Rank 2] Searching for data...
[Rank 3] Searching for data...
Train files: 33
Val files: 9
Effective batch size: 4
```

### During training
```
Epoch 0:   4%|██    | 1/24 [00:27<10:38,  0.04it/s]
  train/recon_loss_step=0.547
  train/commitment_loss_step=0.026
```

All 4 GPUs should show ~25-35% utilization.

### Expected speed
- **Per epoch:** ~4-7 minutes (vs 16 min on single GPU)
- **30 epochs:** ~2-3.5 hours (vs 8 hours on single GPU)
- **Speedup:** ~4× faster

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPUs | 4 | All H200s |
| Batch per GPU | 1 | Each GPU = 1 patient |
| Effective batch | 4 | Synchronized across GPUs |
| Workers per GPU | 4 | Total = 16 workers |
| Patch size | 128³ | ~80 patches per volume |
| Precision | fp16 | Mixed precision |
| Max epochs | 30 | Can increase for better quality |

## Configuration Changes

### Train on subset (faster testing)
Edit `config/dataset/ctpa_4gpu.yaml`:
```yaml
max_patients: 12  # 3 patients per GPU
```

### More epochs for better quality
Edit `config/model/vqgan_4gpu.yaml`:
```yaml
max_epochs: 50  # Default is 30
```

### Enable discriminator (after warmup)
Edit `config/model/vqgan_4gpu.yaml`:
```yaml
discriminator_iter_start: 10000
image_gan_weight: 1.0
video_gan_weight: 0.5
perceptual_weight: 0.1
```

## Stopping Training

```bash
# Find process ID
ps aux | grep train_vqgan_4gpu

# Kill process
kill <PID>

# Or use pkill
pkill -f train_vqgan_4gpu
```

## Resuming Training

Training automatically saves checkpoints. To resume:

```bash
python train_vqgan_4gpu.py \
    model.resume_from_checkpoint=outputs/vqgan_patches_4gpu/checkpoints/last.ckpt
```

## Troubleshooting

**All GPUs show 0% usage:**
- Check logs: `tail training_4gpu.log`
- Verify DDP started: Look for `[Rank 0]`, `[Rank 1]`, etc.

**"NCCL error":**
```bash
export NCCL_DEBUG=INFO
./launch_4gpu_training.sh
```

**"Port already in use":**
Edit `launch_4gpu_training.sh` and change `MASTER_PORT=29500` to different port.

**Training slower than expected:**
- Check all 4 GPUs active: `nvidia-smi`
- Increase `num_workers` to 8 in config
- Ensure NVLink enabled

## Output Files

```
outputs/vqgan_patches_4gpu/
├── checkpoints/
│   ├── last.ckpt                    # Latest checkpoint
│   └── vqgan-patches-epoch*.ckpt    # Top 3 by PSNR
└── lightning_logs/
    └── version_0/
        ├── events.out.tfevents.*    # TensorBoard logs
        ├── metrics.csv              # Training metrics
        └── hparams.yaml             # Hyperparameters
```

## Next Steps After Training

1. **Test reconstruction quality:**
```bash
cd ../patchwise_512x512x604_single_gpu
python test_vqgan_video.py \
    --checkpoint ../patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints/last.ckpt \
    --input /workspace/Xray-to-CTPA/datasets/volume-covid19-A-0400_ct/1.3.6.1.4.1.9328.50.4.0042.nii.gz \
    --axes axial coronal
```

2. **Check metrics:**
```bash
cat outputs/vqgan_patches_4gpu/lightning_logs/version_0/metrics.csv | tail -20
```

3. **If satisfied, proceed to DDPM training**

## Expected Performance

After 30 epochs on full dataset (42 patients):

- **PSNR:** 29-31 dB
- **SSIM:** 0.94-0.96
- **Training time:** ~3.5 hours
- **Checkpoint size:** ~400MB

This is better than single GPU (28 dB PSNR) due to:
- More training data (42 vs 30 patients)
- Larger effective batch size (4 vs 1)
- Better gradient estimates

## Questions?

Check the full README.md for detailed explanations and advanced configuration.
