# VQ-GAN Training with Discriminator

## Overview

This configuration enables **adversarial training** with discriminators to achieve sharper, more realistic reconstructions beyond what pure L1 loss can provide.

## Key Differences from Base Training

### Loss Configuration

**Without Discriminator (Current):**
```yaml
image_gan_weight: 0.0      # Disabled
video_gan_weight: 0.0      # Disabled
perceptual_weight: 0.0     # Disabled
l1_weight: 1.0             # Only reconstruction loss
```

**With Discriminator (New):**
```yaml
image_gan_weight: 1.0      # 2D adversarial loss
video_gan_weight: 0.5      # 3D temporal consistency
perceptual_weight: 0.1     # LPIPS perceptual loss
gan_feat_weight: 4.0       # Feature matching (stability)
l1_weight: 1.0             # Still use reconstruction
discriminator_iter_start: 10000  # Warmup period
```

### Training Strategy

1. **Warmup Phase (0-10k steps / ~5 epochs):**
   - Only L1 + commitment loss (like current training)
   - Discriminators loaded but not active
   - Builds good initial representations

2. **Adversarial Phase (10k+ steps):**
   - Discriminators activated
   - Generator learns to fool discriminators
   - Results in sharper, more realistic outputs
   - Potential PSNR: 30-32 dB (vs current 28 dB)

### Expected Improvements

| Metric | Without Disc | With Disc | Improvement |
|--------|--------------|-----------|-------------|
| PSNR | ~28 dB | ~30-32 dB | +7-14% |
| SSIM | ~0.93 | ~0.95+ | +2% |
| Visual Quality | Smooth/blurry | Sharp/realistic | Significant |
| Training Time | ~16 min/epoch | ~20-25 min/epoch | +25-50% |

## Usage

### Option 1: Resume from Current Checkpoint

Continue training with discriminators enabled (recommended):

```bash
cd patchwise_512x512x604_single_gpu
chmod +x train_with_discriminator.sh
./train_with_discriminator.sh
```

This will:
- Load your current VQ-GAN weights (28 dB baseline)
- Enable discriminators after warmup
- Train for 50 total epochs
- Output to `outputs/vqgan_patches_with_disc/`

### Option 2: Start Fresh

Train from scratch with discriminators:

```bash
# Remove or rename checkpoint reference in script
vim train_with_discriminator.sh  # Comment out CHECKPOINT line
./train_with_discriminator.sh
```

### Option 3: Manual Command

```bash
python train/train_vqgan_distributed.py \
    --config-path ../config \
    --config-name base_cfg \
    dataset=full_resolution_ctpa \
    model=vq_gan_3d_with_discriminator \
    dataset.max_patients=30 \
    model.resume_from_checkpoint=outputs/vqgan_patches_distributed/checkpoints/last.ckpt
```

## Monitoring

Watch for these changes after discriminator activation (step 10k):

```bash
tail -f training_with_disc.log
```

**Look for:**
- New losses appear: `train/disc_loss`, `train/gen_loss`, `train/perceptual_loss`
- PSNR starts increasing again (was plateaued at 28)
- Training slows down slightly (discriminator forward/backward)

**Check TensorBoard:**
```bash
tensorboard --logdir=outputs/vqgan_patches_with_disc/lightning_logs
```

## Training Schedule

**Total epochs: 50**
- Epochs 0-5: Warmup (L1 only)
- Epochs 6-50: Full adversarial training
- Expected total time: ~15-20 hours (vs 8 hours for 30 epochs without disc)

## Memory Considerations

**Discriminators add:**
- Image discriminator: 2.8M params
- Video discriminator: 11.0M params
- Total: +13.8M params (~27% increase)

**VRAM impact:**
- Without disc: ~23% utilization (32GB / 140GB)
- With disc: ~30-35% utilization (estimated 42-49GB)
- Still safe on H200 (140GB capacity)

If you hit OOM:
1. Reduce `video_gan_weight` to 0.0 (disable 3D disc, keep 2D)
2. Reduce `perceptual_weight` to 0.0 (LPIPS is 14.7M params)
3. Further reduce micro_batch_size if needed

## Loss Weight Tuning

If training becomes unstable (losses diverge):

**Make discriminator weaker:**
```yaml
image_gan_weight: 0.5      # Reduce from 1.0
video_gan_weight: 0.25     # Reduce from 0.5
```

**Make discriminator stronger:**
```yaml
image_gan_weight: 1.5      # Increase from 1.0
gan_feat_weight: 8.0       # More feature matching
```

## Expected Results

**After 50 epochs with discriminator:**
- **PSNR:** 30-32 dB (currently 28 dB)
- **SSIM:** 0.95+ (currently 0.93)
- **Visual quality:** Sharper edges, better textures, less blur
- **Codebook usage:** More efficient code utilization

**When to use this:**
- ✅ Current model plateaued
- ✅ Want photorealistic reconstructions
- ✅ Have time for longer training
- ✅ DDPM needs high-quality latents

**When NOT to use:**
- ❌ Current quality sufficient for your task
- ❌ Limited training time
- ❌ Unstable GAN training concerns
- ❌ Just need compression (current works fine)

## Next Steps After Training

1. Test reconstruction quality:
```bash
python test_vqgan_video.py \
    --checkpoint outputs/vqgan_patches_with_disc/checkpoints/last.ckpt \
    --input /path/to/test_volume.nii.gz
```

2. Compare with non-discriminator version visually

3. If satisfied, proceed to DDPM training using the better VQ-GAN

## Troubleshooting

**Problem: Discriminator loss explodes**
- Solution: Reduce `discriminator_iter_start` warmup or lower GAN weights

**Problem: Generator loss explodes**
- Solution: Increase `gan_feat_weight` for stability

**Problem: Training much slower**
- Solution: Normal with discriminators; reduce `video_gan_weight` to 0 for speedup

**Problem: No improvement over baseline**
- Solution: Ensure `discriminator_iter_start` has been reached; check logs for disc activation

## Files Created

1. `config/model/vq_gan_3d_with_discriminator.yaml` - Model config with discriminators enabled
2. `train_with_discriminator.sh` - Training launch script
3. `README_DISCRIMINATOR_TRAINING.md` - This guide
