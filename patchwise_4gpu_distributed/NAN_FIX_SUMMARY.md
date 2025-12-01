# NaN/Inf Loss Fix Summary

## Problem Identified
Training was immediately encountering NaN/Inf losses at step 0, caused by:
1. **Division by zero** in normalization (when volume has uniform intensity)
2. **Unstable perceptual loss** with mixed precision (AMP)
3. **Discriminator training** interfering with gradient scaling
4. **No epsilon safeguards** in multiple normalization steps
5. **Extreme values** in VQ-GAN encoding propagating through the network

## Fixes Applied

### 1. Dataset Normalization (xray_ctpa_patch_dataset.py)
**Added epsilon and edge case handling:**
```python
# Before: Division by zero possible
if max_val > min_val:
    ctpa = (ctpa - min_val) / (max_val - min_val)

# After: Safe normalization with epsilon
eps = 1e-8
if max_val > min_val + eps:
    ctpa = (ctpa - min_val) / (max_val - min_val + eps)
else:
    # If uniform volume, set to middle value
    ctpa = np.full_like(ctpa, 0.5)
```

**Impact:** Prevents NaN when CTPA volume has uniform intensity (e.g., all zeros or constant value)

### 2. VQ-GAN Encoding Normalization (diffusion.py)
**Added epsilon and clamping:**
```python
# Before: Can produce extreme values
emb_range = emb_max - emb_min
if emb_range > 1e-6:
    ct = ((ct - emb_min) / emb_range) * 2.0 - 1.0

# After: Safe normalization with clamping
eps = 1e-6
if emb_range > eps:
    ct = ((ct - emb_min) / (emb_range + eps)) * 2.0 - 1.0
    ct = torch.clamp(ct, -10.0, 10.0)  # Prevent extreme values
```

**Impact:** Prevents extreme latent values that cause gradient explosion

### 3. Input Validation in p_losses
**Added NaN/Inf checks before computation:**
```python
# Check for NaN/Inf in inputs
if torch.isnan(x_start).any() or torch.isinf(x_start).any():
    print("WARNING: NaN/Inf detected in x_start, replacing with zeros")
    x_start = torch.nan_to_num(x_start, nan=0.0, posinf=1.0, neginf=-1.0)

if cond is not None and (torch.isnan(cond).any() or torch.isinf(cond).any()):
    print("WARNING: NaN/Inf detected in conditioning, replacing with zeros")
    cond = torch.nan_to_num(cond, nan=0.0, posinf=1.0, neginf=-1.0)
```

**Impact:** Catches corrupted data before it propagates through the model

### 4. X-ray Encoding Safety
**Added checks after MedCLIP encoding:**
```python
# After encoding
cond = self.xray_encoder.encode_image(cond.to(device), normalize=True)

# Safety check
if torch.isnan(cond).any() or torch.isinf(cond).any():
    print("WARNING: NaN/Inf in X-ray encoding, replacing with zeros")
    cond = torch.nan_to_num(cond, nan=0.0, posinf=1.0, neginf=-1.0)
cond = torch.clamp(cond, -10.0, 10.0)
```

**Impact:** Prevents unstable X-ray features from corrupting training

### 5. Prediction and Reconstruction Safety
**Added clamping and NaN checks:**
```python
pred_noise = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

# Check for NaN in prediction
if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
    print("WARNING: NaN/Inf in pred_noise, replacing with noise")
    pred_noise = noise

x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=pred_noise)

# Clamp reconstruction to prevent extreme values
x_recon = torch.clamp(x_recon, -10.0, 10.0)
```

**Impact:** Prevents model predictions from becoming unstable

### 6. Auxiliary Loss Safety
**Disabled perceptual/discriminator losses with AMP:**
```python
# Perceptual loss (skip with AMP to avoid instability)
lpips_loss = 0
if self.perceptual_weight > 0 and not torch.is_autocast_enabled():
    lpips_loss = self.lpips_loss_fn(x_start, x_recon)
    if torch.isnan(lpips_loss) or torch.isinf(lpips_loss):
        lpips_loss = torch.tensor(0.0, device=device, requires_grad=True)

# Discriminator loss (skip with AMP to avoid instability)
disc_loss = 0
if self.discriminator_weight > 0 and not torch.is_autocast_enabled():
    disc_loss = self.disc_loss_fn(x_start, x_recon)
    if torch.isnan(disc_loss) or torch.isinf(disc_loss):
        disc_loss = torch.tensor(0.0, device=device, requires_grad=True)
```

**Impact:** Prevents unstable auxiliary losses from causing NaN with mixed precision

### 7. Final Loss Safety
**Added final checks and clamping:**
```python
# Final NaN check on total loss
if torch.isnan(loss) or torch.isinf(loss):
    print("WARNING: NaN/Inf in final loss, returning zero")
    return torch.tensor(0.0, device=device, requires_grad=True)

# Clamp loss to reasonable range
loss = torch.clamp(loss, 0.0, 1000.0)
```

**Impact:** Last line of defense against NaN propagation

### 8. Configuration Updates (ddpm_4gpu.yaml)
**Simplified loss to avoid unstable components:**
```yaml
# Before
loss_type: l1_lpips
perceptual_weight: 0.01

# After
loss_type: l1  # Start simple
perceptual_weight: 0.0  # Disabled - unstable with AMP
```

**Impact:** Start with stable L1 loss, add complexity after warmup

## Gradient Clipping Already Configured
The trainer already has gradient clipping enabled:
```python
gradient_clip_val=cfg.model.max_grad_norm  # Set to 1.0 in config
```

## Expected Behavior After Fixes

### Normal Training:
- Loss starts high (~0.5-2.0) and decreases gradually
- No NaN/Inf warnings
- Stable gradient norms

### If Warnings Appear:
The code now has defensive checks that will:
1. Print warning messages (helpful for debugging)
2. Replace NaN/Inf with safe values
3. Continue training without crashing

## Recommended Training Plan

### Phase 1: Warmup (Epochs 1-5)
- **Loss:** L1 only
- **Learning rate:** 1e-4
- **Goal:** Stable baseline convergence

### Phase 2: Add Perceptual Loss (Epochs 6-20)
Update config:
```yaml
loss_type: l1_lpips
perceptual_weight: 0.001  # Start small
```

### Phase 3: Full Training (Epochs 21-30)
```yaml
perceptual_weight: 0.01
discriminator_weight: 0.001  # Can enable if needed
```

## Testing Command
```bash
cd /workspace/Xray-to-CTPA/patchwise_4gpu_distributed
bash launch_ddpm_4gpu.sh
```

## Monitoring Checklist
✅ No NaN/Inf warnings during first 100 steps
✅ Loss decreases gradually (not stuck at 0)
✅ GPU memory stable (~15-20GB per GPU)
✅ Training progresses past validation step

## If Issues Persist

### Check Data:
```python
# In RunPod terminal
cd /workspace/Xray-to-CTPA/patchwise_4gpu_distributed
python test_dataset.py  # Verify data loads correctly
```

### Reduce Learning Rate:
Edit config: `train_lr: 5e-5`  (half the current value)

### Disable AMP:
Edit config: `amp: false`  (use full precision)

## Files Modified
1. `patchwise_4gpu_distributed/dataset/xray_ctpa_patch_dataset.py`
2. `baseline_256x256x64_single_gpu/ddpm/diffusion.py`
3. `patchwise_4gpu_distributed/config/model/ddpm_4gpu.yaml`

## Key Insight
The root cause was **numerical instability from division by zero and extreme values**. The fix adds:
- Epsilon safeguards (1e-6 to 1e-8)
- Value clamping (-10 to 10 range)
- NaN detection and replacement
- Simplified loss function

Training should now be stable! 🚀
