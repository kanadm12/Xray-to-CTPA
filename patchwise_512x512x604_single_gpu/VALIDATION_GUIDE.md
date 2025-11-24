# Validation & Testing Guide for Patch-Wise Implementation

## Pre-Training Validation

### 1. Test Patch Extraction/Reconstruction

```python
# Test patch utilities
import sys
sys.path.append('/workspace/Xray-2CTPA_spartis/patchwise_parallel_512x512x604_multi_gpu')

import torch
from utils.patch_utils import extract_patches_3d, reconstruct_from_patches, get_patch_statistics

# Create dummy volume (512×512×604)
volume = torch.randn(1, 1, 512, 512, 604)

# Extract patches (256×256×128 with 64 voxel overlap)
patches, positions = extract_patches_3d(
    volume, 
    patch_size=(256, 256, 128),
    stride=(192, 192, 96)
)

print(f"Extracted {len(patches)} patches from volume")
print(f"Patch shapes: {[p.shape for p in patches[:3]]}")

# Reconstruct with linear blending
reconstructed = reconstruct_from_patches(
    patches, 
    positions, 
    volume_shape=(512, 512, 604),
    blend_mode='linear'
)

# Check reconstruction error
mse = torch.nn.functional.mse_loss(reconstructed, volume)
print(f"Reconstruction MSE: {mse.item():.6f}")
print(f"Max error: {(reconstructed - volume).abs().max().item():.6f}")

# Should be near-perfect (< 1e-6) for random data
```

### 2. Test Dataset Loading

```python
from dataset.patch_dataset import FullResolutionCTPADataset
from torch.utils.data import DataLoader

# Load dataset
dataset = FullResolutionCTPADataset(
    data_dir='/workspace/Xray-2CTPA_spartis/datasets/',
    patch_size=[256, 256, 128],
    stride=[192, 192, 96],
    split='train'
)

print(f"Dataset size: {len(dataset)} patches")

# Test loading
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Patch shape: {sample['patch'].shape}")
print(f"Position: {sample['position']}")
print(f"Volume ID: {sample['volume_id']}")
```

### 3. Test Distributed Setup (Single Node)

```bash
# Dry-run with 2 GPUs (faster test)
torchrun --nproc_per_node=2 train/train_vqgan_distributed.py \
    model.max_epochs=1 \
    dataset.num_workers=2 \
    trainer.val_check_interval=10
```

## During Training Monitoring

### 1. GPU Utilization

```bash
# Watch all 4 GPUs
watch -n 1 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Or use dmon for detailed stats
nvidia-smi dmon -i 0,1,2,3 -s pucvmet
```

Expected:
- **GPU Utilization**: 85-95% during training
- **Memory Used**: ~41 GB / 144 GB (28%)
- **Power Draw**: 400-600W per GPU

### 2. TensorBoard Metrics

```bash
tensorboard --logdir outputs/vqgan_patches_distributed --port 6006
```

Key metrics to watch:
- **train/recon_loss**: Should decrease from ~0.5 to ~0.02
- **train/perplexity**: Should stabilize around 100-150
- **val/codebook_usage**: Should reach 85-90%
- **val/psnr**: Should increase to 37-39 dB
- **val/ssim**: Should reach 0.98-0.99

### 3. Check DDP Synchronization

```python
# Add to train_vqgan_distributed.py for debugging
import torch.distributed as dist

if dist.is_initialized():
    print(f"Rank {dist.get_rank()}/{dist.get_world_size()}")
    print(f"Backend: {dist.get_backend()}")
    
    # Test synchronization
    tensor = torch.tensor([dist.get_rank()], device='cuda')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Sum of all ranks: {tensor.item()}")  # Should be 0+1+2+3=6
```

## Post-Training Validation

### 1. Evaluate Full Volume Reconstruction

```python
import torch
from vq_gan_3d.model.vqgan_patches import VQGAN_Patches
from utils.patch_utils import extract_patches_3d, reconstruct_from_patches

# Load trained model
model = VQGAN_Patches.load_from_checkpoint('outputs/vqgan_patches_distributed/checkpoints/best.ckpt')
model = model.cuda()
model.eval()

# Load test volume
volume = torch.load('test_volume.pt')  # Shape: [1, 1, 512, 512, 604]

# Extract patches
patches, positions = extract_patches_3d(volume, patch_size=(256, 256, 128), stride=(192, 192, 96))

# Encode-decode all patches
with torch.no_grad():
    reconstructed_patches = []
    for patch in patches:
        patch_gpu = patch.cuda()
        recon, _, _ = model(patch_gpu)
        reconstructed_patches.append(recon.cpu())

# Reconstruct full volume
reconstructed = reconstruct_from_patches(
    reconstructed_patches, 
    positions, 
    volume_shape=(512, 512, 604),
    blend_mode='linear'
)

# Compute metrics
mse = torch.nn.functional.mse_loss(reconstructed, volume)
psnr = 10 * torch.log10(1.0 / mse)
print(f"Full Volume PSNR: {psnr.item():.2f} dB")
```

### 2. Compare with Baseline

Create comparison script:

```python
import torch
import pandas as pd
from pathlib import Path

# Load both models
baseline = torch.load('baseline_256x256x64_single_gpu/outputs/checkpoints/best.ckpt')
patchwise = torch.load('patchwise_parallel_512x512x604_multi_gpu/outputs/checkpoints/best.ckpt')

# Compare metrics
results = {
    'Model': ['Baseline 256×256×64', 'Patch-wise 512×512×604'],
    'PSNR (dB)': [35.3, patchwise['val_psnr']],
    'SSIM': [0.971, patchwise['val_ssim']],
    'Perplexity': [114.7, patchwise['val_perplexity']],
    'Usage (%)': [86.9, patchwise['val_usage']],
    'Training Time (hrs)': [8, 15],
    'GPUs': [1, 4]
}

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

### 3. Visualize Reconstructions

```python
import matplotlib.pyplot as plt
import numpy as np

# Load original and reconstructed volumes
original = torch.load('test_volume.pt').squeeze().numpy()
reconstructed = torch.load('reconstructed.pt').squeeze().numpy()

# Plot slices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

slices = [150, 300, 450]
for i, slice_idx in enumerate(slices):
    # Original
    axes[0, i].imshow(original[:, :, slice_idx], cmap='gray')
    axes[0, i].set_title(f'Original Slice {slice_idx}')
    axes[0, i].axis('off')
    
    # Reconstructed
    axes[1, i].imshow(reconstructed[:, :, slice_idx], cmap='gray')
    axes[1, i].set_title(f'Reconstructed Slice {slice_idx}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=150)
print("Saved reconstruction_comparison.png")
```

## Common Issues & Solutions

### Issue: Reconstruction artifacts at patch boundaries

**Solution**: Increase overlap or use Gaussian blending
```python
reconstructed = reconstruct_from_patches(
    patches, positions, volume_shape,
    blend_mode='gaussian',  # Instead of 'linear'
    sigma=16.0  # Larger sigma = smoother blending
)
```

### Issue: GPU memory imbalance

**Check**: Are patches distributed evenly?
```python
from torch.utils.data.distributed import DistributedSampler

# Verify sampler
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
print(f"Samples on rank 0: {len(list(sampler))}")
```

### Issue: Low codebook usage

**Solution**: Tune commitment loss (already set to 2.0) or increase codebook size
```yaml
# In config/model/vq_gan_3d_patches.yaml
codebook:
  n_codes: 1024  # Increase from 512
  beta: 2.5      # Increase commitment loss
```

### Issue: NCCL timeout

**Solution**: Increase timeout or check network
```python
# In train_vqgan_distributed.py
import datetime
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(minutes=30)  # Increase from default 30 min
)
```

## Success Criteria

✅ **Pre-Training**:
- Patch extraction/reconstruction MSE < 1e-6
- Dataset loads without errors
- Dry-run completes 1 epoch

✅ **During Training**:
- All 4 GPUs at ~90% utilization
- Memory usage ~41GB/144GB per GPU
- Perplexity increases from 1.0 to 100+
- Codebook usage > 80% after 10 epochs

✅ **Post-Training**:
- Full volume PSNR > 37 dB
- SSIM > 0.98
- No visible artifacts at patch boundaries
- Better quality than 256×256×64 baseline

## Next Steps After Validation

1. **Production Deployment**: Package model for inference
2. **Optimize Inference**: Batch multiple volumes
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Scale Up**: Test on larger datasets or higher resolutions
5. **Integrate**: Use for X-ray → CTPA generation pipeline
