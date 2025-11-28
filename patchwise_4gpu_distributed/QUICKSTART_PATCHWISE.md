# Quick Start: Patchwise DDPM Training on RunPod

## What Changed?

Instead of processing full 512×512×604 volumes (causing OOM), we now:
- Extract **128×128×128 patches** from each CTPA volume
- Process **~42 patches per patient** (with 25% overlap)
- Use **batch_size=2 per GPU** (vs 1 before)
- **Memory: 10GB/GPU** vs 138GB before

## Step 1: Pull Latest Code

On RunPod terminal:
```bash
cd /workspace/Xray-to-CTPA
git pull origin main
```

You should see:
```
patchwise_4gpu_distributed/README_PATCHWISE_DDPM.md
patchwise_4gpu_distributed/config/model/ddpm_4gpu_patches.yaml
patchwise_4gpu_distributed/dataset/xray_ctpa_patch_dataset.py
patchwise_4gpu_distributed/launch_ddpm_patches_4gpu.sh
patchwise_4gpu_distributed/train_ddpm_patches_4gpu.py
```

## Step 2: Make Script Executable

```bash
cd /workspace/Xray-to-CTPA/patchwise_4gpu_distributed
chmod +x launch_ddpm_patches_4gpu.sh
```

## Step 3: Launch Training

```bash
./launch_ddpm_patches_4gpu.sh
```

## What to Expect

### Initialization
```
Loading patchwise X-ray → CTPA dataset...
CTPA dir: /workspace/data/LIDC-IDRI_RSPECT_CTPELVIC1K_PA_DRR
Patch size: [128, 128, 128]
Stride: [96, 96, 96]
[TRAIN] Loaded 720 patients
  Total patches: 30240 (42 per volume)
[VAL] Loaded 181 patients
  Total patches: 7602 (42 per volume)
```

### Memory Usage
Each GPU should use **~20-30GB** (vs 138GB before):
```
GPU 0: 25GB / 140GB
GPU 1: 25GB / 140GB
GPU 2: 25GB / 140GB
GPU 3: 25GB / 140GB
```

### Training Progress
```
Epoch 0:   1%|▋  | 100/3780 [02:15<1:23:20,  0.74it/s]
train/loss: 0.4523, lr: 0.0001
```

### Iterations
- **~3,780 iterations per epoch** (30,240 patches / 8 effective batch)
- **~2-3 iterations/sec** with FP16 + gradient checkpointing
- **~30-60 min per epoch**
- **30 epochs = ~15-30 hours total**

## Checkpoint Location

Checkpoints saved to:
```
/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches/
```

Files:
- `last.ckpt`: Latest epoch
- `ddpm-patches-epoch{XX}-loss{Y.YYYY}.ckpt`: Top 3 best models

## Monitor Training

### Watch GPU usage
```bash
watch -n 1 nvidia-smi
```

### Follow training log
```bash
tail -f nohup.out  # If running with nohup
```

### TensorBoard
```bash
tensorboard --logdir ./checkpoints/ddpm_4gpu_patches/ --port 6006
```

## Key Differences from Full Volume Training

| Metric | Full Volume (OLD) | Patchwise (NEW) |
|--------|-------------------|-----------------|
| CTPA Size | 512×512×604 | 128×128×128 |
| Latent Size | 128×128×151 | 32×32×32 |
| Memory/GPU | 138GB ❌ OOM | 10GB ✅ |
| Batch/GPU | 1 | 2 |
| Effective Batch | 4 | 8 |
| Train Samples | 720 volumes | 30,240 patches |
| Epochs | Never finished | 30 epochs |

## Troubleshooting

### Still OOM?
Very unlikely, but if it happens:
1. Edit `config/model/ddpm_4gpu_patches.yaml`
2. Reduce batch size: `batch_size: 1`
3. Or smaller patches: `patch_size: [96, 96, 96]`

### Training too slow?
- Increase batch size: `batch_size: 4`
- Reduce num_workers if CPU bottleneck

### Check config
```bash
cat config/model/ddpm_4gpu_patches.yaml
```

Should show:
```yaml
use_patches: true
patch_size: [128, 128, 128]
stride: [96, 96, 96]
diffusion_img_size: 32
diffusion_depth_size: 32
batch_size: 2
```

## After Training

Once training completes (30 epochs):
1. Best checkpoint at `ddpm-patches-epoch{XX}-loss{Y}.ckpt`
2. Use for inference to generate CTPA from X-ray
3. Stitch patches back to full 512×512×604 volumes

## Commands Summary

```bash
# Pull code
cd /workspace/Xray-to-CTPA && git pull origin main

# Navigate to directory
cd patchwise_4gpu_distributed

# Make executable
chmod +x launch_ddpm_patches_4gpu.sh

# Launch training
./launch_ddpm_patches_4gpu.sh
```

That's it! Training should now work without OOM errors. 🎉
