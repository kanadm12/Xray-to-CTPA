# Patch-Wise Training Guide (Single GPU)

## Quick Start

### 1. Environment Setup

```bash
# Navigate to patch-wise implementation
cd /workspace/Xray-2CTPA_spartis/patchwise_512x512x604_single_gpu

# Install dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
# Should show your GPU (24GB+ recommended)
```

### 2. Prepare Data

Your full-resolution CTPA volumes should be in:
```
/workspace/Xray-2CTPA_spartis/datasets/
├── volume_001.nii.gz  (512×512×604)
├── volume_002.nii.gz
├── ...
└── volume_1349.nii.gz
```

### 3. Launch Training

```bash
# Option 1: Use launch script
chmod +x launch_distributed_training.sh
./launch_distributed_training.sh

# Option 2: Direct Python command
export PYTHONPATH=/workspace/Xray-2CTPA_spartis/patchwise_512x512x604_single_gpu:$PYTHONPATH
python train/train_vqgan_distributed.py
```

### 4. Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/vqgan_patches_distributed

# Watch GPU utilization
watch -n 1 nvidia-smi
```

## Architecture Details

### Patch Extraction Strategy

**Full Volume**: 512 × 512 × 604 voxels

**Patch Configuration**:
- Patch Size: 256 × 256 × 128
- Stride: 192 × 192 × 96
- Overlap: 64 voxels (25%) per dimension
- Patches per Volume: ~24 (3×3×3 grid)

**Why this configuration?**
1. ✅ Patches fit comfortably in GPU memory (32 MB each)
2. ✅ 25% overlap ensures smooth reconstruction
3. ✅ Compatible with baseline model architecture (256×256×128 vs 256×256×64)
4. ✅ Can train on full resolution with a single GPU

### Single GPU Training Strategy

**Data Processing**:
- Each volume → 24 patches
- DataLoader provides patches sequentially
- Batch size of 4 patches processed simultaneously

**Memory Efficiency**:
- Patches processed one batch at a time
- No need for distributed communication overhead
- Simpler code, easier to debug

**Memory Usage**:
```
Model parameters:     ~500 MB
Batch (4 patches):    ~128 MB
Activations:          ~20 GB
Gradients:            ~20 GB
-------------------------
Total:                ~41 GB (fits in 48GB or 80GB GPU)
```

### Reconstruction Pipeline

**Training**: Patches processed independently

**Validation/Inference**:
1. Extract all 24 patches from volume
2. Process patches in parallel across GPUs
3. Reconstruct using weighted overlap blending
4. Linear weights smooth transitions in overlap regions

## Performance Expectations

### Training Speed
- **Baseline (256×256×64)**: ~4 seconds/volume
- **Patch-wise (512×512×604)**: ~16 seconds/volume (4× more data)
- **Trade-off**: Slower but 97.4% more training data

### Convergence
- **Epochs**: 30 (similar to baseline)
- **Time per Epoch**: ~6 hours (1349 volumes)
- **Total Training Time**: ~180 hours (~7.5 days)

### Quality Metrics
Expected improvements over 256×256×64 baseline:

| Metric | Baseline | Expected (Full Res) |
|--------|----------|---------------------|
| PSNR | 35.3 dB | 37-39 dB |
| SSIM | 0.971 | 0.98-0.99 |
| Perplexity | 114.7 | 120-150 |
| Usage | 86.9% | 85-90% |

## Advantages

### vs. Baseline (256×256×64)

1. **97.4% More Data**: Training on full 512×512×604 instead of downsampled 256×256×64
2. **No Information Loss**: Preserves fine anatomical details
3. **Higher Quality**: Better PSNR/SSIM due to full resolution
4. **Production Ready**: Handles real CTPA dimensions
5. **Single GPU**: No need for multi-GPU cluster

### Trade-offs

**Pros**:
- ✅ Full resolution training
- ✅ Only requires 1 GPU (24GB+ recommended)
- ✅ Better reconstruction quality
- ✅ No downsampling artifacts
- ✅ Simpler code than multi-GPU

**Cons**:
- ❌ Slower than multi-GPU (but still feasible)
- ❌ ~7.5 days training time vs ~8 hours for baseline
- ❌ Need to handle patch boundaries carefully
- ❌ Requires larger GPU memory (48GB+ ideal)

## Troubleshooting

### OOM Errors
If you still get OOM:
1. Reduce `batch_size` from 2 to 1 per GPU
2. Reduce `patch_size` to [192, 192, 96]
3. Increase `stride` to reduce overlap

### Slow Training
Check:
1. `num_workers` set appropriately (8-12)
2. Data on fast SSD storage
3. Batch size can be increased if you have more VRAM

### Poor Reconstruction Quality
Tune overlap blending:
1. Increase overlap (reduce `stride`)
2. Try `blend_mode='gaussian'` instead of `'linear'`
3. Ensure patches align correctly during reconstruction

### Low GPU Utilization
- Increase batch_size if you have more VRAM
- Increase num_workers for faster data loading
- Monitor with `nvidia-smi dmon`

## Next Steps After Training

1. **Evaluate**: Run validation on full volumes
2. **Export**: Save trained model for inference
3. **Benchmark**: Compare quality metrics vs baseline
4. **Optimize**: Fine-tune overlap and blending parameters
5. **Deploy**: Use for X-ray → CTPA generation pipeline

## Directory Structure

```
patchwise_512x512x604_single_gpu/
├── README.md                          # Overview
├── TRAINING_GUIDE.md                  # This file
├── requirements.txt                   # Dependencies
├── launch_distributed_training.sh     # Launch script
│
├── config/                            # Configuration files
│   ├── base_cfg.yaml
│   ├── dataset/
│   │   └── full_resolution_ctpa.yaml
│   └── model/
│       └── vq_gan_3d_patches.yaml
│
├── dataset/                           # Data loading
│   └── patch_dataset.py
│
├── vq_gan_3d/                         # Model code
│   └── model/
│       └── vqgan_patches.py
│
├── train/                             # Training scripts
│   └── train_vqgan_distributed.py
│
└── utils/                             # Utilities
    ├── patch_utils.py                 # Patch extraction/merging
    └── overlap_blend.py               # Blending functions
```
