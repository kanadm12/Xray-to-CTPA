# Dataset Setup Guide

## Current Dataset Location

**Path:** `/workspace/Xray-to-CTPA/datasets/rsna_drrs_and_nifti`

This directory contains **7,279 patient folders** with the following structure:

```
rsna_drrs_and_nifti/
├── patient_001/
│   ├── patient_001.nii.gz          # CTPA volume
│   └── patient_001_pa_drr.png      # X-ray image (PA view)
├── patient_002/
│   ├── patient_002.nii.gz
│   └── patient_002_pa_drr.png
└── ...
```

## Data Characteristics

### CTPA Volumes (NIfTI files)

- **Format:** `.nii.gz` (gzipped NIfTI)
- **Shape:** `(1, 512, 512, D)` where D varies (typically 169-604 slices)
  - Format: `(C, H, W, D)` - Channel, Height, Width, Depth
  - Channel dimension is automatically handled by the data loaders
- **Spacing:** Variable (typically 0.6-0.8 mm in-plane, 0.8-1.25 mm slice thickness)
- **Intensity:** Hounsfield Units (HU) for CT data

### X-ray Images (PNG files)

- **Format:** `.png` (grayscale)
- **Naming:** `{patient_id}_pa_drr.png` (PA = Posterior-Anterior view)
- **Resolution:** Resized to 224×224 during loading for MedCLIP encoder

## Recent Code Changes

### 1. Support for Both `.nii` and `.nii.gz` Files

All dataset loaders now properly handle both file extensions without duplicates:

**Modified Files:**
- `patchwise_4gpu_distributed/dataset/xray_ctpa_patch_dataset.py`
- `patchwise_4gpu_distributed/dataset/xray_ctpa_dataset.py`
- `patchwise_4gpu_distributed/resample_nii_to_604.py`

**Change:** Added filter to prevent `.nii.gz` files from being matched twice by `*.nii` glob pattern.

### 2. Automatic Channel Dimension Handling

Data loaders now automatically handle NIfTI files with an extra channel dimension:

**Modified Files:**
- `patchwise_4gpu_distributed/dataset/xray_ctpa_patch_dataset.py`
- `patchwise_4gpu_distributed/dataset/xray_ctpa_dataset.py`
- `patchwise_512x512x604_single_gpu/dataset/patch_dataset.py`

**Change:** If a volume has shape `(1, H, W, D)` or `(C, D, H, W)` with `C=1`, the channel dimension is automatically squeezed before processing.

```python
# Handle files with channel dimension
if ctpa.ndim == 4 and ctpa.shape[0] == 1:
    ctpa = ctpa.squeeze(0)  # Remove channel dimension
```

### 3. Updated Data Size Checker

`check_data_size.py` now:
- Checks the new dataset location first
- Reports channel dimension handling
- Shows cropping/padding for 512×512×604 target (instead of 256×256×32)
- Provides training configuration recommendations

## Training Configurations

### Recommended: Patchwise 512×512×604

Your data is already at 512×512 resolution. Use this configuration to preserve detail:

```bash
cd patchwise_512x512x604_single_gpu
# or
cd patchwise_4gpu_distributed  # For multi-GPU training
```

**Target Size:** 512×512×604
- ✅ Preserves full H×W resolution (512×512)
- ✅ Standardizes depth to 604 slices
- ✅ Minimal data loss

### Alternative: Baseline 256×256×64

For faster training with lower resolution:

```bash
cd baseline_256x256x64_single_gpu
```

**Target Size:** 256×256×64
- ⚠️ Downsamples to 256×256 (loses 50% resolution)
- ⚠️ Heavy depth reduction (most volumes will be heavily cropped)
- ✅ Faster training, lower memory usage

## Data Processing Pipeline

1. **Load:** NIfTI files loaded with SimpleITK or nibabel
2. **Channel Handling:** Squeeze channel dimension if present
3. **Resizing:**
   - **Depth (D):** Pad or center-crop to target (604 or 64)
   - **Height/Width:** Pad or center-crop to target (512×512 or 256×256)
4. **Normalization:**
   - Min-max: Scale to [0, 1]
   - Standardization: (x - mean) / std
5. **Patch Extraction:** (for patchwise training only)
   - Extract overlapping 3D patches
   - Typical patch size: 256×256×128
   - Typical stride: 192×192×96

## Verification

Run the data checker to verify your setup:

```bash
python check_data_size.py
```

Expected output:
- Found data in: `/workspace/Xray-to-CTPA/datasets/rsna_drrs_and_nifti`
- 7,279 files
- Shape: (1, 512, 512, D) with automatic channel handling
- Recommendations for training configuration

## Notes

- Files with `swapped` in the filename are automatically excluded
- All spatial transformations use center cropping to preserve anatomical features
- Padding uses minimum intensity value to avoid introducing artifacts
- Data loaders automatically shuffle and split into train/val sets (default 80/20)
