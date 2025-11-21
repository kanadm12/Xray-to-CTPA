#!/usr/bin/env python3
"""Quick script to check input data dimensions"""

import glob
import os
import torchio as tio

# Find all .nii and .nii.gz files
data_dir = '../datasets'  # Adjust if needed
files = glob.glob(f'{data_dir}/**/*.nii.gz', recursive=True)
if not files:
    files = glob.glob(f'{data_dir}/**/*.nii', recursive=True)

if not files:
    print(f"No .nii/.nii.gz files found in {data_dir}")
    exit(1)

print(f"Found {len(files)} files")
print(f"\nChecking first 5 files:")
print("-" * 80)

for i, fpath in enumerate(files[:5]):
    img = tio.ScalarImage(fpath)
    print(f"\n{i+1}. {os.path.basename(fpath)}")
    print(f"   Shape: {img.shape} (C, H, W, D)")
    print(f"   Spacing: {img.spacing}")
    
    # Calculate what gets cropped
    crop_h = max(0, img.shape[1] - 256)
    crop_w = max(0, img.shape[2] - 256)
    crop_d = max(0, img.shape[3] - 32)
    
    if crop_h > 0 or crop_w > 0 or crop_d > 0:
        print(f"   ⚠️  CROPPING: Losing {crop_h}x{crop_w}x{crop_d} voxels")
    else:
        pad_h = max(0, 256 - img.shape[1])
        pad_w = max(0, 256 - img.shape[2])
        pad_d = max(0, 32 - img.shape[3])
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            print(f"   ℹ️  PADDING: Adding {pad_h}x{pad_w}x{pad_d} voxels")
        else:
            print(f"   ✓ Perfect fit!")

print("\n" + "=" * 80)
print(f"Current training configuration: 256x256x32")
print(f"All images will be cropped/padded to this size")
