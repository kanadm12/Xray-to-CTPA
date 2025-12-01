#!/usr/bin/env python3
"""Quick script to check input data dimensions"""

import glob
import os
import torchio as tio

# Find all .nii and .nii.gz files - try multiple common locations
possible_dirs = [
    '/workspace/Xray-to-CTPA/datasets/rsna_drrs_and_nifti',
    './datasets/rsna_drrs_and_nifti',
    './datasets', 
    '../datasets', 
    '/workspace/datasets', 
    './'
]
data_dir = None

for d in possible_dirs:
    if os.path.exists(d):
        test_files = glob.glob(f'{d}/**/*.nii.gz', recursive=True) or glob.glob(f'{d}/**/*.nii', recursive=True)
        if test_files:
            data_dir = d
            print(f"Found data in: {d}")
            break

if not data_dir:
    print("Searching for .nii files in current directory and common locations...")
    data_dir = '.'

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
    
    # Handle channel dimension
    if img.shape[0] == 1:
        print(f"   ℹ️  Single channel - will be handled automatically")
    
    # Calculate what gets cropped for 512x512x604 target
    crop_h = max(0, img.shape[1] - 512)
    crop_w = max(0, img.shape[2] - 512)
    crop_d = max(0, img.shape[3] - 604)
    
    if crop_h > 0 or crop_w > 0 or crop_d > 0:
        print(f"   ⚠️  CROPPING: Will lose {crop_h}x{crop_w}x{crop_d} voxels")
    else:
        pad_h = max(0, 512 - img.shape[1])
        pad_w = max(0, 512 - img.shape[2])
        pad_d = max(0, 604 - img.shape[3])
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            print(f"   ℹ️  PADDING: Will add {pad_h}x{pad_w}x{pad_d} voxels")
        else:
            print(f"   ✓ Perfect match for 512x512x604!")

print("\n" + "=" * 80)
print(f"Training configurations:")
print(f"  • baseline_256x256x64: 256x256x64 (heavy cropping/downsampling)")
print(f"  • patchwise_512x512x604: 512x512x604 (recommended for your data)")
print(f"  • patchwise_4gpu: 512x512x604 with distributed training")
print(f"\nAll images will be automatically cropped/padded to target size.")
print(f"Channel dimensions (if present) are handled automatically.")
