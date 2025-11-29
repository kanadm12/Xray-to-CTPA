"""
Resample .nii/.nii.gz files to have 604 slices to match training data dimensions.

Usage:
    python resample_nii_to_604.py --data_dir /workspace/Xray-to-CTPA/datasets
"""

import os
import argparse
import SimpleITK as sitk
import numpy as np
from glob import glob
from tqdm import tqdm


def resample_to_target_depth(input_path, output_path, target_depth=604):
    """
    Resample a NIFTI volume to have exactly target_depth slices.
    
    Args:
        input_path: Path to input .nii or .nii.gz file
        output_path: Path to save resampled file
        target_depth: Target number of slices (default 604)
    """
    # Load image
    image = sitk.ReadImage(input_path)
    original_size = image.GetSize()  # (W, H, D)
    original_spacing = image.GetSpacing()
    
    current_depth = original_size[2]
    
    if current_depth == target_depth:
        print(f"  Already {target_depth} slices, skipping")
        return False
    
    # Calculate new spacing to achieve target depth
    new_spacing = list(original_spacing)
    new_spacing[2] = (current_depth * original_spacing[2]) / target_depth
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([original_size[0], original_size[1], target_depth])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    
    # Resample
    resampled = resampler.Execute(image)
    
    # Save
    sitk.WriteImage(resampled, output_path)
    print(f"  Resampled from {current_depth} to {target_depth} slices")
    return True


def process_dataset(data_dir, target_depth=604, in_place=False):
    """
    Process all .nii/.nii.gz files in dataset directory.
    
    Args:
        data_dir: Root directory containing patient folders
        target_depth: Target number of slices
        in_place: If True, replace original files. If False, create *_resampled.nii.gz
    """
    # Find all patient folders
    patient_folders = sorted([
        d for d in glob(os.path.join(data_dir, '*'))
        if os.path.isdir(d)
    ])
    
    print(f"Found {len(patient_folders)} patient folders")
    
    total_resampled = 0
    
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_id = os.path.basename(patient_folder)
        
        # Find all .nii and .nii.gz files (exclude swapped and already resampled)
        nii_files = []
        
        # .nii.gz files
        nii_gz = [
            f for f in glob(os.path.join(patient_folder, '*.nii.gz'))
            if 'swapped' not in os.path.basename(f).lower()
            and 'resampled' not in os.path.basename(f).lower()
        ]
        nii_files.extend(nii_gz)
        
        # .nii files
        nii = [
            f for f in glob(os.path.join(patient_folder, '*.nii'))
            if 'swapped' not in os.path.basename(f).lower()
            and 'resampled' not in os.path.basename(f).lower()
        ]
        nii_files.extend(nii)
        
        for nii_path in nii_files:
            basename = os.path.basename(nii_path)
            print(f"\n{patient_id}/{basename}")
            
            if in_place:
                # Create temp file, then replace original
                temp_path = nii_path + '.temp.nii.gz'
                resampled = resample_to_target_depth(nii_path, temp_path, target_depth)
                if resampled:
                    os.replace(temp_path, nii_path)
                    total_resampled += 1
            else:
                # Create new file with _resampled suffix
                if nii_path.endswith('.nii.gz'):
                    output_path = nii_path.replace('.nii.gz', '_resampled.nii.gz')
                else:
                    output_path = nii_path.replace('.nii', '_resampled.nii.gz')
                
                if os.path.exists(output_path):
                    print(f"  Resampled file already exists, skipping")
                    continue
                
                resampled = resample_to_target_depth(nii_path, output_path, target_depth)
                if resampled:
                    total_resampled += 1
    
    print(f"\n{'='*60}")
    print(f"Total files resampled: {total_resampled}")
    if not in_place:
        print(f"Original files preserved. New files have '_resampled' suffix.")
        print(f"After verification, you can:")
        print(f"  1. Delete originals: find {data_dir} -name '*.nii' -o -name '*.nii.gz' | grep -v resampled | xargs rm")
        print(f"  2. Rename resampled: find {data_dir} -name '*_resampled.nii.gz' -exec rename 's/_resampled//' {{}} \\;")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Resample NIFTI files to 604 depth')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing patient folders')
    parser.add_argument('--target_depth', type=int, default=604,
                        help='Target depth (default 604)')
    parser.add_argument('--in_place', action='store_true',
                        help='Replace original files (default: create *_resampled.nii.gz)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    print("="*60)
    print("NIFTI Volume Resampling to 604 Depth")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Target depth: {args.target_depth}")
    print(f"Mode: {'IN-PLACE (will replace originals)' if args.in_place else 'SAFE (will create new files)'}")
    print("="*60)
    
    if args.in_place:
        response = input("\nWARNING: This will REPLACE original files. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    process_dataset(args.data_dir, args.target_depth, args.in_place)


if __name__ == "__main__":
    main()
