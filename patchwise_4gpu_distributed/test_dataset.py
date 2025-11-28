"""
Quick test script to verify X-ray → CTPA dataset loading

Tests that paired X-ray PNG and CTPA .nii.gz files can be loaded correctly.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dataset.xray_ctpa_dataset import XrayCTPADataset

def test_dataset():
    """Test dataset loading with actual data."""
    
    print("=" * 80)
    print("TESTING X-RAY → CTPA DATASET")
    print("=" * 80)
    
    # Dataset configuration
    ctpa_dir = "/workspace/Xray-to-CTPA/datasets/"
    xray_pattern = "*_pa_drr.png"
    
    print(f"\nCTPA directory: {ctpa_dir}")
    print(f"X-ray pattern: {xray_pattern}")
    
    # Create dataset
    print("\nCreating train dataset...")
    train_dataset = XrayCTPADataset(
        ctpa_dir=ctpa_dir,
        xray_pattern=xray_pattern,
        split='train',
        train_split=0.8,
        max_patients=5,  # Test with just 5 patients
        use_latent=True,
        normalization='min_max'
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("\nERROR: No paired data found!")
        print("Check that X-ray PNG files exist alongside CTPA .nii.gz files")
        return
    
    # Test loading first sample
    print("\nLoading first sample...")
    sample = train_dataset[0]
    
    print(f"\nSample keys: {sample.keys()}")
    print(f"X-ray shape: {sample['xray'].shape}")
    print(f"X-ray range: [{sample['xray'].min():.3f}, {sample['xray'].max():.3f}]")
    print(f"CTPA shape: {sample['ctpa'].shape}")
    print(f"CTPA range: [{sample['ctpa'].min():.3f}, {sample['ctpa'].max():.3f}]")
    print(f"Patient ID: {sample['patient_id']}")
    
    # Test loading a few more samples
    print("\nTesting batch loading...")
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"  Sample {i}: X-ray {sample['xray'].shape}, CTPA {sample['ctpa'].shape}")
    
    print("\n" + "=" * 80)
    print("DATASET TEST PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()
