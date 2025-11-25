"""
Full-resolution CTPA dataset with patch extraction for distributed training.

Loads 512×512×604 volumes and extracts patches on-the-fly.
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DistributedSampler
from typing import Tuple, Optional, List
import sys
sys.path.append('..')
from utils.patch_utils import extract_patches_3d


class FullResolutionCTPADataset(Dataset):
    """
    Dataset for full-resolution 512×512×604 CTPA volumes with patch extraction.
    
    Each volume is divided into overlapping patches during training.
    For validation, we can optionally return full volumes.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        patch_size: Tuple[int, int, int] = (256, 256, 128),
        stride: Tuple[int, int, int] = (192, 192, 96),
        train_split: float = 0.8,
        normalize: bool = True,
        return_full_volume: bool = False  # For validation
    ):
        """
        Args:
            root_dir: Directory containing .nii.gz files
            mode: 'train' or 'val'
            patch_size: Size of extracted patches
            stride: Stride for patch extraction (controls overlap)
            train_split: Fraction of data for training
            normalize: Whether to normalize to [-1, 1]
            return_full_volume: If True, return full volume instead of patches
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.return_full_volume = return_full_volume
        
        # Find all .nii.gz files
        self.file_paths = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.nii.gz')
        ])
        
        # Split into train/val
        split_idx = int(len(self.file_paths) * train_split)
        if mode == 'train':
            self.file_paths = self.file_paths[:split_idx]
        else:
            self.file_paths = self.file_paths[split_idx:]
        
        print(f"{mode.upper()} dataset: {len(self.file_paths)} volumes")
        
        # Pre-calculate number of patches per volume for indexing
        if not return_full_volume:
            self._calculate_patch_indices()
    
    def _calculate_patch_indices(self):
        """Calculate global patch indices for efficient access."""
        # For simplicity, assume all volumes have same dimensions
        # In production, you might want to handle variable sizes
        sample_volume = self._load_volume(self.file_paths[0])
        patches, _ = extract_patches_3d(
            sample_volume,
            patch_size=self.patch_size,
            stride=self.stride
        )
        self.patches_per_volume = patches.shape[0]
        self.total_patches = len(self.file_paths) * self.patches_per_volume
        
        print(f"  Patches per volume: {self.patches_per_volume}")
        print(f"  Total patches: {self.total_patches}")
    
    def _load_volume(self, file_path: str) -> torch.Tensor:
        """Load and preprocess a single volume."""
        # Load NIfTI file
        nii = nib.load(file_path)
        volume = nii.get_fdata()
        
        # Convert to tensor and add channel dimension
        volume = torch.from_numpy(volume).float()
        volume = volume.unsqueeze(0)  # [1, D, H, W]
        
        # Normalize to [-1, 1]
        if self.normalize:
            volume_min = volume.min()
            volume_max = volume.max()
            if volume_max > volume_min:
                volume = 2 * (volume - volume_min) / (volume_max - volume_min) - 1
        
        return volume
    
    def __len__(self):
        if self.return_full_volume:
            return len(self.file_paths)
        else:
            return self.total_patches
    
    def __getitem__(self, idx):
        if self.return_full_volume:
            # Return full volume for validation
            volume = self._load_volume(self.file_paths[idx])
            return {
                'volume': volume,
                'file_path': self.file_paths[idx]
            }
        else:
            # Return a single patch
            volume_idx = idx // self.patches_per_volume
            patch_idx = idx % self.patches_per_volume
            
            # Load volume
            volume = self._load_volume(self.file_paths[volume_idx])
            
            # Extract patches
            patches, coordinates = extract_patches_3d(
                volume,
                patch_size=self.patch_size,
                stride=self.stride
            )
            
            # Return specific patch
            patch = patches[patch_idx]
            coord = coordinates[patch_idx]
            
            return {
                'patch': patch,
                'volume_idx': volume_idx,
                'patch_idx': patch_idx,
                'coordinate': coord
            }


class PatchDataset(Dataset):
    """
    Memory-efficient patch dataset that extracts patches on-the-fly.
    
    This version doesn't pre-calculate indices but loads volumes as needed.
    Better for large datasets.
    """
    
    def __init__(
        self,
        volume_paths: List[str],
        patch_size: Tuple[int, int, int] = (256, 256, 128),
        stride: Tuple[int, int, int] = (192, 192, 96),
        normalize: bool = True,
        cache_volumes: bool = False
    ):
        self.volume_paths = volume_paths
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.cache_volumes = cache_volumes
        
        if cache_volumes:
            print("Pre-loading all volumes into memory...")
            self.volumes = [self._load_volume(p) for p in volume_paths]
        else:
            self.volumes = None
    
    def _load_volume(self, file_path: str) -> torch.Tensor:
        """Load and preprocess a single volume."""
        nii = nib.load(file_path)
        volume = nii.get_fdata()
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        
        # Pad to standard size (512x512x604) to ensure all volumes same size
        target_shape = (512, 512, 604)
        current_shape = volume.shape[1:]  # Remove channel dimension
        
        # Calculate padding needed
        pad_d = max(0, target_shape[2] - current_shape[2])
        pad_h = max(0, target_shape[0] - current_shape[0])
        pad_w = max(0, target_shape[1] - current_shape[1])
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Pad symmetrically (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            volume = torch.nn.functional.pad(
                volume,
                (0, pad_w, 0, pad_h, 0, pad_d),
                mode='constant',
                value=volume.min()  # Use min value for padding
            )
        
        # Crop if larger than target
        volume = volume[:, :target_shape[0], :target_shape[1], :target_shape[2]]
        
        if self.normalize:
            volume_min = volume.min()
            volume_max = volume.max()
            if volume_max > volume_min:
                volume = 2 * (volume - volume_min) / (volume_max - volume_min) - 1
        
        return volume
    
    def __len__(self):
        return len(self.volume_paths)
    
    def __getitem__(self, idx):
        # Load volume
        if self.volumes is not None:
            volume = self.volumes[idx]
        else:
            volume = self._load_volume(self.volume_paths[idx])
        
        # Extract all patches from this volume
        patches, coordinates = extract_patches_3d(
            volume,
            patch_size=self.patch_size,
            stride=self.stride
        )
        
        return {
            'patches': patches,  # [N, C, D, H, W]
            'coordinates': coordinates,
            'volume_shape': volume.shape,
            'volume_idx': idx
        }


def collate_patches(batch):
    """
    Custom collate function to handle batches of patches.
    
    Flattens all patches from all volumes into a single batch.
    """
    all_patches = []
    all_coords = []
    volume_indices = []
    
    for item in batch:
        patches = item['patches']
        coords = item['coordinates']
        vol_idx = item['volume_idx']
        
        all_patches.append(patches)
        all_coords.extend(coords)
        volume_indices.extend([vol_idx] * len(patches))
    
    # Stack all patches
    all_patches = torch.cat(all_patches, dim=0)
    
    return {
        'patches': all_patches,
        'coordinates': all_coords,
        'volume_indices': torch.tensor(volume_indices)
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing FullResolutionCTPADataset...")
    
    # Create dummy dataset directory for testing
    import tempfile
    import nibabel as nib
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy volumes
        for i in range(5):
            volume = np.random.randn(512, 512, 604).astype(np.float32)
            nii = nib.Nifti1Image(volume, np.eye(4))
            nib.save(nii, os.path.join(tmpdir, f'volume_{i:03d}.nii.gz'))
        
        # Test dataset
        dataset = FullResolutionCTPADataset(
            root_dir=tmpdir,
            mode='train',
            patch_size=(256, 256, 128),
            stride=(192, 192, 96),
            return_full_volume=False
        )
        
        print(f"\nDataset length: {len(dataset)}")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Patch shape: {sample['patch'].shape}")
        print(f"Coordinate: {sample['coordinate']}")
        
        # Test PatchDataset
        print("\n\nTesting PatchDataset...")
        volume_paths = [os.path.join(tmpdir, f'volume_{i:03d}.nii.gz') for i in range(3)]
        
        patch_dataset = PatchDataset(
            volume_paths=volume_paths,
            patch_size=(256, 256, 128),
            stride=(192, 192, 96)
        )
        
        sample = patch_dataset[0]
        print(f"Patches shape: {sample['patches'].shape}")
        print(f"Number of patches: {len(sample['coordinates'])}")
