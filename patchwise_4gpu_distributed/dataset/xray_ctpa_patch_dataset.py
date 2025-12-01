"""
X-ray → CTPA Patchwise Paired Dataset for DDPM Training

Loads paired X-ray DRR PNG images (full) and CTPA .nii.gz volumes (as patches).
This enables training on full 604-slice volumes by extracting smaller patches.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import SimpleITK as sitk
from PIL import Image
from typing import Tuple


def extract_patches_3d(
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    stride: Tuple[int, int, int] = (96, 96, 96)
) -> Tuple[torch.Tensor, list]:
    """
    Extract 3D patches from volume.
    
    Args:
        volume: [C, D, H, W] tensor
        patch_size: (depth, height, width) of patches
        stride: (depth, height, width) stride for extraction
    
    Returns:
        patches: [N, C, D_patch, H_patch, W_patch] tensor
        coordinates: List of (d, h, w) start coordinates
    """
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    patches = []
    coordinates = []
    
    # Extract patches with sliding window
    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patch = volume[:, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
                coordinates.append((d, h, w))
    
    # Handle remaining slices at boundaries
    # Add patches at the end if we didn't reach the edge
    if len(patches) == 0 or coordinates[-1][0] + pd < D:
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                d = D - pd
                patch = volume[:, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
                coordinates.append((d, h, w))
    
    if len(patches) == 0:
        # Volume too small - just take center crop
        d = max(0, (D - pd) // 2)
        h = max(0, (H - ph) // 2)
        w = max(0, (W - pw) // 2)
        patch = volume[:, d:min(d+pd, D), h:min(h+ph, H), w:min(w+pw, W)]
        # Pad if necessary
        if patch.shape[1] < pd or patch.shape[2] < ph or patch.shape[3] < pw:
            pad_d = max(0, pd - patch.shape[1])
            pad_h = max(0, ph - patch.shape[2])
            pad_w = max(0, pw - patch.shape[3])
            patch = torch.nn.functional.pad(
                patch, 
                (0, pad_w, 0, pad_h, 0, pad_d),
                mode='constant',
                value=patch.min()
            )
        patches.append(patch)
        coordinates.append((d, h, w))
    
    patches = torch.stack(patches)  # [N, C, D, H, W]
    return patches, coordinates


class XrayCTPAPatchDataset(Dataset):
    """
    Patchwise dataset for paired X-ray (2D PNG) and CTPA (3D .nii.gz) data.
    
    CTPA volumes are divided into patches to reduce memory usage.
    X-rays are kept full for MedCLIP encoding.
    
    Args:
        ctpa_dir: Root directory containing patient folders with .nii.gz files
        xray_pattern: Pattern for X-ray files (e.g., '*_pa_drr.png')
        split: 'train' or 'val'
        train_split: Fraction of data for training (default 0.8)
        patch_size: Size of CTPA patches (depth, height, width)
        stride: Stride for patch extraction (controls overlap)
        max_patients: Maximum number of patients to load (default None = all)
        normalization: How to normalize data ('min_max' or 'standard')
    """
    
    def __init__(
        self,
        ctpa_dir,
        xray_pattern='*_pa_drr.png',
        split='train',
        train_split=0.8,
        patch_size=(128, 128, 128),  # D, H, W for patches
        stride=(96, 96, 96),  # 25% overlap
        max_patients=None,
        normalization='min_max'
    ):
        super().__init__()
        
        self.ctpa_dir = ctpa_dir
        self.xray_pattern = xray_pattern
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.normalization = normalization
        
        # Find all patient folders
        patient_folders = sorted([
            d for d in glob(os.path.join(ctpa_dir, '*'))
            if os.path.isdir(d)
        ])
        
        if max_patients is not None and max_patients > 0:
            patient_folders = patient_folders[:max_patients]
        
        # Collect paired files
        paired_files = []
        missing_xrays = 0
        
        for patient_folder in patient_folders:
            patient_id = os.path.basename(patient_folder)
            
            # Find CTPA files: both .nii and .nii.gz (exclude *swapped*)
            ctpa_files = []
            
            # Check for .nii.gz files (exclude swapped)
            nii_gz_files = [
                f for f in glob(os.path.join(patient_folder, '*.nii.gz'))
                if 'swapped' not in os.path.basename(f).lower()
            ]
            ctpa_files.extend(nii_gz_files)
            
            # Check for .nii files only (not .nii.gz) (exclude swapped)
            nii_files = [
                f for f in glob(os.path.join(patient_folder, '*.nii'))
                if 'swapped' not in os.path.basename(f).lower()
                and not f.endswith('.nii.gz')
            ]
            ctpa_files.extend(nii_files)
            
            if not ctpa_files:
                continue
                
            ctpa_path = ctpa_files[0]  # Take first valid CTPA file
            
            # Find corresponding X-ray PNG in same patient folder
            xray_path = os.path.join(patient_folder, f"{patient_id}_pa_drr.png")
            
            if os.path.exists(xray_path):
                paired_files.append({
                    'xray': xray_path,
                    'ctpa': ctpa_path,
                    'patient_id': patient_id
                })
            else:
                missing_xrays += 1
        
        # Split train/val
        split_idx = int(len(paired_files) * train_split)
        if split == 'train':
            self.paired_files = paired_files[:split_idx]
        else:
            self.paired_files = paired_files[split_idx:]
        
        # Calculate patches per volume (for indexing)
        self._calculate_patch_indices()
        
        print(f"[{split.upper()}] Loaded {len(self.paired_files)} patients")
        print(f"  Total patches: {self.total_patches} ({self.patches_per_volume} per volume)")
        if missing_xrays > 0:
            print(f"  Warning: {missing_xrays} CTPA files had no matching X-ray")
    
    def _calculate_patch_indices(self):
        """Pre-calculate number of patches per volume."""
        # Load a sample volume to determine patch count
        if len(self.paired_files) == 0:
            self.patches_per_volume = 0
            self.total_patches = 0
            return
        
        sample_ctpa = self._load_ctpa(self.paired_files[0]['ctpa'])
        patches, _ = extract_patches_3d(
            sample_ctpa,
            patch_size=self.patch_size,
            stride=self.stride
        )
        self.patches_per_volume = patches.shape[0]
        self.total_patches = len(self.paired_files) * self.patches_per_volume
    
    def __len__(self):
        return self.total_patches
    
    def _load_xray(self, xray_path):
        """Load X-ray PNG and convert to tensor."""
        # Load PNG image
        img = Image.open(xray_path).convert('L')  # Grayscale
        
        # Resize to 224x224 for MedCLIP (expects 224x224 input)
        img = img.resize((224, 224), Image.BILINEAR)
        
        xray = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        if self.normalization == 'min_max':
            xray = xray / 255.0  # [0, 1]
        else:
            xray = (xray / 255.0) * 2.0 - 1.0  # [-1, 1]
        
        # Convert to 3-channel for MedCLIP: (H, W) → (3, H, W)
        xray = torch.from_numpy(xray).unsqueeze(0).repeat(3, 1, 1).float()
        
        return xray
    
    def _load_ctpa(self, ctpa_path):
        """Load CTPA .nii.gz volume and convert to tensor."""
        # Load NIFTI file
        image = sitk.ReadImage(ctpa_path)
        ctpa = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Handle files with channel dimension: (1, H, W, D) or (C, D, H, W)
        if ctpa.ndim == 4 and ctpa.shape[0] == 1:
            ctpa = ctpa.squeeze(0)  # Remove channel dimension
        
        # Target shape: 512x512x604
        target_shape = (604, 512, 512)  # D, H, W
        current_shape = ctpa.shape
        
        # Pad or crop depth (dimension 0)
        if current_shape[0] < target_shape[0]:
            pad_d = target_shape[0] - current_shape[0]
            ctpa = np.pad(ctpa, ((0, pad_d), (0, 0), (0, 0)), mode='constant', constant_values=ctpa.min())
        elif current_shape[0] > target_shape[0]:
            start = (current_shape[0] - target_shape[0]) // 2
            ctpa = ctpa[start:start + target_shape[0]]
        
        # Pad or crop height (dimension 1)
        if current_shape[1] < target_shape[1]:
            pad_h = target_shape[1] - current_shape[1]
            ctpa = np.pad(ctpa, ((0, 0), (0, pad_h), (0, 0)), mode='constant', constant_values=ctpa.min())
        elif current_shape[1] > target_shape[1]:
            start = (current_shape[1] - target_shape[1]) // 2
            ctpa = ctpa[:, start:start + target_shape[1]]
        
        # Pad or crop width (dimension 2)
        if current_shape[2] < target_shape[2]:
            pad_w = target_shape[2] - current_shape[2]
            ctpa = np.pad(ctpa, ((0, 0), (0, 0), (0, pad_w)), mode='constant', constant_values=ctpa.min())
        elif current_shape[2] > target_shape[2]:
            start = (current_shape[2] - target_shape[2]) // 2
            ctpa = ctpa[:, :, start:start + target_shape[2]]
        
        # Normalize
        if self.normalization == 'min_max':
            min_val = ctpa.min()
            max_val = ctpa.max()
            if max_val > min_val:
                ctpa = (ctpa - min_val) / (max_val - min_val)  # [0, 1]
        else:
            mean = ctpa.mean()
            std = ctpa.std()
            if std > 0:
                ctpa = (ctpa - mean) / std
        
        # Convert to tensor: [D, H, W] → [C, D, H, W]
        ctpa = torch.from_numpy(ctpa).unsqueeze(0).float()
        
        return ctpa
    
    def __getitem__(self, idx):
        """
        Returns a single patch with its paired X-ray.
        
        Returns:
            dict with keys:
                'ct': CTPA patch [1, D_patch, H_patch, W_patch]
                'cxr': Full X-ray [3, 224, 224]
                'target': Same as 'ct' (for compatibility)
                'patient_id': Patient identifier
                'patch_coord': Patch coordinate (d, h, w)
        """
        # Determine which volume and which patch
        volume_idx = idx // self.patches_per_volume
        patch_idx = idx % self.patches_per_volume
        
        # Get file paths
        pair = self.paired_files[volume_idx]
        
        # Load X-ray (same for all patches from this volume)
        xray = self._load_xray(pair['xray'])
        
        # Load CTPA volume and extract patches
        ctpa_volume = self._load_ctpa(pair['ctpa'])
        patches, coordinates = extract_patches_3d(
            ctpa_volume,
            patch_size=self.patch_size,
            stride=self.stride
        )
        
        # Get specific patch
        ctpa_patch = patches[patch_idx]  # [1, D_patch, H_patch, W_patch]
        patch_coord = coordinates[patch_idx]
        
        return {
            'ct': ctpa_patch,
            'cxr': xray,
            'target': ctpa_patch,
            'patient_id': pair['patient_id'],
            'patch_coord': patch_coord,
            'volume_idx': volume_idx,
            'patch_idx': patch_idx
        }
