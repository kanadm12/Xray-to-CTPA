"""
X-ray → CTPA Paired Dataset for DDPM Training

Loads paired X-ray DRR PNG images and CTPA .nii.gz volumes.
X-rays are co-located with CTPA files as {patient_id}_pa_drr.png
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import SimpleITK as sitk
from PIL import Image


class XrayCTPADataset(Dataset):
    """
    Dataset for paired X-ray (2D PNG) and CTPA (3D .nii.gz) data.
    
    Args:
        ctpa_dir: Root directory containing patient folders with .nii.gz files
        xray_pattern: Pattern for X-ray files (e.g., '*_pa_drr.png')
        split: 'train' or 'val'
        train_split: Fraction of data for training (default 0.8)
        max_patients: Maximum number of patients to load (default None = all)
        use_latent: Whether data will be encoded to VQ-GAN latent space (default True)
        normalization: How to normalize data ('min_max' or 'standard')
    """
    
    def __init__(
        self,
        ctpa_dir,
        xray_pattern='*_pa_drr.png',
        split='train',
        train_split=0.8,
        max_patients=None,
        use_latent=True,
        normalization='min_max'
    ):
        super().__init__()
        
        self.ctpa_dir = ctpa_dir
        self.xray_pattern = xray_pattern
        self.split = split
        self.use_latent = use_latent
        self.normalization = normalization
        
        # Find all patient folders
        patient_folders = sorted([
            d for d in glob(os.path.join(ctpa_dir, '*'))
            if os.path.isdir(d)
        ])
        
        if max_patients is not None and max_patients > 0:
            patient_folders = patient_folders[:max_patients]
        
        # Collect paired files
        self.paired_files = []
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
            
            # Check for .nii files (exclude swapped)
            nii_files = [
                f for f in glob(os.path.join(patient_folder, '*.nii'))
                if 'swapped' not in os.path.basename(f).lower()
            ]
            ctpa_files.extend(nii_files)
            
            if not ctpa_files:
                continue
                
            ctpa_path = ctpa_files[0]  # Take first valid CTPA file
            
            # Find corresponding X-ray PNG in same patient folder
            xray_path = os.path.join(patient_folder, f"{patient_id}_pa_drr.png")
            
            if os.path.exists(xray_path):
                self.paired_files.append({
                    'xray': xray_path,
                    'ctpa': ctpa_path,
                    'patient_id': patient_id
                })
            else:
                missing_xrays += 1
        
        # Split train/val
        split_idx = int(len(self.paired_files) * train_split)
        if split == 'train':
            self.paired_files = self.paired_files[:split_idx]
        else:
            self.paired_files = self.paired_files[split_idx:]
        
        print(f"[{split.upper()}] Loaded {len(self.paired_files)} paired X-ray/CTPA samples")
        if missing_xrays > 0:
            print(f"  Warning: {missing_xrays} CTPA files had no matching X-ray")
    
    def __len__(self):
        return len(self.paired_files)
    
    def _load_xray(self, xray_path):
        """Load X-ray PNG and convert to tensor."""
        # Load PNG image
        img = Image.open(xray_path).convert('L')  # Grayscale
        
        # Resize to 224x224 for MedCLIP (expects 224x224 input)
        img = img.resize((224, 224), Image.BILINEAR)
        
        xray = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] or [-1, 1]
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
        
        # Pad or crop to target depth of 604 (will become 151 after VQ-GAN encoding with downsample=4)
        target_depth = 604
        current_depth = ctpa.shape[0]
        
        if current_depth < target_depth:
            # Pad to target depth (pad at the end)
            pad_amount = target_depth - current_depth
            ctpa = np.pad(ctpa, ((0, pad_amount), (0, 0), (0, 0)), mode='constant', constant_values=ctpa.min())
        elif current_depth > target_depth:
            # Crop to target depth (center crop)
            start = (current_depth - target_depth) // 2
            ctpa = ctpa[start:start + target_depth]
        
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
                ctpa = (ctpa - mean) / std  # Standardized
        
        # Add channel dimension: (D, H, W) → (1, D, H, W)
        ctpa = torch.from_numpy(ctpa).unsqueeze(0).float()
        
        return ctpa
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'ct': Tensor of shape (1, D, H, W) - CTPA volume (expected by GaussianDiffusion)
                'cxr': Tensor of shape (1, H, W) - X-ray image (expected by GaussianDiffusion)
                'target': Tensor of shape (1,) - Label for classification (dummy value)
                'patient_id': str - Patient identifier
        """
        sample = self.paired_files[idx]
        
        xray = self._load_xray(sample['xray'])
        ctpa = self._load_ctpa(sample['ctpa'])
        
        # Create dummy target label (not used when classification_weight=0)
        target = torch.tensor([0.0], dtype=torch.float32)
        
        return {
            'ct': ctpa,           # Match expected key in forward()
            'cxr': xray,          # Match expected key in forward()
            'target': target,     # Match expected key in forward()
            'patient_id': sample['patient_id']
        }


def get_xray_ctpa_dataloaders(cfg, batch_size=1, num_workers=4):
    """
    Helper function to create train/val dataloaders.
    
    Args:
        cfg: Hydra config with dataset parameters
        batch_size: Batch size per GPU (default 1)
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = XrayCTPADataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        xray_pattern=cfg.dataset.get('xray_pattern', '*_pa_drr.png'),
        split='train',
        train_split=cfg.dataset.get('train_split', 0.8),
        max_patients=cfg.dataset.get('max_patients', None),
        use_latent=cfg.dataset.get('use_latent', True),
        normalization=cfg.dataset.get('normalization', 'min_max')
    )
    
    val_dataset = XrayCTPADataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        xray_pattern=cfg.dataset.get('xray_pattern', '*_pa_drr.png'),
        split='val',
        train_split=cfg.dataset.get('train_split', 0.8),
        max_patients=cfg.dataset.get('max_patients', None),
        use_latent=cfg.dataset.get('use_latent', True),
        normalization=cfg.dataset.get('normalization', 'min_max')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2 if num_workers > 0 else 0,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader
