from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(384, 384, 256))  # Reduced for H200 144GB memory constraints
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = 'train', train_split: float = 0.8, augmentation: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS if (augmentation and mode == 'train') else tio.Compose([])
        self.file_paths = self.get_data_files()
        
        # Split data into train/val
        import random
        random.seed(1234)  # For reproducibility
        random.shuffle(self.file_paths)
        split_idx = int(len(self.file_paths) * train_split)
        
        if mode == 'train':
            self.file_paths = self.file_paths[:split_idx]
        elif mode == 'val':
            self.file_paths = self.file_paths[split_idx:]
        
        print(f"{mode.upper()} dataset: {len(self.file_paths)} files")

    def get_data_files(self):
        """Recursively find all .nii and .nii.gz files, excluding those with 'swapped' in the filename."""
        file_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # Include both .nii and .nii.gz files, exclude files with 'swapped' in the name
                if (file.endswith('.nii.gz') or file.endswith('.nii')) and 'swapped' not in file.lower():
                    file_paths.append(os.path.join(root, file))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(self.file_paths[idx])
        img = self.preprocessing(img)
        img = self.transforms(img)
        # Return data with 'ct' key expected by VQGAN model (C, T, H, W)
        return {'ct': img.data}
