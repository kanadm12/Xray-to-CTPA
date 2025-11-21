from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(256, 256, 32))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class DEFAULTDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.file_paths = self.get_data_files()

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
