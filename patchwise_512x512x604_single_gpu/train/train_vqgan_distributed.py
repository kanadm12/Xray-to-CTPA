"""
Training script for VQ-GAN on full-resolution CTPA with patch-wise processing.

Runs on single GPU, trains on full 512×512×604 resolution using patches.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('..')
from dataset.patch_dataset import PatchDataset, collate_patches
from vq_gan_3d.model.vqgan_patches import VQGAN_Patches


def get_dataloaders(cfg):
    """Create dataloaders for single-GPU training."""
    # Get file paths
    data_dir = cfg.dataset.root_dir
    all_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.nii.gz')
    ])
    
    # Split train/val
    split_idx = int(len(all_files) * cfg.dataset.train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    
    # Create datasets
    train_dataset = PatchDataset(
        volume_paths=train_files,
        patch_size=tuple(cfg.dataset.patch_size),
        stride=tuple(cfg.dataset.stride),
        normalize=True,
        cache_volumes=False
    )
    
    val_dataset = PatchDataset(
        volume_paths=val_files,
        patch_size=tuple(cfg.dataset.patch_size),
        stride=tuple(cfg.dataset.stride),
        normalize=True,
        cache_volumes=False
    )
    
    # Create dataloaders (no distributed sampler needed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers,
        collate_fn=collate_patches,
        pin_memory=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers // 2,
        collate_fn=collate_patches,
        pin_memory=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False
    )
    
    return train_loader, val_loader


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function for single-GPU patch-wise training.
    """
    print("=" * 80)
    print("PATCH-WISE VQ-GAN TRAINING (Single GPU)")
    print("=" * 80)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print("=" * 80)
    
    # Set up training
    pl.seed_everything(cfg.model.seed)
    
    # Create model
    model = VQGAN_Patches(cfg)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.model.default_root_dir + '/checkpoints',
            filename='vqgan-patches-{epoch:02d}-{val/psnr:.2f}',
            monitor='val/psnr',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create trainer (single GPU, no DDP)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,  # Single GPU
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=cfg.model.default_root_dir
    )
    
    # Train
    trainer.fit(model)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
