"""
Training script for VQ-GAN with 4-GPU DDP (DistributedDataParallel).

Key features:
- Each GPU processes 1 patient volume (batch_size=1 per GPU)
- Total effective batch_size = 4 (1 per GPU × 4 GPUs)
- Gradient synchronization across all GPUs
- Patch-wise processing within each volume
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

# Add paths for imports - use single-GPU implementation
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
single_gpu_dir = os.path.join(repo_root, 'patchwise_512x512x604_single_gpu')

sys.path.insert(0, single_gpu_dir)
sys.path.insert(0, repo_root)

from dataset.patch_dataset import PatchDataset, collate_patches
from vq_gan_3d.model.vqgan_patches import VQGAN_Patches


def get_dataloaders(cfg):
    """
    Create dataloaders for 4-GPU distributed training.
    
    Each GPU will get 1 volume per batch.
    Total effective batch size = 4 volumes across 4 GPUs.
    """
    data_dir = cfg.dataset.root_dir
    
    print(f"[Rank {os.environ.get('LOCAL_RANK', 0)}] Searching for data in: {data_dir}")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Optional: Limit to subset of patient folders
    max_patients = cfg.dataset.get('max_patients', None)
    
    # Get all patient folders
    patient_folders = sorted([
        os.path.join(data_dir, d) 
        for d in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    
    if max_patients is not None and max_patients > 0:
        patient_folders = patient_folders[:max_patients]
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(f"Limited to first {max_patients} patient folders")
    
    # Collect all .nii.gz files from selected patient folders
    all_files = []
    for patient_folder in patient_folders:
        for root, dirs, files in os.walk(patient_folder):
            for f in files:
                if (f.endswith('.nii.gz') or f.endswith('.nii')) and 'swapped' not in f.lower():
                    all_files.append(os.path.join(root, f))
    
    all_files = sorted(all_files)
    
    if len(all_files) == 0:
        raise ValueError(f"No NIfTI files found in {data_dir}")
    
    # Split train/val
    split_idx = int(len(all_files) * cfg.dataset.train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(f"Total files: {len(all_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Val files: {len(val_files)}")
        print(f"Effective batch size: {cfg.model.batch_size} × 4 GPUs = {cfg.model.batch_size * 4}")
    
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
    
    # Create dataloaders
    # PyTorch Lightning will automatically add DistributedSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,  # 1 volume per GPU
        shuffle=True,
        num_workers=cfg.model.num_workers,
        collate_fn=collate_patches,
        pin_memory=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False,
        drop_last=True  # Important for DDP stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers // 2 if cfg.model.num_workers > 0 else 0,
        collate_fn=collate_patches,
        pin_memory=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False
    )
    
    return train_loader, val_loader


# Get absolute config path
_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

@hydra.main(config_path=_config_dir, config_name="base_cfg", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function for 4-GPU distributed training.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print("=" * 80)
        print("PATCH-WISE VQ-GAN TRAINING (4-GPU DDP)")
        print("=" * 80)
        print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        print("=" * 80)
    
    # Set up training
    pl.seed_everything(cfg.model.seed)
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(cfg)
    
    # Create model
    model = VQGAN_Patches(cfg)
    
    # Set up callbacks (only on rank 0 to avoid conflicts)
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.model.default_root_dir + '/checkpoints',
            filename='vqgan-patches-epoch{epoch:02d}-psnr{val_psnr:.2f}',  # Fixed: no slash
            monitor='val/psnr',
            mode='max',
            save_top_k=3,
            save_last=True,
            save_on_train_epoch_end=False  # Save on validation end
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # DDP Strategy with optimizations
    ddp_strategy = DDPStrategy(
        find_unused_parameters=cfg.model.get('find_unused_parameters', False),
        gradient_as_bucket_view=True,  # Memory optimization
        static_graph=False,  # Dynamic graph for flexibility
    )
    
    # Create trainer for 4-GPU DDP
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,  # 4 GPUs
        strategy=ddp_strategy,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        accumulate_grad_batches=cfg.model.get('accumulate_grad_batches', 1),
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        enable_progress_bar=True,
        enable_model_summary=local_rank == 0,  # Only on rank 0
        default_root_dir=cfg.model.default_root_dir,
        sync_batchnorm=cfg.model.get('sync_batchnorm', True),  # Sync BN across GPUs
        use_distributed_sampler=True,  # Explicit DDP sampler
    )
    
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if local_rank == 0:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    main()
