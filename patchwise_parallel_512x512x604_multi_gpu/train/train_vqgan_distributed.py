"""
Distributed training script for VQ-GAN on full-resolution CTPA with patch-wise parallelism.

Runs on 4 H200 GPUs using PyTorch DDP.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('..')
from dataset.patch_dataset import PatchDataset, collate_patches
from vq_gan_3d.model.vqgan_patches import VQGAN_Patches


class DistributedTrainer:
    """
    Distributed trainer for patch-wise VQ-GAN training.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
    def setup_distributed(self, rank, world_size):
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(rank)
        
    def cleanup_distributed(self):
        """Clean up distributed training."""
        dist.destroy_process_group()
    
    def get_dataloaders(self, rank, world_size):
        """Create distributed dataloaders."""
        # Get file paths
        data_dir = self.cfg.dataset.root_dir
        all_files = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.nii.gz')
        ])
        
        # Split train/val
        split_idx = int(len(all_files) * self.cfg.dataset.train_split)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        if rank == 0:
            print(f"Total files: {len(all_files)}")
            print(f"Train files: {len(train_files)}")
            print(f"Val files: {len(val_files)}")
        
        # Create datasets
        train_dataset = PatchDataset(
            volume_paths=train_files,
            patch_size=tuple(self.cfg.dataset.patch_size),
            stride=tuple(self.cfg.dataset.stride),
            normalize=True,
            cache_volumes=False
        )
        
        val_dataset = PatchDataset(
            volume_paths=val_files,
            patch_size=tuple(self.cfg.dataset.patch_size),
            stride=tuple(self.cfg.dataset.stride),
            normalize=True,
            cache_volumes=False
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.model.batch_size,
            sampler=train_sampler,
            num_workers=self.cfg.model.num_workers,
            collate_fn=collate_patches,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.model.batch_size,
            sampler=val_sampler,
            num_workers=self.cfg.model.num_workers // 2,
            collate_fn=collate_patches,
            pin_memory=True
        )
        
        return train_loader, val_loader


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function using PyTorch Lightning with DDP.
    """
    print("=" * 80)
    print("PATCH-WISE PARALLEL VQ-GAN TRAINING")
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
    
    # Set up DDP strategy
    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        strategy=strategy,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(model)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Ensure proper environment variables for distributed training
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    main()
