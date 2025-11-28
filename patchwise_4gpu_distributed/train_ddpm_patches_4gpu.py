"""
DDPM Patchwise Training Script for 4-GPU Distributed Setup

Trains diffusion model in VQ-GAN latent space for X-ray → CTPA generation.
Uses patch-based approach to handle full 604-slice volumes efficiently.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
baseline_dir = os.path.join(parent_dir, 'baseline_256x256x64_single_gpu')

sys.path.insert(0, baseline_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion
from dataset.xray_ctpa_patch_dataset import XrayCTPAPatchDataset
from torch.utils.data import DataLoader


def get_dataloaders(cfg):
    """
    Create dataloaders for X-ray → CTPA patchwise paired data.
    
    Each sample contains:
    - X-ray: Full 224×224 image (for MedCLIP)
    - CTPA patch: 128×128×128 (or configured size)
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print(f"Loading patchwise X-ray → CTPA dataset...")
        print(f"CTPA dir: {cfg.dataset.ctpa_dir}")
        print(f"Patch size: {cfg.model.patch_size}")
        print(f"Stride: {cfg.model.stride}")
    
    # Create datasets using patchwise implementation
    train_dataset = XrayCTPAPatchDataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        xray_pattern=cfg.dataset.get('xray_pattern', '*_pa_drr.png'),
        split='train',
        train_split=cfg.dataset.get('train_split', 0.8),
        patch_size=tuple(cfg.model.patch_size),
        stride=tuple(cfg.model.stride),
        max_patients=cfg.dataset.get('max_patients', None),
        normalization=cfg.dataset.get('normalization', 'min_max')
    )
    
    val_dataset = XrayCTPAPatchDataset(
        ctpa_dir=cfg.dataset.ctpa_dir,
        xray_pattern=cfg.dataset.get('xray_pattern', '*_pa_drr.png'),
        split='val',
        train_split=cfg.dataset.get('train_split', 0.8),
        patch_size=tuple(cfg.model.patch_size),
        stride=tuple(cfg.model.stride),
        max_patients=cfg.dataset.get('max_patients', None),
        normalization=cfg.dataset.get('normalization', 'min_max')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers // 2 if cfg.model.num_workers > 0 else 0,
        pin_memory=True,
        persistent_workers=True if cfg.model.num_workers > 0 else False
    )
    
    return train_loader, val_loader


def main_wrapper():
    """Wrapper to set absolute config path for Hydra."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config")
    
    @hydra.main(config_path=config_path, config_name="base_cfg_ddpm", version_base=None)
    def main(cfg: DictConfig):
        """
        Main training function for 4-GPU patchwise DDPM training.
        """
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if local_rank == 0:
            print("=" * 80)
            print("DDPM PATCHWISE TRAINING (4-GPU DDP) - X-ray → CTPA Generation")
            print("=" * 80)
            print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
            print("=" * 80)
        
        # Set seed
        pl.seed_everything(cfg.model.seed)
        
        # Check VQ-GAN checkpoint exists
        if not os.path.exists(cfg.model.vqgan_ckpt):
            raise FileNotFoundError(
                f"VQ-GAN checkpoint not found: {cfg.model.vqgan_ckpt}\n"
                "Train VQ-GAN first using launch_4gpu_vqgan_disc.sh"
            )
        
        # Create dataloaders
        train_loader, val_loader = get_dataloaders(cfg)
        
        # Create UNet model
        if local_rank == 0:
            print("Creating UNet3D model...")
        
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            cond_dim=cfg.model.cond_dim,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            resnet_groups=8,
            classifier_free_guidance=cfg.model.classifier_free_guidance,
            medclip=cfg.model.medclip
        )
        
        # Create Gaussian Diffusion wrapper
        if local_rank == 0:
            print("Creating Gaussian Diffusion...")
        
        diffusion = GaussianDiffusion(
            model,
            vqgan_ckpt=cfg.model.vqgan_ckpt,
            vae_ckpt=cfg.model.vae_ckpt,
            image_size=cfg.model.diffusion_img_size,
            num_frames=cfg.model.diffusion_depth_size,
            channels=cfg.model.diffusion_num_channels,
            timesteps=cfg.model.timesteps,
            img_cond=True,  # Condition on X-ray
            loss_type=cfg.model.loss_type,
            l1_weight=cfg.model.l1_weight,
            perceptual_weight=cfg.model.perceptual_weight,
            discriminator_weight=cfg.model.discriminator_weight,
            classification_weight=cfg.model.classification_weight,
            classifier_free_guidance=cfg.model.classifier_free_guidance,
            medclip=cfg.model.medclip,
            name_dataset=cfg.model.name_dataset,
            dataset_min_value=cfg.model.dataset_min_value,
            dataset_max_value=cfg.model.dataset_max_value,
        )
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=cfg.model.results_folder,
                filename='ddpm-patches-epoch{epoch:02d}-loss{val/loss:.4f}',
                monitor='val/loss',
                mode='min',
                save_top_k=3,
                save_last=True,
                every_n_epochs=1
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        # DDP Strategy (enable find_unused_parameters for discriminator/classifier)
        ddp_strategy = DDPStrategy(
            find_unused_parameters=True,  # Required: discriminator/classifier not always used
            gradient_as_bucket_view=True,
            static_graph=False,
        )
        
        # Enable gradient checkpointing if specified
        if cfg.model.get('gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            if local_rank == 0:
                print("Gradient checkpointing enabled to reduce memory usage")
        
        # Create trainer
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=4,
            strategy=ddp_strategy,
            max_epochs=cfg.model.get('max_epochs', 30),
            precision=16 if cfg.model.amp else 32,
            accumulate_grad_batches=cfg.model.gradient_accumulate_every,
            callbacks=callbacks,
            log_every_n_steps=50,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            enable_model_summary=local_rank == 0,
            gradient_clip_val=cfg.model.max_grad_norm,
            sync_batchnorm=cfg.model.get('sync_batchnorm', True),
        )
        
        # Train
        if local_rank == 0:
            print("\nStarting patchwise DDPM training...")
            print(f"Training for {cfg.model.get('max_epochs', 30)} epochs")
            print(f"Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")
            print(f"Patch size: {cfg.model.patch_size} → Latent: {cfg.model.diffusion_depth_size}×{cfg.model.diffusion_img_size}×{cfg.model.diffusion_img_size}")
        
        trainer.fit(diffusion, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        if local_rank == 0:
            print("\n" + "=" * 80)
            print("PATCHWISE DDPM TRAINING COMPLETE!")
            print("=" * 80)

    return main
    
if __name__ == "__main__":
    main_fn = main_wrapper()
    main_fn()
