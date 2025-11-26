"""
DDPM Training Script for 4-GPU Distributed Setup

Trains diffusion model in VQ-GAN latent space for X-ray → CTPA generation.
Uses trained VQ-GAN encoder/decoder as frozen feature extractors.
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
repo_root = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, repo_root)

from ddpm.unet import Unet3D
from ddpm.diffusion import GaussianDiffusion
from dataset.xray_ctpa_dataset import XrayCTPADataset
from torch.utils.data import DataLoader


def get_dataloaders(cfg):
    """
    Create dataloaders for X-ray → CTPA paired data.
    
    Each GPU gets 1 X-ray/CTPA pair per batch.
    Total effective batch = 4 pairs across 4 GPUs.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print(f"Loading paired X-ray → CTPA dataset...")
        print(f"X-ray dir: {cfg.dataset.xray_dir}")
        print(f"CTPA dir: {cfg.dataset.ctpa_dir}")
    
    # Get all patient folders
    ctpa_root = cfg.dataset.ctpa_dir
    patient_folders = sorted([
        os.path.join(ctpa_root, d) 
        for d in os.listdir(ctpa_root) 
        if os.path.isdir(os.path.join(ctpa_root, d))
    ])
    
    # Limit dataset if specified
    max_patients = cfg.dataset.get('max_patients', None)
    if max_patients is not None and max_patients > 0:
        patient_folders = patient_folders[:max_patients]
    
    # Collect paired X-ray and CTPA files
    paired_files = []
    for patient_folder in patient_folders:
        # Assuming X-ray files have matching names in xray_dir
        patient_id = os.path.basename(patient_folder)
        xray_path = os.path.join(cfg.dataset.xray_dir, f"{patient_id}_xray.nii.gz")
        
        # Find CTPA file in patient folder
        for root, dirs, files in os.walk(patient_folder):
            for f in files:
                if f.endswith('.nii.gz') and 'swapped' not in f.lower():
                    ctpa_path = os.path.join(root, f)
                    if os.path.exists(xray_path):
                        paired_files.append((xray_path, ctpa_path))
                    break
    
    # Split train/val
    split_idx = int(len(paired_files) * cfg.dataset.train_split)
    train_pairs = paired_files[:split_idx]
    val_pairs = paired_files[split_idx:]
    
    if local_rank == 0:
        print(f"Total paired files: {len(paired_files)}")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Val pairs: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = XrayCTPADataset(
        paired_files=train_pairs,
        patch_size=tuple(cfg.dataset.patch_size),
        vqgan_checkpoint=cfg.model.vqgan_ckpt,
        normalize=True
    )
    
    val_dataset = XrayCTPADataset(
        paired_files=val_pairs,
        patch_size=tuple(cfg.dataset.patch_size),
        vqgan_checkpoint=cfg.model.vqgan_ckpt,
        normalize=True
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


@hydra.main(config_path="config", config_name="base_cfg_ddpm", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function for 4-GPU DDPM training.
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if local_rank == 0:
        print("=" * 80)
        print("DDPM TRAINING (4-GPU DDP) - X-ray → CTPA Generation")
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
            filename='ddpm-step{step:06d}-loss{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_train_steps=cfg.model.save_and_sample_every
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # DDP Strategy
    ddp_strategy = DDPStrategy(
        find_unused_parameters=cfg.model.get('find_unused_parameters', False),
        gradient_as_bucket_view=True,
        static_graph=False,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=ddp_strategy,
        max_steps=cfg.model.train_num_steps,
        precision=16 if cfg.model.amp else 32,
        accumulate_grad_batches=cfg.model.gradient_accumulate_every,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=cfg.model.save_and_sample_every,
        enable_progress_bar=True,
        enable_model_summary=local_rank == 0,
        gradient_clip_val=cfg.model.max_grad_norm,
        sync_batchnorm=cfg.model.get('sync_batchnorm', True),
    )
    
    # Train
    if local_rank == 0:
        print("\nStarting DDPM training...")
        print(f"Training for {cfg.model.train_num_steps} steps")
        print(f"Saving checkpoints every {cfg.model.save_and_sample_every} steps")
    
    trainer.fit(diffusion, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if local_rank == 0:
        print("\n" + "=" * 80)
        print("DDPM TRAINING COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    main()
