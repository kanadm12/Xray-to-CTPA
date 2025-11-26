"""
Patch-aware VQ-GAN model for distributed training.

Handles patches during training and provides full-volume reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
import sys
import os

# Add paths for imports
current_file = os.path.abspath(__file__)
model_dir = os.path.dirname(current_file)
vqgan_3d_dir = os.path.dirname(model_dir)
patchwise_dir = os.path.dirname(vqgan_3d_dir)
repo_root = os.path.dirname(patchwise_dir)

sys.path.insert(0, patchwise_dir)
sys.path.insert(0, repo_root)

# Import from baseline implementation
from baseline_256x256x64_single_gpu.vq_gan_3d.model.vqgan import (
    Encoder, Decoder, NLayerDiscriminator, NLayerDiscriminator3D,
    SamePadConv3d
)
from baseline_256x256x64_single_gpu.vq_gan_3d.model.codebook import Codebook
from baseline_256x256x64_single_gpu.vq_gan_3d.model.lpips import LPIPS


class VQGAN_Patches(pl.LightningModule):
    """
    VQ-GAN adapted for patch-wise training on full-resolution volumes.
    
    Architecture is identical to baseline but handles distributed training
    and patch-based data loading.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes
        self.automatic_optimization = False
        self.patch_micro_batch_size = 1  # Process 1 patch at a time (training needs more memory than validation)
        
        # Encoder/Decoder (same as baseline)
        self.encoder = Encoder(
            cfg.model.n_hiddens, 
            cfg.model.downsample,
            1,  # Single channel for medical images
            cfg.model.norm_type, 
            cfg.model.padding_type,
            cfg.model.num_groups
        )
        
        self.decoder = Decoder(
            cfg.model.n_hiddens, 
            cfg.model.downsample, 
            1,  # Single channel
            cfg.model.norm_type, 
            cfg.model.num_groups
        )
        
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, 
            cfg.model.embedding_dim, 
            1, 
            padding_type=cfg.model.padding_type
        )
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, 
            self.enc_out_ch, 
            1
        )
        
        # Codebook
        self.codebook = Codebook(
            cfg.model.n_codes, 
            cfg.model.embedding_dim,
            no_random_restart=cfg.model.no_random_restart,
            restart_thres=cfg.model.restart_thres
        )
        
        # Discriminators (disabled for initial training)
        self.image_discriminator = NLayerDiscriminator(
            1, cfg.model.disc_channels, cfg.model.disc_layers, 
            norm_layer=nn.BatchNorm2d
        )
        self.video_discriminator = NLayerDiscriminator3D(
            1, cfg.model.disc_channels, cfg.model.disc_layers,
            norm_layer=nn.BatchNorm3d
        )
        
        # Perceptual loss
        self.perceptual_model = LPIPS().eval()
        
        # Loss weights
        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight
        self.perceptual_weight = cfg.model.perceptual_weight
        self.l1_weight = cfg.model.l1_weight
        
        self.save_hyperparameters()
    
    def encode(self, x):
        """Encode patches to latent codes."""
        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        return vq_output['encodings']
    
    def decode(self, encodings):
        """Decode latent codes to patches."""
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(torch.permute(h, (0, 4, 1, 2, 3)))
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass through encoder, codebook, decoder."""
        # Simple forward pass without gradient checkpointing
        # (checkpointing was causing OOM during backward pass)
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        
        # Clamp output
        x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        return x_recon, vq_output
    
    def training_step(self, batch, batch_idx):
        """Training step for patch-based training with micro-batching."""
        # Extract patches from batch
        patches = batch['patches']  # [B*N, C, D, H, W]
        
        # Check for invalid inputs
        if torch.isnan(patches).any() or torch.isinf(patches).any():
            print(f"Warning: Invalid input data in batch {batch_idx}, skipping...")
            return None
        
        # Get optimizers
        opt_ae, opt_disc = self.optimizers()
        opt_ae.zero_grad()
        
        # Process patches in micro-batches to save memory
        num_patches = patches.shape[0]
        all_recons = []
        all_commitments = []
        all_perplexities = []
        all_usages = []
        
        for i in range(0, num_patches, self.patch_micro_batch_size):
            end_idx = min(i + self.patch_micro_batch_size, num_patches)
            patch_batch = patches[i:end_idx]
            
            # Forward pass on micro-batch
            x_recon, vq_output = self.forward(patch_batch)
            
            # Compute losses for this micro-batch
            recon_loss = F.l1_loss(x_recon, patch_batch) * self.l1_weight
            commitment_loss = vq_output['commitment_loss']
            ae_loss = recon_loss + commitment_loss
            
            # Scale loss by micro-batch proportion
            micro_batch_size = end_idx - i
            loss_scale = micro_batch_size / num_patches
            ae_loss = ae_loss * loss_scale
            
            # Skip NaN micro-batches
            if torch.isnan(ae_loss):
                print(f"Warning: NaN detected in micro-batch {i//self.patch_micro_batch_size}, skipping...")
                continue
            
            # Backward pass
            self.manual_backward(ae_loss)
            
            # Store for metrics
            all_recons.append(x_recon.detach())
            all_commitments.append(commitment_loss.detach())
            all_perplexities.append(vq_output['perplexity'].detach())
            all_usages.append(vq_output['codebook_usage'])
        
        # Optimize after processing all micro-batches
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt_ae.step()
        
        # Compute aggregated metrics
        if len(all_recons) > 0:
            all_recons = torch.cat(all_recons, dim=0)
            with torch.no_grad():
                recon_loss_avg = F.l1_loss(all_recons, patches) * self.l1_weight
                commitment_loss_avg = torch.stack(all_commitments).mean()
                perplexity_avg = torch.stack(all_perplexities).mean()
                codebook_usage_avg = sum(all_usages) / len(all_usages)
                mse = F.mse_loss(all_recons, patches)
                psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
            
            # Log metrics
            self.log('train/recon_loss', recon_loss_avg, prog_bar=True, 
                     logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/commitment_loss', commitment_loss_avg, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/perplexity', perplexity_avg, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/codebook_usage', float(codebook_usage_avg),
                     prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log('train/psnr', psnr, prog_bar=True, logger=True, 
                     on_step=True, on_epoch=True, sync_dist=True)
            
            return recon_loss_avg + commitment_loss_avg
        else:
            return None
    
    def validation_step(self, batch, batch_idx):
        """Validation step with micro-batching."""
        patches = batch['patches']
        
        # Process patches in micro-batches to save memory
        num_patches = patches.shape[0]
        all_recons = []
        all_perplexities = []
        all_usages = []
        all_commitments = []
        
        for i in range(0, num_patches, self.patch_micro_batch_size):
            end_idx = min(i + self.patch_micro_batch_size, num_patches)
            patch_batch = patches[i:end_idx]
            
            # Forward pass on micro-batch
            with torch.no_grad():
                x_recon, vq_output = self.forward(patch_batch)
            
            # Store results
            all_recons.append(x_recon)
            all_perplexities.append(vq_output['perplexity'])
            all_usages.append(vq_output['codebook_usage'])
            all_commitments.append(vq_output['commitment_loss'])
        
        # Concatenate all reconstructions
        all_recons = torch.cat(all_recons, dim=0)
        
        # Compute metrics on all patches
        recon_loss = F.l1_loss(all_recons, patches)
        mse = F.mse_loss(all_recons, patches)
        psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
        
        # SSIM approximation
        x_flat = patches.reshape(patches.size(0), -1)
        x_recon_flat = all_recons.reshape(all_recons.size(0), -1)
        correlation = F.cosine_similarity(x_flat, x_recon_flat, dim=1).mean()
        
        # Average metrics across micro-batches
        avg_perplexity = torch.stack(all_perplexities).mean()
        avg_usage = sum(all_usages) / len(all_usages)
        avg_commitment = torch.stack(all_commitments).mean()
        codebook_usage_pct = (avg_usage / self.n_codes) * 100.0
        
        # Log metrics
        self.log('val/recon_loss', recon_loss, prog_bar=True, sync_dist=True)
        self.log('val/psnr', psnr, prog_bar=True, sync_dist=True)
        self.log('val/ssim', correlation, prog_bar=True, sync_dist=True)
        self.log('val/perplexity', avg_perplexity, sync_dist=True)
        self.log('val/codebook_usage_%', codebook_usage_pct, sync_dist=True)
        self.log('val/codebook_usage_count', float(avg_usage), sync_dist=True)
        self.log('val/commitment_loss', avg_commitment, sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizers."""
        lr = self.cfg.model.learning_rate
        
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.pre_vq_conv.parameters()) +
            list(self.post_vq_conv.parameters()) +
            list(self.codebook.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        opt_disc = torch.optim.Adam(
            list(self.image_discriminator.parameters()) +
            list(self.video_discriminator.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        
        return [opt_ae, opt_disc]
