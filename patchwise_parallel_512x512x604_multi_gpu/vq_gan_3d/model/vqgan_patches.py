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
sys.path.append('../..')

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
        # Use gradient checkpointing to save memory
        if self.training:
            z = torch.utils.checkpoint.checkpoint(
                lambda x: self.pre_vq_conv(self.encoder(x)), 
                x, 
                use_reentrant=False
            )
            vq_output = self.codebook(z)
            x_recon = torch.utils.checkpoint.checkpoint(
                lambda emb: self.decoder(self.post_vq_conv(emb)), 
                vq_output['embeddings'], 
                use_reentrant=False
            )
        else:
            z = self.pre_vq_conv(self.encoder(x))
            vq_output = self.codebook(z)
            x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        
        # Clamp output
        x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        return x_recon, vq_output
    
    def training_step(self, batch, batch_idx):
        """Training step for patch-based training."""
        # Extract patches from batch
        patches = batch['patches']  # [B*N, C, D, H, W]
        
        # Check for invalid inputs
        if torch.isnan(patches).any() or torch.isinf(patches).any():
            print(f"Warning: Invalid input data in batch {batch_idx}, skipping...")
            return None
        
        # Get optimizers
        opt_ae, opt_disc = self.optimizers()
        
        # Forward pass
        x_recon, vq_output = self.forward(patches)
        
        # Compute losses
        recon_loss = F.l1_loss(x_recon, patches) * self.l1_weight
        commitment_loss = vq_output['commitment_loss']
        
        # Total autoencoder loss
        ae_loss = recon_loss + commitment_loss
        
        # Skip NaN batches
        if torch.isnan(ae_loss):
            print(f"Warning: NaN detected in batch {batch_idx}, skipping...")
            return None
        
        # Backward and optimize
        opt_ae.zero_grad()
        self.manual_backward(ae_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt_ae.step()
        
        # Compute metrics
        with torch.no_grad():
            mse = F.mse_loss(x_recon, patches)
            psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
        
        # Log metrics
        self.log('train/recon_loss', recon_loss, prog_bar=True, 
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/commitment_loss', commitment_loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/perplexity', vq_output['perplexity'], prog_bar=True,
                 logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/codebook_usage', float(vq_output['codebook_usage']),
                 prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/psnr', psnr, prog_bar=True, logger=True, 
                 on_step=True, on_epoch=True, sync_dist=True)
        
        return ae_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        patches = batch['patches']
        
        # Forward pass
        x_recon, vq_output = self.forward(patches)
        
        # Compute metrics
        recon_loss = F.l1_loss(x_recon, patches)
        mse = F.mse_loss(x_recon, patches)
        psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
        
        # SSIM approximation
        x_flat = patches.reshape(patches.size(0), -1)
        x_recon_flat = x_recon.reshape(x_recon.size(0), -1)
        correlation = F.cosine_similarity(x_flat, x_recon_flat, dim=1).mean()
        
        # Codebook usage
        codebook_usage_pct = (vq_output['codebook_usage'] / self.n_codes) * 100.0
        
        # Log metrics
        self.log('val/recon_loss', recon_loss, prog_bar=True, sync_dist=True)
        self.log('val/psnr', psnr, prog_bar=True, sync_dist=True)
        self.log('val/ssim', correlation, prog_bar=True, sync_dist=True)
        self.log('val/perplexity', vq_output['perplexity'], sync_dist=True)
        self.log('val/codebook_usage_%', codebook_usage_pct, sync_dist=True)
        self.log('val/codebook_usage_count', float(vq_output['codebook_usage']), sync_dist=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], sync_dist=True)
    
    def configure_optimizers(self):
        """Configure optimizers."""
        lr = self.cfg.model.lr
        
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
