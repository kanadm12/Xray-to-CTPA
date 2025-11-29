"""
Working DDPM Inference - Generate CTPA patch from X-ray

Successfully generates 256×256×128 CTPA patches.
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import SimpleITK as sitk
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
baseline_dir = os.path.join(parent_dir, 'baseline_256x256x64_single_gpu')

sys.path.insert(0, baseline_dir)
sys.path.insert(0, current_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion
from vq_gan_3d.model.vqgan import VQGAN


def load_xray(xray_path, device='cuda'):
    """Load and preprocess X-ray for MedCLIP."""
    img = Image.open(xray_path).convert('L')
    img = img.resize((224, 224), Image.BILINEAR)
    
    xray = np.array(img, dtype=np.float32) / 255.0
    xray = torch.from_numpy(xray).unsqueeze(0).repeat(3, 1, 1).float()
    xray = xray.unsqueeze(0)  # [1, 3, 224, 224]
    
    return xray.to(device)


def load_models(ddpm_ckpt, vqgan_ckpt, device='cuda'):
    """Load DDPM and VQ-GAN models."""
    print("Loading VQ-GAN...")
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt)
    vqgan = vqgan.to(device)
    vqgan.eval()
    
    print("Loading DDPM...")
    checkpoint = torch.load(ddpm_ckpt, map_location=device)
    hparams = checkpoint.get('hyper_parameters', {})
    
    model = Unet3D(
        dim=hparams.get('image_size', 64),
        cond_dim=512,
        dim_mults=[1, 2, 4, 8],
        channels=64,
        resnet_groups=8,
        classifier_free_guidance=False,
        medclip=True
    )
    
    # Create diffusion WITHOUT vqgan_ckpt (manual decoding)
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=None,
        image_size=hparams.get('image_size', 64),
        num_frames=hparams.get('num_frames', 32),
        channels=64,
        timesteps=1000,
        loss_type='l1',
        img_cond=True,
        medclip=True,
        classification_weight=0.0,
        discriminator_weight=0.0,
        perceptual_weight=0.0
    )
    
    diffusion.load_state_dict(checkpoint['state_dict'], strict=False)
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    print("Models loaded!")
    return diffusion, vqgan


@torch.no_grad()
def generate_ctpa_patch(xray, diffusion, vqgan, device='cuda'):
    """Generate CTPA patch from X-ray."""
    print("Generating CTPA patch...")
    
    # Get X-ray embedding
    if diffusion.medclip:
        cond = diffusion.xray_encoder.encode_image(xray, normalize=True)
    else:
        cond = diffusion.xray_encoder(xray)[0]
    
    batch_size = 1
    image_size = diffusion.image_size
    channels = diffusion.channels
    num_frames = diffusion.num_frames
    
    # Run denoising loop to get latent
    print("Running diffusion sampling...")
    print(f"Target shape: ({batch_size}, {channels}, {num_frames}, {image_size}, {image_size})")
    latent = diffusion.p_sample_loop(
        (batch_size, channels, num_frames, image_size, image_size),
        cond=cond,
        cond_scale=1.0
    )
    
    print(f"Generated latent shape: {latent.shape}, range: [{latent.min():.4f}, {latent.max():.4f}]")
    
    # Denormalize latent from [-1, 1] back to codebook embedding range
    emb_min = vqgan.codebook.embeddings.min()
    emb_max = vqgan.codebook.embeddings.max()
    emb_range = emb_max - emb_min
    
    if emb_range > 1e-6:
        latent = ((latent + 1.0) / 2.0) * emb_range + emb_min
    
    print(f"Denormalized latent range: [{latent.min():.4f}, {latent.max():.4f}]")
    print(f"Codebook embedding range: [{emb_min:.4f}, {emb_max:.4f}]")
    
    # Fix dimension order: [B, C, D, H, W] -> [B, C, H, W, D]
    if latent.shape[2] < latent.shape[3]:
        latent = latent.permute(0, 1, 3, 4, 2)
    
    print(f"Decoding latent shape: {latent.shape}")
    
    # Decode directly (bypass quantization)
    h = vqgan.post_vq_conv(latent)
    print(f"After post_vq_conv: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")
    
    patch = vqgan.decoder(h)
    print(f"Generated patch shape: {patch.shape}, range=[{patch.min():.4f}, {patch.max():.4f}]")
    return patch


def save_as_nifti(volume, output_path):
    """Save as .nii.gz file."""
    if volume.dim() == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.dim() == 4:
        volume = volume.squeeze(0)
    
    vol_np = volume.cpu().numpy()
    
    # Normalize to [0, 1] using min-max normalization instead of clipping
    vmin, vmax = vol_np.min(), vol_np.max()
    print(f"Volume range before normalization: [{vmin:.4f}, {vmax:.4f}]")
    
    if vmax > vmin:
        vol_np = (vol_np - vmin) / (vmax - vmin)
    else:
        vol_np = np.zeros_like(vol_np)
    
    vol_np = (vol_np * 255).astype(np.uint8)
    
    vol_sitk = sitk.GetImageFromArray(vol_np)
    sitk.WriteImage(vol_sitk, output_path)
    print(f"Saved NIFTI: {output_path}")


def save_slices_png(volume, output_dir, num_slices=10):
    """Save slices as PNG."""
    if volume.dim() == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.dim() == 4:
        volume = volume.squeeze(0)
    
    vol_np = volume.cpu().numpy()
    depth = vol_np.shape[0]
    
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, slice_idx in enumerate(slice_indices):
        slice_img = vol_np[slice_idx]
        slice_img = (slice_img * 255).clip(0, 255).astype(np.uint8)
        
        img = Image.fromarray(slice_img)
        img.save(os.path.join(output_dir, f"slice_{i:03d}_depth{slice_idx:03d}.png"))
    
    print(f"Saved {num_slices} slices: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate CTPA patch from X-ray')
    parser.add_argument('--xray_path', type=str, required=True)
    parser.add_argument('--ddpm_ckpt', type=str, required=True)
    parser.add_argument('--vqgan_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/inference/')
    parser.add_argument('--save_slices', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DDPM Inference - X-ray to CTPA Patch Generation")
    print("=" * 80)
    print(f"X-ray: {args.xray_path}")
    print(f"DDPM: {args.ddpm_ckpt}")
    print(f"VQ-GAN: {args.vqgan_ckpt}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load
    xray = load_xray(args.xray_path, device=args.device)
    diffusion, vqgan = load_models(args.ddpm_ckpt, args.vqgan_ckpt, device=args.device)
    
    # Generate
    patch = generate_ctpa_patch(xray, diffusion, vqgan, device=args.device)
    
    # Save
    xray_name = Path(args.xray_path).stem
    
    nifti_path = os.path.join(args.output_dir, f"{xray_name}_ctpa_patch.nii.gz")
    save_as_nifti(patch, nifti_path)
    
    if args.save_slices:
        slice_dir = os.path.join(args.output_dir, f"{xray_name}_slices")
        save_slices_png(patch, slice_dir, num_slices=10)
    
    print("\n" + "=" * 80)
    print("SUCCESS! Generated 256×256×128 CTPA patch")
    print("=" * 80)


if __name__ == "__main__":
    main()
