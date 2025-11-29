"""
DDPM Inference Script - Generate CTPA from X-ray

Usage:
    python inference_ddpm.py --xray_path /path/to/xray.png --output_dir ./outputs/inference/
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import SimpleITK as sitk
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
baseline_dir = os.path.join(parent_dir, 'baseline_256x256x64_single_gpu')

sys.path.insert(0, baseline_dir)
sys.path.insert(0, current_dir)

from ddpm.diffusion import Unet3D, GaussianDiffusion
from vq_gan_3d.model.vqgan import VQGAN


def load_xray(xray_path, device='cuda'):
    """Load and preprocess X-ray image for MedCLIP."""
    img = Image.open(xray_path).convert('L')
    img = img.resize((224, 224), Image.BILINEAR)
    
    xray = np.array(img, dtype=np.float32)
    xray = xray / 255.0  # [0, 1]
    
    # Convert to 3-channel: (H, W) → (3, H, W)
    xray = torch.from_numpy(xray).unsqueeze(0).repeat(3, 1, 1).float()
    xray = xray.unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
    
    return xray.to(device)


def load_models(ddpm_ckpt_path, vqgan_ckpt_path, device='cuda'):
    """Load DDPM and VQ-GAN models."""
    print("Loading VQ-GAN...")
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt_path)
    vqgan = vqgan.to(device)
    vqgan.eval()
    
    print("Loading DDPM...")
    # Load checkpoint
    checkpoint = torch.load(ddpm_ckpt_path, map_location=device)
    
    # Extract hyperparameters from checkpoint if available
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create UNet3D model
    model = Unet3D(
        dim=hparams.get('image_size', 64),
        cond_dim=512,  # MedCLIP output dim
        dim_mults=[1, 2, 4, 8],
        channels=64,
        resnet_groups=8,
        classifier_free_guidance=False,
        medclip=True
    )
    
    # Create GaussianDiffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=vqgan_ckpt_path,
        image_size=hparams.get('image_size', 64),
        num_frames=hparams.get('num_frames', 32),
        channels=64,
        timesteps=1000,
        loss_type='l1',
        img_cond=True,
        medclip=True
    )
    
    # Load state dict
    diffusion.load_state_dict(checkpoint['state_dict'])
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    print("Models loaded successfully!")
    return diffusion, vqgan


@torch.no_grad()
def generate_ctpa(xray, diffusion, vqgan, num_samples=1, device='cuda'):
    """
    Generate CTPA volume from X-ray using DDPM.
    
    Args:
        xray: X-ray tensor [1, 3, 224, 224]
        diffusion: GaussianDiffusion model
        vqgan: VQ-GAN model for decoding
        num_samples: Number of samples to generate
        device: cuda or cpu
    
    Returns:
        Generated CTPA volumes [num_samples, 1, D, H, W]
    """
    print("Generating CTPA from X-ray...")
    
    batch_size = xray.shape[0]
    
    # Sample from DDPM (generates latent representation)
    # This performs the denoising process
    latent_samples = diffusion.sample(
        cond=xray,
        batch_size=batch_size,
        return_all_timesteps=False
    )
    
    print(f"Generated latent shape: {latent_samples.shape}")
    
    # Decode latent to full CTPA volume using VQ-GAN
    print("Decoding latent with VQ-GAN...")
    
    # Check if we need to permute dimensions back
    # DDPM outputs [B, C, D, H, W], VQ-GAN decoder expects [B, C, H, W, D]
    if latent_samples.shape[2] < latent_samples.shape[3]:
        # Likely [B, C, D, H, W], permute to [B, C, H, W, D]
        latent_samples = latent_samples.permute(0, 1, 3, 4, 2)
    
    ctpa_volumes = vqgan.decode(latent_samples)
    
    print(f"Decoded CTPA shape: {ctpa_volumes.shape}")
    
    return ctpa_volumes


def save_ctpa_volume(ctpa_tensor, output_path):
    """Save CTPA tensor as .nii.gz file."""
    # Convert from [1, C, D, H, W] to [D, H, W]
    if ctpa_tensor.dim() == 5:
        ctpa_tensor = ctpa_tensor.squeeze(0).squeeze(0)  # Remove batch and channel
    elif ctpa_tensor.dim() == 4:
        ctpa_tensor = ctpa_tensor.squeeze(0)
    
    # Convert to numpy
    ctpa_np = ctpa_tensor.cpu().numpy()
    
    # Denormalize if needed (assuming [0, 1] range)
    ctpa_np = (ctpa_np * 255).clip(0, 255).astype(np.uint8)
    
    # Create SimpleITK image
    ctpa_sitk = sitk.GetImageFromArray(ctpa_np)
    
    # Save as .nii.gz
    sitk.WriteImage(ctpa_sitk, output_path)
    print(f"Saved CTPA volume to: {output_path}")


def save_slices_as_png(ctpa_tensor, output_dir, num_slices=10):
    """Save representative slices as PNG for quick visualization."""
    if ctpa_tensor.dim() == 5:
        ctpa_tensor = ctpa_tensor.squeeze(0).squeeze(0)
    elif ctpa_tensor.dim() == 4:
        ctpa_tensor = ctpa_tensor.squeeze(0)
    
    ctpa_np = ctpa_tensor.cpu().numpy()
    depth = ctpa_np.shape[0]
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, slice_idx in enumerate(slice_indices):
        slice_img = ctpa_np[slice_idx]
        slice_img = (slice_img * 255).clip(0, 255).astype(np.uint8)
        
        img = Image.fromarray(slice_img)
        img.save(os.path.join(output_dir, f"slice_{i:03d}_depth{slice_idx:03d}.png"))
    
    print(f"Saved {num_slices} slices to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate CTPA from X-ray using DDPM')
    parser.add_argument('--xray_path', type=str, required=True, help='Path to input X-ray PNG')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='./checkpoints/ddpm_4gpu_patches/last.ckpt',
                        help='Path to DDPM checkpoint')
    parser.add_argument('--vqgan_ckpt', type=str,
                        default='./outputs/vqgan_patches_4gpu/checkpoints/last.ckpt',
                        help='Path to VQ-GAN checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference/',
                        help='Output directory for generated CTPA')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--save_slices', action='store_true',
                        help='Save PNG slices for visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(args.xray_path):
        raise FileNotFoundError(f"X-ray not found: {args.xray_path}")
    if not os.path.exists(args.ddpm_ckpt):
        raise FileNotFoundError(f"DDPM checkpoint not found: {args.ddpm_ckpt}")
    if not os.path.exists(args.vqgan_ckpt):
        raise FileNotFoundError(f"VQ-GAN checkpoint not found: {args.vqgan_ckpt}")
    
    print("=" * 80)
    print("DDPM Inference - X-ray to CTPA Generation")
    print("=" * 80)
    print(f"X-ray: {args.xray_path}")
    print(f"DDPM checkpoint: {args.ddpm_ckpt}")
    print(f"VQ-GAN checkpoint: {args.vqgan_ckpt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load X-ray
    print("\n1. Loading X-ray...")
    xray = load_xray(args.xray_path, device=args.device)
    print(f"   X-ray shape: {xray.shape}")
    
    # Load models
    print("\n2. Loading models...")
    diffusion, vqgan = load_models(args.ddpm_ckpt, args.vqgan_ckpt, device=args.device)
    
    # Generate CTPA
    print("\n3. Generating CTPA...")
    with torch.no_grad():
        ctpa_volumes = generate_ctpa(xray, diffusion, vqgan, 
                                     num_samples=args.num_samples, 
                                     device=args.device)
    
    # Save results
    print("\n4. Saving results...")
    xray_name = Path(args.xray_path).stem
    
    for i in range(args.num_samples):
        # Save full volume as .nii.gz
        output_path = os.path.join(args.output_dir, f"{xray_name}_generated_{i:02d}.nii.gz")
        save_ctpa_volume(ctpa_volumes[i], output_path)
        
        # Optionally save slices as PNG
        if args.save_slices:
            slice_dir = os.path.join(args.output_dir, f"{xray_name}_slices_{i:02d}")
            save_slices_as_png(ctpa_volumes[i], slice_dir, num_slices=10)
    
    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
