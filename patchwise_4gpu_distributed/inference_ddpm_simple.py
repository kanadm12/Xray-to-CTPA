"""
Simple DDPM Inference Script - Generate single patch from X-ray

This generates a 256x256x128 CTPA patch (not full volume) from an X-ray.
For full volume generation, multiple patches would need to be stitched.
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


def load_xray(xray_path, device='cuda'):
    """Load and preprocess X-ray for MedCLIP."""
    img = Image.open(xray_path).convert('L')
    img = img.resize((224, 224), Image.BILINEAR)
    
    xray = np.array(img, dtype=np.float32) / 255.0
    xray = torch.from_numpy(xray).unsqueeze(0).repeat(3, 1, 1).float()
    xray = xray.unsqueeze(0)  # [1, 3, 224, 224]
    
    return xray.to(device)


def load_ddpm_model(ckpt_path, vqgan_ckpt_path, device='cuda'):
    """Load DDPM model from checkpoint."""
    print("Loading DDPM checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create model
    model = Unet3D(
        dim=hparams.get('image_size', 64),
        cond_dim=512,
        dim_mults=[1, 2, 4, 8],
        channels=64,
        resnet_groups=8,
        classifier_free_guidance=False,
        medclip=True
    )
    
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=vqgan_ckpt_path,
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
    
    diffusion.load_state_dict(checkpoint['state_dict'])
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    print("Model loaded!")
    return diffusion


@torch.no_grad()
def generate_patch(xray, diffusion, device='cuda'):
    """Generate CTPA patch from X-ray."""
    print("Generating CTPA patch...")
    
    # Sample - this handles encoding, denoising, and decoding
    patch = diffusion.sample(cond=xray, batch_size=1)
    
    print(f"Generated patch shape: {patch.shape}")
    return patch


def save_patch(patch_tensor, output_path):
    """Save patch as .nii.gz."""
    # Remove batch and channel dims
    if patch_tensor.dim() == 5:
        patch_tensor = patch_tensor.squeeze(0).squeeze(0)
    elif patch_tensor.dim() == 4:
        patch_tensor = patch_tensor.squeeze(0)
    
    # Convert to numpy
    patch_np = patch_tensor.cpu().numpy()
    
    # Clip to valid range
    patch_np = patch_np.clip(0, 1)
    patch_np = (patch_np * 255).astype(np.uint8)
    
    # Save as NIFTI
    patch_sitk = sitk.GetImageFromArray(patch_np)
    sitk.WriteImage(patch_sitk, output_path)
    print(f"Saved to: {output_path}")


def save_slices(patch_tensor, output_dir, num_slices=10):
    """Save representative slices as PNG."""
    if patch_tensor.dim() == 5:
        patch_tensor = patch_tensor.squeeze(0).squeeze(0)
    elif patch_tensor.dim() == 4:
        patch_tensor = patch_tensor.squeeze(0)
    
    patch_np = patch_tensor.cpu().numpy()
    depth = patch_np.shape[0]
    
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, slice_idx in enumerate(slice_indices):
        slice_img = patch_np[slice_idx]
        slice_img = (slice_img * 255).clip(0, 255).astype(np.uint8)
        
        img = Image.fromarray(slice_img)
        img.save(os.path.join(output_dir, f"slice_{i:03d}_depth{slice_idx:03d}.png"))
    
    print(f"Saved {num_slices} slices to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate CTPA patch from X-ray')
    parser.add_argument('--xray_path', type=str, required=True)
    parser.add_argument('--ddpm_ckpt', type=str, required=True)
    parser.add_argument('--vqgan_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/inference_simple/')
    parser.add_argument('--save_slices', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DDPM Simple Inference - Generate CTPA Patch from X-ray")
    print("=" * 80)
    print(f"X-ray: {args.xray_path}")
    print(f"DDPM: {args.ddpm_ckpt}")
    print(f"VQ-GAN: {args.vqgan_ckpt}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load X-ray
    print("\n1. Loading X-ray...")
    xray = load_xray(args.xray_path, device=args.device)
    print(f"   Shape: {xray.shape}")
    
    # Load model
    print("\n2. Loading DDPM...")
    diffusion = load_ddpm_model(args.ddpm_ckpt, args.vqgan_ckpt, device=args.device)
    
    # Generate
    print("\n3. Generating patch...")
    patch = generate_patch(xray, diffusion, device=args.device)
    
    # Save
    print("\n4. Saving results...")
    xray_name = Path(args.xray_path).stem
    
    output_path = os.path.join(args.output_dir, f"{xray_name}_patch.nii.gz")
    save_patch(patch, output_path)
    
    if args.save_slices:
        slice_dir = os.path.join(args.output_dir, f"{xray_name}_slices")
        save_slices(patch, slice_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
