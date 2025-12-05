"""
DDPM Inference Script - Generate Full CTPA Volume from PA X-ray

Takes a single PA chest X-ray and generates a complete 512x512x604 CTPA volume.
Reconstructs from 128³ patches and saves 3 orthogonal views (axial, coronal, sagittal).
"""

import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from PIL import Image
import argparse
from pathlib import Path
import yaml

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
baseline_dir = os.path.join(parent_dir, 'baseline_256x256x64_single_gpu')

sys.path.insert(0, baseline_dir)
sys.path.insert(0, current_dir)

from ddpm.diffusion import GaussianDiffusion
from dataset.xray_ctpa_patch_dataset import extract_patches_3d


def load_xray(xray_path, flip=True):
    """Load and preprocess X-ray image for inference."""
    img = Image.open(xray_path).convert('L')
    
    # Fix upside-down X-rays
    if flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Resize to 224x224 for MedCLIP
    img = img.resize((224, 224), Image.BILINEAR)
    
    xray = np.array(img, dtype=np.float32)
    xray = xray / 255.0  # [0, 1]
    
    # Convert to 3-channel tensor
    xray = torch.from_numpy(xray).unsqueeze(0).repeat(3, 1, 1).float()
    xray = xray.unsqueeze(0)  # [1, 3, 224, 224]
    
    return xray


def generate_patches(diffusion_model, xray_condition, num_patches=80, device='cuda'):
    """
    Generate CTPA patches conditioned on X-ray.
    
    Args:
        diffusion_model: Loaded GaussianDiffusion model
        xray_condition: X-ray tensor [1, 3, 224, 224]
        num_patches: Number of patches to generate (default 80 for 512x512x604)
        device: Device to run inference on
    
    Returns:
        patches: List of generated patches [N, 1, 128, 128, 128]
    """
    print(f"Generating {num_patches} patches...")
    
    patches = []
    xray_condition = xray_condition.to(device)
    
    with torch.no_grad():
        for i in range(num_patches):
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_patches} patches...")
            
            # Sample from diffusion model
            # Shape: [1, 64, 32, 32, 32] in latent space
            sample = diffusion_model.sample(
                batch_size=1,
                img_cond=xray_condition,
                return_all_timesteps=False
            )
            
            patches.append(sample.cpu())
    
    print(f"✓ Generated all {num_patches} patches")
    return torch.cat(patches, dim=0)  # [N, 1, 128, 128, 128]


def reconstruct_volume(patches, target_shape=(604, 512, 512), patch_size=(128, 128, 128), stride=(128, 128, 128)):
    """
    Reconstruct full volume from overlapping patches.
    
    Args:
        patches: Tensor [N, 1, D, H, W] of patches
        target_shape: Final volume shape (D, H, W)
        patch_size: Size of each patch
        stride: Stride used during extraction
    
    Returns:
        volume: Reconstructed volume [1, D, H, W]
    """
    print("Reconstructing volume from patches...")
    
    D, H, W = target_shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Initialize volume and count map for averaging overlaps
    volume = torch.zeros((1, D, H, W), dtype=torch.float32)
    count = torch.zeros((1, D, H, W), dtype=torch.float32)
    
    patch_idx = 0
    
    # Reconstruct with same pattern as extraction
    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                if patch_idx < patches.shape[0]:
                    volume[:, d:d+pd, h:h+ph, w:w+pw] += patches[patch_idx]
                    count[:, d:d+pd, h:h+ph, w:w+pw] += 1
                    patch_idx += 1
    
    # Average overlapping regions
    volume = volume / (count + 1e-8)
    
    # Handle any remaining boundary patches
    if patch_idx < patches.shape[0]:
        print(f"  Note: Used {patch_idx}/{patches.shape[0]} patches")
    
    print(f"✓ Reconstructed volume: {volume.shape}")
    return volume


def save_orthogonal_views(volume, output_dir, prefix="generated"):
    """
    Save 3 orthogonal views (axial, coronal, sagittal) as images.
    
    Args:
        volume: Volume tensor [1, D, H, W]
        output_dir: Directory to save views
        prefix: Prefix for output filenames
    """
    print("Saving orthogonal views...")
    
    volume_np = volume[0].numpy()  # [D, H, W]
    D, H, W = volume_np.shape
    
    # Normalize to [0, 255]
    vmin, vmax = volume_np.min(), volume_np.max()
    volume_norm = ((volume_np - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    
    # 1. Axial view (transverse, XY plane) - middle slice
    axial_slice = volume_norm[D // 2, :, :]
    axial_img = Image.fromarray(axial_slice)
    axial_path = os.path.join(output_dir, f"{prefix}_axial.png")
    axial_img.save(axial_path)
    print(f"  ✓ Axial view: {axial_path}")
    
    # 2. Coronal view (frontal, XZ plane) - middle slice
    coronal_slice = volume_norm[:, H // 2, :]
    coronal_img = Image.fromarray(coronal_slice)
    coronal_path = os.path.join(output_dir, f"{prefix}_coronal.png")
    coronal_img.save(coronal_path)
    print(f"  ✓ Coronal view: {coronal_path}")
    
    # 3. Sagittal view (lateral, YZ plane) - middle slice
    sagittal_slice = volume_norm[:, :, W // 2]
    sagittal_img = Image.fromarray(sagittal_slice)
    sagittal_path = os.path.join(output_dir, f"{prefix}_sagittal.png")
    sagittal_img.save(sagittal_path)
    print(f"  ✓ Sagittal view: {sagittal_path}")


def save_volume_nifti(volume, output_path, spacing=(1.0, 1.0, 1.0)):
    """
    Save volume as NIfTI file.
    
    Args:
        volume: Volume tensor [1, D, H, W]
        output_path: Path to save .nii.gz file
        spacing: Voxel spacing (D, H, W)
    """
    print(f"Saving NIfTI volume: {output_path}")
    
    volume_np = volume[0].numpy()  # [D, H, W]
    
    # Create SimpleITK image
    sitk_img = sitk.GetImageFromArray(volume_np)
    sitk_img.SetSpacing(spacing)
    
    # Save as NIfTI
    sitk.WriteImage(sitk_img, output_path)
    print(f"✓ Saved NIfTI: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate CTPA volume from PA X-ray')
    parser.add_argument('--xray', type=str, required=True, help='Path to input PA X-ray (.png)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/ddpm_4gpu/last.ckpt',
                       help='Path to DDPM checkpoint')
    parser.add_argument('--config', type=str, default='./config/model/ddpm_4gpu.yaml',
                       help='Path to model config')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--num_patches', type=int, default=80,
                       help='Number of patches to generate (default: 80)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference (cuda/cpu)')
    parser.add_argument('--flip_xray', action='store_true', default=True,
                       help='Flip X-ray vertically (fix upside-down)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DDPM Inference - X-ray → CTPA Generation")
    print("=" * 80)
    print(f"Input X-ray: {args.xray}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Load config
    print("[1/6] Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")
    
    # Load X-ray
    print("\n[2/6] Loading X-ray image...")
    xray = load_xray(args.xray, flip=args.flip_xray)
    print(f"✓ X-ray loaded: {xray.shape}")
    
    # Load diffusion model
    print("\n[3/6] Loading diffusion model from checkpoint...")
    
    # First, create the UNet model
    from ddpm.diffusion import Unet3D
    
    # Config is flat, not nested
    cfg = config
    model = Unet3D(
        dim=cfg['diffusion_img_size'],
        cond_dim=cfg['cond_dim'],
        dim_mults=cfg['dim_mults'],
        channels=cfg['diffusion_num_channels'],
        resnet_groups=8,
        classifier_free_guidance=cfg.get('classifier_free_guidance', False),
        medclip=cfg.get('medclip', True)
    )
    
    # Then create GaussianDiffusion with the model
    diffusion_model = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg['vqgan_ckpt'],
        vae_ckpt=cfg.get('vae_ckpt', None),
        image_size=cfg['diffusion_img_size'],
        num_frames=cfg['diffusion_depth_size'],
        channels=cfg['diffusion_num_channels'],
        timesteps=cfg['timesteps'],
        img_cond=True,
        loss_type=cfg.get('loss_type', 'l1'),
        l1_weight=cfg.get('l1_weight', 1.0),
        perceptual_weight=cfg.get('perceptual_weight', 0.0),
        discriminator_weight=cfg.get('discriminator_weight', 0.0),
        classification_weight=cfg.get('classification_weight', 0.0),
        classifier_free_guidance=cfg.get('classifier_free_guidance', False),
        medclip=cfg.get('medclip', True),
        name_dataset=cfg.get('name_dataset', 'CTPA'),
        dataset_min_value=cfg.get('dataset_min_value', -12.911299),
        dataset_max_value=cfg.get('dataset_max_value', 9.596558),
    )
    
    # Load checkpoint weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    diffusion_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    diffusion_model.eval()
    diffusion_model = diffusion_model.to(args.device)
    print("✓ Model loaded and ready")
    
    # Generate patches
    print(f"\n[4/6] Generating CTPA patches...")
    patches = generate_patches(
        diffusion_model, 
        xray, 
        num_patches=args.num_patches,
        device=args.device
    )
    print(f"✓ Patches shape: {patches.shape}")
    
    # Reconstruct full volume
    print(f"\n[5/6] Reconstructing full volume...")
    volume = reconstruct_volume(
        patches,
        target_shape=(604, 512, 512),
        patch_size=(128, 128, 128),
        stride=(128, 128, 128)
    )
    
    # Save outputs
    print(f"\n[6/6] Saving outputs...")
    
    # Get base name for output files
    xray_name = Path(args.xray).stem
    
    # Save orthogonal views
    save_orthogonal_views(volume, args.output_dir, prefix=xray_name)
    
    # Save full volume as NIfTI
    nifti_path = os.path.join(args.output_dir, f"{xray_name}_volume.nii.gz")
    save_volume_nifti(volume, nifti_path)
    
    print("\n" + "=" * 80)
    print("✓ INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - {xray_name}_axial.png (XY plane)")
    print(f"  - {xray_name}_coronal.png (XZ plane)")
    print(f"  - {xray_name}_sagittal.png (YZ plane)")
    print(f"  - {xray_name}_volume.nii.gz (full volume)")
    print()


if __name__ == "__main__":
    main()
