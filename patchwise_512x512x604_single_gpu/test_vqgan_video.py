"""
Test VQ-GAN on unseen data and create video visualization of CT volume slices.
Compares original vs reconstructed volumes across all slices.
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vq_gan_3d.model.vqgan_patches import VQGAN_Patches

# Import patch utilities directly
import importlib.util
spec = importlib.util.spec_from_file_location("patch_utils", os.path.join(current_dir, "utils", "patch_utils.py"))
patch_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_utils)
extract_patches_3d = patch_utils.extract_patches_3d
reconstruct_from_patches = patch_utils.reconstruct_from_patches


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load VQ-GAN model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get hyperparameters
    hparams = ckpt['hyper_parameters']
    
    # Handle different checkpoint structures
    if 'cfg' in hparams:
        # Use the cfg object directly (Hydra config)
        cfg = hparams['cfg']
        print(f"Model config: embedding_dim={cfg.model.embedding_dim}, n_codes={cfg.model.n_codes}")
    else:
        cfg = hparams
        print(f"Model config keys: {list(cfg.keys())[:10]}")
    
    # Initialize model - pass cfg instead of hparams
    model = VQGAN_Patches(cfg)
    
    # Load state dict
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = ckpt.get('epoch', 'unknown')
    global_step = ckpt.get('global_step', 'unknown')
    print(f"Model loaded from epoch {epoch}, global step {global_step}")
    
    return model, cfg


def load_volume(nifti_path):
    """Load NIfTI volume and convert to tensor with correct dimension order."""
    print(f"Loading volume: {nifti_path}")
    
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()  # [H, W, D]
    
    # Convert to tensor and fix dimension order for Conv3d
    volume = torch.from_numpy(volume).float()
    volume = volume.permute(2, 0, 1)  # [D, H, W]
    volume = volume.unsqueeze(0)  # [1, D, H, W]
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    return volume


def normalize_volume(volume):
    """Normalize volume to [-1, 1] range."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)  # [0, 1]
        volume = volume * 2 - 1  # [-1, 1]
    return volume


def reconstruct_volume_patches(model, volume, patch_size=(128, 128, 128), stride=(128, 128, 128), device='cuda'):
    """Reconstruct full volume using patch-based processing."""
    print(f"\nExtracting patches with size={patch_size}, stride={stride}")
    
    # Extract patches
    patches, positions = extract_patches_3d(volume, patch_size, stride)
    print(f"Extracted {len(patches)} patches")
    
    # Reconstruct each patch
    reconstructed_patches = []
    
    with torch.no_grad():
        for i, patch in enumerate(tqdm(patches, desc="Reconstructing patches")):
            patch = patch.unsqueeze(0).to(device)  # [1, 1, D, H, W]
            
            # Forward pass through VQ-GAN
            recon_patch, vq_output = model(patch)
            
            reconstructed_patches.append(recon_patch.cpu())
    
    # Reconstruct full volume from patches
    print("Reconstructing full volume from patches...")
    recon_volume = reconstruct_from_patches(
        reconstructed_patches, 
        positions, 
        volume.shape, 
        patch_size, 
        stride
    )
    
    return recon_volume


def create_comparison_video(original, reconstructed, output_path, fps=10, axis='axial'):
    """
    Create side-by-side video comparing original and reconstructed volumes.
    
    Args:
        original: [1, D, H, W] tensor
        reconstructed: [1, D, H, W] tensor
        output_path: Path to save video
        fps: Frames per second
        axis: 'axial' (D axis), 'coronal' (H axis), or 'sagittal' (W axis)
    """
    print(f"\nCreating {axis} video: {output_path}")
    
    # Remove batch dimension
    original = original.squeeze(0).cpu().numpy()  # [D, H, W]
    reconstructed = reconstructed.squeeze(0).cpu().numpy()  # [D, H, W]
    
    # Normalize to [0, 255] for video
    def normalize_for_display(arr):
        arr = np.clip(arr, -1, 1)
        arr = (arr + 1) / 2  # [0, 1]
        arr = (arr * 255).astype(np.uint8)
        return arr
    
    original_norm = normalize_for_display(original)
    reconstructed_norm = normalize_for_display(reconstructed)
    
    # Select axis and get dimensions
    if axis == 'axial':
        num_slices = original_norm.shape[0]
        get_slice = lambda vol, i: vol[i, :, :]
        axis_name = "Axial (Depth)"
    elif axis == 'coronal':
        num_slices = original_norm.shape[1]
        get_slice = lambda vol, i: vol[:, i, :]
        axis_name = "Coronal (Height)"
    elif axis == 'sagittal':
        num_slices = original_norm.shape[2]
        get_slice = lambda vol, i: vol[:, :, i]
        axis_name = "Sagittal (Width)"
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    # Get first slice to determine video dimensions
    first_orig = get_slice(original_norm, 0)
    h, w = first_orig.shape
    
    # Create video writer (side-by-side width, add space for text)
    video_width = w * 2 + 20  # 20 pixels between images
    video_height = h + 60  # Extra space for text at top
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
    
    # Calculate metrics
    mse_list = []
    ssim_list = []
    
    # Generate frames
    for i in tqdm(range(num_slices), desc=f"Generating {axis} frames"):
        orig_slice = get_slice(original_norm, i)
        recon_slice = get_slice(reconstructed_norm, i)
        
        # Calculate MSE for this slice
        mse = np.mean((orig_slice.astype(float) - recon_slice.astype(float)) ** 2)
        mse_list.append(mse)
        
        # Calculate SSIM (simplified version)
        def ssim_simple(img1, img2):
            img1 = img1.astype(float)
            img2 = img2.astype(float)
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            return ssim
        
        ssim = ssim_simple(orig_slice, recon_slice)
        ssim_list.append(ssim)
        
        # Create frame with white background
        frame = np.ones((video_height, video_width, 3), dtype=np.uint8) * 255
        
        # Convert grayscale slices to RGB
        orig_rgb = cv2.cvtColor(orig_slice, cv2.COLOR_GRAY2RGB)
        recon_rgb = cv2.cvtColor(recon_slice, cv2.COLOR_GRAY2RGB)
        
        # Place images side by side (with 20px gap)
        frame[60:60+h, 0:w] = orig_rgb
        frame[60:60+h, w+20:w+20+w] = recon_rgb
        
        # Add text annotations
        text_y = 25
        cv2.putText(frame, f"{axis_name} Slice: {i+1}/{num_slices}", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"MSE: {mse:.2f}  SSIM: {ssim:.4f}", 
                   (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(frame, "Original", (w//2-40, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, "Reconstructed", (w + 20 + w//2-60, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        writer.write(frame)
    
    writer.release()
    
    # Print summary statistics
    print(f"\n{axis_name} Video Statistics:")
    print(f"  Average MSE: {np.mean(mse_list):.2f} ± {np.std(mse_list):.2f}")
    print(f"  Average SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"  Video saved to: {output_path}")


def main():
    """Main testing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VQ-GAN and create visualization videos")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input NIfTI volume (.nii.gz)')
    parser.add_argument('--output_dir', type=str, default='./test_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                       help='Patch size (D H W)')
    parser.add_argument('--stride', type=int, nargs=3, default=[128, 128, 128],
                       help='Stride for patch extraction (D H W)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output videos')
    parser.add_argument('--axes', type=str, nargs='+', default=['axial'],
                       choices=['axial', 'coronal', 'sagittal'],
                       help='Which axes to create videos for')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, hparams = load_checkpoint(args.checkpoint, device=args.device)
    
    # Load input volume
    volume = load_volume(args.input)
    
    # Normalize volume
    volume_normalized = normalize_volume(volume)
    
    # Reconstruct volume
    reconstructed = reconstruct_volume_patches(
        model, 
        volume_normalized,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=args.device
    )
    
    # Calculate overall metrics
    original_flat = volume_normalized.flatten()
    reconstructed_flat = reconstructed.flatten()
    
    mse = torch.mean((original_flat - reconstructed_flat) ** 2).item()
    mae = torch.mean(torch.abs(original_flat - reconstructed_flat)).item()
    
    print(f"\n{'='*60}")
    print(f"Overall Reconstruction Metrics:")
    print(f"{'='*60}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"PSNR: {-10 * np.log10(mse):.2f} dB")
    
    # Save reconstructed volume as NIfTI
    volume_name = Path(args.input).stem.replace('.nii', '')
    output_nifti = os.path.join(args.output_dir, f"{volume_name}_reconstructed.nii.gz")
    
    # Convert back to numpy and original dimension order
    recon_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, D]
    recon_nii = nib.Nifti1Image(recon_np, affine=np.eye(4))
    nib.save(recon_nii, output_nifti)
    print(f"\nReconstructed volume saved to: {output_nifti}")
    
    # Create videos for each requested axis
    for axis in args.axes:
        output_video = os.path.join(args.output_dir, f"{volume_name}_{axis}_comparison.mp4")
        create_comparison_video(
            volume_normalized,
            reconstructed,
            output_video,
            fps=args.fps,
            axis=axis
        )
    
    print(f"\n{'='*60}")
    print(f"Testing complete! All outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
