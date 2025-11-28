"""
Test VQ-GAN reconstruction on CTPA volumes.

Usage:
    python test_vqgan.py --checkpoint path/to/checkpoint.ckpt
"""

import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'patchwise_512x512x604_single_gpu'))

from vq_gan_3d.model.vqgan_patches import VQGAN_Patches
from dataset.patch_dataset import PatchDataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained VQ-GAN model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters from checkpoint
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Create model
    model = VQGAN_Patches.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  Codebook size: {model.n_codes}")
    print(f"  Downsample: {model.downsample}")
    
    return model


def reconstruct_volume(model, volume_path, patch_size=(128, 128, 128), stride=(128, 128, 128), device='cuda'):
    """
    Reconstruct a full volume using patch-wise processing.
    
    Args:
        model: Trained VQ-GAN model
        volume_path: Path to input .nii.gz file
        patch_size: Size of patches (D, H, W)
        stride: Stride for patch extraction (D, H, W)
        device: Device to use
    
    Returns:
        original_volume: Original volume array
        reconstructed_volume: Reconstructed volume array
        metrics: Dictionary of metrics (PSNR, SSIM, etc.)
    """
    print(f"\nProcessing: {os.path.basename(volume_path)}")
    
    # Load volume
    nii = nib.load(volume_path)
    original_volume = nii.get_fdata()
    
    # Normalize to [-1, 1]
    vol_min, vol_max = original_volume.min(), original_volume.max()
    original_normalized = 2.0 * (original_volume - vol_min) / (vol_max - vol_min + 1e-8) - 1.0
    
    print(f"  Volume shape: {original_volume.shape}")
    print(f"  Value range: [{vol_min:.2f}, {vol_max:.2f}]")
    
    # Convert to tensor (add batch and channel dims)
    volume_tensor = torch.from_numpy(original_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract patches
    D, H, W = volume_tensor.shape[2:]
    patch_d, patch_h, patch_w = patch_size
    stride_d, stride_h, stride_w = stride
    
    patches = []
    positions = []
    
    for d in range(0, D - patch_d + 1, stride_d):
        for h in range(0, H - patch_h + 1, stride_h):
            for w in range(0, W - patch_w + 1, stride_w):
                patch = volume_tensor[:, :, d:d+patch_d, h:h+patch_h, w:w+patch_w]
                patches.append(patch)
                positions.append((d, h, w))
    
    patches = torch.cat(patches, dim=0)
    print(f"  Extracted {len(patches)} patches of size {patch_size}")
    
    # Reconstruct patches in batches
    reconstructed_patches = []
    batch_size = 4  # Process 4 patches at a time
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="  Reconstructing"):
            batch = patches[i:i+batch_size]
            recon_batch, _ = model(batch)
            reconstructed_patches.append(recon_batch.cpu())
    
    reconstructed_patches = torch.cat(reconstructed_patches, dim=0)
    
    # Reassemble volume
    reconstructed_volume = torch.zeros_like(volume_tensor).cpu()
    count_map = torch.zeros_like(volume_tensor).cpu()
    
    for patch, (d, h, w) in zip(reconstructed_patches, positions):
        reconstructed_volume[0, 0, d:d+patch_d, h:h+patch_h, w:w+patch_w] += patch[0]
        count_map[0, 0, d:d+patch_d, h:h+patch_h, w:w+patch_w] += 1
    
    # Average overlapping regions
    reconstructed_volume = reconstructed_volume / (count_map + 1e-8)
    reconstructed_volume = reconstructed_volume[0, 0].numpy()
    
    # Denormalize back to original range
    reconstructed_volume = (reconstructed_volume + 1.0) / 2.0 * (vol_max - vol_min) + vol_min
    
    # Calculate metrics
    mse = np.mean((original_volume - reconstructed_volume) ** 2)
    psnr = 10 * np.log10((vol_max - vol_min) ** 2 / (mse + 1e-8))
    
    # SSIM (simple version)
    def ssim(img1, img2):
        C1 = (0.01 * (vol_max - vol_min)) ** 2
        C2 = (0.03 * (vol_max - vol_min)) ** 2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.var()
        sigma2 = img2.var()
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        return ssim_val
    
    ssim_val = ssim(original_volume, reconstructed_volume)
    
    metrics = {
        'psnr': psnr,
        'ssim': ssim_val,
        'mse': mse,
        'mae': np.mean(np.abs(original_volume - reconstructed_volume))
    }
    
    print(f"  Metrics:")
    print(f"    PSNR: {psnr:.2f} dB")
    print(f"    SSIM: {ssim_val:.4f}")
    print(f"    MSE:  {mse:.4f}")
    print(f"    MAE:  {metrics['mae']:.4f}")
    
    return original_volume, reconstructed_volume, metrics


def save_reconstruction(original, reconstructed, output_path, affine=None):
    """Save reconstructed volume as NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    
    recon_nii = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
    nib.save(recon_nii, output_path)
    print(f"✓ Saved reconstruction to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test VQ-GAN reconstruction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input .nii.gz file (if None, uses validation data)')
    parser.add_argument('--output-dir', type=str, default='./test_outputs',
                        help='Directory to save outputs')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Get test files
    if args.input:
        test_files = [args.input]
    else:
        # Use validation data
        data_dir = "/workspace/Xray-to-CTPA/datasets/"
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.nii.gz') and 'swapped' not in f.lower():
                    all_files.append(os.path.join(root, f))
        
        # Use validation split (last 20%)
        split_idx = int(len(all_files) * 0.8)
        val_files = sorted(all_files)[split_idx:]
        test_files = val_files[:args.num_samples]
    
    print(f"\nTesting on {len(test_files)} volumes")
    print("=" * 80)
    
    # Process each file
    all_metrics = []
    
    for volume_path in test_files:
        try:
            # Load original volume to get affine
            nii = nib.load(volume_path)
            
            # Reconstruct
            original, reconstructed, metrics = reconstruct_volume(
                model, volume_path, device=args.device
            )
            
            all_metrics.append(metrics)
            
            # Save reconstruction
            output_name = os.path.basename(volume_path).replace('.nii.gz', '_recon.nii.gz')
            output_path = os.path.join(args.output_dir, output_name)
            save_reconstruction(original, reconstructed, output_path, nii.affine)
            
        except Exception as e:
            print(f"✗ Error processing {volume_path}: {e}")
            continue
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if all_metrics:
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['ssim'] for m in all_metrics])
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average MSE:  {avg_mse:.4f}")
        print(f"Average MAE:  {avg_mae:.4f}")
        print(f"\nProcessed {len(all_metrics)} volumes successfully")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("No volumes processed successfully")


if __name__ == "__main__":
    main()
