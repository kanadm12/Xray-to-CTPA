"""
Full Volume DDPM Inference - Generate complete 512×512×604 CTPA from X-ray

Generates multiple overlapping patches and stitches them into a full volume.
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

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
def generate_single_patch(xray, diffusion, vqgan, xray_cond=None, device='cuda'):
    """Generate one 256×256×128 CTPA patch from X-ray."""
    # Get X-ray embedding (reuse if provided)
    if xray_cond is None:
        if diffusion.medclip:
            xray_cond = diffusion.xray_encoder.encode_image(xray, normalize=True)
        else:
            xray_cond = diffusion.xray_encoder(xray)[0]
    
    batch_size = 1
    image_size = diffusion.image_size
    channels = diffusion.channels
    num_frames = diffusion.num_frames
    
    # Run denoising loop
    latent = diffusion.p_sample_loop(
        (batch_size, channels, num_frames, image_size, image_size),
        cond=xray_cond,
        cond_scale=1.0
    )
    
    # Denormalize latent
    emb_min = vqgan.codebook.embeddings.min()
    emb_max = vqgan.codebook.embeddings.max()
    emb_range = emb_max - emb_min
    
    if emb_range > 1e-6:
        latent = ((latent + 1.0) / 2.0) * emb_range + emb_min
    
    # Fix dimension order: [B, C, D, H, W] -> [B, C, H, W, D]
    if latent.shape[2] < latent.shape[3]:
        latent = latent.permute(0, 1, 3, 4, 2)
    
    # Decode directly
    h = vqgan.post_vq_conv(latent)
    patch = vqgan.decoder(h)
    
    return patch


def stitch_patches_with_overlap(patches, patch_positions, output_shape, patch_size, stride):
    """
    Stitch overlapping patches into full volume with weighted blending.
    
    Args:
        patches: List of patch tensors [1, 1, D, H, W]
        patch_positions: List of (d_start, h_start, w_start) tuples
        output_shape: (D, H, W) of final volume
        patch_size: (pd, ph, pw) size of each patch
        stride: (sd, sh, sw) stride between patches
    
    Returns:
        Full volume [D, H, W]
    """
    D, H, W = output_shape
    pd, ph, pw = patch_size
    
    # Initialize output volume and weight map
    volume = np.zeros((D, H, W), dtype=np.float32)
    weights = np.zeros((D, H, W), dtype=np.float32)
    
    # Create weight map for each patch (higher weight in center, lower at edges)
    patch_weight = np.ones((pd, ph, pw), dtype=np.float32)
    
    # Apply Gaussian weighting to reduce edge artifacts
    for i in range(3):
        size = [pd, ph, pw][i]
        sigma = size / 6.0
        axis_weights = np.exp(-0.5 * ((np.arange(size) - size/2) / sigma) ** 2)
        
        if i == 0:  # depth
            patch_weight *= axis_weights[:, None, None]
        elif i == 1:  # height
            patch_weight *= axis_weights[None, :, None]
        else:  # width
            patch_weight *= axis_weights[None, None, :]
    
    # Stitch patches
    for idx, (patch, (d_start, h_start, w_start)) in enumerate(zip(patches, patch_positions)):
        # Convert patch to numpy
        if torch.is_tensor(patch):
            patch = patch.squeeze(0).squeeze(0).cpu().numpy()
        
        # Debug: print actual patch shape
        if idx == 0:
            print(f"First patch shape: {patch.shape}")
            print(f"Expected patch_size (pd, ph, pw): {(pd, ph, pw)}")
            print(f"First position (d, h, w): {(d_start, h_start, w_start)}")
        
        # Calculate valid region
        d_end = min(d_start + pd, D)
        h_end = min(h_start + ph, H)
        w_end = min(w_start + pw, W)
        
        # Debug problematic patches
        if w_end - w_start != pw or h_end - h_start != ph or d_end - d_start != pd:
            print(f"Patch {idx}: position ({d_start}, {h_start}, {w_start}), "
                  f"valid region: {d_end-d_start}×{h_end-h_start}×{w_end-w_start}, "
                  f"expected: {pd}×{ph}×{pw}")
        
        # Extract valid portion of patch and weights
        valid_d = d_end - d_start
        valid_h = h_end - h_start
        valid_w = w_end - w_start
        
        patch_valid = patch[:valid_d, :valid_h, :valid_w]
        weight_valid = patch_weight[:valid_d, :valid_h, :valid_w]
        
        # Add to volume with weights
        volume[d_start:d_end, h_start:h_end, w_start:w_end] += patch_valid * weight_valid
        weights[d_start:d_end, h_start:h_end, w_start:w_end] += weight_valid
    
    # Normalize by weights
    weights[weights == 0] = 1  # Avoid division by zero
    volume /= weights
    
    return volume


def generate_full_volume(xray, diffusion, vqgan, 
                        output_shape=(512, 512, 256),
                        patch_size=(256, 256, 128),
                        stride=(192, 192, 96),
                        device='cuda'):
    """
    Generate full CTPA volume by generating and stitching multiple patches.
    
    IMPORTANT: Generated patches are [1, 1, D, H, W] = [1, 1, 256, 256, 128]
    
    Args:
        xray: X-ray image tensor
        diffusion: DDPM model
        vqgan: VQ-GAN model
        output_shape: (H, W, D) of final volume in user coordinates
        patch_size: (ph, pw, pd) in user coordinates - will be converted
        stride: (sh, sw, sd) in user coordinates - will be converted
        device: cuda or cpu
    
    Returns:
        Full volume [1, 1, D, H, W]
    """
    H_out, W_out, D_out = output_shape  # User-specified output (H, W, D)
    ph_user, pw_user, pd_user = patch_size  # User-specified patch (H, W, D)
    sh_user, sw_user, sd_user = stride  # User-specified stride (H, W, D)
    
    # Generated patches are ALWAYS [1, 1, D, H, W] = [1, 1, 256, 256, 128]
    # So actual patch shape in (D, H, W) order is:
    pd_actual = 256  # depth
    ph_actual = 256  # height
    pw_actual = 128  # width
    
    # Pre-compute X-ray conditioning (same for all patches)
    print("Encoding X-ray...")
    if diffusion.medclip:
        xray_cond = diffusion.xray_encoder.encode_image(xray, normalize=True)
    else:
        xray_cond = diffusion.xray_encoder(xray)[0]
    
    # Calculate patch positions in (D, H, W) storage order
    # We need to cover output volume of (H_out, W_out, D_out) stored as (D_out, H_out, W_out)
    patch_positions = []
    d_starts = list(range(0, D_out - pd_actual + 1, sd_user))
    if D_out > pd_actual and d_starts[-1] < D_out - pd_actual:
        d_starts.append(D_out - pd_actual)
    
    h_starts = list(range(0, H_out - ph_actual + 1, sh_user))
    if H_out > ph_actual and h_starts[-1] < H_out - ph_actual:
        h_starts.append(H_out - ph_actual)
    
    w_starts = list(range(0, W_out - pw_actual + 1, sw_user))
    if W_out > pw_actual and w_starts[-1] < W_out - pw_actual:
        w_starts.append(W_out - pw_actual)
    
    for d in d_starts:
        for h in h_starts:
            for w in w_starts:
                patch_positions.append((d, h, w))
    
    num_patches = len(patch_positions)
    print(f"Generating {num_patches} patches to cover {H_out}×{W_out}×{D_out} volume...")
    print(f"Each patch: {pd_actual}×{ph_actual}×{pw_actual} (D×H×W)")
    
    # Generate all patches
    patches = []
    for i, (d, h, w) in enumerate(tqdm(patch_positions, desc="Generating patches")):
        patch = generate_single_patch(xray, diffusion, vqgan, xray_cond, device)
        patches.append(patch)
        
        # Free memory periodically
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    print("Stitching patches into full volume...")
    # Patches are [1, 1, 256, 256, 128] in (D, H, W) order
    # Output volume shape in storage order (D, H, W)
    volume = stitch_patches_with_overlap(
        patches, 
        patch_positions,
        output_shape=(D_out, H_out, W_out),  # Storage order (D, H, W)
        patch_size=(pd_actual, ph_actual, pw_actual),  # (256, 256, 128)
        stride=(sd_user, sh_user, sw_user)  # (D, H, W) order
    )
    
    # Convert back to torch tensor [1, 1, D, H, W]
    volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
    
    return volume


def save_as_nifti(volume, output_path):
    """Save as .nii.gz file."""
    if volume.dim() == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.dim() == 4:
        volume = volume.squeeze(0)
    
    vol_np = volume.cpu().numpy()
    vol_np = vol_np.clip(0, 1)
    vol_np = (vol_np * 255).astype(np.uint8)
    
    vol_sitk = sitk.GetImageFromArray(vol_np)
    sitk.WriteImage(vol_sitk, output_path)
    print(f"Saved NIFTI: {output_path}")


def save_summary_slices(volume, output_dir, num_slices=20):
    """Save representative slices as PNG."""
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
    parser = argparse.ArgumentParser(description='Generate full CTPA volume from X-ray')
    parser.add_argument('--xray_path', type=str, required=True)
    parser.add_argument('--ddpm_ckpt', type=str, required=True)
    parser.add_argument('--vqgan_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/inference_full/')
    parser.add_argument('--output_shape', type=int, nargs=3, default=[512, 512, 256],
                        help='Output volume shape (H W D)')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[256, 256, 128],
                        help='Patch size (H W D)')
    parser.add_argument('--stride', type=int, nargs=3, default=[192, 192, 96],
                        help='Stride between patches (H W D)')
    parser.add_argument('--save_slices', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Full Volume DDPM Inference - X-ray to Complete CTPA")
    print("=" * 80)
    print(f"X-ray: {args.xray_path}")
    print(f"Output shape: {args.output_shape[0]}×{args.output_shape[1]}×{args.output_shape[2]}")
    print(f"Patch size: {args.patch_size[0]}×{args.patch_size[1]}×{args.patch_size[2]}")
    print(f"Stride: {args.stride[0]}×{args.stride[1]}×{args.stride[2]}")
    print("=" * 80)
    
    # Load
    xray = load_xray(args.xray_path, device=args.device)
    diffusion, vqgan = load_models(args.ddpm_ckpt, args.vqgan_ckpt, device=args.device)
    
    # Generate full volume
    volume = generate_full_volume(
        xray, diffusion, vqgan,
        output_shape=tuple(args.output_shape),
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=args.device
    )
    
    # Save
    xray_name = Path(args.xray_path).stem
    
    nifti_path = os.path.join(args.output_dir, f"{xray_name}_full_volume.nii.gz")
    save_as_nifti(volume, nifti_path)
    
    if args.save_slices:
        slice_dir = os.path.join(args.output_dir, f"{xray_name}_volume_slices")
        save_summary_slices(volume, slice_dir, num_slices=20)
    
    print("\n" + "=" * 80)
    print(f"SUCCESS! Generated full {args.output_shape[0]}×{args.output_shape[1]}×{args.output_shape[2]} CTPA volume")
    print("=" * 80)


if __name__ == "__main__":
    main()
