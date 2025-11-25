"""
Patch extraction and reconstruction utilities for full-resolution CTPA volumes.

Handles:
- Extracting overlapping 3D patches from 512×512×604 volumes
- Reconstructing full volumes from patches with overlap blending
- Coordinate management for distributed processing
"""

import torch
import numpy as np
from typing import Tuple, List


def extract_patches_3d(
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (256, 256, 256),
    stride: Tuple[int, int, int] = (256, 256, 256),
    padding: str = 'constant'
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Extract overlapping 3D patches from a volume.
    
    Args:
        volume: Input volume [C, D, H, W] (Channel, Depth, Height, Width)
        patch_size: Size of each patch (D, H, W) matching volume dims
        stride: Stride between patches (D, H, W) - allows overlap
        padding: Padding mode for edges
        
    Returns:
        patches: Tensor of shape [N, C, D_patch, H_patch, W_patch]
        coordinates: List of (d, h, w) start coordinates for each patch
    """
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size  # patch depth, height, width
    sd, sh, sw = stride  # stride depth, height, width
    
    # Calculate padding needed to ensure full coverage
    pad_d = (pd - (D - pd) % sd) % sd if D > pd else 0
    pad_h = (ph - (H - ph) % sh) % sh if H > ph else 0
    pad_w = (pw - (W - pw) % sw) % sw if W > pw else 0
    
    # Pad volume if needed
    # F.pad order for 5D: (W_left, W_right, H_left, H_right, D_left, D_right)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = torch.nn.functional.pad(
            volume, 
            (0, pad_w, 0, pad_h, 0, pad_d),  # Pad order: W, H, D (last to first spatial dims)
            mode=padding,
            value=0
        )
    
    C, D, H, W = volume.shape
    patches = []
    coordinates = []
    
    # Extract patches with stride
    d_positions = list(range(0, D - pd + 1, sd))
    h_positions = list(range(0, H - ph + 1, sh))
    w_positions = list(range(0, W - pw + 1, sw))
    
    # Handle edge cases where volume is smaller than patch
    if not d_positions:
        d_positions = [0]
    if not h_positions:
        h_positions = [0]
    if not w_positions:
        w_positions = [0]
    
    # Ensure we capture the last patches if needed
    if d_positions[-1] + pd < D:
        d_positions.append(D - pd)
    if h_positions[-1] + ph < H:
        h_positions.append(H - ph)
    if w_positions[-1] + pw < W:
        w_positions.append(W - pw)
    
    for d in d_positions:
        for h in h_positions:
            for w in w_positions:
                patch = volume[:, d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)
                coordinates.append((d, h, w))
    
    patches = torch.stack(patches, dim=0)
    return patches, coordinates


def reconstruct_from_patches(
    patches: torch.Tensor,
    coordinates: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int, int],
    patch_size: Tuple[int, int, int] = (256, 256, 128),
    blend_mode: str = 'linear'
) -> torch.Tensor:
    """
    Reconstruct full volume from overlapping patches using weighted blending.
    
    Args:
        patches: Tensor of patches [N, C, D_patch, H_patch, W_patch]
        coordinates: List of (d, h, w) start coordinates for each patch
        output_shape: Target shape [C, D, H, W]
        patch_size: Size of each patch
        blend_mode: Blending method ('linear', 'gaussian', 'avg')
        
    Returns:
        volume: Reconstructed volume [C, D, H, W]
    """
    C, D, H, W = output_shape
    pd, ph, pw = patch_size
    
    # Initialize output volume and weight accumulator
    volume = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)
    weights = torch.zeros(output_shape, dtype=patches.dtype, device=patches.device)
    
    # Create blending weight for patches
    if blend_mode == 'linear':
        # Linear ramp from edges to center
        weight_d = torch.linspace(0, 1, pd // 2).repeat(2)[:pd]
        weight_h = torch.linspace(0, 1, ph // 2).repeat(2)[:ph]
        weight_w = torch.linspace(0, 1, pw // 2).repeat(2)[:pw]
        
        # 3D weight grid
        weight_grid = (
            weight_d.view(-1, 1, 1) *
            weight_h.view(1, -1, 1) *
            weight_w.view(1, 1, -1)
        )
        weight_grid = weight_grid.unsqueeze(0)  # [1, D, H, W]
        
    elif blend_mode == 'gaussian':
        # Gaussian falloff from center
        d_coords = torch.linspace(-1, 1, pd)
        h_coords = torch.linspace(-1, 1, ph)
        w_coords = torch.linspace(-1, 1, pw)
        
        weight_d = torch.exp(-d_coords**2 / 0.5)
        weight_h = torch.exp(-h_coords**2 / 0.5)
        weight_w = torch.exp(-w_coords**2 / 0.5)
        
        weight_grid = (
            weight_d.view(-1, 1, 1) *
            weight_h.view(1, -1, 1) *
            weight_w.view(1, 1, -1)
        )
        weight_grid = weight_grid.unsqueeze(0)
        
    else:  # 'avg'
        weight_grid = torch.ones(1, pd, ph, pw)
    
    weight_grid = weight_grid.to(patches.device)
    
    # Add weighted patches to output
    for patch, (d, h, w) in zip(patches, coordinates):
        d_end = min(d + pd, D)
        h_end = min(h + ph, H)
        w_end = min(w + pw, W)
        
        pd_actual = d_end - d
        ph_actual = h_end - h
        pw_actual = w_end - w
        
        patch_slice = patch[:, :pd_actual, :ph_actual, :pw_actual]
        weight_slice = weight_grid[:, :pd_actual, :ph_actual, :pw_actual]
        
        volume[:, d:d_end, h:h_end, w:w_end] += patch_slice * weight_slice
        weights[:, d:d_end, h:h_end, w:w_end] += weight_slice
    
    # Normalize by accumulated weights
    volume = volume / (weights + 1e-8)
    
    return volume


def get_patch_statistics(volume_shape: Tuple[int, int, int],
                         patch_size: Tuple[int, int, int],
                         stride: Tuple[int, int, int]) -> dict:
    """
    Calculate statistics about patch extraction.
    
    Returns:
        dict with keys: num_patches, overlap_percent, memory_per_patch_mb
    """
    D, H, W = volume_shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Calculate number of patches
    nd = (D - pd) // sd + 1 + (1 if (D - pd) % sd > 0 else 0)
    nh = (H - ph) // sh + 1 + (1 if (H - ph) % sh > 0 else 0)
    nw = (W - pw) // sw + 1 + (1 if (W - pw) % sw > 0 else 0)
    
    num_patches = nd * nh * nw
    
    # Calculate overlap percentage
    overlap_d = ((pd - sd) / pd * 100) if sd < pd else 0
    overlap_h = ((ph - sh) / ph * 100) if sh < ph else 0
    overlap_w = ((pw - sw) / pw * 100) if sw < pw else 0
    
    avg_overlap = (overlap_d + overlap_h + overlap_w) / 3
    
    # Memory per patch (assuming float32)
    memory_per_patch_mb = (pd * ph * pw * 4) / (1024 ** 2)
    
    return {
        'num_patches': num_patches,
        'patches_per_dim': (nd, nh, nw),
        'overlap_percent': avg_overlap,
        'memory_per_patch_mb': memory_per_patch_mb,
        'total_memory_mb': memory_per_patch_mb * num_patches
    }


if __name__ == "__main__":
    # Test patch extraction and reconstruction
    print("Testing patch extraction on 512×512×604 volume...")
    
    # Create dummy volume
    volume = torch.randn(1, 512, 512, 604)
    
    # Extract patches
    patches, coords = extract_patches_3d(
        volume,
        patch_size=(256, 256, 128),
        stride=(192, 192, 96)
    )
    
    print(f"Input volume shape: {volume.shape}")
    print(f"Number of patches: {patches.shape[0]}")
    print(f"Patch shape: {patches.shape[1:]}")
    
    # Reconstruct
    reconstructed = reconstruct_from_patches(
        patches,
        coords,
        volume.shape,
        patch_size=(256, 256, 128),
        blend_mode='linear'
    )
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Calculate reconstruction error
    mse = torch.mean((volume - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    
    # Get statistics
    stats = get_patch_statistics(
        volume_shape=(1, 604, 512, 512),  # [C, D, H, W]
        patch_size=(256, 256, 128),
        stride=(192, 192, 96)
    )
    print("\nPatch Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
