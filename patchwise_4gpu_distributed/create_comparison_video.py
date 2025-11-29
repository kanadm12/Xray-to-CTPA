"""
Create side-by-side video comparison of generated vs ground truth CTPA

Usage:
    python create_comparison_video.py \
        --generated /path/to/generated.nii.gz \
        --ground_truth /path/to/ground_truth.nii.gz \
        --output comparison.mp4
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm


def load_volume(path):
    """Load NIFTI volume."""
    if path.endswith('.npy'):
        volume = np.load(path)
    else:
        sitk_img = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(sitk_img)
    return volume


def normalize_volume(volume):
    """Normalize to [0, 1] range."""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    return volume


def create_comparison_frame(gen_slice, gt_slice, slice_idx, total_slices):
    """Create side-by-side comparison frame with metrics."""
    # Normalize both slices
    gen_slice = normalize_volume(gen_slice)
    gt_slice = normalize_volume(gt_slice)
    
    # Resize if dimensions don't match
    if gen_slice.shape != gt_slice.shape:
        gen_slice = cv2.resize(gen_slice, (gt_slice.shape[1], gt_slice.shape[0]))
    
    # Convert to uint8
    gen_uint8 = (gen_slice * 255).astype(np.uint8)
    gt_uint8 = (gt_slice * 255).astype(np.uint8)
    
    # Apply colormap for better visualization
    gen_color = cv2.applyColorMap(gen_uint8, cv2.COLORMAP_BONE)
    gt_color = cv2.applyColorMap(gt_uint8, cv2.COLORMAP_BONE)
    
    # Create difference map
    diff = np.abs(gen_slice - gt_slice)
    diff_uint8 = (diff * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    
    # Calculate metrics for this slice
    mse = np.mean((gen_slice - gt_slice) ** 2)
    mae = np.mean(np.abs(gen_slice - gt_slice))
    
    # Create side-by-side layout
    h, w = gt_color.shape[:2]
    frame = np.zeros((h + 100, w * 3, 3), dtype=np.uint8)
    
    # Place images
    frame[50:50+h, 0:w] = gt_color
    frame[50:50+h, w:2*w] = gen_color
    frame[50:50+h, 2*w:3*w] = diff_color
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(frame, 'Ground Truth', (10, 30), font, font_scale, color, thickness)
    cv2.putText(frame, 'Generated', (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(frame, 'Difference (Abs)', (2*w + 10, 30), font, font_scale, color, thickness)
    
    # Add slice info and metrics at bottom
    info_y = h + 70
    cv2.putText(frame, f'Slice: {slice_idx + 1}/{total_slices}', (10, info_y), 
                font, 0.6, color, 1)
    cv2.putText(frame, f'MSE: {mse:.4f}  MAE: {mae:.4f}', (w + 10, info_y), 
                font, 0.6, color, 1)
    
    return frame


def create_comparison_video(gen_path, gt_path, output_path, fps=10):
    """Create video comparing generated and ground truth volumes."""
    print("Loading volumes...")
    gen_vol = load_volume(gen_path)
    gt_vol = load_volume(gt_path)
    
    print(f"Generated shape: {gen_vol.shape}")
    print(f"Ground truth shape: {gt_vol.shape}")
    
    # Normalize volumes
    gen_vol = normalize_volume(gen_vol)
    gt_vol = normalize_volume(gt_vol)
    
    # Handle different depths
    min_depth = min(gen_vol.shape[0], gt_vol.shape[0])
    
    # Get frame dimensions
    h, w = gt_vol.shape[1], gt_vol.shape[2]
    if gen_vol.shape[1:] != gt_vol.shape[1:]:
        print(f"Resizing generated volume from {gen_vol.shape[1:]} to {gt_vol.shape[1:]}")
    
    frame_height = h + 100
    frame_width = w * 3
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Creating video with {min_depth} frames at {fps} FPS...")
    
    # Calculate overall metrics
    all_mse = []
    all_mae = []
    
    # Generate frames
    for i in range(min_depth):
        gen_slice = gen_vol[i] if i < gen_vol.shape[0] else np.zeros_like(gt_vol[0])
        gt_slice = gt_vol[i]
        
        frame = create_comparison_frame(gen_slice, gt_slice, i, min_depth)
        out.write(frame)
        
        # Track metrics
        if gen_slice.shape == gt_slice.shape:
            all_mse.append(np.mean((gen_slice - gt_slice) ** 2))
            all_mae.append(np.mean(np.abs(gen_slice - gt_slice)))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{min_depth} frames")
    
    out.release()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Video created successfully!")
    print(f"Output: {output_path}")
    print(f"Duration: {min_depth / fps:.1f} seconds")
    print(f"\nOverall Metrics:")
    print(f"  Average MSE: {np.mean(all_mse):.4f}")
    print(f"  Average MAE: {np.mean(all_mae):.4f}")
    print(f"  Min MSE: {np.min(all_mse):.4f}")
    print(f"  Max MSE: {np.max(all_mse):.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Create comparison video')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to generated CTPA (.nii.gz or .npy)')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth CTPA (.nii.gz or .npy)')
    parser.add_argument('--output', type=str, default='comparison.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.generated):
        raise FileNotFoundError(f"Generated file not found: {args.generated}")
    if not os.path.exists(args.ground_truth):
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    
    # Create video
    create_comparison_video(args.generated, args.ground_truth, args.output, args.fps)


if __name__ == "__main__":
    main()
