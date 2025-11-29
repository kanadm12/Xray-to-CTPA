"""
DDPM Inference - Generate latent only (no VQ-GAN decoding)

This script generates the latent representation and returns it without decoding.
Useful for debugging dimension mismatches.
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
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
    """Load both models separately."""
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
    
    # Create diffusion WITHOUT vqgan_ckpt to avoid auto-decoding
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=None,  # Don't load - we'll decode manually
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
def generate_latent(xray, diffusion, device='cuda'):
    """Generate latent representation from X-ray."""
    print("Generating latent...")
    
    # Get X-ray embedding
    if diffusion.medclip:
        cond = diffusion.xray_encoder.encode_image(xray, normalize=True)
    else:
        cond = diffusion.xray_encoder(xray)[0]
    
    batch_size = 1
    image_size = diffusion.image_size
    channels = diffusion.channels
    num_frames = diffusion.num_frames
    
    # Run denoising loop
    latent = diffusion.p_sample_loop(
        (batch_size, channels, num_frames, image_size, image_size),
        cond=cond,
        cond_scale=1.0
    )
    
    print(f"Generated latent shape: {latent.shape}")
    return latent


@torch.no_grad()
def decode_latent(latent, vqgan):
    """Decode latent with VQ-GAN."""
    print("Decoding with VQ-GAN...")
    print(f"Input latent shape: {latent.shape}")
    
    # Denormalize from [-1, 1] to codebook range
    emb_min = vqgan.codebook.embeddings.min()
    emb_max = vqgan.codebook.embeddings.max()
    emb_range = emb_max - emb_min
    
    if emb_range > 1e-6:
        latent = ((latent + 1.0) / 2.0) * emb_range + emb_min
    
    print(f"After denormalization: min={latent.min():.3f}, max={latent.max():.3f}")
    
    # Check dimension order - VQ-GAN expects [B, C, H, W, D]
    if latent.dim() == 5:
        print(f"Latent dims: B={latent.shape[0]}, C={latent.shape[1]}, "
              f"D={latent.shape[2]}, H={latent.shape[3]}, W={latent.shape[4]}")
        
        # If [B, C, D, H, W], permute to [B, C, H, W, D]
        if latent.shape[2] < latent.shape[3]:
            print("Permuting [B,C,D,H,W] -> [B,C,H,W,D]")
            latent = latent.permute(0, 1, 3, 4, 2)
    
    print(f"VQ-GAN input shape: {latent.shape}")
    
    try:
        # Don't quantize - we have continuous latent features, not discrete codes
        decoded = vqgan.decode(latent, quantize=False)
        print(f"Decoded shape: {decoded.shape}")
        return decoded
    except Exception as e:
        print(f"ERROR during decoding: {e}")
        print(f"Latent shape was: {latent.shape}")
        print("\nTrying direct decoder call...")
        try:
            # Skip the embedding lookup, go straight to decoder
            h = vqgan.post_vq_conv(latent)
            decoded = vqgan.decoder(h)
            print(f"Direct decode worked! Shape: {decoded.shape}")
            return decoded
        except Exception as e2:
            print(f"Direct decode also failed: {e2}")
            return None


def save_volume(volume, output_path):
    """Save as numpy file."""
    if volume.dim() == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.dim() == 4:
        volume = volume.squeeze(0)
    
    vol_np = volume.cpu().numpy()
    np.save(output_path, vol_np)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xray_path', type=str, required=True)
    parser.add_argument('--ddpm_ckpt', type=str, required=True)
    parser.add_argument('--vqgan_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/inference_debug/')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DDPM Debug Inference")
    print("=" * 80)
    
    # Load
    xray = load_xray(args.xray_path, device=args.device)
    print(f"X-ray shape: {xray.shape}")
    
    diffusion, vqgan = load_models(args.ddpm_ckpt, args.vqgan_ckpt, device=args.device)
    
    # Generate latent
    latent = generate_latent(xray, diffusion, device=args.device)
    
    # Save latent
    xray_name = Path(args.xray_path).stem
    latent_path = os.path.join(args.output_dir, f"{xray_name}_latent.npy")
    save_volume(latent, latent_path)
    
    # Try to decode
    decoded = decode_latent(latent, vqgan)
    
    if decoded is not None:
        decoded_path = os.path.join(args.output_dir, f"{xray_name}_decoded.npy")
        save_volume(decoded, decoded_path)
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
