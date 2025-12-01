#!/usr/bin/env python3
"""
Download checkpoints from Google Drive to RunPod using rclone
Automatically handles configuration and downloads VQ-GAN and DDPM checkpoints
"""

import os
import subprocess
import sys
from pathlib import Path

# Target directories
VQGAN_DIR = "/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints"
DDPM_DIR = "/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches"

# Colors
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

def print_color(text, color):
    print(f"{color}{text}{NC}")

def run_command(cmd, check=True, capture=False):
    """Run shell command"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            print_color(f"Error running command: {cmd}", RED)
            print_color(f"Error: {e}", RED)
            sys.exit(1)
        return False

def check_rclone():
    """Check if rclone is installed, install if not"""
    print_color("[1/5] Checking rclone installation...", BLUE)
    
    if run_command("which rclone", check=False):
        version = run_command("rclone --version | head -n 1", capture=True)
        print_color(f"✓ rclone already installed ({version})", GREEN)
        return True
    
    print_color("rclone not found. Installing...", YELLOW)
    run_command("curl https://rclone.org/install.sh | sudo bash")
    print_color("✓ rclone installed", GREEN)
    return True

def check_gdrive_config():
    """Check if Google Drive remote is configured"""
    print_color("\n[2/5] Checking Google Drive configuration...", BLUE)
    
    remotes = run_command("rclone listremotes", capture=True)
    
    if "gdrive:" in remotes:
        print_color("✓ Google Drive remote 'gdrive' already configured", GREEN)
        return True
    
    print_color("Google Drive not configured. Setting up...", YELLOW)
    print_color("\nTo authenticate with Google Drive:", YELLOW)
    print("1. Run: rclone config")
    print("2. Choose 'n' for new remote")
    print("3. Name it 'gdrive'")
    print("4. Choose 'drive' for Google Drive")
    print("5. Follow prompts (use defaults for most)")
    print("6. For 'Auto config', choose 'n' and use browser authentication")
    print("\nOr run this script again after configuration.")
    
    response = input("\nPress Enter to open rclone config now, or 'q' to quit: ")
    if response.lower() == 'q':
        sys.exit(0)
    
    run_command("rclone config")
    return True

def list_gdrive_folders():
    """List folders in Google Drive"""
    print_color("\n[3/5] Listing Google Drive folders...", BLUE)
    
    print_color("Root folders in your Google Drive:", YELLOW)
    folders = run_command("rclone lsd gdrive:", capture=True)
    print(folders)
    
    print_color("\nSearching for checkpoint folders...", YELLOW)
    checkpoint_search = run_command("rclone lsf gdrive: --dirs-only | grep -i checkpoint", 
                                   check=False, capture=True)
    if checkpoint_search:
        print_color("Found checkpoint folders:", GREEN)
        print(checkpoint_search)

def download_checkpoints(gdrive_path, local_path, name):
    """Download checkpoints from Google Drive"""
    print_color(f"\nDownloading {name} checkpoints...", BLUE)
    print(f"  From: gdrive:{gdrive_path}")
    print(f"  To:   {local_path}")
    
    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    # Download using rclone
    cmd = f"""rclone copy "gdrive:{gdrive_path}" "{local_path}" \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --stats 5s \
        --include "*.ckpt" \
        --include "*.pt" \
        --include "*.pth" \
        --include "*.yaml" \
        --include "*.json" \
        --verbose
    """
    
    run_command(cmd)
    
    # List downloaded files
    print_color(f"\n✓ {name} checkpoints downloaded", GREEN)
    print(f"Files in {local_path}:")
    run_command(f"ls -lh {local_path}")

def main():
    print("=" * 50)
    print("Google Drive Checkpoint Downloader")
    print("=" * 50)
    
    # Check rclone
    check_rclone()
    
    # Check Google Drive config
    check_gdrive_config()
    
    # List folders
    list_gdrive_folders()
    
    # Download VQ-GAN checkpoints
    print_color("\n[4/5] VQ-GAN Checkpoint Download", BLUE)
    print("Enter the Google Drive path to your VQ-GAN checkpoints")
    print("Examples:")
    print("  - checkpoints/vqgan_patches_4gpu")
    print("  - X-ray2CTPA/vqgan_checkpoints")
    print("  - outputs/vqgan_patches_4gpu/checkpoints")
    
    vqgan_path = input("\nVQ-GAN path (or press Enter to skip): ").strip()
    
    if vqgan_path:
        download_checkpoints(vqgan_path, VQGAN_DIR, "VQ-GAN")
    else:
        print_color("Skipping VQ-GAN download", YELLOW)
    
    # Download DDPM checkpoints
    print_color("\n[5/5] DDPM Checkpoint Download", BLUE)
    print("Enter the Google Drive path to your DDPM checkpoints")
    print("Examples:")
    print("  - checkpoints/ddpm_4gpu_patches")
    print("  - X-ray2CTPA/ddpm_checkpoints")
    
    ddpm_path = input("\nDDPM path (or press Enter to skip): ").strip()
    
    if ddpm_path:
        download_checkpoints(ddpm_path, DDPM_DIR, "DDPM")
    else:
        print_color("Skipping DDPM download", YELLOW)
    
    # Summary
    print_color("\n" + "=" * 50, GREEN)
    print_color("✓ Download Complete!", GREEN)
    print_color("=" * 50, GREEN)
    print(f"\nCheckpoint locations:")
    print(f"  VQ-GAN: {VQGAN_DIR}")
    print(f"  DDPM:   {DDPM_DIR}")
    
    print("\nTo verify downloads:")
    print(f"  ls -lh {VQGAN_DIR}")
    print(f"  ls -lh {DDPM_DIR}")
    
    print("\nTo sync again later:")
    print("  python3 download_gdrive_checkpoints.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_color("\n\nDownload cancelled by user", YELLOW)
        sys.exit(0)
    except Exception as e:
        print_color(f"\nError: {e}", RED)
        sys.exit(1)
