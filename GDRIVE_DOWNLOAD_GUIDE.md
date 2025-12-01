# Quick Guide: Download Checkpoints from Google Drive

## Method 1: Using the Python Script (Recommended)

```bash
cd /workspace/Xray-to-CTPA
python3 download_gdrive_checkpoints.py
```

The script will:
1. Install rclone if needed
2. Guide you through Google Drive authentication (one-time)
3. Show your Google Drive folders
4. Download checkpoints to the correct locations

## Method 2: Using the Bash Script

```bash
cd /workspace/Xray-to-CTPA
chmod +x download_gdrive_checkpoints.sh
./download_gdrive_checkpoints.sh
```

## Method 3: Manual rclone Commands

### First-time setup:
```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config
# Choose: n (new remote)
# Name: gdrive
# Storage: drive (Google Drive)
# Follow prompts and authenticate
```

### Download VQ-GAN checkpoints:
```bash
rclone copy \
  "gdrive:YOUR_GDRIVE_PATH/vqgan_checkpoints" \
  "/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints" \
  --progress --include "*.ckpt" --include "*.pt"
```

### Download DDPM checkpoints:
```bash
rclone copy \
  "gdrive:YOUR_GDRIVE_PATH/ddpm_checkpoints" \
  "/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches" \
  --progress --include "*.ckpt" --include "*.pt"
```

## Finding Your Google Drive Paths

```bash
# List root folders
rclone lsd gdrive:

# List all folders recursively
rclone lsf gdrive: --dirs-only -R

# Search for checkpoint folders
rclone lsf gdrive: --dirs-only -R | grep checkpoint
```

## Common Google Drive Paths

If you previously uploaded checkpoints, they might be at:
- `X-ray2CTPA/checkpoints/`
- `checkpoints/vqgan_patches_4gpu/`
- `checkpoints/ddpm_4gpu_patches/`
- `outputs/vqgan_patches_4gpu/checkpoints/`

## Useful rclone Commands

```bash
# List files in a Google Drive folder
rclone ls "gdrive:YOUR_PATH"

# Copy specific file
rclone copy "gdrive:path/to/file.ckpt" /local/path/

# Sync (keeps directories identical)
rclone sync "gdrive:YOUR_PATH" /local/path/

# Check transfer speed
rclone copy "gdrive:YOUR_PATH" /local/path/ --progress --stats 1s

# Resume interrupted transfer
rclone copy "gdrive:YOUR_PATH" /local/path/ --progress --checksum
```

## Troubleshooting

### Authentication Issues
```bash
# Reconnect to Google Drive
rclone config reconnect gdrive:

# Or delete and reconfigure
rclone config delete gdrive
rclone config
```

### Check rclone configuration
```bash
rclone config show
rclone listremotes
```

### Test connection
```bash
rclone lsd gdrive:
```

### Transfer too slow?
```bash
# Increase parallel transfers
rclone copy "gdrive:PATH" /local/path/ \
  --transfers 16 \
  --checkers 32 \
  --drive-chunk-size 128M
```

## Target Directories

After download, your checkpoints will be at:

**VQ-GAN:**
```
/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints/
├── last.ckpt           # Latest checkpoint
├── best.ckpt           # Best performing
└── epoch=N-step=M.ckpt # Specific epochs
```

**DDPM:**
```
/workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches/
├── model_latest.pt     # Latest checkpoint
├── model_best.pt       # Best performing
└── model_epoch_N.pt    # Specific epochs
```

## Verify Downloads

```bash
# Check VQ-GAN checkpoints
ls -lh /workspace/Xray-to-CTPA/patchwise_4gpu_distributed/outputs/vqgan_patches_4gpu/checkpoints/

# Check DDPM checkpoints
ls -lh /workspace/Xray-to-CTPA/patchwise_4gpu_distributed/checkpoints/ddpm_4gpu_patches/

# Check file integrity
md5sum /path/to/checkpoint.ckpt
```
