# Complete Patch-Wise Training Setup Guide

## Step 1: Environment Preparation (RunPod)

If training on RunPod, start with this template:

```bash
# SSH into RunPod terminal
cd /workspace

# Clone or navigate to your repository
# If not already present:
# git clone https://github.com/kanadm12/Xray-to-CTPA.git
# cd Xray-to-CTPA

# Check GPU availability
nvidia-smi
# Should show H100 or RTX 6000 with 24GB+ VRAM
```

## Step 2: Install Core Dependencies

```bash
# Navigate to patch-wise directory
cd patchwise_512x512x604_single_gpu

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# Or on Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Packages Being Installed:
- `torch>=2.0.0` - PyTorch deep learning framework
- `pytorch-lightning>=2.0.0` - Training utilities and callbacks
- `torchvision>=0.15.0` - Vision utilities
- `numpy<2.0` - Numerical computations
- `nibabel>=5.0.0` - NIfTI medical image format support
- `scipy>=1.10.0` - Scientific computing
- `omegaconf>=2.3.0` - Configuration management
- `hydra-core>=1.3.0` - Experiment configuration framework
- `einops>=0.6.0` - Tensor operations
- `tensorboard>=2.12.0` - Training visualization
- `tqdm>=4.65.0` - Progress bars
- `scikit-image>=0.20.0` - Image processing utilities

**Installation Time**: ~5-10 minutes depending on internet speed

## Step 3: Verify Installation

```bash
# Test PyTorch and GPU
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Test other critical imports
python -c "import pytorch_lightning; import nibabel; import omegaconf; print('All imports successful!')"

# Check GPU memory
nvidia-smi --query-gpu=memory.total --format=csv,noheader
# Should show 24GB or higher
```

## Step 4: Verify Data Structure

After preprocessing, verify your NIfTI files:

```bash
# Check if NIfTI files were created
ls -lh /workspace/datasets_nifti/ | head -20
# Should show .nii.gz files

# Check dimensions of a sample volume
python -c "
import nibabel as nib
import glob
nifti_files = glob.glob('/workspace/datasets_nifti/*.nii.gz')
if nifti_files:
    img = nib.load(nifti_files[0])
    print(f'Sample file: {nifti_files[0]}')
    print(f'Shape: {img.shape}')
    print(f'Data type: {img.get_fdata().dtype}')
    print(f'Total NIfTI files: {len(nifti_files)}')
else:
    print('No NIfTI files found - check preprocessing step')
"
```

Expected output:
```
Sample file: /workspace/datasets_nifti/patient_001.nii.gz
Shape: (512, 512, 604)
Data type: float32
Total NIfTI files: 42
```

## Step 6: Customize Training

```yaml
# Example customizations:
training:
  batch_size: 4  # Number of patches per batch
  num_epochs: 30
  learning_rate: 4.5e-6
  
model:
  embedding_dim: 64
  num_res_blocks: 2
  codebook_size: 512
  
hardware:
  device: cuda
  num_workers: 4
```

## Step 7: Launch Training

### Option 1: Using Launch Script (Recommended)

```bash
# Make executable (Linux/Mac)
chmod +x launch_distributed_training.sh

# Run training
./launch_distributed_training.sh

# On RunPod, you can detach with screen:
screen -S training
chmod +x launch_distributed_training.sh
./launch_distributed_training.sh
# Press Ctrl+A then D to detach
# screen -r training  # To reattach
```

### Option 2: Direct Python Command

```bash
# Set Python path
export PYTHONPATH=/workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu:$PYTHONPATH

# Run training
python train/train_vqgan_distributed.py
```

### Option 3: With Custom Parameters

```bash
python train/train_vqgan_distributed.py \
  --data_dir ../datasets_nifti \
  --batch_size 4 \
  --num_epochs 30 \
  --learning_rate 4.5e-6 \
  --checkpoint_dir outputs/checkpoints
```

## Step 8: Monitor Training Progress

### Real-Time Monitoring

```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: View training logs
tail -f outputs/logs/training.log

# Terminal 3: Launch TensorBoard
tensorboard --logdir outputs/tensorboard_logs --port 6006
# Access at: http://localhost:6006 or http://runpod_ip:6006
```

### Key Metrics to Watch

```
Codebook Usage: Should be >80% (aim for 85-95%)
Reconstruction Loss: Should decrease monotonically
PSNR: Should improve from baseline's 35.3 dB
Perplexity: Should stabilize around 256-512
```

## Step 9: Checkpointing and Recovery

Training will automatically save checkpoints:

```
outputs/
├── checkpoints/
│   ├── epoch_00_step_0000.ckpt
│   ├── epoch_05_step_0100.ckpt
│   └── ...
├── tensorboard_logs/
├── logs/
│   └── training.log
└── ...
```

**Resume from checkpoint**:

```bash
python train/train_vqgan_distributed.py \
  --resume_from outputs/checkpoints/epoch_05_step_0100.ckpt
```

## Step 10: Validation After Training

```bash
# Run validation on held-out test set
python train/validate_patches.py \
  --checkpoint outputs/checkpoints/final.ckpt \
  --data_dir ../datasets_nifti \
  --output_dir validation_results

# Generate sample reconstructions
python train/generate_samples.py \
  --checkpoint outputs/checkpoints/final.ckpt \
  --num_samples 5 \
  --output_dir sample_reconstructions
```

## Troubleshooting

### Out of Memory (OOM) Error
```
Error: CUDA out of memory
Solution 1: Reduce batch_size from 4 to 2
Solution 2: Reduce patch size from 256×256×128 to 192×192×96
Solution 3: Ensure no other GPU processes are running (nvidia-smi)
```

### Data Loading Issues
```
Error: No DICOM files found
Solution 1: Verify data_dir path in config file is correct
Solution 2: Check file format - must be .nii.gz
Solution 3: Run: find ../datasets -name "*.nii.gz" | head
```

### Training Loss Not Decreasing
```
Cause: Learning rate too high or too low
Solution 1: Check TensorBoard for learning rate schedule
Solution 2: Restart with learning_rate: 1.5e-6 or 9.0e-6
Solution 3: Verify data normalization is correct
```

### GPU Not Detected
```
Error: CUDA not available
Solution 1: Check nvidia-smi shows GPU
Solution 2: Verify PyTorch was installed with CUDA support
Solution 3: Reinstall PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Reference Commands

```bash
# Check everything is ready
python -c "import torch; print(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 24e9)"

# Start training (from patchwise directory)
python train/train_vqgan_distributed.py

# Stop training gracefully
# Press Ctrl+C in terminal

# View tensorboard in browser
tensorboard --logdir outputs/tensorboard_logs

# Archive results after training
tar -czf vqgan_patch_results_epoch30.tar.gz outputs/

# Download results locally (from local machine)
scp -r user@runpod_ip:/workspace/Xray-to-CTPA/patchwise_512x512x604_single_gpu/outputs ./results_patchwise
```

## Expected Training Timeline

- **Data Preparation**: 30 minutes - 2 hours (depending on dataset size)
- **First Epoch**: 10-15 minutes
- **Full Training (30 epochs)**: 5-7.5 days on H100/RTX 6000
- **Validation**: 15-30 minutes

**Total Wall Time**: ~7-8 days for complete training + validation

## What to Expect at Each Stage

### Epoch 1-3
- High reconstruction loss
- Codebook usage ramping up
- PSNR ~20-25 dB

### Epoch 10
- Loss settling into pattern
- Codebook usage 50-70%
- PSNR ~30-33 dB

### Epoch 30 (Final)
- Stable loss
- Codebook usage >85%
- PSNR >35 dB (targeting 37+ dB)

## Next Steps After Training

1. **Evaluate Quality**: Compare reconstructions with baseline (35.3 dB PSNR)
2. **Analyze Boundary Artifacts**: Check for visible seams in patch reconstructions
3. **Test DDPM Integration**: Use encoded latents for diffusion model training
4. **Fine-tune if Needed**: Adjust hyperparameters based on results
5. **Deploy Model**: Export trained checkpoint for inference

---

**Questions?** Check the README.md or VALIDATION_GUIDE.md in this directory.
