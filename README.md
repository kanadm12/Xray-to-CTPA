# X-ray2CTPA: Full Resolution Medical Image Synthesis

This repository contains the complete implementation for training VQ-GAN models to generate high-resolution 3D CTPA (CT Pulmonary Angiography) scans from 2D X-ray conditioning, with both baseline and full-resolution patch-wise approaches.

**Paper**: [X-ray2CTPA: Generating 3D CTPA scans from 2D X-ray conditioning](https://arxiv.org/abs/2406.16109)

---

## 🎯 Project Overview

### Problem Statement

- **X-ray imaging**: Lower resolution, 2D, but low cost and radiation exposure
- **CT imaging**: Higher resolution, 3D, detailed anatomy, but expensive and high radiation
- **CTPA scans**: Gold standard for pulmonary embolism diagnosis, very expensive

**Goal**: Generate high-quality 3D CTPA volumes from 2D X-ray inputs using generative AI, enabling:
- More accessible diagnostic tools
- Reduced radiation exposure when combined with X-rays
- Cost-effective advanced imaging

### Solution: Diffusion-Based Cross-Modal Synthesis

1. **Stage 1: VQ-GAN Training** (this repo) - Learns compressed latent representations of 3D CTPA volumes
2. **Stage 2: Diffusion Model** - Learns to generate latent codes conditioned on X-ray inputs
3. **Stage 3: Inference** - X-ray → latent code → decoded CTPA volume

---

## 📁 Repository Structure

```
X-ray2CTPA/
├── baseline_256x256x64_single_gpu/          # Baseline VQ-GAN (256×256×64 resolution)
│   ├── train/train_vqgan.py                 # Training script
│   ├── dataset/                             # Dataset loaders
│   ├── vq_gan_3d/model/vqgan.py             # VQ-GAN architecture
│   ├── config/                              # Hydra configuration files
│   ├── README.md                            # Baseline documentation
│   ├── RUNPOD_TRAINING_GUIDE.md            # Training guide (baseline)
│   └── requirements.txt                     # Dependencies
│
├── patchwise_512x512x604_single_gpu/        # Full-resolution VQ-GAN (patch-wise)
│   ├── train/train_vqgan_distributed.py     # Single-GPU training
│   ├── dataset/patch_dataset.py             # Patch extraction & blending
│   ├── utils/patch_utils.py                 # Patch utilities
│   ├── vq_gan_3d/model/vqgan_patches.py    # Patch-based model
│   ├── config/                              # Configuration files
│   ├── RUNPOD_TRAINING_GUIDE.md            # Training guide (full-res)
│   ├── TRAINING_GUIDE.md                    # Technical architecture details
│   ├── VALIDATION_GUIDE.md                  # Validation procedures
│   ├── README.md                            # Architecture overview
│   ├── launch_distributed_training.sh       # Training launcher
│   ├── setup_training.sh                    # Automated setup
│   └── requirements.txt                     # Dependencies
│
├── ddpm/                                    # Diffusion model (Stage 2)
│   ├── diffusion.py
│   ├── unet.py
│   ├── text.py
│   ├── time_embedding.py
│   ├── lora.py
│   └── classifier/
│
├── dataset/                                 # Data loading utilities
├── preprocess/                              # Data preprocessing scripts
├── config/                                  # Root configuration files
├── SETUP_PATCHWISE_TRAINING.md             # Simplified setup guide
├── README.md                                # This file
├── params.py                                # Global parameters
└── requirements.txt                         # Root dependencies
```

---

## 🚀 Quick Start

### Option 1: Baseline Training (Fast, Lower Resolution)

Best for prototyping and validation on limited hardware.

```bash
cd baseline_256x256x64_single_gpu
pip install -r requirements.txt
python train/train_vqgan.py
```

**Specifications**:
- Resolution: 256×256×64 voxels
- Training Time: ~8 hours (30 epochs)
- GPU Memory: 12 GB
- Expected PSNR: 35.3 dB
- Codebook Usage: 87%

📖 See `baseline_256x256x64_single_gpu/RUNPOD_TRAINING_GUIDE.md` for detailed instructions.

### Option 2: Full-Resolution Training (Comprehensive)

Production-quality results with patch-wise decomposition.

```bash
cd patchwise_512x512x604_single_gpu
pip install -r requirements.txt
./launch_distributed_training.sh
```

**Specifications**:
- Resolution: 512×512×604 voxels (full clinical resolution)
- Training Time: 5-7 days (30 epochs)
- GPU Memory: 24 GB (H100/RTX 6000)
- Expected PSNR: >37 dB
- Patch Strategy: 256×256×128 with 25% overlap blending
- Patches per Volume: ~24 (3×3×3 grid)

📖 See `patchwise_512x512x604_single_gpu/RUNPOD_TRAINING_GUIDE.md` for step-by-step guide.

---

## 📊 Model Architectures

### Baseline VQ-GAN (256×256×64)

| Component | Specification |
|-----------|---------------|
| **Input** | 512 3D volumes, 256×256×64 voxels |
| **Encoder** | 3 resolution-reducing blocks |
| **Quantization** | 512 codebook vectors, embedding_dim=64 |
| **Decoder** | 3 resolution-increasing blocks |
| **Parameters** | ~36M |
| **Codebook Usage** | 87% |

**Performance**:
- PSNR: 35.3 dB
- SSIM: 0.971
- Training: 8 hours on single GPU

### Full-Resolution VQ-GAN (512×512×604)

| Component | Specification |
|-----------|---------------|
| **Input** | 512×512×604 volumes via 256×256×128 patches |
| **Patches per Volume** | ~24 (3×3×3 grid) |
| **Patch Overlap** | 64 voxels (25%) for boundary blending |
| **Architecture** | Same 36M parameter model as baseline |
| **Blending** | Linear/Gaussian weighted reconstruction |
| **Memory** | ~24 GB per GPU (batch_size=4) |

**Performance** (Expected):
- PSNR: >37 dB
- Codebook Usage: >85%
- Training: 5-7 days on single H100/RTX 6000

---

## 💾 Dataset

### Source
RSNA CT dataset: ~42 patient folders with multiple DICOM series each

### Location (on RunPod)
```
/workspace/datasets/        # Pre-converted NIfTI files
└── *.nii.gz               # 512×512×604 volumes
```

### Dataset Characteristics
- **Image Format**: 512×512 pixels (consistent)
- **Volume Depth**: 604-800 slices (variable)
- **Tube Voltage (kVp)**: 80, 100, 120 kV (heterogeneous)
- **Reconstruction Kernels**: STANDARD, B, B30f, FC08-H (4 types)
- **Slice Thickness**: 1.0, 1.25, 2.0 mm (variable)
- **Most Common**: 120 kVp, STANDARD kernel, 1.25 mm

See `azure_patient_analysis.json` for detailed statistics on 20 sampled patient folders.

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 24GB+ VRAM GPU (H100, RTX 6000, A100 recommended)

### Setup (on RunPod or Local)

```bash
# Navigate to repository
cd /workspace/Xray-to-CTPA

# For baseline training
cd baseline_256x256x64_single_gpu
pip install -r requirements.txt

# For full-resolution training
cd patchwise_512x512x604_single_gpu
pip install -r requirements.txt
```

**Key Dependencies**:
- torch >= 2.0.0
- pytorch-lightning >= 2.0.0
- nibabel >= 5.0.0 (medical image format)
- hydra-core >= 1.3.0 (configuration)
- tensorboard >= 2.12.0 (monitoring)

---

## 🎓 Training Pipeline

### Stage 1: VQ-GAN Training (This Repository)

**Baseline (256×256×64)**:
```bash
cd baseline_256x256x64_single_gpu
python train/train_vqgan.py
```

**Full-Resolution (512×512×604)**:
```bash
cd patchwise_512x512x604_single_gpu
./launch_distributed_training.sh
```

### Stage 2: Diffusion Model Training (Future)

```bash
cd diffusion_training
python train_ddpm.py --pretrained_vqgan ../patchwise_512x512x604_single_gpu/outputs/final_model.ckpt
```

### Stage 3: Inference

```python
from vqgan_inference import X-Ray2CTPA

model = X-Ray2CTPA.load_checkpoint('final_model.ckpt')
xray = load_xray('patient_xray.png')
ctpa_volume = model.generate(xray)
```

---

## 📈 Monitoring Training

### Real-Time Metrics

```bash
# TensorBoard visualization
tensorboard --logdir outputs/tensorboard_logs --port 6006
# Access at: http://localhost:6006
```

**Key Metrics**:
- **Reconstruction Loss**: Should decrease smoothly
- **Codebook Usage**: Target >85% (avoid collapse)
- **Perplexity**: Indicates codebook diversity
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)

### Logging

```bash
# View training logs
tail -f outputs/logs/training.log

# GPU utilization
watch -n 1 nvidia-smi
```

---

## 🔄 Checkpointing and Recovery

### Automatic Saves

Checkpoints save every 5 epochs:
```
outputs/
├── checkpoints/
│   ├── epoch_00_step_0000.ckpt
│   ├── epoch_05_step_0100.ckpt
│   └── epoch_30_step_0600.ckpt  (final)
├── tensorboard_logs/
└── logs/
```

### Resume Training

```bash
python train/train_vqgan_distributed.py \
  ckpt_path=outputs/checkpoints/epoch_15_step_0300.ckpt
```

---

## 📋 Configuration

### Using Hydra for Configuration

Edit `config/base_cfg.yaml`:

```yaml
defaults:
  - dataset: full_resolution_ctpa
  - model: vq_gan_3d_patches

training:
  batch_size: 4
  num_epochs: 30
  learning_rate: 4.5e-6
  num_workers: 4

hardware:
  device: cuda
  mixed_precision: true
```

### Override from Command Line

```bash
python train/train_vqgan_distributed.py \
  training.batch_size=6 \
  training.num_epochs=50 \
  hardware.device=cuda:0
```

---

## 🧪 Validation and Evaluation

### Validate After Training

```bash
python train/validate_patches.py \
  --checkpoint outputs/checkpoints/final.ckpt \
  --data_dir /workspace/datasets
```

### Generate Samples

```bash
python train/generate_samples.py \
  --checkpoint outputs/checkpoints/final.ckpt \
  --num_samples 10
```

### Compute Metrics

```bash
python train/compute_metrics.py \
  --checkpoint outputs/checkpoints/final.ckpt \
  --test_data /workspace/datasets \
  --output metrics_report.json
```

---

## 🔬 Architecture Comparison

| Aspect | Baseline | Full-Resolution |
|--------|----------|-----------------|
| **Resolution** | 256×256×64 | 512×512×604 |
| **Approach** | Direct encoding | Patch-wise |
| **Patches** | N/A | 256×256×128, 25% overlap |
| **Training Time** | 8 hours | 5-7 days |
| **Memory** | 12 GB | 24 GB |
| **PSNR** | 35.3 dB | >37 dB |
| **Use Case** | Prototyping | Production |
| **Quality** | Good | Excellent |

---

## 📚 Documentation

- **Baseline**: `baseline_256x256x64_single_gpu/RUNPOD_TRAINING_GUIDE.md`
- **Full-Resolution**: `patchwise_512x512x604_single_gpu/RUNPOD_TRAINING_GUIDE.md`
- **Architecture Details**: `patchwise_512x512x604_single_gpu/TRAINING_GUIDE.md`
- **Validation**: `patchwise_512x512x604_single_gpu/VALIDATION_GUIDE.md`
- **Quick Setup**: `SETUP_PATCHWISE_TRAINING.md`

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train/train_vqgan_distributed.py training.batch_size=2
```

### Training Loss Not Decreasing
- Verify data is normalized to [0, 1]
- Check learning rate: try `9.0e-6` or `1.5e-6`
- Reduce batch size to detect batch-specific issues

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Outputs

After training, you'll have:

```
outputs/
├── checkpoints/final.ckpt           # Trained VQ-GAN model
├── tensorboard_logs/                # Training metrics
├── logs/training.log                # Training log
├── validation_results/              # PSNR, SSIM, codebook stats
└── sample_reconstructions/          # Visual validation samples
```

---

## 🎯 Next Steps

1. **Train baseline** (recommended first): 8 hours, validate approach
2. **Train full-resolution**: 5-7 days, production-quality results
3. **Evaluate quality**: Compare PSNR, SSIM with baseline
4. **Extract latents**: Use trained VQ-GAN for diffusion model training
5. **Train diffusion model**: Generate CTPA from X-ray conditioning
6. **Validate clinically**: Radiologist review and diagnostic assessment

---

## 📖 References

- **VQ-VAE**: [Neural Discrete Representation Learning (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- **VQ-GAN**: [Taming Transformers for Realistic Image Synthesis (Esser et al., 2021)](https://arxiv.org/abs/2012.09841)
- **Diffusion Models**: [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

---

## 📝 Citation

If you use this code or model, please cite:

```bibtex
@article{xray2ctpa2024,
  title={X-ray2CTPA: Generating 3D CTPA scans from 2D X-ray conditioning},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 👤 Author

**Kanad M** - Medical AI Research

---

## 📄 License

See `LICENSE` file for details.

---

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for bugs/suggestions.

---

**Last Updated**: November 2025  
**Repository**: https://github.com/kanadm12/Xray-to-CTPA
