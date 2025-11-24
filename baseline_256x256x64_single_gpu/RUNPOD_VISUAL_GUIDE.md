# RunPod Training - Step-by-Step Visual Guide

## 🎬 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Launch RunPod Pod                                        │
│    - Go to runpod.io                                        │
│    - Choose GPU (RTX4090 recommended)                       │
│    - Click Connect → JupyterLab or SSH                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Clone Repository                                         │
│    cd /workspace                                            │
│    git clone https://github.com/kanadm12/Xray-2CTPA_spartis│
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Install Dependencies                                     │
│    pip install -r requirements.txt                          │
│    (Takes 5-10 minutes)                                     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Upload Dataset                                           │
│    Place at: /workspace/datasets/data_new/                  │
│    (Or download from cloud storage)                         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Update Configuration                                     │
│    Edit: config/dataset/custom_data.yaml                    │
│    Set correct path to dataset                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Start Training                                           │
│    bash train/scripts/train_vqgan_custom.sh                 │
│    Or: bash setup_and_train.sh (auto everything!)           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Monitor & Wait                                           │
│    Watch GPU usage: nvidia-smi -l 1                         │
│    Check logs in: lightning_logs/                           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. Download Results                                         │
│    Via RunPod Files or sync to cloud storage                │
│    Model saved in: lightning_logs/version_X/checkpoints/    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Three Ways to Get Started

### 🟢 Way 1: Fully Automated (Recommended)
**Best for: Everyone - handles everything**

```bash
cd /workspace
git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
cd Xray-2CTPA_spartis

# Upload dataset first to /workspace/datasets/data_new/

bash setup_and_train.sh
# ✅ Clones → ✅ Installs → ✅ Verifies → ✅ Trains
```

**Time:** ~15 min setup + training time

---

### 🟡 Way 2: Manual Command Line
**Best for: Advanced users - full control**

```bash
cd /workspace/Xray-2CTPA_spartis
export PYTHONPATH=$PWD
python train/train_vqgan.py dataset=custom_data model=vq_gan_3d model.gpus=1
```

**Time:** ~5 min setup + training time

---

### 🔵 Way 3: Minimal One-Liner
**Best for: Impatient users**

```bash
cd /workspace/Xray-2CTPA_spartis && export PYTHONPATH=$PWD && bash train/scripts/train_vqgan_custom.sh
```

**Time:** ~2 min setup + training time

---

## 📊 Training Timeline Example

```
⏱️  0:00  - Script starts
⏱️  0:30  - Dependencies installed, dataset verified
⏱️  1:00  - GPU initialized, training begins
⏱️  1:05  - First epoch starting (loss: 0.523)
⏱️  5:00  - Multiple epochs completed
⏱️ 15:00  - Model improving (loss: 0.401)
⏱️ 30:00  - Checkpoint saved
...
⏱️  N:NN  - Training complete (depends on dataset size)
```

---

## 🔧 Customization Options

### Small Dataset/Limited VRAM
```bash
model.batch_size=1 model.num_workers=0 model.precision=32
```

### Large Dataset/High VRAM
```bash
model.batch_size=4 model.num_workers=8 model.num_accumulate_grad_batches=2
```

### Faster Training (Lower Quality)
```bash
model.embedding_dim=4 model.n_codes=8192 model.downsample=[4,4,4]
```

### Better Quality (Slower)
```bash
model.embedding_dim=16 model.n_codes=32768 model.downsample=[2,2,2]
```

---

## 📂 Directory Structure After Setup

```
/workspace/
├── Xray-2CTPA_spartis/                    ← Your repo
│   ├── dataset/                           ← Dataset classes
│   ├── train/
│   │   ├── train_vqgan.py                 ← Main training script
│   │   └── scripts/
│   │       └── train_vqgan_custom.sh      ← Your training script
│   ├── config/
│   │   └── dataset/
│   │       └── custom_data.yaml           ← Dataset config
│   ├── setup_and_train.sh                 ← Auto-setup script
│   ├── lightning_logs/                    ← Training outputs (created)
│   │   └── version_0/
│   │       ├── checkpoints/
│   │       │   ├── latest_checkpoint.ckpt ← Best model
│   │       │   └── epoch-N-step-M.ckpt
│   │       └── logs/
│   └── RUNPOD_TRAINING_GUIDE.md           ← Full guide
│
└── datasets/
    └── data_new/                          ← Your uploaded dataset
        ├── patient_001/
        │   ├── patient_001.nii.gz         ✅ Used
        │   ├── patient_001_lat_drr.png    ✅ Used
        │   ├── patient_001_pa_drr.png     ✅ Used
        │   └── patient_001_swapped.nii.gz ❌ Excluded
        ├── patient_002/
        └── ...
```

---

## 🎯 Essential Commands Reference

| Task | Command |
|------|---------|
| **Clone repo** | `cd /workspace && git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git` |
| **Install deps** | `pip install -r requirements.txt` |
| **Upload dataset** | Via RunPod Files GUI or: `rclone sync gdrive:dataset /workspace/datasets/data_new/` |
| **Verify dataset** | `find /workspace/datasets/data_new -name "*.nii.gz" -not -name "*swapped*" \| wc -l` |
| **Start training** | `bash setup_and_train.sh` |
| **Manual training** | `export PYTHONPATH=$PWD && bash train/scripts/train_vqgan_custom.sh` |
| **Monitor GPU** | `nvidia-smi -l 1` |
| **View logs** | `tail -f lightning_logs/version_0/*/metrics.csv` |
| **Stop training** | `pkill -f train_vqgan` |
| **Save model** | `tar -czf model.tar.gz lightning_logs/` |

---

## ❌ Common Mistakes & How to Avoid

| Mistake | How to Avoid |
|---------|-------------|
| Forgot to upload dataset | ✅ Upload to `/workspace/datasets/data_new/` **before** starting |
| Wrong config path | ✅ Verify: `cat config/dataset/custom_data.yaml` shows correct path |
| CUDA out of memory | ✅ Reduce batch size: `model.batch_size=1` |
| Connection drops mid-training | ✅ Use `tmux` or `nohup`: `nohup bash setup_and_train.sh &` |
| Ran out of disk space | ✅ Check: `df -h /workspace` (need >50GB) |

---

## 🎬 Real Example: From Start to Training

```bash
# 1. SSH into your RunPod (replace IP)
ssh root@123.456.789.012

# 2. Go to workspace
cd /workspace

# 3. Clone everything
git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
cd Xray-2CTPA_spartis

# 4. [PAUSE] Upload your dataset to /workspace/datasets/data_new/
#    (Use RunPod Files GUI or rclone)
#    Then verify: ls -la /workspace/datasets/data_new/

# 5. Run the automated setup
bash setup_and_train.sh

# 6. Watch training
# In another terminal:
nvidia-smi -l 1

# 7. When done, download your model
# Via RunPod Files GUI: Download lightning_logs.tar.gz
# Or: tar -czf model.tar.gz lightning_logs/
```

**Total time:** ~30 minutes for setup + training time depends on dataset

---

## 📚 Where to Find Help

| Need | Look Here |
|------|-----------|
| **Full setup guide** | `RUNPOD_TRAINING_GUIDE.md` |
| **Quick reference** | `RUNPOD_QUICK_START.md` |
| **Custom dataset help** | `CUSTOM_DATASET_TRAINING.md` |
| **Main README** | `README.md` |
| **Training troubleshooting** | See "Phase 5" in `RUNPOD_TRAINING_GUIDE.md` |

---

## ✅ Final Checklist Before Starting

```
Pre-Training:
  □ RunPod pod is running (check dashboard)
  □ GPU is available (nvidia-smi works)
  □ Repository cloned to /workspace/Xray-2CTPA_spartis/
  □ Dependencies installed (pip list shows torch, etc)
  □ Dataset uploaded to /workspace/datasets/data_new/
  □ Config file shows correct path
  □ At least 50GB disk space available
  □ Dataset has 5+ patient folders with .nii.gz files

During Training:
  □ GPU usage 80-100% (nvidia-smi)
  □ Training loss decreasing over time
  □ Checkpoints being saved (watch lightning_logs/)
  □ Memory usage stable

After Training:
  □ Final checkpoint saved
  □ Download model or backup to cloud
  □ Keep logs for future reference
```

---

**You're all set! Happy training! 🚀**
