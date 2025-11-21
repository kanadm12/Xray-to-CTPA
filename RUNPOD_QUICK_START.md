# Quick Reference: RunPod Training Steps

## ⚡ Super Quick Start (Copy-Paste Ready)

```bash
# 1. SSH into RunPod pod (replace with your IP)
ssh root@YOUR_POD_IP

# 2. Clone and setup (runs everything automatically)
cd /workspace
git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
cd Xray-2CTPA_spartis
bash setup_and_train.sh
```

**That's it! The script handles everything automatically.**

---

## 📋 Step-by-Step Manual Setup (If needed)

### 1️⃣ Clone Repository
```bash
cd /workspace
git clone https://github.com/kanadm12/Xray-2CTPA_spartis.git
cd Xray-2CTPA_spartis
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Upload/Download Your Dataset
Place your dataset at: `/workspace/datasets/data_new/`

**Option A - RunPod Files Upload:**
- Click "Files" in RunPod
- Create: `/workspace/datasets/`
- Upload your `data_new` folder

**Option B - Google Drive:**
```bash
pip install gdown
gdown --id YOUR_FILE_ID -O /workspace/dataset.tar.gz
mkdir -p /workspace/datasets && tar -xzf /workspace/dataset.tar.gz -C /workspace/datasets/
```

**Option C - Verify Dataset:**
```bash
ls -la /workspace/datasets/data_new/
find /workspace/datasets/data_new -name "*.nii.gz" ! -name "*swapped*" | wc -l
```

### 4️⃣ Check Configuration
```bash
cat config/dataset/custom_data.yaml
# Should show: root_dir: /workspace/datasets/data_new/
```

### 5️⃣ Start Training
```bash
export PYTHONPATH=$PWD
bash train/scripts/train_vqgan_custom.sh
```

---

## 🎯 Minimal Quick Command (One-Liner)

```bash
cd /workspace/Xray-2CTPA_spartis && export PYTHONPATH=$PWD && python train/train_vqgan.py dataset=custom_data model=vq_gan_3d model.gpus=1 model.batch_size=2 model.default_root_dir_postfix='custom_data'
```

---

## 📊 Monitor Training

```bash
# GPU Usage (Real-time)
nvidia-smi -l 1

# Training Logs
tail -f /workspace/Xray-2CTPA_spartis/lightning_logs/version_*/checkpoints/*.log

# Disk Space
df -h /workspace

# Running Processes
ps aux | grep train_vqgan
```

---

## 🆘 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Add: `model.batch_size=1` |
| **Dataset not found** | Check: `ls /workspace/datasets/data_new/` |
| **Slow training** | Add: `model.num_workers=8` |
| **Connection dropped** | Use: `tmux` or `nohup bash ...` |
| **Config error** | Verify: `cat config/dataset/custom_data.yaml` |

---

## 💾 Save Your Trained Model

```bash
# Zip checkpoints
cd /workspace/Xray-2CTPA_spartis
tar -czf lightning_logs.tar.gz lightning_logs/

# Option 1: Download via RunPod Files (click Files > lightning_logs.tar.gz)
# Option 2: Upload to cloud
pip install rclone
rclone sync lightning_logs/ gdrive:backups/lightning_logs/
```

---

## 📁 Key File Locations

```
/workspace/
├── Xray-2CTPA_spartis/           # Main repo
│   ├── config/dataset/custom_data.yaml   # Dataset config
│   ├── train/scripts/train_vqgan_custom.sh  # Training script
│   ├── lightning_logs/            # Training outputs
│   └── setup_and_train.sh         # Auto setup script
│
└── datasets/
    └── data_new/                  # Your dataset
        ├── patient_001/
        ├── patient_002/
        └── ...
```

---

## ✅ Pre-Training Checklist

- [ ] Pod is running (check RunPod dashboard)
- [ ] GPU available (`nvidia-smi` shows GPU)
- [ ] Repository cloned
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Dataset uploaded (`ls /workspace/datasets/data_new/`)
- [ ] Config updated
- [ ] Disk space > 50GB available

---

## 🚀 TL;DR - Absolute Minimum Steps

1. SSH into pod: `ssh root@IP`
2. Setup: `cd /workspace && git clone ... && cd Xray-2CTPA_spartis && bash setup_and_train.sh`
3. Monitor: `nvidia-smi -l 1`
4. Wait for training to complete
5. Download from Files or backup to cloud

That's literally it! 🎉
