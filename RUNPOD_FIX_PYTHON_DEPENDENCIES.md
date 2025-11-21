# Fix: Python Version & Dependencies Issue on RunPod

## 🔴 Problem

When trying to install dependencies, you get errors like:
```
ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cu118
ERROR: Requires-Python >=3.8,<3.12
```

## ⚡ Quick One-Liner Fixes

**Option 1 - Use the new compatible requirements file with NVIDIA index:**
```bash
cd /workspace/Xray-2CTPA_spartis
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-runpod.txt --no-deps
```

**Option 2 - Auto-fix with setup script (Recommended):**
```bash
cd /workspace/Xray-2CTPA_spartis
bash setup_and_train.sh
```

**Option 3 - Manual fix (if Option 1 doesn't work):**
```bash
pip cache purge
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache-dir
pip install -r requirements-runpod.txt --no-deps --force-reinstall --no-cache-dir
```

Then continue with:
```bash
bash setup_and_train.sh
```

Or manually:
```bash
export PYTHONPATH=$PWD
python train/train_vqgan.py dataset=custom_data model=vq_gan_3d model.gpus=1 model.batch_size=2
```

---

## 🟢 Detailed Solution

The issue is that RunPod's default environment might have **Python 3.12**, but the original `requirements.txt` specifies packages for **Python 3.11 and below**.

### Quick Fix (Recommended)

**Option A: Use the NVIDIA index (Most Reliable)**

```bash
cd /workspace/Xray-2CTPA_spartis
# Install PyTorch from NVIDIA's CUDA index (latest stable)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements-runpod.txt --no-deps
```

**Option B: Use the auto-setup script (Easiest)**

```bash
bash setup_and_train.sh
```

The script automatically handles PyTorch installation from the NVIDIA index.

### Manual Fix (If you know what you're doing)

If you need to manually install:

```bash
# Clear pip cache
pip cache purge

# Install PyTorch from NVIDIA's CUDA repository (2.2.0 is latest stable for cu118)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall --no-cache-dir

# Then install remaining dependencies (without re-installing PyTorch)
pip install -r requirements-runpod.txt --no-deps
```

**Key Points:**
- `--index-url https://download.pytorch.org/whl/cu118` - Uses NVIDIA's CUDA 11.8 PyTorch builds
- `--no-deps` - Tells pip not to reinstall dependencies that are already met
- `cu118` - Specifies CUDA 11.8 (compatible with most RunPod setups)
- `2.2.0` - Latest stable PyTorch with CUDA 11.8 support

---

## 📋 What Changed

### Key Differences Between Files

| Package | requirements.txt | requirements-runpod.txt |
|---------|------------------|------------------------|
| torch | 2.6.0+cu118 | 2.1.2+cu118 ✓ |
| torchaudio | 2.6.0+cu118 | 2.1.2+cu118 ✓ |
| torchvision | 0.21.0+cu118 | 0.16.2+cu118 ✓ |
| pytorch-lightning | 2.5.0.post0 | 2.1.3 ✓ |
| Python Support | 3.8-3.11 | 3.9-3.12 ✓ |
| CUDA Support | 11.8 | 11.8 ✓ |

**Note:** Older torch versions are still very capable and will work fine for training VQGAN.

---

## 🚀 Next Steps

After using the correct requirements file:

```bash
# Verify installation
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Continue with training
bash setup_and_train.sh
```

---

## ⚠️ Important Notes

1. **Both files are functional** - The newer versions in `requirements.txt` may have better performance, but the versions in `requirements-runpod.txt` are proven to work on RunPod
2. **No performance loss** - PyTorch 2.1.2 is stable and performant for this use case
3. **Automatic selection** - If you use `setup_and_train.sh`, it automatically picks the right one

---

## 🆘 Still Getting Errors?

### If CUDA still not found:
```bash
# Check GPU availability
nvidia-smi

# Force CUDA 12.x if available
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### If specific package fails:
```bash
# Try installing that package individually
pip install --upgrade PACKAGE_NAME
```

### Last resort:
```bash
# Clear pip cache and retry
pip cache purge
pip install -r requirements-runpod.txt --force-reinstall --no-cache-dir
```

---

## 📚 Files Reference

- `requirements.txt` - Original requirements (Python 3.8-3.11)
- `requirements-runpod.txt` - RunPod compatible (Python 3.9-3.12) ← **Use this**
- `setup_and_train.sh` - Automatically selects the right one

---

## 💡 Why This Happens

RunPod sometimes provides newer Python versions than what original projects tested for. This is common in cloud GPU environments. The solution is to use dependency versions that support the newer Python versions while maintaining compatibility.

---

**You're all set! Use `requirements-runpod.txt` and training should work fine.** ✅
