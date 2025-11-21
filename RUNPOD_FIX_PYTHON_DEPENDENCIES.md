# Fix: Python Version & Dependencies Issue on RunPod

## 🔴 Problem

When trying to install dependencies, you get errors like:
```
ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cu118
ERROR: Requires-Python >=3.8,<3.12
```

## 🟢 Solution

The issue is that RunPod's default environment might have **Python 3.12**, but the original `requirements.txt` specifies packages for **Python 3.11 and below**.

### Quick Fix (Recommended)

**Option 1: Use the new RunPod requirements file**

```bash
cd /workspace/Xray-2CTPA_spartis
pip install -r requirements-runpod.txt
```

The `requirements-runpod.txt` file is already compatible with Python 3.11/3.12.

**Option 2: Use the auto-setup script**

```bash
bash setup_and_train.sh
```

The script automatically detects and uses the correct requirements file.

### Manual Fix (If you know what you're doing)

If you need to manually install:

```bash
# Check your Python version first
python --version

# If Python 3.12, use requirements-runpod.txt
# If Python 3.11, use requirements-runpod.txt
# If Python 3.10, use requirements.txt

pip install -r requirements-runpod.txt
```

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
