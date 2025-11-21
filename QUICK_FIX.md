# 🔧 One-Line Fixes for Common RunPod Issues

## Python/Dependency Errors

### The Error You Got:
```
ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cu118
ERROR: Requires-Python >=3.8,<3.12
```

### The Fix:

**Option 1 - Use the new compatible requirements file:**
```bash
cd /workspace/Xray-2CTPA_spartis
pip install -r requirements-runpod.txt
```

**Option 2 - Auto-fix with setup script:**
```bash
cd /workspace/Xray-2CTPA_spartis
bash setup_and_train.sh
```

**Option 3 - Manual fix (if Option 1 doesn't work):**
```bash
pip cache purge && pip install --force-reinstall --no-cache-dir -r requirements-runpod.txt
```

---

## That's It! 

After running one of the above, continue with:
```bash
bash setup_and_train.sh
```

Or manually:
```bash
export PYTHONPATH=$PWD
python train/train_vqgan.py dataset=custom_data model=vq_gan_3d model.gpus=1 model.batch_size=2
```

---

## Why This Works

- **requirements-runpod.txt** = PyTorch 2.1.2 (supports Python 3.9-3.12)
- **requirements.txt** = PyTorch 2.6.0 (supports Python 3.8-3.11)

RunPod often has Python 3.12, so we use the compatible version.

**No quality loss** - Both versions work equally well for VQGAN training!
