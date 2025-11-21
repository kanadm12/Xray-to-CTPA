# 🚀 RunPod Training: Your Quick Action Plan

## ⚡ TL;DR - Do This Now

After cloning the repo on RunPod, follow these **4 simple steps**:

### Step 0: Check Python Version (Important!)
```bash
python --version
# If Python 3.12 or 3.11, continue to Step 1
# If Python 3.10 or lower, you might need adjustments
```

### Step 1: Upload Your Dataset
- Use RunPod Files GUI to upload `data_new/` folder to `/workspace/datasets/`
- **OR** download from cloud using provided commands
- Verify: `ls -la /workspace/datasets/data_new/`

### Step 2: Run Setup
```bash
cd /workspace/Xray-2CTPA_spartis
bash setup_and_train.sh
```

That's it! The script handles:
- ✅ Installing dependencies
- ✅ Verifying dataset
- ✅ Updating configuration
- ✅ Starting training

### Step 3: Monitor Progress
```bash
# In a new terminal
nvidia-smi -l 1
```

### Step 4: Download Results
When done, download `lightning_logs/` folder via RunPod Files

---

## 📖 When You Need More Details

| Scenario | Read This |
|----------|-----------|
| I want step-by-step walkthrough | `RUNPOD_TRAINING_GUIDE.md` |
| Show me a quick reference | `RUNPOD_QUICK_START.md` |
| I want to see a flowchart/visual | `RUNPOD_VISUAL_GUIDE.md` |
| I need troubleshooting help | See "Phase 5" in `RUNPOD_TRAINING_GUIDE.md` |
| Getting Python/dependency errors | `RUNPOD_FIX_PYTHON_DEPENDENCIES.md` ← **Start here if error** |
| How do I setup my custom dataset? | `CUSTOM_DATASET_TRAINING.md` |

---

## 🎯 Key Points

### What Gets Trained
- ✅ All `.nii.gz` files in your dataset
- ❌ Files containing "swapped" in name (automatically excluded)
- Works with any patient folder structure

### What You Need
- GPU pod running (RTX4090 or better recommended)
- Dataset uploaded
- That's literally all

### Output
- Trained model saved in: `/workspace/Xray-2CTPA_spartis/lightning_logs/version_X/checkpoints/`
- Best checkpoint: `latest_checkpoint.ckpt`

### Time to Start Training
- Setup: ~5-10 minutes
- Training: Depends on dataset size (hours to days)

---

## 🆘 Quick Troubleshooting

**"ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cu118"**
→ See `RUNPOD_FIX_PYTHON_DEPENDENCIES.md` - Use `requirements-runpod.txt` instead

**"Dataset not found"**
→ Make sure it's at `/workspace/datasets/data_new/`

**"CUDA out of memory"**
→ In the training script, change `model.batch_size=2` to `model.batch_size=1`

**"Connection dropped"**
→ Use `tmux new-session -d -s training 'bash setup_and_train.sh'`

**"Training too slow"**
→ Check `nvidia-smi` - GPU usage should be 80%+

For more help, see the guides above.

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `setup_and_train.sh` | 🤖 Auto-setup script (use this!) |
| `train/scripts/train_vqgan_custom.sh` | Training script |
| `config/dataset/custom_data.yaml` | Dataset configuration |
| `RUNPOD_TRAINING_GUIDE.md` | Full detailed guide |
| `RUNPOD_QUICK_START.md` | Quick command reference |
| `RUNPOD_VISUAL_GUIDE.md` | Flowcharts and visuals |

---

## 💡 Pro Tips

1. **Keep monitoring:** Open `nvidia-smi -l 1` in a separate terminal while training
2. **Backup often:** Periodically download `lightning_logs/` in case pod restarts
3. **Save for later:** Once training finishes, save your model somewhere safe
4. **Next step:** After VQGAN training, you can train DDPM (diffusion) model on the latent space

---

## 🎬 One-Command Start (All-in-One)

Copy-paste this if everything is already setup:

```bash
cd /workspace/Xray-2CTPA_spartis && bash setup_and_train.sh
```

---

## 📞 Need Help?

1. **Check the relevant guide** above
2. **Search for your error message** in the troubleshooting section
3. **Re-read the setup steps** - usually a missed step
4. **Check GPU memory:** `nvidia-smi` - is GPU being used?

---

**Happy training! 🎉**
