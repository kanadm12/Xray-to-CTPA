# Launch VQ-GAN Patch-wise Training on Single GPU
# PowerShell script for Windows

# Set PYTHONPATH to include current directory
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Launching VQ-GAN Patch-wise Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  GPUs: 1x GPU" -ForegroundColor White
Write-Host "  Input Resolution: 512×512×604" -ForegroundColor White
Write-Host "  Patch Size: 256×256×128" -ForegroundColor White
Write-Host "  Overlap: 25% (64 voxels)" -ForegroundColor White
Write-Host "  Batch Size: 4" -ForegroundColor White
Write-Host ""

# Check for GPU availability
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
try {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host ""
} catch {
    Write-Host "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?" -ForegroundColor Red
    exit 1
}

# Launch training
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train/train_vqgan_distributed.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
