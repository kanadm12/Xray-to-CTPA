# Launch Distributed VQ-GAN Training on 4 H200 GPUs
# PowerShell script for Windows

# Set PYTHONPATH to include current directory
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Set distributed training environment variables
$env:MASTER_ADDR = "localhost"
$env:MASTER_PORT = "29500"

# Optional: Enable NCCL debugging (comment out for production)
# $env:NCCL_DEBUG = "INFO"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Launching VQ-GAN Distributed Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  GPUs: 4× H200 (144GB each)" -ForegroundColor White
Write-Host "  Input Resolution: 512×512×604" -ForegroundColor White
Write-Host "  Patch Size: 256×256×128" -ForegroundColor White
Write-Host "  Overlap: 25% (64 voxels)" -ForegroundColor White
Write-Host "  Batch Size: 2 per GPU (8 global)" -ForegroundColor White
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

# Launch training with torchrun
Write-Host "Starting training with torchrun..." -ForegroundColor Green
Write-Host ""

torchrun `
    --nproc_per_node=4 `
    --master_addr=$env:MASTER_ADDR `
    --master_port=$env:MASTER_PORT `
    train/train_vqgan_distributed.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
