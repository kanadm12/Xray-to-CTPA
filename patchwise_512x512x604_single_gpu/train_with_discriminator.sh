#!/bin/bash

# Train VQ-GAN with discriminator enabled for better reconstruction quality
# This script enables adversarial training after warmup for sharper, more realistic reconstructions

echo "=========================================="
echo "VQ-GAN Training with Discriminator"
echo "=========================================="

# Configuration
CHECKPOINT="outputs/vqgan_patches_distributed/checkpoints/last.ckpt"  # Resume from existing checkpoint
DATASET_CONFIG="config/dataset/full_resolution_ctpa.yaml"
MODEL_CONFIG="config/model/vq_gan_3d_with_discriminator.yaml"

# Check if resuming from checkpoint
if [ -f "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    RESUME_FLAG="model.resume_from_checkpoint=$CHECKPOINT"
else
    echo "Starting fresh training (no checkpoint found)"
    RESUME_FLAG=""
fi

# Launch training
nohup python -u train/train_vqgan_distributed.py \
    --config-path ../config \
    --config-name base_cfg \
    dataset=full_resolution_ctpa \
    model=vq_gan_3d_with_discriminator \
    dataset.max_patients=30 \
    dataset.patch_size=[128,128,128] \
    dataset.stride=[128,128,128] \
    model.batch_size=1 \
    model.num_workers=0 \
    model.precision=16 \
    model.max_epochs=50 \
    model.discriminator_iter_start=10000 \
    model.image_gan_weight=1.0 \
    model.video_gan_weight=0.5 \
    model.perceptual_weight=0.1 \
    model.gan_feat_weight=4.0 \
    model.learning_rate=1e-4 \
    model.default_root_dir=./outputs/vqgan_patches_with_disc/ \
    $RESUME_FLAG \
    > training_with_disc.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Logs: training_with_disc.log"
echo ""
echo "Monitor training with:"
echo "  tail -f training_with_disc.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo "=========================================="
