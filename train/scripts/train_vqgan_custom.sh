#!/bin/bash
# Script to train VQGAN on custom dataset

# Set environment variables
export PYTHONPATH=$PWD

# Configuration
DATASET_NAME="custom_data"  # or 'CUSTOM_DATA' - matches your config file
CUDA_DEVICE=0
EMBEDDING_DIM=8
N_HIDDENS=16
DOWNSAMPLE="[2,2,2]"  # Adjust based on your data resolution
NUM_WORKERS=4  # Reduce if you have limited RAM
BATCH_SIZE=2
LEARNING_RATE=3e-4
ACCUMULATE_GRAD_BATCHES=1

# Run training
PL_TORCH_DISTRIBUTED_BACKEND=gloo \
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
python train/train_vqgan.py \
    dataset=$DATASET_NAME \
    model=vq_gan_3d \
    model.gpus=1 \
    model.precision=16 \
    model.embedding_dim=$EMBEDDING_DIM \
    model.n_hiddens=$N_HIDDENS \
    model.downsample=$DOWNSAMPLE \
    model.num_workers=$NUM_WORKERS \
    model.gradient_clip_val=1.0 \
    model.lr=$LEARNING_RATE \
    model.discriminator_iter_start=10000 \
    model.perceptual_weight=4 \
    model.image_gan_weight=1 \
    model.video_gan_weight=1 \
    model.gan_feat_weight=4 \
    model.batch_size=$BATCH_SIZE \
    model.n_codes=16384 \
    model.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
    model.default_root_dir_postfix='custom_data'
