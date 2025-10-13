#!/bin/bash
# Training script for TinyStories dataset
# Following the hyperparameter specifications from the assignment

# This script trains a Transformer language model on TinyStories with:
# - vocab_size: 10000
# - context_length: 256
# - d_model: 512
# - num_layers: 4
# - num_heads: 16
# - d_ff: 1344 (≈2.625×d_model, multiple of 64)
# - Total tokens: 327,680,000
# - Batch size: 32 → 40,000 training steps

uv run python train.py \
  --train_data tokenizer/encode/tinystories_train.npy \
  --val_data tokenizer/encode/tinystories_val.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000.0 \
  --batch_size 32 \
  --max_iterations 40000 \
  --learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_iters 2000 \
  --cosine_cycle_iters 38000 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --grad_clip 1.0 \
  --save_interval 5000 \
  --eval_interval 1000 \
  --log_interval 100 \
  --experiment_name "tinystories_baseline" \
  --experiment_description "TinyStories baseline: 4L/16H/512D, 40K steps, 327M tokens" \
  --experiment_tags tinystories baseline 4layer \
  --device auto \
  --dtype bfloat16
  --wandb \
  --wandb_project "transformer-lm-training" \
  --wandb_run_name "tinystories_test_$(date +%Y%m%d_%H%M%S)"
