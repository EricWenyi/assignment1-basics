#!/bin/bash
# Local validation script for testing training infrastructure
# Uses small subset of data for quick iteration

uv run python train.py \
  --train_data tokenizer/encode/tinystories_mini_train.npy \
  --val_data tokenizer/encode/tinystories_mini_val.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000.0 \
  --batch_size 32 \
  --max_iterations 500 \
  --learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_iters 50 \
  --cosine_cycle_iters 450 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --grad_clip 1.0 \
  --save_interval 200 \
  --eval_interval 50 \
  --log_interval 10 \
  --experiment_name "local_test" \
  --experiment_description "Local test run with mini dataset to validate training infrastructure" \
  --experiment_tags local test validation \
  --device auto \
  --dtype bfloat16 \
  --wandb \
  --wandb_project "transformer-lm-training" \
  --wandb_run_name "local_test_$(date +%Y%m%d_%H%M%S)"
