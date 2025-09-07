#!/usr/bin/env python3
"""
Training script for Transformer Language Model

This script implements a complete training loop that brings together all components:
- BPE tokenization
- Transformer model
- AdamW optimizer
- Cosine learning rate scheduling
- Gradient clipping
- Checkpointing
- Memory-efficient data loading
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Import our custom implementations
from adamw_optimizer import AdamW
from experiment_tracker import ExperimentTracker
from tests.adapters import (
    TransformerLM,
    run_get_batch,
    run_gradient_clipping, 
    run_get_lr_cosine_schedule,
    run_save_checkpoint,
    run_load_checkpoint,
    run_cross_entropy
)


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy file with token IDs)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data (.npy file with token IDs)")
    
    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=50000,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_iterations", type=int, default=100000,
                        help="Maximum training iterations")
    parser.add_argument("--learning_rate", type=float, default=6e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=6e-5,
                        help="Minimum learning rate for cosine schedule")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=80000,
                        help="Number of iterations for cosine annealing")
    
    # Optimizer hyperparameters
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate on validation set every N iterations")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log training metrics every N iterations")
    
    # Device and precision
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('cpu', 'cuda', 'mps', or 'auto')")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for training ('float32', 'float16', 'bfloat16')")
    
    # Experiment tracking
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment (auto-generated if not provided)")
    parser.add_argument("--experiment_description", type=str, default="",
                        help="Description of the experiment")
    parser.add_argument("--experiment_tags", type=str, nargs="*", default=[],
                        help="Tags for categorizing the experiment")
    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Directory to store experiment logs and results")
    
    # Weights & Biases logging (optional)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the best available device."""
    if device_arg != "auto":
        return device_arg
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def load_data(data_path: str) -> np.ndarray:
    """Load tokenized data using memory mapping for efficiency."""
    print(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Use memory mapping to avoid loading entire dataset into memory
    data = np.memmap(data_path, dtype=np.int32, mode='r')
    print(f"Loaded {len(data):,} tokens")
    
    return data


def calculate_perplexity(model, val_data: np.ndarray, batch_size: int, 
                        context_length: int, device: str, max_batches: int = 100) -> float:
    """Calculate perplexity on validation data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for _ in range(min(max_batches, len(val_data) // (batch_size * context_length))):
            # Get a batch of validation data
            inputs, targets = run_get_batch(val_data, batch_size, context_length, device)
            
            # Forward pass
            logits = model(inputs)  # Shape: [batch_size, context_length, vocab_size]
            
            # Reshape for cross-entropy: flatten batch and sequence dimensions
            logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * context_length, vocab_size]
            targets_flat = targets.view(-1)  # [batch_size * context_length]
            
            # Compute cross-entropy loss
            loss = run_cross_entropy(logits_flat, targets_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return float('inf')
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return perplexity


def main():
    args = parse_args()
    
    # Set up device and data type
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"transformer_{args.d_model}d_{args.num_layers}l_{args.batch_size}b"
    
    # Initialize experiment tracker
    experiment = ExperimentTracker(
        experiment_name=args.experiment_name,
        base_dir=args.experiments_dir,
        description=args.experiment_description,
        tags=args.experiment_tags,
        auto_save_interval=args.log_interval
    )
    
    # Create checkpoint directory (use experiment directory)
    checkpoint_dir = experiment.exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            print("Initialized Weights & Biases logging")
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
            args.wandb = False
    
    # Load training and validation data
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    print(f"Training data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device).to(dtype)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Log hyperparameters and model config to experiment tracker
    experiment.log_hyperparameters({
        'learning_rate': args.learning_rate,
        'min_learning_rate': args.min_learning_rate,
        'warmup_iters': args.warmup_iters,
        'cosine_cycle_iters': args.cosine_cycle_iters,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'grad_clip': args.grad_clip,
        'max_iterations': args.max_iterations,
        'device': device,
        'dtype': str(dtype)
    })
    
    experiment.log_model_config({
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    })
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Initialize training state
    start_iter = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
        start_iter = run_load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Save configuration
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    start_time = time.time()
    running_loss = 0.0
    
    for iteration in range(start_iter, args.max_iterations):
        iter_start_time = time.time()
        
        # Get learning rate for this iteration
        lr = run_get_lr_cosine_schedule(
            iteration,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get training batch
        inputs, targets = run_get_batch(
            train_data, 
            args.batch_size, 
            args.context_length, 
            device
        )
        
        # Forward pass
        logits = model(inputs)  # Shape: [batch_size, context_length, vocab_size]
        
        # Reshape for cross-entropy: flatten batch and sequence dimensions
        logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * context_length, vocab_size]
        targets_flat = targets.view(-1)  # [batch_size * context_length]
        
        loss = run_cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        run_gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Calculate iteration time
        iter_time = time.time() - iter_start_time
        
        # Logging
        if (iteration + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            perplexity = math.exp(avg_loss)
            elapsed_time = time.time() - start_time
            tokens_per_sec = (iteration + 1 - start_iter) * args.batch_size * args.context_length / elapsed_time
            
            # Log to experiment tracker
            experiment.log_metrics(
                step=iteration + 1,
                metrics={
                    'loss': avg_loss,
                    'perplexity': perplexity,
                    'learning_rate': lr,
                    'tokens_per_second': tokens_per_sec,
                    'iter_time_ms': iter_time * 1000
                },
                split='train'
            )
            
            # Optional Weights & Biases logging
            if args.wandb and wandb_run:
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/perplexity": perplexity,
                    "train/learning_rate": lr,
                    "train/tokens_per_second": tokens_per_sec,
                    "iteration": iteration + 1
                })
            
            running_loss = 0.0
        
        # Validation evaluation
        if (iteration + 1) % args.eval_interval == 0:
            print("Evaluating on validation set...")
            val_perplexity = calculate_perplexity(
                model, val_data, args.batch_size, args.context_length, device
            )
            val_loss = math.log(val_perplexity)
            
            # Log validation metrics to experiment tracker
            experiment.log_metrics(
                step=iteration + 1,
                metrics={
                    'loss': val_loss,
                    'perplexity': val_perplexity
                },
                split='val'
            )
            
            # Optional Weights & Biases logging
            if args.wandb and wandb_run:
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity,
                    "iteration": iteration + 1
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / "best_model.pt"
                run_save_checkpoint(model, optimizer, iteration + 1, best_checkpoint_path)
                print(f"Saved new best model to {best_checkpoint_path}")
                
                # Generate loss curves when we find a new best model
                experiment.plot_loss_curves(save_plot=True, show_plot=False)
        
        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration + 1:06d}.pt"
            run_save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping check
        if math.isnan(loss.item()):
            print("Loss became NaN, stopping training")
            break
    
    # Final evaluation
    print("Final evaluation on validation set...")
    final_val_perplexity = calculate_perplexity(
        model, val_data, args.batch_size, args.context_length, device
    )
    final_val_loss = math.log(final_val_perplexity)
    
    print(f"Final Validation | Loss: {final_val_loss:.4f} | PPL: {final_val_perplexity:.2f}")
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    run_save_checkpoint(model, optimizer, args.max_iterations, final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")
    
    total_time = time.time() - start_time
    total_tokens = args.max_iterations * args.batch_size * args.context_length
    
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Processed {total_tokens:,} tokens ({total_tokens / total_time:.0f} tokens/sec)")
    
    # Log final metrics to experiment tracker
    experiment.log_metrics(
        step=args.max_iterations,
        metrics={
            'final_val_loss': final_val_loss,
            'final_val_perplexity': final_val_perplexity,
            'total_time_seconds': total_time,
            'total_tokens': total_tokens,
            'final_tokens_per_second': total_tokens / total_time
        },
        split='final',
        force_save=True
    )
    
    # Finalize experiment (saves all data and generates final plots)
    experiment.finalize(create_plots=True)
    
    # Optional Weights & Biases logging
    if args.wandb and wandb_run:
        wandb_run.log({
            "final/val_loss": final_val_loss,
            "final/val_perplexity": final_val_perplexity,
            "final/total_time": total_time,
            "final/total_tokens": total_tokens
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()