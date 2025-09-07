#!/usr/bin/env python3
"""
Text Generation and Decoding for Transformer Language Model

This module implements advanced decoding strategies including:
- Temperature scaling for controlling randomness
- Top-p (nucleus) sampling for quality improvement
- Interactive text completion from prompts

Key concepts:
- Temperature τ: Controls randomness in sampling (lower = more deterministic)
- Top-p sampling: Only samples from tokens that comprise top p% of probability mass
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import our implementations
from adamw_optimizer import AdamW
from tests.adapters import (
    TransformerLM,
    run_load_checkpoint
)
from bpe_tokenizer import Tokenizer


def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling to logits before softmax.
    
    Temperature scaling modifies the softmax distribution:
    - temperature > 1.0: Makes distribution more uniform (more random)
    - temperature < 1.0: Makes distribution more peaked (more deterministic)
    - temperature → 0: Approaches argmax (completely deterministic)
    - temperature = 1.0: No change (standard softmax)
    
    Mathematical formula:
    softmax(v, τ)_i = exp(v_i/τ) / Σ_j exp(v_j/τ)
    
    Args:
        logits: Raw model outputs, shape [vocab_size]
        temperature: Temperature parameter τ (must be positive)
        
    Returns:
        Temperature-scaled logits, same shape as input
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Divide logits by temperature before applying softmax
    # This effectively "sharpens" (temp < 1) or "flattens" (temp > 1) the distribution
    return logits / temperature


def top_p_sampling(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to probability distribution.
    
    Top-p sampling improves text quality by:
    1. Sorting tokens by probability (highest first)
    2. Finding the smallest set V(p) where Σ_{i∈V(p)} prob_i ≥ p  
    3. Setting probabilities outside V(p) to zero
    4. Renormalizing the remaining probabilities
    
    This removes the "tail" of low-probability tokens that often generate
    nonsensical text, while keeping enough diversity for natural generation.
    
    Args:
        probs: Probability distribution from softmax, shape [vocab_size]
        p: Nucleus parameter (0 < p ≤ 1), e.g. p=0.9 keeps top 90% probability mass
        
    Returns:
        Modified probability distribution, same shape as input
    """
    if not (0 < p <= 1):
        raise ValueError(f"Top-p value must be in (0, 1], got {p}")
    
    # Step 1: Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Step 2: Calculate cumulative probabilities
    # cumsum[i] = sum of probabilities from rank 0 to rank i
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Step 3: Find cutoff point where cumulative probability >= p
    # We want the smallest set V(p) such that Σ_{i∈V(p)} prob_i ≥ p
    cutoff_mask = cumulative_probs <= p
    
    # Ensure we keep at least one token (the highest probability one)
    cutoff_mask[0] = True
    
    # Step 4: Zero out probabilities below the cutoff
    filtered_probs = torch.zeros_like(probs)
    filtered_probs[sorted_indices[cutoff_mask]] = sorted_probs[cutoff_mask]
    
    # Step 5: Renormalize to make it a valid probability distribution
    # After filtering, probabilities may not sum to 1, so we normalize
    total_prob = filtered_probs.sum()
    if total_prob > 0:
        filtered_probs = filtered_probs / total_prob
    
    return filtered_probs


def sample_next_token(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    top_p: Optional[float] = None
) -> int:
    """
    Sample the next token from model logits using temperature and top-p sampling.
    
    Pipeline:
    1. Apply temperature scaling to logits
    2. Convert to probabilities via softmax  
    3. Apply top-p filtering if specified
    4. Sample from the resulting distribution
    
    Args:
        logits: Raw model outputs for vocabulary, shape [vocab_size]
        temperature: Temperature for scaling (default: 1.0 = no scaling)
        top_p: Top-p threshold for nucleus sampling (default: None = no filtering)
        
    Returns:
        Token ID of the sampled token
    """
    # Step 1: Apply temperature scaling
    if temperature != 1.0:
        logits = apply_temperature_scaling(logits, temperature)
    
    # Step 2: Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)
    
    # Step 3: Apply top-p filtering if specified
    if top_p is not None:
        probs = top_p_sampling(probs, top_p)
    
    # Step 4: Sample from the probability distribution
    # torch.multinomial samples indices according to the probability weights
    token_id = torch.multinomial(probs, num_samples=1).item()
    
    return token_id


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    end_token: str = "<|endoftext|>",
    device: str = "cpu"
) -> str:
    """
    Generate text completion given a prompt using the trained language model.
    
    This function implements autoregressive generation:
    1. Encode the prompt into token IDs
    2. For each new token position:
       a. Run forward pass through model
       b. Get logits for next token
       c. Apply temperature scaling and top-p sampling
       d. Sample next token
       e. Add token to sequence
    3. Stop when hitting end token or max length
    4. Decode final sequence back to text
    
    Args:
        model: Trained transformer language model
        tokenizer: BPE tokenizer for encoding/decoding
        prompt: Input text to complete
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling control
        top_p: Top-p threshold for nucleus sampling
        end_token: Token that signals end of generation
        device: Device to run inference on
        
    Returns:
        Complete text (prompt + generated completion)
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    # Step 1: Encode prompt to token IDs
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("Empty prompt after tokenization")
    
    # Convert to tensor and move to device
    # Shape: [sequence_length]
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    
    # Get special token IDs
    end_token_id = None
    if end_token in tokenizer.special_tokens:
        end_token_id = tokenizer.special_tokens[end_token]
    
    print(f"Generating completion for: '{prompt}'")
    print(f"Settings - Temperature: {temperature}, Top-p: {top_p}")
    print(f"Initial tokens: {len(prompt_tokens)}, Max new tokens: {max_new_tokens}")
    print("-" * 60)
    
    # Step 2: Autoregressive generation loop
    generated_tokens = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for step in range(max_new_tokens):
            # Ensure we don't exceed model's context length
            context_length = model.context_length
            if len(input_ids) >= context_length:
                # Take the most recent (context_length - 1) tokens to leave room for next prediction
                input_ids = input_ids[-(context_length - 1):]
            
            # Step 2a: Forward pass through model
            # Add batch dimension: [sequence_length] -> [1, sequence_length]
            input_batch = input_ids.unsqueeze(0)
            
            # Get logits: [1, sequence_length, vocab_size]
            logits = model(input_batch)
            
            # Step 2b: Get logits for the last position (next token prediction)
            # Shape: [vocab_size]
            next_token_logits = logits[0, -1, :]
            
            # Step 2c & 2d: Apply sampling strategy and sample next token
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p
            )
            
            # Step 2e: Add sampled token to sequence
            input_ids = torch.cat([input_ids, torch.tensor([next_token_id], device=device)])
            generated_tokens.append(next_token_id)
            
            # Print progress (decode current token to show what was generated)
            try:
                current_token_text = tokenizer.decode([next_token_id])
                print(f"Step {step+1:3d}: Token {next_token_id:5d} -> '{current_token_text}'")
            except:
                print(f"Step {step+1:3d}: Token {next_token_id:5d} -> [decode error]")
            
            # Step 3: Check for early stopping
            if end_token_id is not None and next_token_id == end_token_id:
                print(f"Generated end token, stopping early at step {step+1}")
                break
    
    # Step 4: Decode complete sequence back to text
    all_tokens = prompt_tokens + generated_tokens
    generated_text = tokenizer.decode(all_tokens)
    
    print("-" * 60)
    print(f"Generation complete! Generated {len(generated_tokens)} new tokens.")
    
    return generated_text


def main():
    """Main function for interactive text generation."""
    parser = argparse.ArgumentParser(description="Generate text with trained Transformer model")
    
    # Model and data arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Path to BPE vocabulary file (.json)")
    parser.add_argument("--merges_file", type=str, required=True,
                        help="Path to BPE merges file (.txt)")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="Special tokens for tokenizer")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                        help="Input prompt for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p threshold for nucleus sampling")
    parser.add_argument("--end_token", type=str, default="<|endoftext|>",
                        help="Token that signals end of generation")
    
    # Model architecture (needed to initialize model)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for inference ('cpu', 'cuda', 'mps', or 'auto')")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading BPE tokenizer...")
    tokenizer = Tokenizer.from_files(
        args.vocab_file, 
        args.merges_file, 
        special_tokens=args.special_tokens
    )
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    
    # Initialize model with same architecture as training
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    # Load trained weights
    print(f"Loading model weights from {args.model_checkpoint}...")
    # Create dummy optimizer for checkpoint loading (we only need model weights)
    dummy_optimizer = AdamW(model.parameters())
    iteration = run_load_checkpoint(args.model_checkpoint, model, dummy_optimizer)
    print(f"Loaded model from iteration {iteration}")
    
    # Generate text
    print("=" * 80)
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token=args.end_token,
        device=device
    )
    
    # Print final result
    print("=" * 80)
    print("FINAL GENERATED TEXT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)


if __name__ == "__main__":
    main()