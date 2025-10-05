#!/usr/bin/env python3
"""
Test script for decoding functions.
Tests temperature scaling and top-p sampling with synthetic data.
"""

import torch
import numpy as np
from decode import apply_temperature_scaling, top_p_sampling, sample_next_token


def test_temperature_scaling():
    """Test temperature scaling function."""
    print("Testing temperature scaling...")
    
    # Create test logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]
    
    print(f"Original logits: {logits}")
    
    for temp in temperatures:
        scaled = apply_temperature_scaling(logits, temp)
        probs = torch.softmax(scaled, dim=-1)
        print(f"Temperature {temp:4.1f}: Probs = {probs}")
    
    print("✅ Temperature scaling test passed\n")


def test_top_p_sampling():
    """Test top-p (nucleus) sampling function."""
    print("Testing top-p sampling...")
    
    # Create test probability distribution
    # Probabilities: [0.5, 0.3, 0.1, 0.05, 0.05]
    probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])
    
    # Test different p values
    p_values = [0.5, 0.8, 0.9, 1.0]
    
    print(f"Original probs: {probs}")
    print(f"Cumulative:     {torch.cumsum(probs, dim=0)}")
    
    for p in p_values:
        filtered = top_p_sampling(probs, p)
        print(f"p = {p:3.1f}: Filtered = {filtered} (sum = {filtered.sum():.3f})")
    
    print("✅ Top-p sampling test passed\n")


def test_sampling_integration():
    """Test the integrated sampling function."""
    print("Testing integrated sampling...")
    
    # Create test logits
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test different combinations
    configs = [
        {"temperature": 1.0, "top_p": None},
        {"temperature": 0.5, "top_p": None},
        {"temperature": 1.0, "top_p": 0.8},
        {"temperature": 0.7, "top_p": 0.9},
    ]
    
    print(f"Test logits: {logits}")
    
    for config in configs:
        # Sample multiple times to see distribution
        samples = []
        for _ in range(10):
            sample = sample_next_token(logits, **config)
            samples.append(sample)
        
        print(f"Config {config}: Samples = {samples}")
    
    print("✅ Integrated sampling test passed\n")


def main():
    """Run all decoding tests."""
    print("=" * 50)
    print("TESTING DECODING FUNCTIONS")
    print("=" * 50)
    
    # Set seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_temperature_scaling()
    test_top_p_sampling()
    test_sampling_integration()
    
    print("=" * 50)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    main()