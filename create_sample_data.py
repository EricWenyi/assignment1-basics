#!/usr/bin/env python3
"""
Create sample data for testing the training script.
This generates small synthetic datasets for training and validation.
"""

import numpy as np
import os

def create_sample_data():
    """Create sample training and validation data."""
    
    # Create sample data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # Generate synthetic token sequences
    vocab_size = 1000  # Small vocabulary for testing
    seq_length = 50000  # 50K tokens for training
    val_seq_length = 10000  # 10K tokens for validation
    
    # Generate random token sequences
    np.random.seed(42)  # For reproducibility
    
    # Training data - mix of random tokens with some patterns
    train_tokens = []
    for i in range(seq_length):
        if i % 100 == 0:
            # Add some special tokens periodically
            train_tokens.append(0)  # Special token
        else:
            # Random token from vocabulary
            train_tokens.append(np.random.randint(1, vocab_size))
    
    train_data = np.array(train_tokens, dtype=np.int32)
    
    # Validation data - similar pattern but different seed
    np.random.seed(123)
    val_tokens = []
    for i in range(val_seq_length):
        if i % 100 == 0:
            val_tokens.append(0)  # Special token
        else:
            val_tokens.append(np.random.randint(1, vocab_size))
    
    val_data = np.array(val_tokens, dtype=np.int32)
    
    # Save as .npy files
    train_path = "sample_data/train.npy"
    val_path = "sample_data/val.npy"
    
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    
    print(f"Created training data: {train_path} ({len(train_data):,} tokens)")
    print(f"Created validation data: {val_path} ({len(val_data):,} tokens)")
    print(f"Vocabulary size: {vocab_size}")
    
    # Verify the data can be loaded
    loaded_train = np.load(train_path)
    loaded_val = np.load(val_path)
    
    print(f"Verification - Train shape: {loaded_train.shape}, dtype: {loaded_train.dtype}")
    print(f"Verification - Val shape: {loaded_val.shape}, dtype: {loaded_val.dtype}")
    print(f"Sample train tokens: {loaded_train[:10]}")
    print(f"Sample val tokens: {loaded_val[:10]}")

if __name__ == "__main__":
    create_sample_data()