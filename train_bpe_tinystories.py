#!/usr/bin/env python3
"""
Train BPE tokenizer on TinyStories dataset.

This script trains a byte-level BPE tokenizer on the TinyStories dataset using a maximum
vocabulary size of 10,000, includes the <|endoftext|> special token, and serializes
the resulting vocabulary and merges to disk.
"""

import time
import json
import os
from bpe_tokenizer import train_bpe


def train_bpe_tinystories(data_path="../data/TinyStoriesV2-GPT4-train.txt", vocab_size=32000):
    """
    Train a BPE tokenizer on the TinyStories dataset.
    
    Args:
        data_path: Path to the TinyStories training data
        vocab_size: Maximum vocabulary size (default: 10000)
        
    Returns:
        tuple: (vocab, merges, training_time_hours, longest_token)
    """
    print(f"Training BPE tokenizer on {data_path}")
    print(f"Target vocabulary size: {vocab_size}")
    
    # Define special tokens (TinyStories uses <|endoftext|> as document delimiter)
    special_tokens = ["<|endoftext|>"]
    
    # Record start time
    start_time = time.time()
    
    # Train the BPE tokenizer with multiprocessing enabled
    print("Starting BPE training...")
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        parallel=True
    )
    
    # Calculate training time
    end_time = time.time()
    training_time_seconds = end_time - start_time
    training_time_hours = training_time_seconds / 3600
    
    print(f"Training completed in {training_time_seconds:.2f} seconds ({training_time_hours:.4f} hours)")
    print(f"Final vocabulary size: {len(vocab)}")
    
    # Find the longest token in the vocabulary
    longest_token = b""
    longest_length = 0
    
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > longest_length:
            longest_length = len(token_bytes)
            longest_token = token_bytes
    
    # Try to decode longest token for inspection
    try:
        longest_token_str = longest_token.decode('utf-8')
        print(f"Longest token: '{longest_token_str}' (length: {longest_length} bytes)")
    except UnicodeDecodeError:
        print(f"Longest token: {longest_token!r} (length: {longest_length} bytes)")
    
    # Serialize vocabulary to JSON
    vocab_file = "tinystories_vocab.json"
    print(f"Saving vocabulary to {vocab_file}")
    
    # Convert vocab to JSON-serializable format (token_str -> token_id)
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        try:
            # Try to decode as UTF-8 for JSON serialization
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # For non-UTF-8 bytes, use Latin-1 encoding to preserve all bytes
            token_str = token_bytes.decode('latin-1')
        vocab_json[token_str] = token_id
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    # Serialize merges to text file
    merges_file = "tinystories_merges.txt"
    print(f"Saving merges to {merges_file}")
    
    with open(merges_file, 'w', encoding='utf-8') as f:
        for merge1, merge2 in merges:
            try:
                # Try to decode as UTF-8
                merge1_str = merge1.decode('utf-8')
                merge2_str = merge2.decode('utf-8')
            except UnicodeDecodeError:
                # Use Latin-1 encoding to preserve all bytes
                merge1_str = merge1.decode('latin-1')
                merge2_str = merge2.decode('latin-1')
            f.write(f"{merge1_str} {merge2_str}\n")
    
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Merges saved to {merges_file}")
    
    return vocab, merges, training_time_hours, longest_token


if __name__ == "__main__":
    # Check if data file exists
    data_file = "../data/owt_valid.txt"
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        exit(1)
    
    # Train the tokenizer
    vocab, merges, training_time, longest_token = train_bpe_tinystories(data_file)
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Training time: {training_time:.4f} hours")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    try:
        longest_str = longest_token.decode('utf-8')
        print(f"Longest token: '{longest_str}' ({len(longest_token)} bytes)")
    except UnicodeDecodeError:
        print(f"Longest token: {longest_token!r} ({len(longest_token)} bytes)")
    
    # Test the tokenizer with a sample
    print("\n=== Testing Tokenizer ===")
    from bpe_tokenizer import Tokenizer
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    
    # Test with a sample story
    sample_text = "Once upon a time there was a little boy named Ben.<|endoftext|>"
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Sample text: {sample_text}")
    print(f"Encoded tokens: {tokens}")
    print(f"Decoded text: {decoded}")
    print(f"Number of tokens: {len(tokens)}")