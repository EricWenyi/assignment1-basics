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
import psutil
from collections import Counter
from bpe_tokenizer import (
    train_bpe, read_text_file, process_chunk, find_chunk_boundaries,
    get_byte_pair_counts, merge_pair_in_word_counts, PAT
)
from multiprocessing import Pool, cpu_count
import regex as re


def train_bpe_with_progress(input_path, vocab_size, special_tokens, parallel=True, num_processes=None):
    """
    Train BPE tokenizer with simple console progress tracking.
    
    This is a modified version of the original train_bpe function that adds
    progress monitoring without external dependencies.
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"🚀 Starting BPE training with progress tracking")
    print(f"📁 Input: {input_path}")
    print(f"🎯 Target vocab: {vocab_size:,}")
    print()
    
    # Step 1: Pre-tokenization
    print("🔄 Phase 1: Pre-tokenization...")
    pre_start = time.time()
    
    file_size = os.path.getsize(input_path)
    use_parallel = parallel and num_processes > 1 and file_size > 1024 * 1024
    
    if use_parallel:
        print(f"   Using {num_processes} processes for parallel pre-tokenization")
        word_counts = Counter()
        split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
        split_token_bytes = split_token.encode('utf-8')
        
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)
        
        if len(boundaries) > 2:
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((input_path, start, end, special_tokens))
            
            with Pool(num_processes) as pool:
                chunk_results = pool.map(process_chunk, chunk_args)
            
            for chunk_word_counts in chunk_results:
                for word_tuple, count in chunk_word_counts.items():
                    word_counts[word_tuple] += count
        else:
            use_parallel = False
    
    if not use_parallel:
        print("   Using serial pre-tokenization")
        text = read_text_file(input_path)
        
        if special_tokens:
            delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
            text_chunks = re.split(delimiter_pattern, text)
        else:
            text_chunks = [text]
        
        word_counts = Counter()
        for chunk in text_chunks:
            if not chunk:
                continue
            for match in re.finditer(PAT, chunk):
                pre_token_bytes = match.group().encode('utf-8')
                word_tuple = tuple(pre_token_bytes)
                word_counts[word_tuple] += 1
    
    pre_time = time.time() - pre_start
    unique_words = len(word_counts)
    total_words = sum(word_counts.values())
    
    print(f"✅ Pre-tokenization done in {pre_time:.1f}s")
    print(f"   Found {unique_words:,} unique words, {total_words:,} total")
    print()
    
    # Step 2: Initialize vocabulary
    print("🏗️  Phase 2: Initializing vocabulary...")
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    
    merges = []
    initial_vocab_size = len(vocab)
    target_merges = vocab_size - initial_vocab_size
    
    print(f"   Initial vocab: {initial_vocab_size} tokens")
    print(f"   Need {target_merges:,} merges")
    print()
    
    # Step 3: BPE training loop with progress
    print("🔥 Phase 3: BPE merge training...")
    training_start = time.time()
    iteration = 0
    last_progress_time = time.time()
    progress_interval = 10  # Print every 10 seconds
    
    while len(vocab) < vocab_size:
        iter_start = time.time()
        iteration += 1
        current_vocab_size = len(vocab)
        
        # Count pairs
        pair_counts = get_byte_pair_counts(word_counts)
        if not pair_counts:
            break
        
        # Find most frequent pair
        max_frequency = pair_counts.most_common(1)[0][1]
        max_pairs = [(pair, count) for pair, count in pair_counts.items() if count == max_frequency]
        
        # Tie-breaking
        def get_pair_bytes(pair):
            token_id1, token_id2 = pair
            bytes1 = vocab[token_id1]
            bytes2 = vocab[token_id2]
            return (bytes1, bytes2)
        
        max_pairs.sort(key=lambda x: get_pair_bytes(x[0]), reverse=True)
        most_frequent_pair = max_pairs[0][0]
        
        # Create new token
        new_token_id = next_token_id
        next_token_id += 1
        
        pair_bytes1 = vocab[most_frequent_pair[0]]
        pair_bytes2 = vocab[most_frequent_pair[1]]
        vocab[new_token_id] = pair_bytes1 + pair_bytes2
        
        merges.append((pair_bytes1, pair_bytes2))
        
        # Update word counts
        word_counts = merge_pair_in_word_counts(word_counts, most_frequent_pair, new_token_id)
        
        # Progress tracking
        iter_time = time.time() - iter_start
        progress = (current_vocab_size - initial_vocab_size) / target_merges * 100
        elapsed = time.time() - training_start
        
        # Time estimation
        if iteration > 10:
            avg_time = elapsed / iteration
            remaining = target_merges - iteration
            eta_seconds = remaining * avg_time
            if eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
        else:
            eta_str = "..."
        
        # Print progress periodically
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval or iteration % 100 == 0:
            memory_mb = psutil.Process().memory_info().rss / 1024**2
            print(f"📊 Step {iteration:6d}/{target_merges:,} | "
                  f"Vocab: {current_vocab_size:,}/{vocab_size:,} | "
                  f"Progress: {progress:5.1f}% | "
                  f"ETA: {eta_str:>6s} | "
                  f"Freq: {max_frequency:,} | "
                  f"Mem: {memory_mb:.0f}MB")
            last_progress_time = current_time
    
    total_time = time.time() - training_start
    print()
    print(f"✅ Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"📊 Final vocab size: {len(vocab):,}")
    print(f"🔗 Total merges: {len(merges):,}")
    print()
    
    return vocab, merges


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
    
    # Train the BPE tokenizer with progress tracking
    vocab, merges = train_bpe_with_progress(
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
    data_file = "../data/TinyStoriesV2-GPT4-train.txt"
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please ensure the TinyStories dataset is available at ../data/")
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