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
import pickle
import hashlib
from collections import Counter
from pathlib import Path
# All training utilities are now defined in this file
# Import regex pattern from our new tokenizer module  
from tokenizer.tokenizer import PAT
from multiprocessing import Pool, cpu_count
import regex as re
from typing import Dict, List, Tuple, BinaryIO


# ============================================================================
# BPE Training Core Functions
# ============================================================================

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    
    This function finds chunk boundaries that align with special token boundaries
    to ensure no merging occurs across special tokens.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args: Tuple[str, int, int, List[str]]) -> Counter:
    """
    Process a single chunk of the file and return word counts.
    
    Args:
        args: Tuple containing (file_path, start_byte, end_byte, special_tokens)
        
    Returns:
        Counter of word tuples to their counts
    """
    file_path, start, end, special_tokens = args
    
    # Read the chunk
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    # Split on special tokens to prevent cross-boundary merging
    if special_tokens:
        delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(delimiter_pattern, chunk_text)
    else:
        text_chunks = [chunk_text]
    
    # Pre-tokenize each sub-chunk and count words
    word_counts = Counter()
    for text_chunk in text_chunks:
        if not text_chunk:
            continue
            
        for match in re.finditer(PAT, text_chunk):
            pre_token_bytes = match.group().encode('utf-8')
            word_tuple = tuple(pre_token_bytes)
            word_counts[word_tuple] += 1
    
    return word_counts


def read_text_file(file_path: str, special_tokens: List[str] = []) -> Counter:
    """
    Read and pre-tokenize a text file, returning word counts.
    
    Args:
        file_path: Path to the text file to read
        special_tokens: List of special tokens for splitting
        
    Returns:
        Counter of word tuples to their counts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text on special tokens to prevent cross-boundary merging
    if special_tokens:
        delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(delimiter_pattern, text)
    else:
        text_chunks = [text]
    
    # Pre-tokenize each chunk separately using regex and count word frequencies
    word_counts = Counter()
    
    for chunk in text_chunks:
        if not chunk:
            continue
            
        for match in re.finditer(PAT, chunk):
            pre_token_bytes = match.group().encode('utf-8')
            word_tuple = tuple(pre_token_bytes)
            word_counts[word_tuple] += 1
    
    return word_counts


def get_byte_pair_counts(word_counts: Dict[Tuple[int, ...], int]) -> Counter:
    """
    Count all consecutive byte pairs in the tokenized text using word counts.
    
    This function examines each unique word (represented as a tuple of bytes) and 
    counts how many times each consecutive pair of bytes appears, weighted by
    the frequency of each word.
    
    Args:
        word_counts: Dict mapping word tuples to their occurrence counts
        
    Returns:
        Counter object mapping (byte1, byte2) tuples to their frequencies
    """
    pair_counts = Counter()
    
    # Iterate through each unique word and its count
    for word_tuple, count in word_counts.items():
        # For each word, iterate through consecutive pairs of bytes
        for i in range(len(word_tuple) - 1):
            # Create a pair from current byte and next byte
            pair = (word_tuple[i], word_tuple[i + 1])
            # Count each pair weighted by word frequency
            pair_counts[pair] += count
    
    return pair_counts


def merge_pair_in_word_counts(word_counts: Dict[Tuple[int, ...], int], pair: Tuple[int, int], new_token_id: int) -> Dict[Tuple[int, ...], int]:
    """
    OPTIMIZED: Merge all instances of a specific byte pair in word counts dictionary.
    
    This function finds every occurrence of the specified pair in each word
    and replaces it with a single new token ID, preserving the word counts.
    
    PERFORMANCE OPTIMIZATIONS:
    - Skip words that don't contain the target pair (fast path)
    - Use in-place list operations to reduce memory allocations
    - Minimize tuple/list conversions
    
    Args:
        word_counts: Dict mapping word tuples to their occurrence counts
        pair: The (byte1, byte2) tuple to merge
        new_token_id: The new token ID to use for the merged pair
        
    Returns:
        Updated word_counts dictionary with the pair merged in all words
    """
    new_word_counts = {}
    pair_a, pair_b = pair
    
    for word_tuple, count in word_counts.items():
        # OPTIMIZATION: Quick check for words that can't contain the pair
        if len(word_tuple) < 2 or pair_a not in word_tuple or pair_b not in word_tuple:
            new_word_counts[word_tuple] = count
            continue
        
        # OPTIMIZATION: More sophisticated pair existence check
        contains_pair = False
        for i in range(len(word_tuple) - 1):
            if word_tuple[i] == pair_a and word_tuple[i + 1] == pair_b:
                contains_pair = True
                break
        
        if not contains_pair:
            # FAST PATH: Word doesn't contain pair, keep as-is
            new_word_counts[word_tuple] = count
            continue
        
        # SLOW PATH: Word contains pair, perform merge with in-place operations
        word_list = list(word_tuple)
        i = 0
        
        while i < len(word_list) - 1:
            if word_list[i] == pair_a and word_list[i + 1] == pair_b:
                # OPTIMIZATION: Replace pair with new token using list slicing
                word_list[i:i+2] = [new_token_id]
                # No need to increment i since we just shortened the list
            else:
                i += 1
        
        new_word_counts[tuple(word_list)] = count
    
    return new_word_counts


def update_pair_counts_after_merge(
    pair_counts: Counter, 
    word_counts: Dict[Tuple[int, ...], int], 
    old_word_counts: Dict[Tuple[int, ...], int],
    merged_pair: Tuple[int, int], 
    new_token_id: int
) -> Counter:
    """
    Incrementally update pair counts after a merge operation.
    
    This is the key optimization: instead of recalculating all pair counts,
    we only update the counts for pairs that are affected by the merge.
    
    Args:
        pair_counts: Current pair counts to update
        word_counts: New word counts after merge
        old_word_counts: Word counts before merge  
        merged_pair: The pair that was merged (A, B)
        new_token_id: The new token ID C that replaced (A, B)
        
    Returns:
        Updated pair_counts
    """
    a, b = merged_pair
    c = new_token_id
    
    # Find all words that were affected by the merge
    affected_old_words = set()
    affected_new_words = set()
    
    # Collect old words that contained the merged pair
    for word_tuple in old_word_counts:
        if len(word_tuple) >= 2:
            for i in range(len(word_tuple) - 1):
                if word_tuple[i] == a and word_tuple[i + 1] == b:
                    affected_old_words.add(word_tuple)
                    break
    
    # Collect corresponding new words
    for word_tuple in word_counts:
        if c in word_tuple:
            affected_new_words.add(word_tuple)
    
    # Remove pair counts from old affected words
    for word_tuple in affected_old_words:
        count = old_word_counts[word_tuple]
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] -= count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
    
    # Add pair counts from new affected words
    for word_tuple in affected_new_words:
        count = word_counts[word_tuple]
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += count
    
    return pair_counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], parallel: bool = True, num_processes: int = None, show_progress: bool = True) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer with optimized algorithm and optional progress tracking.
    
    This unified function combines the best features:
    - Optimized incremental pair count updates (fast performance)
    - Optional progress tracking with ETA and memory monitoring
    - Pre-tokenization caching for faster re-runs
    - Memory-efficient streaming for large files
    - Parallel processing for faster pre-tokenization
    
    Args:
        input_path: Path to the training text file
        vocab_size: Maximum vocabulary size (including bytes and special tokens)
        special_tokens: List of special tokens to add to vocabulary
        parallel: Whether to use parallel processing for pre-tokenization (default: True)
        num_processes: Number of processes to use (default: cpu_count())
        show_progress: Whether to show progress tracking (default: True)
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: Dict mapping token IDs to their byte representation
        - merges: List of merge operations as (token1_bytes, token2_bytes) tuples
    """
    import time
    import psutil
    
    if num_processes is None:
        num_processes = cpu_count()
    
    if show_progress:
        print(f"🚀 Starting BPE training")
        print(f"📁 Input: {input_path}")
        print(f"🎯 Target vocab: {vocab_size:,}")
        print()
    
    # Step 1: Pre-tokenization with caching
    if show_progress:
        print("🔄 Phase 1: Pre-tokenization...")
    pre_start = time.time()
    
    # Check for cached pre-tokenization results
    word_counts = load_pretokenization_cache(input_path, special_tokens)
    
    if word_counts is not None:
        # Cache hit! Use cached results
        pre_time = time.time() - pre_start
        if show_progress:
            print(f"⚡ Using cached pre-tokenization results (loaded in {pre_time:.1f}s)")
    else:
        # Cache miss, need to process file
        if show_progress:
            print("🔄 No valid cache found, processing file...")
        processing_start = time.time()
        
        file_size = os.path.getsize(input_path)
        file_size_gb = file_size / (1024**3)
        
        # Use streaming for large files (>5GB) to avoid memory issues
        if file_size_gb > 5.0:
            if show_progress:
                print(f"   Large file detected ({file_size_gb:.1f}GB), using memory-efficient streaming")
            word_counts = stream_process_file(input_path, special_tokens, chunk_size_mb=50)
        elif parallel and num_processes > 1 and file_size > 1024 * 1024:
            if show_progress:
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
                # Fall back to streaming for safety
                word_counts = stream_process_file(input_path, special_tokens, chunk_size_mb=100)
        else:
            if show_progress:
                print("   Using streaming processing for small files")
            word_counts = stream_process_file(input_path, special_tokens, chunk_size_mb=100)
        
        processing_time = time.time() - processing_start
        pre_time = time.time() - pre_start
        
        # Save to cache for future runs
        save_pretokenization_cache(input_path, special_tokens, word_counts, processing_time)
    
    unique_words = len(word_counts)
    total_words = sum(word_counts.values())
    
    if show_progress:
        print(f"✅ Pre-tokenization done in {pre_time:.1f}s")
        print(f"   Found {unique_words:,} unique words, {total_words:,} total")
        print()
    
    # Step 2: Initialize vocabulary
    if show_progress:
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
    
    if show_progress:
        print(f"   Initial vocab: {initial_vocab_size} tokens")
        print(f"   Need {target_merges:,} merges")
        print()
    
    # Step 3: OPTIMIZED BPE training loop with incremental pair count updates
    if show_progress:
        print("🔥 Phase 3: BPE merge training (OPTIMIZED)...")
    training_start = time.time()
    iteration = 0
    last_progress_time = time.time()
    progress_interval = 10  # Print every 10 seconds
    
    # KEY OPTIMIZATION: Initialize pair counts cache once, then update incrementally
    if show_progress:
        print("   Initializing pair counts cache...")
    pair_counts = get_byte_pair_counts(word_counts)
    if show_progress:
        print(f"   Cached {len(pair_counts):,} unique pairs")
    
    while len(vocab) < vocab_size:
        iter_start = time.time()
        iteration += 1
        current_vocab_size = len(vocab)
        
        # NO MORE EXPENSIVE RECALCULATION! Use cached pair counts
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
        
        # Store old word counts for incremental update
        old_word_counts = dict(word_counts)
        
        # Create new token
        new_token_id = next_token_id
        next_token_id += 1
        
        pair_bytes1 = vocab[most_frequent_pair[0]]
        pair_bytes2 = vocab[most_frequent_pair[1]]
        vocab[new_token_id] = pair_bytes1 + pair_bytes2
        
        merges.append((pair_bytes1, pair_bytes2))
        
        # Update word counts
        word_counts = merge_pair_in_word_counts(word_counts, most_frequent_pair, new_token_id)
        
        # KEY OPTIMIZATION: Incrementally update pair counts (much faster!)
        pair_counts = update_pair_counts_after_merge(
            pair_counts, word_counts, old_word_counts, most_frequent_pair, new_token_id
        )
        
        # Progress tracking (only if enabled)
        if show_progress:
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
    
    if show_progress:
        total_time = time.time() - training_start
        print()
        print(f"✅ Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"📊 Final vocab size: {len(vocab):,}")
        print(f"🔗 Total merges: {len(merges):,}")
        print()
    
    return vocab, merges


# ============================================================================
# Caching Functions
# ============================================================================

def get_file_hash(file_path):
    """Generate a hash of the file content for cache validation."""
    print(f"   Computing file hash for cache validation...")
    hash_md5 = hashlib.md5()
    
    # For large files, hash first and last chunks + file size for speed
    file_size = os.path.getsize(file_path)
    
    with open(file_path, "rb") as f:
        # Hash file size
        hash_md5.update(str(file_size).encode())
        
        # Hash first 1MB
        chunk = f.read(1024 * 1024)
        hash_md5.update(chunk)
        
        # Hash last 1MB (if file is large enough)
        if file_size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)  # Seek to 1MB before end
            chunk = f.read(1024 * 1024)
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def get_cache_path(file_path, special_tokens):
    """Generate cache file path based on input file and configuration."""
    cache_dir = Path("tokenizer/cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache filename based on input file and special tokens
    file_name = Path(file_path).name
    special_tokens_str = "_".join(special_tokens) if special_tokens else "none"
    cache_name = f"{file_name}_{special_tokens_str}_pretokenized.pkl"
    
    return cache_dir / cache_name


def save_pretokenization_cache(file_path, special_tokens, word_counts, processing_time):
    """Save pre-tokenization results to cache."""
    cache_path = get_cache_path(file_path, special_tokens)
    
    # Get file hash for validation
    file_hash = get_file_hash(file_path)
    
    cache_data = {
        'file_path': file_path,
        'file_hash': file_hash,
        'file_size': os.path.getsize(file_path),
        'special_tokens': special_tokens,
        'word_counts': word_counts,
        'processing_time': processing_time,
        'cache_created': time.time(),
        'unique_words': len(word_counts),
        'total_words': sum(word_counts.values())
    }
    
    print(f"💾 Saving pre-tokenization cache to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"   Cache saved: {len(word_counts):,} unique words, {sum(word_counts.values()):,} total")


def load_pretokenization_cache(file_path, special_tokens):
    """Load pre-tokenization results from cache if valid."""
    cache_path = get_cache_path(file_path, special_tokens)
    
    if not cache_path.exists():
        print(f"📂 No cache found at {cache_path}")
        return None
    
    print(f"📂 Found pre-tokenization cache: {cache_path}")
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache
        print(f"   Validating cache...")
        
        # Check if file path matches
        if cache_data['file_path'] != file_path:
            print(f"   ❌ Cache invalid: file path mismatch")
            return None
        
        # Check if special tokens match
        if cache_data['special_tokens'] != special_tokens:
            print(f"   ❌ Cache invalid: special tokens mismatch")
            return None
        
        # Check if file size matches
        current_size = os.path.getsize(file_path)
        if cache_data['file_size'] != current_size:
            print(f"   ❌ Cache invalid: file size changed ({cache_data['file_size']} -> {current_size})")
            return None
        
        # Check file hash for content changes
        current_hash = get_file_hash(file_path)
        if cache_data['file_hash'] != current_hash:
            print(f"   ❌ Cache invalid: file content changed")
            return None
        
        # Cache is valid
        cache_age = time.time() - cache_data['cache_created']
        print(f"   ✅ Cache valid! Created {cache_age/3600:.1f} hours ago")
        print(f"   📊 Cached: {cache_data['unique_words']:,} unique words, {cache_data['total_words']:,} total")
        print(f"   ⚡ Original processing took {cache_data['processing_time']:.1f}s")
        
        return cache_data['word_counts']
        
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        return None


def list_cache_files():
    """List all cached pre-tokenization files."""
    cache_dir = Path("tokenizer/cache")
    if not cache_dir.exists():
        print("No cache directory found")
        return []
    
    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        print("No cache files found")
        return []
    
    print(f"📂 Found {len(cache_files)} cache files:")
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            file_size_mb = cache_data['file_size'] / (1024**2)
            cache_age = time.time() - cache_data['cache_created']
            
            print(f"  📄 {cache_file.name}")
            print(f"     Original file: {cache_data['file_path']} ({file_size_mb:.1f}MB)")
            print(f"     Created: {cache_age/3600:.1f} hours ago")
            print(f"     Words: {cache_data['unique_words']:,} unique, {cache_data['total_words']:,} total")
            print(f"     Processing time: {cache_data['processing_time']:.1f}s")
            print()
        except Exception as e:
            print(f"  ❌ {cache_file.name} (corrupted: {e})")
    
    return cache_files


def clean_cache(max_age_hours=168):  # Default: 1 week
    """Clean old cache files."""
    cache_dir = Path("tokenizer/cache")
    if not cache_dir.exists():
        return
    
    cache_files = list(cache_dir.glob("*.pkl"))
    current_time = time.time()
    cleaned_count = 0
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            cache_age = current_time - cache_data['cache_created']
            age_hours = cache_age / 3600
            
            if age_hours > max_age_hours:
                cache_file.unlink()
                print(f"🗑️  Cleaned old cache: {cache_file.name} (age: {age_hours:.1f}h)")
                cleaned_count += 1
                
        except Exception as e:
            # Remove corrupted cache files
            cache_file.unlink()
            print(f"🗑️  Removed corrupted cache: {cache_file.name}")
            cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} cache files")
    else:
        print("✅ No cache files needed cleaning")


def stream_process_file(file_path, special_tokens, chunk_size_mb=100):
    """
    Memory-efficient streaming file processing that doesn't load entire file into memory.
    
    Args:
        file_path: Path to input file
        special_tokens: List of special tokens
        chunk_size_mb: Size of chunks to process at a time (MB)
    
    Returns:
        Counter of word frequencies
    """
    print(f"   Using streaming processing (chunks of {chunk_size_mb}MB)")
    
    word_counts = Counter()
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # Prepare special token regex pattern
    if special_tokens:
        delimiter_pattern = "|".join(re.escape(token) for token in special_tokens)
    else:
        delimiter_pattern = None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = ""
        chunk_num = 0
        
        while True:
            # Read chunk
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                # Process final buffer
                if buffer:
                    process_text_chunk(buffer, word_counts, delimiter_pattern)
                break
            
            chunk_num += 1
            if chunk_num % 10 == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024**2
                print(f"   Processed chunk {chunk_num}, Memory: {memory_mb:.0f}MB, Words: {len(word_counts):,}")
            
            # Add to buffer
            buffer += chunk
            
            # Find last complete line to avoid splitting words
            last_newline = buffer.rfind('\n')
            if last_newline != -1:
                # Process complete lines
                complete_text = buffer[:last_newline]
                buffer = buffer[last_newline + 1:]
                
                process_text_chunk(complete_text, word_counts, delimiter_pattern)
            
            # If buffer gets too large (no newlines), process anyway
            if len(buffer) > chunk_size_bytes * 2:
                process_text_chunk(buffer, word_counts, delimiter_pattern)
                buffer = ""
    
    return word_counts


def process_text_chunk(text, word_counts, delimiter_pattern):
    """Process a chunk of text and update word counts."""
    if not text.strip():
        return
    
    # Split on special tokens if needed
    if delimiter_pattern:
        text_chunks = re.split(delimiter_pattern, text)
    else:
        text_chunks = [text]
    
    # Pre-tokenize each chunk
    for chunk in text_chunks:
        if not chunk:
            continue
        
        for match in re.finditer(PAT, chunk):
            try:
                pre_token_bytes = match.group().encode('utf-8')
                word_tuple = tuple(pre_token_bytes)
                word_counts[word_tuple] += 1
            except (UnicodeEncodeError, MemoryError):
                # Skip problematic tokens
                continue


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
    
    # Train the BPE tokenizer with unified optimized function
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        parallel=True,
        show_progress=True
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
    vocab_file = "tokenizer/outputs/tinystories_vocab.json"
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
    merges_file = "tokenizer/outputs/tinystories_merges.txt"
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
    
    # Also save in native Python format (pickle) for exact type preservation
    vocab_pickle_file = "tokenizer/outputs/tinystories_vocab.pkl"
    merges_pickle_file = "tokenizer/outputs/tinystories_merges.pkl"
    
    with open(vocab_pickle_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(merges_pickle_file, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"Native formats saved: {vocab_pickle_file}, {merges_pickle_file}")
    
    return vocab, merges, training_time_hours, longest_token


if __name__ == "__main__":
    # Check if data file exists
    data_file = "../data/TinyStoriesV2-GPT4-valid.txt"
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
    from tokenizer.tokenizer import Tokenizer
    
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    
    # Test with a sample story
    sample_text = "Once upon a time there was a little boy named Ben.<|endoftext|>"
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Sample text: {sample_text}")
    print(f"Encoded tokens: {tokens}")
    print(f"Decoded text: {decoded}")
    print(f"Number of tokens: {len(tokens)}")