"""
BPE Tokenizer Training Implementation

This file contains the implementation for training a Byte-Pair Encoding (BPE) tokenizer
from scratch, following the algorithm described in Sennrich et al. (2016).

The BPE algorithm works by:
1. Starting with a vocabulary of all individual bytes (0-255)
2. Pre-tokenizing the text using regex patterns (GPT-2 style)
3. Iteratively finding the most frequent pair of consecutive tokens
4. Merging this pair into a single new token
5. Repeating until desired vocabulary size is reached
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple, BinaryIO
import os
import regex as re
from multiprocessing import Pool, cpu_count

# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def read_text_file(file_path: str) -> str:
    """
    Read the entire content of a text file.
    
    Args:
        file_path: Path to the text file to read
        
    Returns:
        The complete text content as a string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def text_to_bytes(text: str) -> List[int]:
    """
    Convert a text string to a list of byte values.
    
    Args:
        text: Input text string
        
    Returns:
        List of integers representing the UTF-8 byte encoding
    """
    return list(text.encode('utf-8'))


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
    Merge all instances of a specific byte pair in word counts dictionary.
    
    This function finds every occurrence of the specified pair in each word
    and replaces it with a single new token ID, preserving the word counts.
    
    Args:
        word_counts: Dict mapping word tuples to their occurrence counts
        pair: The (byte1, byte2) tuple to merge
        new_token_id: The new token ID to use for the merged pair
        
    Returns:
        Updated word_counts dictionary with the pair merged in all words
    """
    new_word_counts = {}
    
    for word_tuple, count in word_counts.items():
        # Convert tuple to list for easier manipulation
        word_list = list(word_tuple)
        new_word = []
        i = 0
        
        while i < len(word_list):
            # Check if we can form the target pair at current position
            if i < len(word_list) - 1 and word_list[i] == pair[0] and word_list[i + 1] == pair[1]:
                # Found the target pair, replace with new token ID
                new_word.append(new_token_id)
                i += 2  # Skip both elements of the pair
            else:
                # Not a pair or no pair possible, keep current element
                new_word.append(word_list[i])
                i += 1
        
        # Convert back to tuple and preserve count
        new_word_tuple = tuple(new_word)
        new_word_counts[new_word_tuple] = count
    
    return new_word_counts


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], parallel: bool = True, num_processes: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given text corpus using regex pre-tokenization.
    
    The algorithm:
    1. Read the input text and pre-tokenize using regex patterns (with optional parallelization)
    2. Count word frequencies without storing all tokens in memory
    3. Initialize vocabulary with all individual bytes (0-255)
    4. Add special tokens to vocabulary
    5. Iteratively find most frequent byte pairs and merge them
    6. Continue until desired vocabulary size is reached
    
    Args:
        input_path: Path to the training text file
        vocab_size: Maximum vocabulary size (including bytes and special tokens)
        special_tokens: List of special tokens to add to vocabulary
        parallel: Whether to use parallel processing for pre-tokenization (default: True)
        num_processes: Number of processes to use (default: cpu_count())
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: Dict mapping token IDs to their byte representation
        - merges: List of merge operations as (token1_bytes, token2_bytes) tuples
    """
    
    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count()
        print("number of processes to use:", num_processes)
    
    # Step 1: Pre-tokenize text with optional parallelization
    # Check file size - use parallel processing only for larger files
    file_size = os.path.getsize(input_path)
    use_parallel = parallel and num_processes > 1 and file_size > 1024 * 1024  # 1MB threshold
    
    if use_parallel:
        # Parallel processing
        word_counts = Counter()
        
        # Find primary special token for chunking (use first one, or default)
        split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
        split_token_bytes = split_token.encode('utf-8')
        
        # Find chunk boundaries aligned with special tokens
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)
        
        # Only use parallel if we actually get multiple chunks
        if len(boundaries) > 2:  # More than start and end boundary
            # Create arguments for each chunk
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((input_path, start, end, special_tokens))
            
            # Process chunks in parallel
            with Pool(num_processes) as pool:
                chunk_results = pool.map(process_chunk, chunk_args)
            
            # Combine results from all chunks
            for chunk_word_counts in chunk_results:
                for word_tuple, count in chunk_word_counts.items():
                    word_counts[word_tuple] += count
        else:
            # Fall back to serial processing
            use_parallel = False
                
    if not use_parallel:
        # Serial processing (original implementation)
        text = read_text_file(input_path)
        
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
    
    # Step 4: Initialize vocabulary with all possible bytes (0-255)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # Step 5: Add special tokens to vocabulary
    # Special tokens get IDs starting from 256
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    
    # Step 6: Track merge operations
    merges = []
    
    # Step 7: Main BPE training loop
    # Continue until we reach the desired vocabulary size
    while len(vocab) < vocab_size:
        
        # Count all byte pairs in current word counts
        # Note: Pairs don't cross pre-token boundaries since we process each pre-token separately
        pair_counts = get_byte_pair_counts(word_counts)
        
        # If no pairs found, break (shouldn't happen in practice)
        if not pair_counts:
            break
            
        # Find the most frequent pair with deterministic tie-breaking
        # Get the maximum frequency
        max_frequency = pair_counts.most_common(1)[0][1]
        
        # Get all pairs with the maximum frequency
        max_pairs = [(pair, count) for pair, count in pair_counts.items() if count == max_frequency]
        
        # Break ties by preferring lexicographically greater pair based on byte content
        # Convert token IDs to their byte representations for comparison
        def get_pair_bytes(pair):
            token_id1, token_id2 = pair
            bytes1 = vocab[token_id1]
            bytes2 = vocab[token_id2]
            return (bytes1, bytes2)
        
        # Sort by the byte representation of pairs in descending order (lexicographically greater first)
        max_pairs.sort(key=lambda x: get_pair_bytes(x[0]), reverse=True)
        
        # Select the lexicographically greatest pair
        most_frequent_pair = max_pairs[0][0]
        
        # Create new token ID for this merged pair
        new_token_id = next_token_id
        next_token_id += 1
        
        # Add the merged token to vocabulary
        # The merged token's byte representation is the concatenation of the two parts
        pair_bytes1 = vocab[most_frequent_pair[0]]
        pair_bytes2 = vocab[most_frequent_pair[1]]
        vocab[new_token_id] = pair_bytes1 + pair_bytes2
        
        # Record this merge operation
        merges.append((pair_bytes1, pair_bytes2))
        
        # Update word counts by merging this pair
        word_counts = merge_pair_in_word_counts(word_counts, most_frequent_pair, new_token_id)
        
    return vocab, merges

