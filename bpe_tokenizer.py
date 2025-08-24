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
from typing import Dict, List, Tuple, BinaryIO, Iterable, Iterator
import os
import regex as re
from multiprocessing import Pool, cpu_count
import json

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


class Tokenizer:
    """
    BPE Tokenizer class for encoding and decoding text using trained vocabulary and merges.
    
    This class implements the BPE (Byte-Pair Encoding) tokenization algorithm for both
    encoding text to token IDs and decoding token IDs back to text.
    """
    
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Initialize the tokenizer with vocabulary, merges, and optional special tokens.
        
        Args:
            vocab: Dictionary mapping token IDs to their byte representations
            merges: List of merge operations as (token1_bytes, token2_bytes) tuples
            special_tokens: Optional list of special token strings to add to vocabulary
        """
        # Store the vocabulary (token_id -> bytes)
        self.vocab = vocab.copy()
        
        # Create reverse vocabulary (bytes -> token_id) for encoding
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        
        # Store merges in order
        self.merges = merges.copy()
        
        # Create a mapping from merge pairs to their merged result for fast lookup
        self.merge_map = {}
        
        # Process merges in order to build merge_map
        next_token_id = max(self.vocab.keys()) + 1
        for merge_bytes1, merge_bytes2 in self.merges:
            # Find the token IDs for the two parts being merged
            if merge_bytes1 in self.vocab_reverse and merge_bytes2 in self.vocab_reverse:
                token_id1 = self.vocab_reverse[merge_bytes1]
                token_id2 = self.vocab_reverse[merge_bytes2]
                
                # Find the merged result in vocabulary
                merged_bytes = merge_bytes1 + merge_bytes2
                if merged_bytes in self.vocab_reverse:
                    merged_token_id = self.vocab_reverse[merged_bytes]
                    self.merge_map[(token_id1, token_id2)] = merged_token_id
        
        # Add special tokens to vocabulary if provided
        if special_tokens:
            for special_token in special_tokens:
                special_bytes = special_token.encode('utf-8')
                if special_bytes not in self.vocab_reverse:
                    # Add new special token
                    self.vocab[next_token_id] = special_bytes
                    self.vocab_reverse[special_bytes] = next_token_id
                    next_token_id += 1
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        Create a tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to JSON file containing the vocabulary
            merges_filepath: Path to text file containing the merges  
            special_tokens: Optional list of special token strings
            
        Returns:
            Tokenizer instance
        """
        # Load vocabulary from JSON file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        # Convert vocabulary from string keys to int keys and string values to bytes
        vocab = {}
        for token_str, token_id in vocab_json.items():
            token_bytes = token_str.encode('utf-8')
            vocab[token_id] = token_bytes
        
        # Load merges from text file
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)  # Split on first space only
                    if len(parts) == 2:
                        merge1 = parts[0].encode('utf-8')
                        merge2 = parts[1].encode('utf-8')
                        merges.append((merge1, merge2))
        
        return cls(vocab, merges, special_tokens)
    
    def _apply_bpe_to_token(self, token_bytes: bytes) -> List[int]:
        """
        Apply BPE merges to a single pre-token (sequence of bytes).
        
        This function takes a pre-token as bytes and applies the learned BPE merges
        in the order they were learned to produce a sequence of token IDs.
        
        Args:
            token_bytes: The pre-token as a sequence of bytes
            
        Returns:
            List of token IDs after applying BPE merges
        """
        # Start with individual bytes
        word = []
        for byte_val in token_bytes:
            byte_token = bytes([byte_val])
            if byte_token in self.vocab_reverse:
                word.append(byte_token)
            else:
                # This shouldn't happen with a proper vocabulary, but handle gracefully
                word.append(byte_token)
        
        if len(word) == 1:
            # Single character, return its token ID
            if word[0] in self.vocab_reverse:
                return [self.vocab_reverse[word[0]]]
            else:
                return []
        
        # Apply merges in the order they were learned
        for merge_left, merge_right in self.merges:
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # Look for the merge pair starting at position i
                    if i < len(word) - 1 and word[i] == merge_left and word[i + 1] == merge_right:
                        # Found the pair, merge it
                        merged_bytes = merge_left + merge_right
                        new_word.append(merged_bytes)
                        i += 2  # Skip both parts of the pair
                    else:
                        # No merge, keep current element
                        new_word.append(word[i])
                        i += 1
                except:
                    # Handle any errors gracefully
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        
        # Convert final word tokens to token IDs
        token_ids = []
        for token_bytes in word:
            if token_bytes in self.vocab_reverse:
                token_ids.append(self.vocab_reverse[token_bytes])
            else:
                # If token not found, try to break it down to bytes
                for byte_val in token_bytes:
                    byte_token = bytes([byte_val])
                    if byte_token in self.vocab_reverse:
                        token_ids.append(self.vocab_reverse[byte_token])
        
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode input text into a sequence of token IDs.
        
        Steps:
        1. Pre-tokenize the text using regex patterns
        2. Handle special tokens by splitting on them first
        3. Apply BPE merges to each pre-token
        4. Return the concatenated sequence of token IDs
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        token_ids = []
        
        # Step 1: Handle special tokens - use regex to find and preserve them
        special_token_bytes_set = set()
        for token_bytes in self.vocab_reverse.keys():
            try:
                token_str = token_bytes.decode('utf-8')
                # Check if this looks like a special token (contains special characters)
                if '<' in token_str and '>' in token_str:
                    special_token_bytes_set.add(token_bytes)
            except UnicodeDecodeError:
                pass
        
        # Split text using regex, keeping special tokens separate 
        if special_token_bytes_set:
            # Create regex pattern for special tokens
            special_strs = []
            for token_bytes in special_token_bytes_set:
                try:
                    special_strs.append(re.escape(token_bytes.decode('utf-8')))
                except UnicodeDecodeError:
                    pass
            
            if special_strs:
                # Split text on special tokens, keeping them
                special_pattern = '(' + '|'.join(special_strs) + ')'
                text_parts = re.split(special_pattern, text)
            else:
                text_parts = [text]
        else:
            text_parts = [text]
        
        # Step 2: Process each text part
        for text_part in text_parts:
            if not text_part:
                continue
                
            # Check if this part is a special token
            part_bytes = text_part.encode('utf-8')
            if part_bytes in special_token_bytes_set:
                if part_bytes in self.vocab_reverse:
                    token_ids.append(self.vocab_reverse[part_bytes])
                continue
            
            # Step 3: Pre-tokenize using regex pattern
            for match in re.finditer(PAT, text_part):
                pre_token = match.group()
                pre_token_bytes = pre_token.encode('utf-8')
                
                # Step 4: Apply BPE to this pre-token
                pre_token_ids = self._apply_bpe_to_token(pre_token_bytes)
                token_ids.extend(pre_token_ids)
        
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings (e.g., file lines) into token IDs lazily.
        
        This method is memory-efficient and suitable for processing large files
        that cannot be loaded entirely into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle, list of strings)
            
        Yields:
            Token IDs one by one
        """
        for text_chunk in iterable:
            # Encode each chunk and yield the token IDs
            chunk_token_ids = self.encode(text_chunk)
            for token_id in chunk_token_ids:
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        
        This method looks up each token ID in the vocabulary, concatenates
        the resulting byte sequences, and decodes them to Unicode text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string, with malformed bytes replaced by Unicode replacement character
        """
        # Collect all bytes from token IDs
        all_bytes = b""
        
        for token_id in ids:
            if token_id in self.vocab:
                all_bytes += self.vocab[token_id]
            else:
                # Handle unknown token IDs gracefully - skip them or use a placeholder
                # For now, we'll skip unknown token IDs
                pass
        
        # Decode bytes to Unicode string, replacing malformed sequences with replacement character
        try:
            text = all_bytes.decode('utf-8', errors='replace')
        except Exception:
            # Fallback in case of any other decoding issues
            text = all_bytes.decode('utf-8', errors='replace')
        
        return text

