"""
BPE Tokenizer Package

This package provides a complete Byte-Pair Encoding (BPE) tokenizer implementation
with training, caching, and optimization features.

Main Components:
- Tokenizer: The main BPE tokenizer class
- train_bpe: Optimized BPE training functions with caching
- utils: Utility functions for tokenizer creation and management

Usage:
    from tokenizer import Tokenizer
    from tokenizer.train_bpe import train_bpe_tinystories
    
    # Train a tokenizer
    vocab, merges, training_time, longest_token = train_bpe_tinystories(
        data_path="../data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=32000
    )
    
    # Load a trained tokenizer
    tokenizer = Tokenizer.from_pickle(
        vocab_pickle_path="tokenizer/outputs/tinystories_vocab.pkl",
        merges_pickle_path="tokenizer/outputs/tinystories_merges.pkl"
    )
    
    # Use the tokenizer
    tokens = tokenizer.encode("Hello world!")
    text = tokenizer.decode(tokens)
"""

# Import main classes and functions for easy access
from .bpe_tokenizer import (
    Tokenizer,
    train_bpe,
    train_bpe_optimized,
    read_text_file,
    get_byte_pair_counts,
    merge_pair_in_word_counts,
    update_pair_counts_after_merge,
    PAT
)

from .train_bpe import (
    train_bpe_tinystories,
    train_bpe_with_progress,
    get_file_hash,
    save_pretokenization_cache,
    load_pretokenization_cache,
    stream_process_file
)

__version__ = "1.0.0"
__all__ = [
    # Main tokenizer class
    "Tokenizer",
    
    # Training functions
    "train_bpe",
    "train_bpe_optimized", 
    "train_bpe_tinystories",
    "train_bpe_with_progress",
    
    # Core BPE functions
    "read_text_file",
    "get_byte_pair_counts",
    "merge_pair_in_word_counts",
    "update_pair_counts_after_merge",
    
    # Caching functions
    "get_file_hash",
    "save_pretokenization_cache", 
    "load_pretokenization_cache",
    "stream_process_file",
    
    # Constants
    "PAT"
]