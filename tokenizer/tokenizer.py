#!/usr/bin/env python3
"""
BPE Tokenizer for Encoding and Decoding

This module contains only the Tokenizer class for encoding text to token IDs
and decoding token IDs back to text using a trained BPE vocabulary and merges.

For training BPE tokenizers, see train_bpe.py.
"""

import json
import regex as re
from typing import Dict, List, Tuple

# Pre-tokenization pattern for splitting text into tokens
# This pattern matches:
# - Contractions and possessives (e.g., "don't", "John's")  
# - Letter sequences
# - Number sequences
# - Individual non-space characters
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    """
    BPE Tokenizer class for encoding and decoding text using trained vocabulary and merges.
    
    This class implements the BPE (Byte-Pair Encoding) tokenization algorithm for both
    encoding text to token IDs and decoding token IDs back to text.
    
    Usage:
        # Load from pickle files (recommended)
        tokenizer = Tokenizer.from_pickle('vocab.pkl', 'merges.pkl')
        
        # Load from text/JSON files
        tokenizer = Tokenizer.from_files('vocab.json', 'merges.txt')
        
        # Encode text to token IDs
        token_ids = tokenizer.encode("Hello world!")
        
        # Decode token IDs back to text
        text = tokenizer.decode(token_ids)
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
    
    @classmethod
    def from_pickle(cls, vocab_pickle_path: str, merges_pickle_path: str, special_tokens: List[str] = None):
        """
        Create a tokenizer from pickled vocabulary and merges files.
        
        This method loads the vocabulary and merges from pickle files in their
        native Python formats: dict[int, bytes] and list[tuple[bytes, bytes]].
        
        Args:
            vocab_pickle_path: Path to pickle file containing vocabulary (dict[int, bytes])
            merges_pickle_path: Path to pickle file containing merges (list[tuple[bytes, bytes]])
            special_tokens: Optional list of special token strings
            
        Returns:
            Tokenizer instance
        """
        import pickle
        
        # Load vocabulary from pickle file
        with open(vocab_pickle_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Load merges from pickle file  
        with open(merges_pickle_path, 'rb') as f:
            merges = pickle.load(f)
        
        # Validate loaded data types
        if not isinstance(vocab, dict):
            raise ValueError(f"Expected vocab to be dict, got {type(vocab)}")
        if not isinstance(merges, list):
            raise ValueError(f"Expected merges to be list, got {type(merges)}")
        
        # Validate vocab format: dict[int, bytes]
        for token_id, token_bytes in vocab.items():
            if not isinstance(token_id, int):
                raise ValueError(f"Expected vocab keys to be int, got {type(token_id)}")
            if not isinstance(token_bytes, bytes):
                raise ValueError(f"Expected vocab values to be bytes, got {type(token_bytes)}")
        
        # Validate merges format: list[tuple[bytes, bytes]]
        for i, merge in enumerate(merges):
            if not isinstance(merge, tuple) or len(merge) != 2:
                raise ValueError(f"Expected merge {i} to be tuple of length 2, got {type(merge)} of length {len(merge) if hasattr(merge, '__len__') else 'unknown'}")
            merge1, merge2 = merge
            if not isinstance(merge1, bytes) or not isinstance(merge2, bytes):
                raise ValueError(f"Expected merge {i} elements to be bytes, got {type(merge1)}, {type(merge2)}")
        
        return cls(vocab, merges, special_tokens)
    
    def save_pickle(self, vocab_pickle_path: str, merges_pickle_path: str):
        """
        Save the tokenizer's vocabulary and merges to pickle files.
        
        Args:
            vocab_pickle_path: Path to save vocabulary pickle file  
            merges_pickle_path: Path to save merges pickle file
        """
        import pickle
        
        # Save vocabulary (dict[int, bytes])
        with open(vocab_pickle_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        # Save merges (list[tuple[bytes, bytes]])
        with open(merges_pickle_path, 'wb') as f:
            pickle.dump(self.merges, f)
        
        print(f"Tokenizer saved: {vocab_pickle_path}, {merges_pickle_path}")
    
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
        if len(token_bytes) == 0:
            return []
        
        # Start with individual bytes as token IDs
        # Each byte (0-255) has a corresponding token ID (0-255)
        word = list(token_bytes)
        
        # Apply merges iteratively until no more merges can be applied
        while len(word) >= 2:
            # Find pairs that can be merged using the merge_map
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            
            # Check which pairs can be merged
            mergeable_pairs = []
            for i, pair in enumerate(pairs):
                if pair in self.merge_map:
                    mergeable_pairs.append((i, pair, self.merge_map[pair]))
            
            # If no pairs can be merged, we're done
            if not mergeable_pairs:
                break
                
            # Choose the merge that was learned earliest (appears first in merges list)
            # Find the earliest merge by checking the position in self.merges
            earliest_merge = None
            earliest_position = float('inf')
            
            for i, pair, merged_token in mergeable_pairs:
                # Find position of this merge in the original merges list
                pair_bytes1 = self.vocab[pair[0]]
                pair_bytes2 = self.vocab[pair[1]]
                merge_tuple = (pair_bytes1, pair_bytes2)
                
                if merge_tuple in self.merges:
                    position = self.merges.index(merge_tuple)
                    if position < earliest_position:
                        earliest_position = position
                        earliest_merge = (i, pair, merged_token)
            
            if earliest_merge is None:
                break
                
            # Apply the earliest merge
            merge_index, merge_pair, merged_token = earliest_merge
            
            # Create new word with the merge applied
            new_word = word[:merge_index] + [merged_token] + word[merge_index + 2:]
            word = new_word
        
        return word
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs using BPE.
        
        This method:
        1. Pre-tokenizes the text using regex pattern matching
        2. Converts each pre-token to bytes  
        3. Applies BPE merges to each pre-token
        4. Returns the concatenated list of token IDs
        
        Args:
            text: Input text string to encode
            
        Returns:
            List of token IDs representing the encoded text
        """
        if not text:
            return []
        
        # Step 1: Pre-tokenize using regex pattern
        pre_tokens = PAT.findall(text)
        
        # Step 2: Apply BPE to each pre-token
        token_ids = []
        for pre_token in pre_tokens:
            # Convert pre-token to bytes
            pre_token_bytes = pre_token.encode('utf-8')
            
            # Apply BPE merges to get list of token IDs for this pre-token
            pre_token_ids = self._apply_bpe_to_token(pre_token_bytes)
            
            # Add to overall token list
            token_ids.extend(pre_token_ids)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.
        
        This method:
        1. Looks up each token ID in the vocabulary to get bytes
        2. Concatenates all the bytes together
        3. Decodes the concatenated bytes as UTF-8 text
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""
        
        # Convert token IDs to bytes
        byte_sequences = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
            else:
                # Handle unknown token ID - could raise error or skip
                # For now, we'll raise an error for debugging
                raise ValueError(f"Unknown token ID: {token_id}")
        
        # Concatenate all bytes
        concatenated_bytes = b''.join(byte_sequences)
        
        # Decode as UTF-8 text
        try:
            text = concatenated_bytes.decode('utf-8')
            return text
        except UnicodeDecodeError as e:
            # This shouldn't happen with proper BPE, but provide helpful error
            raise ValueError(f"Failed to decode bytes as UTF-8: {e}")