#!/usr/bin/env python3
"""
Create dummy BPE tokenizer files for testing decoding.
"""

import json


def create_dummy_tokenizer():
    """Create minimal tokenizer files for testing."""
    
    # Create a small vocabulary
    vocab = {}
    
    # Add individual characters and common words
    tokens = [
        # Special tokens
        "<|endoftext|>",
        
        # Individual characters
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        ".", ",", "!", "?", "'", '"', "-", ":", ";",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        
        # Common subword units
        "er", "ing", "ed", "ly", "un", "re", "in", "on", "at", "it", "is", "be",
        "to", "of", "and", "the", "that", "have", "for", "not", "with", "he", 
        "as", "you", "do", "we", "his", "from", "they", "she", "or", "an", "will",
        "my", "one", "all", "would", "there", "their",
        
        # Common words
        "hello", "world", "test", "quick", "brown", "fox", "jumps", "over", 
        "lazy", "dog", "this", "example", "text", "generation", "model"
    ]
    
    # Create vocab dictionary mapping token_string -> token_id
    # (This matches the expected format in the tokenizer)
    vocab_for_json = {}
    for i, token in enumerate(tokens):
        vocab_for_json[token] = i
    
    with open('sample_data/vocab.json', 'w') as f:
        json.dump(vocab_for_json, f, indent=2)
    
    # Create simple merges (just a few examples)
    merges = [
        ("t", "h"),      # th
        ("i", "n"),      # in  
        ("e", "r"),      # er
        ("a", "n"),      # an
        ("o", "n"),      # on
    ]
    
    # Save merges
    with open('sample_data/merges.txt', 'w') as f:
        for merge in merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    
    print(f"Created dummy tokenizer:")
    print(f"  Vocabulary: sample_data/vocab.json ({len(vocab)} tokens)")
    print(f"  Merges: sample_data/merges.txt ({len(merges)} merges)")
    print(f"  Special tokens: ['<|endoftext|>']")


if __name__ == "__main__":
    create_dummy_tokenizer()