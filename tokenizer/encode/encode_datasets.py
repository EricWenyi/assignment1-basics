#!/usr/bin/env python3
"""
Script to encode TinyStories and OpenWebText datasets into token sequences.
Saves the encoded data as NumPy arrays with uint16 datatype.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add tokenizer to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer

def estimate_file_size(file_path):
    """Get file size for progress estimation"""
    return os.path.getsize(file_path)

def read_and_encode_file(file_path, tokenizer, output_path, separator=None):
    """
    Read a large text file and encode it chunk by chunk to avoid memory issues.

    Args:
        file_path: Path to input text file
        tokenizer: Tokenizer instance to use for encoding
        output_path: Path to save encoded numpy array
        separator: Document separator (e.g., '<|endoftext|>' for TinyStories)
    """
    print(f"Processing {file_path}")
    print(f"Output will be saved to {output_path}")

    file_size = estimate_file_size(file_path)
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")

    start_time = time.time()
    tokens = []
    current_text = ""
    bytes_processed = 0
    last_progress_time = start_time

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            bytes_processed += len(line.encode('utf-8'))

            if separator and separator in line:
                # Handle documents separated by special tokens (TinyStories)
                parts = line.split(separator)
                current_text += parts[0]

                # Encode current document
                if current_text.strip():
                    doc_tokens = tokenizer.encode(current_text.strip())
                    tokens.extend(doc_tokens)

                # Process remaining parts
                for i in range(1, len(parts)):
                    if i < len(parts) - 1:
                        # Complete document
                        if parts[i].strip():
                            doc_tokens = tokenizer.encode(parts[i].strip())
                            tokens.extend(doc_tokens)
                    else:
                        # Start of next document
                        current_text = parts[i]
            else:
                # Regular line, add to current text
                current_text += line

            # Progress reporting every 10 seconds
            current_time = time.time()
            if current_time - last_progress_time > 10:
                elapsed = current_time - start_time
                progress = bytes_processed / file_size
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = estimated_total - elapsed

                print(f"Progress: {progress*100:.1f}% | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Estimated remaining: {remaining/60:.1f}min | "
                      f"Tokens so far: {len(tokens):,}")
                last_progress_time = current_time

            # Periodic memory management - save intermediate results
            if len(tokens) > 50_000_000:  # Save every 50M tokens
                print(f"Saving intermediate results... ({len(tokens):,} tokens)")
                # Convert to uint16 and save
                token_array = np.array(tokens, dtype=np.uint16)
                intermediate_path = output_path.replace('.npy', f'_part_{len(tokens)}.npy')
                np.save(intermediate_path, token_array)
                print(f"Saved intermediate file: {intermediate_path}")

                # Clear tokens list to free memory
                del token_array
                tokens = []

    # Encode any remaining text
    if current_text.strip():
        doc_tokens = tokenizer.encode(current_text.strip())
        tokens.extend(doc_tokens)

    # Save final results
    total_time = time.time() - start_time
    print(f"\nEncoding completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Throughput: {bytes_processed/total_time:.0f} bytes/second")

    if tokens:
        print(f"Converting to uint16 array and saving...")
        token_array = np.array(tokens, dtype=np.uint16)
        np.save(output_path, token_array)
        print(f"Saved: {output_path}")

        # Verify the saved file
        saved_size = os.path.getsize(output_path) / (1024*1024)
        print(f"Output file size: {saved_size:.1f} MB")
        print(f"Compression ratio: {file_size/saved_size:.1f}x")

def main():
    print("=== Dataset Encoding Script ===\n")

    # Create output directory in tokenizer/encode
    script_dir = Path(__file__).parent
    output_dir = script_dir
    output_dir.mkdir(exist_ok=True)

    # Load tokenizers
    print("Loading tokenizers...")

    # TinyStories tokenizer
    base_dir = script_dir.parent.parent
    tinystories_tokenizer = Tokenizer.from_files(
        str(base_dir / 'tinystories_vocab.json'),
        str(base_dir / 'tinystories_merges.txt')
    )
    print(f"TinyStories tokenizer loaded: {len(tinystories_tokenizer.vocab):,} vocab")

    # OpenWebText tokenizer
    owt_tokenizer = Tokenizer.from_files(
        str(script_dir.parent / 'outputs' / 'owt_vocab.json'),
        str(script_dir.parent / 'outputs' / 'owt_merges.txt')
    )
    print(f"OpenWebText tokenizer loaded: {len(owt_tokenizer.vocab):,} vocab")

    # Check if data files exist
    data_dir = Path("/home/glimmer/Work/data")
    tinystories_path = str(data_dir / "TinyStoriesV2-GPT4-train.txt")
    owt_path = str(data_dir / "owt_train.txt")

    if not os.path.exists(tinystories_path):
        print(f"Warning: TinyStories file not found at {tinystories_path}")

    if not os.path.exists(owt_path):
        print(f"Warning: OpenWebText file not found at {owt_path}")

    # Encode TinyStories
    if os.path.exists(tinystories_path):
        print("\n" + "="*60)
        print("ENCODING TINYSTORIES DATASET")
        print("="*60)
        read_and_encode_file(
            tinystories_path,
            tinystories_tokenizer,
            str(output_dir / "tinystories_train.npy"),
            separator="<|endoftext|>"
        )

    # Encode OpenWebText
    if os.path.exists(owt_path):
        print("\n" + "="*60)
        print("ENCODING OPENWEBTEXT DATASET")
        print("="*60)
        read_and_encode_file(
            owt_path,
            owt_tokenizer,
            str(output_dir / "owt_train.npy")
        )

    print(f"\n=== ENCODING COMPLETE ===")
    print(f"Encoded files saved in: {output_dir.absolute()}")

    # List output files
    output_files = list(output_dir.glob("*.npy"))
    if output_files:
        print(f"\nOutput files:")
        for f in sorted(output_files):
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()