#!/usr/bin/env python3
"""
Fast dataset encoding script using batch processing and multiprocessing.
Optimized for large files like TinyStories and OpenWebText.
"""

import numpy as np
import time
import sys
import os
import mmap
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager, Value, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import gc
import threading

# Add tokenizer to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer

def load_tokenizer(vocab_path, merges_path):
    """Load tokenizer in worker process"""
    return Tokenizer.from_files(vocab_path, merges_path)

def process_chunk(args):
    """Process a chunk of text in a separate process"""
    chunk_data, vocab_path, merges_path, separator, chunk_id, chunk_size_bytes = args

    start_time = time.time()

    # Load tokenizer in worker process
    tokenizer = load_tokenizer(vocab_path, merges_path)

    tokens = []
    documents = []

    if separator:
        # Handle TinyStories with <|endoftext|> separator
        documents = chunk_data.split(separator)
        documents = [doc.strip() for doc in documents if doc.strip()]
    else:
        # Handle OpenWebText - split by double newlines
        documents = chunk_data.split('\n\n')
        documents = [doc.strip() for doc in documents if doc.strip() and len(doc) > 100]

    # Batch encode documents for efficiency
    batch_size = 100
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        for doc in batch:
            doc_tokens = tokenizer.encode(doc)
            tokens.extend(doc_tokens)

        # Progress within chunk
        if batch_idx % 10 == 0 and total_batches > 10:
            batch_progress = (batch_idx + 1) / total_batches
            print(f"  Chunk {chunk_id}: {batch_progress*100:.1f}% ({batch_idx+1}/{total_batches} batches)")

    processing_time = time.time() - start_time
    throughput = chunk_size_bytes / processing_time / (1024*1024) if processing_time > 0 else 0

    print(f"Chunk {chunk_id} DONE: {len(documents)} docs -> {len(tokens):,} tokens "
          f"({processing_time:.1f}s, {throughput:.1f} MB/s)")

    return tokens, chunk_id, len(documents), processing_time

def read_file_in_chunks(file_path, chunk_size_mb=100):
    """Read file in chunks using memory mapping for efficiency"""
    chunks = []
    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024

    print(f"Reading {file_path} in {chunk_size_mb}MB chunks...")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            chunk_id = 0

            while start < file_size:
                end = min(start + chunk_size, file_size)

                # Find a good breaking point (end of line or document)
                if end < file_size:
                    # Read a bit more to find line break
                    f.seek(end)
                    extra = f.read(10000)  # Read up to 10KB more
                    if '\n' in extra:
                        line_break = extra.find('\n')
                        end += line_break + 1
                    elif '<|endoftext|>' in extra:
                        token_end = extra.find('<|endoftext|>') + len('<|endoftext|>')
                        end += token_end

                # Extract chunk
                mm.seek(start)
                chunk_data = mm.read(end - start).decode('utf-8', errors='ignore')
                chunk_size_bytes = end - start
                chunks.append((chunk_data, start, end, chunk_id, chunk_size_bytes))

                start = end
                chunk_id += 1

                if chunk_id % 10 == 0:
                    print(f"  Read {chunk_id} chunks ({start/(1024*1024*1024):.1f}GB processed)")

    print(f"Split into {len(chunks)} chunks")
    return chunks

def progress_monitor(futures, total_chunks, start_time, file_size):
    """Monitor progress of parallel processing in separate thread"""
    completed = 0
    total_tokens = 0
    total_docs = 0
    bytes_processed = 0

    while completed < total_chunks:
        time.sleep(5)  # Update every 5 seconds

        # Count completed futures
        current_completed = sum(1 for f in futures if f.done())

        if current_completed > completed:
            completed = current_completed
            elapsed = time.time() - start_time
            progress = completed / total_chunks

            # Calculate ETA
            if progress > 0:
                eta = (elapsed / progress) - elapsed
                eta_str = f"{eta/60:.1f}min"
            else:
                eta_str = "calculating..."

            # Get results from completed futures to update token count
            current_tokens = 0
            current_docs = 0
            for f in futures:
                if f.done() and not f.exception():
                    try:
                        result = f.result()
                        tokens, chunk_id, docs, proc_time = result
                        current_tokens += len(tokens)
                        current_docs += docs
                    except:
                        pass

            print(f"\n[PROGRESS] {progress*100:.1f}% complete | "
                  f"{completed}/{total_chunks} chunks | "
                  f"Elapsed: {elapsed/60:.1f}min | ETA: {eta_str}")
            print(f"           Tokens so far: {current_tokens:,} | "
                  f"Documents: {current_docs:,}")

            # Estimate throughput
            if elapsed > 0:
                estimated_bytes = file_size * progress
                throughput = estimated_bytes / elapsed / (1024*1024)
                print(f"           Throughput: ~{throughput:.1f} MB/s")

def encode_file_parallel(file_path, vocab_path, merges_path, output_path, separator=None, max_workers=None):
    """Encode file using parallel processing with real-time progress tracking"""
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Don't use too many processes

    print(f"Encoding {file_path} with {max_workers} processes")
    start_time = time.time()
    file_size = os.path.getsize(file_path)

    # Read file in chunks
    chunks = read_file_in_chunks(file_path, chunk_size_mb=200)  # Larger chunks for better efficiency

    # Prepare arguments for parallel processing
    chunk_args = []
    for chunk_data, start, end, chunk_id, chunk_size_bytes in chunks:
        chunk_args.append((chunk_data, vocab_path, merges_path, separator, chunk_id, chunk_size_bytes))

    print(f"Processing {len(chunk_args)} chunks in parallel...")
    print(f"File size: {file_size/(1024*1024*1024):.2f} GB")
    print("="*60)

    # Process chunks in parallel with real-time progress tracking
    all_tokens = []
    chunk_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = [executor.submit(process_chunk, args) for args in chunk_args]

        # Start progress monitoring in separate thread
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(futures, len(chunks), start_time, file_size)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        # Collect results as they complete
        total_docs = 0
        total_processing_time = 0

        for future in as_completed(futures):
            try:
                tokens, chunk_id, docs, proc_time = future.result()
                chunk_results[chunk_id] = tokens
                total_docs += docs
                total_processing_time += proc_time

            except Exception as e:
                print(f"Error processing chunk: {e}")

        # Wait for monitor thread to finish
        monitor_thread.join(timeout=1)

    # Reassemble tokens in correct order
    print("\nReassembling tokens in correct order...")
    for chunk_id in sorted(chunk_results.keys()):
        all_tokens.extend(chunk_results[chunk_id])

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("ENCODING COMPLETED!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total documents: {total_docs:,}")
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"File throughput: {file_size/total_time/1024/1024:.0f} MB/second")
    print(f"Avg processing time per chunk: {total_processing_time/len(chunks):.1f}s")
    print(f"Parallelization efficiency: {total_processing_time/total_time/max_workers*100:.1f}%")

    # Save as uint16 numpy array
    print("Converting to uint16 and saving...")
    token_array = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, token_array)

    # Verify
    saved_size = os.path.getsize(output_path) / (1024*1024)
    print(f"Saved: {output_path}")
    print(f"Output size: {saved_size:.1f} MB")
    print(f"Compression: {file_size/1024/1024/saved_size:.1f}x")

    return len(all_tokens)

def main():
    print("=== Fast Dataset Encoding Script ===\n")
    print(f"Using up to {cpu_count()} CPU cores")

    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    data_dir = Path("/home/glimmer/Work/data")

    # Paths to tokenizers
    tinystories_vocab = str(base_dir / 'tinystories_vocab.json')
    tinystories_merges = str(base_dir / 'tinystories_merges.txt')
    owt_vocab = str(script_dir.parent / 'outputs' / 'owt_vocab.json')
    owt_merges = str(script_dir.parent / 'outputs' / 'owt_merges.txt')

    # Data file paths
    tinystories_path = str(data_dir / "TinyStoriesV2-GPT4-train.txt")
    owt_path = str(data_dir / "owt_train.txt")

    # Output paths
    ts_output = str(script_dir / "tinystories_train.npy")
    owt_output = str(script_dir / "owt_train.npy")

    total_start = time.time()

    # Encode TinyStories
    if os.path.exists(tinystories_path):
        print("\n" + "="*60)
        print("ENCODING TINYSTORIES DATASET")
        print("="*60)

        ts_tokens = encode_file_parallel(
            tinystories_path,
            tinystories_vocab,
            tinystories_merges,
            ts_output,
            separator="<|endoftext|>",
            max_workers=6  # Leave some cores for system
        )

        # Force garbage collection
        gc.collect()

        print(f"TinyStories completed: {ts_tokens:,} tokens")

    # Encode OpenWebText
    if os.path.exists(owt_path):
        print("\n" + "="*60)
        print("ENCODING OPENWEBTEXT DATASET")
        print("="*60)

        owt_tokens = encode_file_parallel(
            owt_path,
            owt_vocab,
            owt_merges,
            owt_output,
            separator=None,
            max_workers=6
        )

        print(f"OpenWebText completed: {owt_tokens:,} tokens")

    total_time = time.time() - total_start
    print(f"\n=== ALL ENCODING COMPLETE ===")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

    # List results
    for output_file in [ts_output, owt_output]:
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024*1024)
            print(f"{os.path.basename(output_file)}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()