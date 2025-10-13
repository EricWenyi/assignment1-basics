#!/usr/bin/env python3
"""Generate TinyStories samples from a trained checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from decode import generate_text
from tests.adapters import TransformerLM
from tokenizer import Tokenizer

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device, dtype: torch.dtype) -> TransformerLM:
    """Instantiate TransformerLM and load weights from the checkpoint."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg.get("rope_theta", 10000.0),
    ).to(device=device).to(dtype=dtype)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    """Return the BPE tokenizer used during training."""
    return Tokenizer.from_files(str(vocab_path), str(merges_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained TinyStories model")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint .pt file")
    parser.add_argument("prompt", type=str, help="Prompt to seed generation")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=120, help="Number of new tokens to sample")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("-p", "--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--end-token", type=str, default="<|endoftext|>", help="Early stop token string")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=DTYPE_MAP.keys(),
        help="Model dtype for inference",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to training config JSON (defaults to checkpoint_dir/config.json)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("tokenizer/outputs/tinystories_vocab.json"),
        help="Path to vocab JSON used by tokenizer",
    )
    parser.add_argument(
        "--merges",
        type=Path,
        default=Path("tokenizer/outputs/tinystories_merges.txt"),
        help="Path to merges TXT used by tokenizer",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path: Optional[Path]
    if args.config:
        config_path = args.config
    else:
        config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Could not locate config.json; pass --config explicitly.")

    model = load_model(config_path, checkpoint_path, device, dtype)
    tokenizer = load_tokenizer(args.vocab, args.merges)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token=args.end_token,
        device=str(device),
    )

    print("\n=== Generated Text ===\n")
    print(generated)


if __name__ == "__main__":
    main()

