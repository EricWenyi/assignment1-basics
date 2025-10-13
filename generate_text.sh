uv run python generate_text.py \
  experiments/20251006_174451_tinystories_baseline_195e0192/checkpoints/best_model.pt \
  "Tom likes to go home" \
  --max-new-tokens 120 --temperature 0.8 --top-p 0.9