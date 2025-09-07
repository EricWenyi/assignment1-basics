#!/usr/bin/env python3
"""
Simple demonstration of decoding functions with a minimal setup.
This shows the decoding logic without requiring a full trained model.
"""

import torch
from decode import apply_temperature_scaling, top_p_sampling, sample_next_token


def demo_temperature_effects():
    """Demonstrate how temperature affects sampling behavior."""
    print("🔥 TEMPERATURE SCALING DEMO")
    print("=" * 50)
    
    # Create example logits (vocabulary of 5 words)
    vocab = ["the", "quick", "brown", "fox", "jumps"]
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # "jumps" has highest logit
    
    print(f"Vocabulary: {vocab}")
    print(f"Raw logits: {logits}")
    print()
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for temp in temperatures:
        print(f"Temperature = {temp}")
        
        # Show probability distribution
        scaled_logits = apply_temperature_scaling(logits, temp)
        probs = torch.softmax(scaled_logits, dim=-1)
        
        print(f"  Probabilities: {[f'{p:.3f}' for p in probs]}")
        
        # Sample multiple times to show distribution
        samples = []
        for _ in range(20):
            token_id = sample_next_token(logits, temperature=temp)
            samples.append(vocab[token_id])
        
        # Count occurrences
        counts = {word: samples.count(word) for word in vocab}
        print(f"  Sample counts: {counts}")
        print()
    
    print("📊 ANALYSIS:")
    print("• Low temperature (0.1): Nearly always picks 'jumps' (highest logit)")
    print("• High temperature (5.0): More uniform distribution, all words possible")
    print("• Temperature = 1.0: Standard softmax behavior")
    print()


def demo_top_p_effects():
    """Demonstrate how top-p sampling affects token selection.""" 
    print("🎯 TOP-P (NUCLEUS) SAMPLING DEMO")
    print("=" * 50)
    
    # Create example with clear probability hierarchy
    vocab = ["very", "quite", "somewhat", "barely", "extremely"]
    # Design logits to create specific probability distribution
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1])
    probs = torch.softmax(logits, dim=-1)
    
    print(f"Vocabulary: {vocab}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probs]}")
    cumulative = torch.cumsum(probs, dim=0)
    print(f"Cumulative:    {[f'{p:.3f}' for p in cumulative]}")
    print()
    
    p_values = [0.3, 0.6, 0.9, 1.0]
    
    for p in p_values:
        print(f"Top-p = {p}")
        
        # Apply top-p filtering
        filtered_probs = top_p_sampling(probs, p)
        
        print(f"  Filtered probs: {[f'{p:.3f}' for p in filtered_probs]}")
        
        # Show which tokens are kept
        kept_tokens = [vocab[i] for i, prob in enumerate(filtered_probs) if prob > 0]
        print(f"  Kept tokens: {kept_tokens}")
        
        # Sample to show effect
        samples = []
        for _ in range(20):
            token_id = sample_next_token(logits, top_p=p)
            samples.append(vocab[token_id])
        
        counts = {word: samples.count(word) for word in vocab if samples.count(word) > 0}
        print(f"  Sample counts: {counts}")
        print()
    
    print("📊 ANALYSIS:")
    print("• p=0.3: Only 'very' (30% isn't enough for top 2, but ensures at least 1)")
    print("• p=0.6: 'very' + 'quite' (together ~60% of probability mass)")
    print("• p=0.9: Excludes only 'extremely' (keeps ~90% of mass)")
    print("• p=1.0: Keeps all tokens (no filtering)")
    print()


def demo_combined_strategies():
    """Show how temperature and top-p work together."""
    print("🎭 COMBINED STRATEGIES DEMO")
    print("=" * 50)
    
    vocab = ["I", "love", "hate", "like", "enjoy"]
    logits = torch.tensor([4.0, 3.0, 1.0, 2.0, 1.5])
    
    print(f"Vocabulary: {vocab}")
    print(f"Raw logits: {logits}")
    print()
    
    strategies = [
        {"name": "Greedy (temp=0.1)", "temperature": 0.1, "top_p": None},
        {"name": "Standard (temp=1.0)", "temperature": 1.0, "top_p": None},
        {"name": "Creative (temp=1.5)", "temperature": 1.5, "top_p": None},
        {"name": "Nucleus (temp=1.0, p=0.7)", "temperature": 1.0, "top_p": 0.7},
        {"name": "Creative + Nucleus (temp=1.2, p=0.8)", "temperature": 1.2, "top_p": 0.8},
    ]
    
    for strategy in strategies:
        print(f"{strategy['name']}:")
        
        # Sample multiple times
        samples = []
        for _ in range(30):
            token_id = sample_next_token(
                logits, 
                temperature=strategy['temperature'], 
                top_p=strategy['top_p']
            )
            samples.append(vocab[token_id])
        
        # Show distribution
        counts = {word: samples.count(word) for word in vocab}
        percentages = {word: f"{count/30*100:.1f}%" for word, count in counts.items() if count > 0}
        print(f"  Distribution: {percentages}")
        print()
    
    print("📊 PRACTICAL GUIDELINES:")
    print("• Greedy: Deterministic, repetitive (good for factual tasks)")
    print("• Creative: High diversity, potentially incoherent")  
    print("• Nucleus: Balanced quality and creativity")
    print("• Combined: Fine-grained control over both aspects")
    print()


def main():
    """Run all decoding demonstrations."""
    print("🤖 ADVANCED DECODING STRATEGIES DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Set seed for reproducible demonstrations
    torch.manual_seed(42)
    
    demo_temperature_effects()
    demo_top_p_effects() 
    demo_combined_strategies()
    
    print("✅ All demonstrations completed!")
    print()
    print("💡 KEY TAKEAWAYS:")
    print("• Temperature controls randomness: lower = more deterministic")
    print("• Top-p removes low-probability 'tail' tokens for better quality")
    print("• Combining both gives precise control over generation behavior")
    print("• Different tasks benefit from different decoding strategies")


if __name__ == "__main__":
    main()