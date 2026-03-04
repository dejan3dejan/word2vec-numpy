# tests/test_model.py
import numpy as np
from src.model import SkipGramNegativeSampling

# Create small model
model = SkipGramNegativeSampling(vocab_size=10, embed_dim=5, negative_samples=3, seed=42)

print(f"Model created!")
print(f"W_in shape: {model.W_in.shape}")
print(f"W_out shape: {model.W_out.shape}")

# Test forward pass
center_idx = 2
context_idx = 5

loss, cache = model.forward(center_idx, context_idx)
print(f"\nForward pass:")
print(f"Loss: {loss:.4f}")
print(f"Negative samples: {cache['negative_indices']}")

# Test backward pass
dW_in, dW_out = model.backward(cache)
print(f"\nBackward pass:")
print(f"dW_in non-zero rows: {np.sum(np.any(dW_in != 0, axis=1))}")  # Should be 1
print(f"dW_out non-zero rows: {np.sum(np.any(dW_out != 0, axis=1))}")  # Should be 4 (1 pos + 3 neg)

print("\nModel class works!")