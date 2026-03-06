import numpy as np
from src.model import SkipGramNegativeSampling
from src.data import build_vocabulary, generate_training_pairs
from src.training import train, find_most_similar

# Small corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends"
]

# Build vocab and pairs
word_to_idx, idx_to_word, vocab_size = build_vocabulary(corpus)
pairs = generate_training_pairs(corpus, word_to_idx, window_size=2)

print(f"Vocabulary: {vocab_size} words")
print(f"Training pairs: {len(pairs)}")

# Create model
model = SkipGramNegativeSampling(vocab_size, embed_dim=10, negative_samples=3, seed=42)

# Train
print("\nTraining...")
losses = train(model, pairs, epochs=50, learning_rate=0.1, verbose=True)

print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {losses[0] - losses[-1]:.4f}")

# Test similarity
embeddings = model.get_embeddings()
similar_to_cat = find_most_similar(word_to_idx["cat"], embeddings, idx_to_word, top_k=3)

print(f"\nMost similar to 'cat':")
for word, sim in similar_to_cat:
    print(f"  {word}: {sim:.4f}")

print("\nTraining module works!")