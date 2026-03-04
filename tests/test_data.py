from src.data import tokenize, build_vocabulary, generate_training_pairs

# Test corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
]

# Test tokenization
tokens = tokenize(corpus[0])
print(f"Tokens: {tokens}")

# Test vocabulary
word_to_idx, idx_to_word, vocab_size = build_vocabulary(corpus)
print(f"\nVocabulary size: {vocab_size}")
print(f"Sample mappings: {list(word_to_idx.items())[:5]}")

# Test pair generation
pairs = generate_training_pairs(corpus, word_to_idx, window_size=2)
print(f"\nGenerated {len(pairs)} pairs")
print(f"Sample pairs: {pairs[:5]}")