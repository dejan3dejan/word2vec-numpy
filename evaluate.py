"""
Evaluate word2vec embeddings on word analogies and similarity tasks.

Usage:
    python evaluate.py --embeddings output/wikitext103/embeddings.npy
"""

import argparse
import numpy as np
from pathlib import Path

from src.training import evaluate_analogy, find_most_similar, cosine_similarity


def load_embeddings(output_dir: str):
    """Load embeddings and vocabulary from output directory."""
    output_dir = Path(output_dir)
    
    embeddings = np.load(output_dir / 'embeddings.npy')
    vocab = np.load(output_dir / 'vocab.npz', allow_pickle=True)
    
    word_to_idx = vocab['word_to_idx'].item()
    idx_to_word = vocab['idx_to_word'].item()
    
    return embeddings, word_to_idx, idx_to_word


def test_word_analogies(embeddings, word_to_idx, idx_to_word):
    """Test classic word analogy examples."""
    
    print("\n" + "="*60)
    print("WORD ANALOGY TESTS")
    print("="*60)
    
    # Classic analogies
    analogies = [
        ("king", "man", "queen", "woman"),
        ("paris", "france", "berlin", "germany"),
        ("paris", "france", "london", "england"),
        ("big", "bigger", "small", "smaller"),
        ("good", "better", "bad", "worse"),
        ("go", "went", "take", "took"),
    ]
    
    for word_a, word_b, word_c, expected in analogies:
        print(f"\n{word_a} is to {word_b} as {word_c} is to ?")
        
        # Check if all words exist
        if not all(w in word_to_idx for w in [word_a, word_b, word_c]):
            missing = [w for w in [word_a, word_b, word_c] if w not in word_to_idx]
            print(f"Words not in vocabulary: {missing}")
            continue
        
        results = evaluate_analogy(word_a, word_b, word_c, embeddings, 
                                  word_to_idx, idx_to_word, top_k=5)
        
        if not results:
            print(f"No results")
            continue
        
        # Print top 5
        for i, (word, score) in enumerate(results, 1):
            marker = "Correct" if word == expected else "  "
            print(f"   {marker} {i}. {word:15s} (similarity: {score:.4f})")
        
        # Check if expected word is in top 5
        top_words = [w for w, _ in results]
        if expected in top_words:
            rank = top_words.index(expected) + 1
            print(f"Expected answer '{expected}' at rank {rank}!")


def test_word_similarity(embeddings, word_to_idx, idx_to_word):
    """Test semantic similarity for interesting words."""
    
    print("\n" + "="*60)
    print("WORD SIMILARITY TESTS")
    print("="*60)
    
    test_words = [
        "king", "queen", "france", "paris", 
        "computer", "science", "cat", "dog"
    ]
    
    for word in test_words:
        if word not in word_to_idx:
            print(f"\n'{word}' not in vocabulary")
            continue
        
        print(f"\nMost similar to '{word}':")
        similar = find_most_similar(word_to_idx[word], embeddings, 
                                   idx_to_word, top_k=10)
        
        for i, (sim_word, score) in enumerate(similar, 1):
            print(f"   {i:2d}. {sim_word:20s} {score:.4f}")


def test_semantic_clusters(embeddings, word_to_idx, idx_to_word):
    """Test if semantic clusters are well-formed."""
    
    print("\n" + "="*60)
    print("SEMANTIC CLUSTER TESTS")
    print("="*60)
    
    clusters = {
        "Countries": ["france", "germany", "england", "spain", "italy"],
        "Cities": ["paris", "berlin", "london", "madrid", "rome"],
        "Animals": ["cat", "dog", "lion", "tiger", "elephant"],
        "Colors": ["red", "blue", "green", "yellow", "black"],
    }
    
    for cluster_name, words in clusters.items():
        print(f"\nCluster: {cluster_name}")
        
        # Find words that exist
        valid_words = [w for w in words if w in word_to_idx]
        
        if len(valid_words) < 2:
            print(f"Not enough words in vocabulary")
            continue
        
        # Compute average pairwise similarity
        similarities = []
        for i, word1 in enumerate(valid_words):
            for word2 in valid_words[i+1:]:
                vec1 = embeddings[word_to_idx[word1]]
                vec2 = embeddings[word_to_idx[word2]]
                sim = cosine_similarity(vec1, vec2)
                similarities.append(sim)
        
        avg_sim = np.mean(similarities)
        print(f"   Words found: {valid_words}")
        print(f"   Average pairwise similarity: {avg_sim:.4f}")
        
        if avg_sim > 0.5:
            print(f"Strong cluster (similarity > 0.5)")
        elif avg_sim > 0.3:
            print(f"Moderate cluster (similarity > 0.3)")
        else:
            print(f"Weak cluster (similarity < 0.3)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate word embeddings')
    parser.add_argument('--embeddings', type=str, 
                       default='output/wikitext103/embeddings.npy',
                       help='Path to embeddings.npy file')
    parser.add_argument('--vocab', type=str,
                       default='output/wikitext103/vocab.npz',
                       help='Path to vocab.npz file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WORD2VEC EMBEDDING EVALUATION")
    print("="*60)
    
    # Infer output dir from embeddings path
    embeddings_path = Path(args.embeddings)
    output_dir = embeddings_path.parent
    
    print(f"\nLoading embeddings from {output_dir}...")
    embeddings, word_to_idx, idx_to_word = load_embeddings(str(output_dir))
    
    print(f"Vocabulary size: {len(word_to_idx):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Run tests
    test_word_analogies(embeddings, word_to_idx, idx_to_word)
    test_word_similarity(embeddings, word_to_idx, idx_to_word)
    test_semantic_clusters(embeddings, word_to_idx, idx_to_word)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()