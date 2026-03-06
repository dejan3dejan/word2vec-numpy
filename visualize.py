#!/usr/bin/env python3
"""
Visualize word embeddings using PCA (pure NumPy implementation).

Usage:
    python visualize.py --embeddings output/text8/embeddings.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_embeddings(output_dir: str):
    """Load embeddings and vocabulary."""
    output_dir = Path(output_dir)
    
    embeddings = np.load(output_dir / 'embeddings.npy')
    vocab = np.load(output_dir / 'vocab.npz', allow_pickle=True)
    
    word_to_idx = vocab['word_to_idx'].item()
    idx_to_word = vocab['idx_to_word'].item()
    
    return embeddings, word_to_idx, idx_to_word


def visualize_pca(embeddings, word_to_idx, idx_to_word, output_path, words_to_plot=None):
    """
    Visualize embeddings using PCA (pure NumPy).
    
    Args:
        embeddings: Word embedding matrix
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
        output_path: Where to save the plot
        words_to_plot: Specific words to highlight
    """
    if words_to_plot is None:
        # Default words
        words_to_plot = [
            'king', 'queen', 'man', 'woman',
            'paris', 'france', 'london', 'england',
            'cat', 'dog'
        ]
    
    # Filter to words that exist
    indices = [word_to_idx[w] for w in words_to_plot if w in word_to_idx]
    
    if len(indices) == 0:
        print("None of the specified words found in vocabulary")
        return
    
    selected_embeddings = embeddings[indices]
    
    print(f"Performing PCA on {len(indices)} words...")
    
    # PCA in pure NumPy
    # Step 1: Center the data
    mean = np.mean(selected_embeddings, axis=0)
    centered = selected_embeddings - mean
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(centered.T)
    
    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Project onto top 2 principal components
    pca_components = eigenvectors[:, :2]
    embeddings_2d = centered @ pca_components
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    for i, word in enumerate([idx_to_word[idx] for idx in indices]):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=12, alpha=0.8)
    
    plt.title('Word Embeddings (PCA - Pure NumPy)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize word embeddings with PCA')
    parser.add_argument('--embeddings', type=str, 
                       default='output/text8/embeddings.npy',
                       help='Path to embeddings.npy file')
    parser.add_argument('--output', type=str,
                       default='output/text8/pca_visualization.png',
                       help='Output path for visualization')
    parser.add_argument('--words', type=str, nargs='+',
                       help='Specific words to visualize')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WORD EMBEDDING VISUALIZATION (PCA)")
    print("="*60)
    
    # Load
    embeddings_path = Path(args.embeddings)
    output_dir = embeddings_path.parent
    
    print(f"\nLoading from {output_dir}...")
    embeddings, word_to_idx, idx_to_word = load_embeddings(str(output_dir))
    
    print(f"   Vocabulary: {len(word_to_idx):,} words")
    print(f"   Embedding dim: {embeddings.shape[1]}")
    
    # Visualize
    visualize_pca(embeddings, word_to_idx, idx_to_word, 
                 args.output, words_to_plot=args.words)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()