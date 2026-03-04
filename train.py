"""
Command-line interface for training word2vec model.

Usage:
    python train.py --corpus data/corpus.txt --epochs 100 --embed-dim 100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import build_vocabulary, generate_training_pairs
from src.model import SkipGramNegativeSampling
from src.training import train, find_most_similar


def load_corpus(filepath: str) -> list:
    """Load corpus from text file (one sentence per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


def save_embeddings(model: SkipGramNegativeSampling, filepath: str):
    """Save embeddings to .npy file."""
    embeddings = model.get_embeddings()
    np.save(filepath, embeddings)
    print(f"Embeddings saved to {filepath}")


def plot_loss(losses: list, save_path: str = None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train word2vec skip-gram model')
    
    # Data arguments
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus file (one sentence per line)')
    parser.add_argument('--min-count', type=int, default=1,
                       help='Minimum word frequency (default: 1)')
    
    # Model arguments
    parser.add_argument('--embed-dim', type=int, default=100,
                       help='Embedding dimension (default: 100)')
    parser.add_argument('--window-size', type=int, default=2,
                       help='Context window size (default: 2)')
    parser.add_argument('--negative-samples', type=int, default=5,
                       help='Number of negative samples (default: 5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for embeddings and plots (default: output)')
    parser.add_argument('--eval-words', type=str, nargs='+',
                       help='Words to evaluate similarity after training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("word2vec Skip-Gram Training")
    print("=" * 60)
    
    # Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"   Loaded {len(corpus)} sentences")
    
    # Build vocabulary
    print(f"\nBuilding vocabulary (min_count={args.min_count})...")
    word_to_idx, idx_to_word, vocab_size = build_vocabulary(corpus, args.min_count)
    print(f"   Vocabulary size: {vocab_size}")
    
    # Generate training pairs
    print(f"\nGenerating skip-gram pairs (window_size={args.window_size})...")
    pairs = generate_training_pairs(corpus, word_to_idx, args.window_size)
    print(f"   Generated {len(pairs)} training pairs")
    
    # Create model
    print(f"\nInitializing model (embed_dim={args.embed_dim}, neg_samples={args.negative_samples})...")
    model = SkipGramNegativeSampling(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        negative_samples=args.negative_samples,
        seed=args.seed
    )
    print(f"   Total parameters: {model.W_in.size + model.W_out.size:,}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs (lr={args.lr})...")
    losses = train(model, pairs, epochs=args.epochs, learning_rate=args.lr, verbose=True)
    
    print(f"\nTraining complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f}")
    
    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    save_embeddings(model, str(embeddings_path))
    
    # Save vocabulary mappings
    vocab_path = output_dir / "vocab.npz"
    np.savez(vocab_path, word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Plot loss
    loss_plot_path = output_dir / "loss_curve.png"
    plot_loss(losses, str(loss_plot_path))
    
    # Evaluate similarity
    if args.eval_words:
        print(f"\n🔍 Evaluating word similarities...")
        embeddings = model.get_embeddings()
        
        for word in args.eval_words:
            if word not in word_to_idx:
                print(f"'{word}' not in vocabulary")
                continue
            
            similar = find_most_similar(word_to_idx[word], embeddings, idx_to_word, top_k=5)
            print(f"\n   Most similar to '{word}':")
            for sim_word, sim_score in similar:
                print(f"      {sim_word}: {sim_score:.4f}")
    
    print(f"\n{'=' * 60}")
    print(f"Training complete! Output saved to {output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()