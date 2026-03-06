"""
Training utilities for word2vec model.
"""
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from .model import SkipGramNegativeSampling


def train(
    model: SkipGramNegativeSampling,
    pairs: List[Tuple[int, int]],
    epochs: int = 100,
    learning_rate: float = 0.1,
    verbose: bool = True
) -> List[float]:
    """
    Train word2vec model using SGD.
    
    Args:
        model: SkipGramNegativeSampling instance
        pairs: List of (center_idx, context_idx) tuples
        epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        verbose: Print progress during training
    
    Returns:
        losses: List of average loss per epoch
    """
    losses = []
    
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for epoch in iterator:
        epoch_loss = 0.0
        
        # Shuffle pairs for stochastic gradient descent
        shuffled_pairs = pairs.copy()
        np.random.shuffle(shuffled_pairs)
        
        for center_idx, context_idx in shuffled_pairs:
            # Forward pass
            loss, cache = model.forward(center_idx, context_idx)
            epoch_loss += loss
            
            # Backward pass
            dW_in, dW_out = model.backward(cache)
            
            # SGD update
            model.W_in -= learning_rate * dW_in
            model.W_out -= learning_rate * dW_out
        
        # Average loss for epoch
        avg_loss = epoch_loss / len(pairs)
        losses.append(avg_loss)
        
        # Update progress bar
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return losses


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity [-1, 1]
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_most_similar(
    word_idx: int,
    embeddings: np.ndarray,
    idx_to_word: dict,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find most similar words to a given word.
    
    Args:
        word_idx: Index of query word
        embeddings: Word embedding matrix (vocab_size, embed_dim)
        idx_to_word: Index to word mapping
        top_k: Number of most similar words to return
    
    Returns:
        List of (word, similarity) tuples
    """
    query_vec = embeddings[word_idx]
    similarities = []
    
    for idx in range(len(embeddings)):
        if idx == word_idx:
            continue  # Skip the word itself
        
        sim = cosine_similarity(query_vec, embeddings[idx])
        similarities.append((idx, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to (word, similarity) tuples
    top_words = [
        (idx_to_word[idx], sim)
        for idx, sim in similarities[:top_k]
    ]
    
    return top_words


def evaluate_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    embeddings: np.ndarray,
    word_to_idx: dict,
    idx_to_word: dict,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Evaluate word analogy: a is to b as c is to ?
    
    Example: "king" is to "man" as "queen" is to "woman"
    
    Args:
        word_a: First word (e.g., "king")
        word_b: Second word (e.g., "man")
        word_c: Third word (e.g., "queen")
        embeddings: Word embedding matrix
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
        top_k: Number of candidates to return
    
    Returns:
        List of (word, score) tuples
    """
    # Get indices
    idx_a = word_to_idx.get(word_a)
    idx_b = word_to_idx.get(word_b)
    idx_c = word_to_idx.get(word_c)
    
    if idx_a is None or idx_b is None or idx_c is None:
        return []
    
    # Compute: vec_c - vec_a + vec_b
    vec_result = embeddings[idx_c] - embeddings[idx_a] + embeddings[idx_b]
    
    # Find closest words
    similarities = []
    for idx in range(len(embeddings)):
        # Skip input words
        if idx in [idx_a, idx_b, idx_c]:
            continue
        
        sim = cosine_similarity(vec_result, embeddings[idx])
        similarities.append((idx, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_words = [
        (idx_to_word[idx], sim)
        for idx, sim in similarities[:top_k]
    ]
    
    return top_words

def train_streaming(
    model: SkipGramNegativeSampling,
    corpus: List[str],
    word_to_idx: dict,
    window_size: int = 2,
    epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 512,
    verbose: bool = True
) -> List[float]:
    """
    Train word2vec model using streaming pairs with mini-batch gradient accumulation.
    
    Args:
        model: SkipGramNegativeSampling instance
        corpus: List of sentences
        word_to_idx: Word to index mapping
        window_size: Context window size
        epochs: Number of training epochs
        learning_rate: Learning rate for SGD
        batch_size: Number of pairs to accumulate before update
        verbose: Print progress during training
    
    Returns:
        losses: List of average loss per epoch
    """
    from .data import stream_training_pairs, count_training_pairs
    
    if verbose:
        print("Counting training pairs...")
        total_pairs = count_training_pairs(corpus, word_to_idx, window_size)
        print(f"   Total pairs: {total_pairs:,}")
    
    losses = []
    
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for epoch in iterator:
        epoch_loss = 0.0
        pair_count = 0
        
        pair_generator = stream_training_pairs(corpus, word_to_idx, window_size, shuffle=True)
        
        if verbose:
            pair_generator = tqdm(
                pair_generator, 
                total=total_pairs, 
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False
            )
        
        grad_in_accumulator = {}
        grad_out_accumulator = {}
        batch_count = 0
        
        for center_idx, context_idx in pair_generator:
            loss, cache = model.forward(center_idx, context_idx)
            epoch_loss += loss
            pair_count += 1
            
            (indices_in, dW_in), (indices_out, dW_out) = model.backward(cache)
            
            for idx, grad in zip(indices_in, dW_in):
                if idx not in grad_in_accumulator:
                    grad_in_accumulator[idx] = np.zeros_like(grad)
                grad_in_accumulator[idx] += grad
            
            for idx, grad in zip(indices_out, dW_out):
                if idx not in grad_out_accumulator:
                    grad_out_accumulator[idx] = np.zeros_like(grad)
                grad_out_accumulator[idx] += grad
            
            batch_count += 1
            
            if batch_count >= batch_size:
                for idx, grad in grad_in_accumulator.items():
                    model.W_in[idx] -= learning_rate * (grad / batch_count)
                
                for idx, grad in grad_out_accumulator.items():
                    model.W_out[idx] -= learning_rate * (grad / batch_count)
                
                grad_in_accumulator.clear()
                grad_out_accumulator.clear()
                batch_count = 0
        
        if batch_count > 0:
            for idx, grad in grad_in_accumulator.items():
                model.W_in[idx] -= learning_rate * (grad / batch_count)
            
            for idx, grad in grad_out_accumulator.items():
                model.W_out[idx] -= learning_rate * (grad / batch_count)
        
        avg_loss = epoch_loss / pair_count
        losses.append(avg_loss)
        
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return losses