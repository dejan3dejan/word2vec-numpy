"""
Data loading and preprocessing utilities.
"""
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization with lowercase.
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens
    """
    return text.lower().split()


def build_vocabulary(corpus: List[str], min_count: int = 1) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build word-to-index and index-to-word mappings.
    
    Args:
        corpus: List of text strings
        min_count: Minimum word frequency to include in vocabulary
    
    Returns:
        word_to_idx: Word to index mapping
        idx_to_word: Index to word mapping
        vocab_size: Vocabulary size
    """
    # Flatten all words
    all_words = []
    for sentence in corpus:
        all_words.extend(tokenize(sentence))
    
    # Count frequencies
    word_counts = Counter(all_words)
    
    # Filter by min_count and sort alphabetically
    vocab = sorted([word for word, count in word_counts.items() if count >= min_count])
    
    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)
    
    return word_to_idx, idx_to_word, vocab_size


def generate_training_pairs(
    corpus: List[str],
    word_to_idx: Dict[str, int],
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """
    Generate skip-gram training pairs.
    
    Args:
        corpus: List of text strings
        word_to_idx: Word to index mapping
        window_size: Context window size
    
    Returns:
        List of (center_idx, context_idx) tuples
    """
    pairs = []
    
    for sentence in corpus:
        words = tokenize(sentence)
        # Filter out words not in vocabulary
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        
        for center_pos, center_idx in enumerate(word_indices):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                
                context_pos = center_pos + offset
                
                if context_pos < 0 or context_pos >= len(word_indices):
                    continue
                
                context_idx = word_indices[context_pos]
                pairs.append((center_idx, context_idx))
    
    return pairs