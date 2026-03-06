"""
Data loading and preprocessing utilities.
"""
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercase."""
    return text.lower().split()


def build_vocabulary(corpus: List[str], min_count: int = 1) -> Tuple[Dict[str, int], Dict[int, str], int, Dict[int, int]]:
    """
    Build vocabulary with word counts for frequency-based negative sampling.
    
    Returns:
        word_to_idx, idx_to_word, vocab_size, word_counts (idx -> count)
    """
    all_words = []
    for sentence in corpus:
        all_words.extend(tokenize(sentence))
    
    word_counts_raw = Counter(all_words)
    vocab = sorted([word for word, count in word_counts_raw.items() if count >= min_count])
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Word counts mapped by index (for negative sampling)
    word_counts = {word_to_idx[word]: count 
                   for word, count in word_counts_raw.items() 
                   if word in word_to_idx}
    
    return word_to_idx, idx_to_word, len(vocab), word_counts


def generate_training_pairs(
    corpus: List[str],
    word_to_idx: Dict[str, int],
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """Generate skip-gram training pairs."""
    pairs = []
    
    for sentence in tqdm(corpus, desc="Generating pairs"):
        words = tokenize(sentence)
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


def stream_training_pairs(
    corpus: List[str],
    word_to_idx: Dict[str, int],
    window_size: int = 2,
    shuffle: bool = True
):
    """Memory-efficient generator that yields training pairs on-the-fly."""
    sentence_indices = list(range(len(corpus)))
    if shuffle:
        np.random.shuffle(sentence_indices)
    
    for sent_idx in sentence_indices:
        sentence = corpus[sent_idx]
        words = tokenize(sentence)
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        
        for center_pos, center_idx in enumerate(word_indices):
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                
                context_pos = center_pos + offset
                if 0 <= context_pos < len(word_indices):
                    context_idx = word_indices[context_pos]
                    yield (center_idx, context_idx)


def count_training_pairs(
    corpus: List[str],
    word_to_idx: Dict[str, int],
    window_size: int = 2
) -> int:
    """Count total number of training pairs without generating them."""
    total = 0
    for sentence in corpus:
        words = tokenize(sentence)
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        n = len(word_indices)
        
        for i in range(n):
            left_context = min(i, window_size)
            right_context = min(n - i - 1, window_size)
            total += left_context + right_context
    
    return total