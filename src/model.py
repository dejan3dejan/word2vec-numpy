"""
Skip-gram word2vec model with negative sampling.
"""
import numpy as np
from typing import Tuple, Dict, Optional


class SkipGramNegativeSampling:
    """
    Skip-gram word2vec with frequency-based negative sampling.
    
    Uses P(w) ∝ count(w)^0.75 as recommended by Mikolov et al.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        negative_samples: int = 5,
        word_counts: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.k = negative_samples
        
        if seed is not None:
            np.random.seed(seed)
        
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # Build frequency-based negative sampling distribution
        self._build_neg_sampling_dist(word_counts)
    
    
    def _build_neg_sampling_dist(self, word_counts: Optional[Dict[int, int]]):
        """
        Build negative sampling distribution: P(w) ∝ count(w)^0.75
        
        This sublinear scaling balances frequent vs rare words.
        """
        if word_counts is not None:
            # Get counts for all indices (default to 1 if missing)
            counts = np.array([word_counts.get(i, 1) for i in range(self.vocab_size)], dtype=np.float64)
            
            # Apply 0.75 power (Mikolov's recommendation)
            self.neg_sampling_probs = np.power(counts, 0.75)
            self.neg_sampling_probs /= np.sum(self.neg_sampling_probs)
        else:
            # Fallback: uniform distribution
            self.neg_sampling_probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size
    
    
    def sample_negatives(self, context_idx: int) -> np.ndarray:
        """
        Sample k negative words using frequency-based distribution.
        
        Excludes the positive context word from sampling.
        """
        # Create temp probabilities excluding positive word
        probs = self.neg_sampling_probs.copy()
        probs[context_idx] = 0
        probs /= np.sum(probs)
        
        # Sample without replacement
        negatives = np.random.choice(
            self.vocab_size,
            size=self.k,
            replace=False,
            p=probs
        )
        
        return negatives
    
    
    def forward(self, center_idx: int, context_idx: int) -> Tuple[float, Dict]:
        """Forward pass with negative sampling."""
        v_center = self.W_in[center_idx]
        v_positive = self.W_out[context_idx]
        
        score_positive = np.dot(v_center, v_positive)
        sigmoid_positive = 1.0 / (1.0 + np.exp(-np.clip(score_positive, -10, 10)))
        
        negative_indices = self.sample_negatives(context_idx)
        v_negatives = self.W_out[negative_indices]
        scores_negative = v_negatives @ v_center
        sigmoid_negative = 1.0 / (1.0 + np.exp(-np.clip(scores_negative, -10, 10)))
        
        loss_positive = -np.log(sigmoid_positive + 1e-10)
        loss_negative = -np.sum(np.log(1.0 - sigmoid_negative + 1e-10))
        loss = loss_positive + loss_negative
        
        cache = {
            'center_idx': center_idx,
            'context_idx': context_idx,
            'negative_indices': negative_indices,
            'v_center': v_center,
            'v_positive': v_positive,
            'v_negatives': v_negatives,
            'sigmoid_positive': sigmoid_positive,
            'sigmoid_negative': sigmoid_negative
        }
        
        return loss, cache
    
    
    def backward(self, cache: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass - compute gradients."""
        center_idx = cache['center_idx']
        context_idx = cache['context_idx']
        negative_indices = cache['negative_indices']
        v_center = cache['v_center']
        v_positive = cache['v_positive']
        v_negatives = cache['v_negatives']
        sigmoid_positive = cache['sigmoid_positive']
        sigmoid_negative = cache['sigmoid_negative']
        
        dscore_positive = sigmoid_positive - 1.0
        dscore_negative = sigmoid_negative
        
        dv_center = v_positive * dscore_positive + v_negatives.T @ dscore_negative
        
        dW_in = np.zeros((1, self.embed_dim), dtype=np.float64)
        dW_in[0] = dv_center
        
        dW_out = np.zeros((1 + self.k, self.embed_dim), dtype=np.float64)
        dW_out[0] = v_center * dscore_positive
        
        for i in range(self.k):
            dW_out[i + 1] = v_center * dscore_negative[i]
        
        indices_in = np.array([center_idx], dtype=np.int32)
        indices_out = np.concatenate([[context_idx], negative_indices]).astype(np.int32)
        
        return (indices_in, dW_in), (indices_out, dW_out)
    
    
    def get_embeddings(self, use_output: bool = False) -> np.ndarray:
        """Get word embeddings."""
        if use_output:
            return self.W_out.copy()
        return self.W_in.copy()