"""
Skip-gram word2vec model with negative sampling.

This module implements the core model class that encapsulates:
- Forward pass (with negative sampling)
- Backward pass (gradient computation)
- Embedding initialization
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SkipGramNegativeSampling:
    """
    Skip-gram word2vec model with negative sampling.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimensionality of word embeddings
        negative_samples: Number of negative samples per positive example
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        negative_samples: int = 5,
        seed: Optional[int] = None
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.k = negative_samples
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize embeddings with small random values
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01
    
    
    def sample_negatives(self, context_idx: int) -> np.ndarray:
        """
        Sample k negative words (not equal to context_idx).
        
        Args:
            context_idx: Index to exclude (positive sample)
        
        Returns:
            Array of k negative word indices
        """
        negatives = []
        while len(negatives) < self.k:
            neg_idx = np.random.randint(0, self.vocab_size)
            if neg_idx != context_idx and neg_idx not in negatives:
                negatives.append(neg_idx)
        
        return np.array(negatives)
    
    
    def forward(
        self,
        center_idx: int,
        context_idx: int
    ) -> Tuple[float, Dict]:
        """
        Forward pass with negative sampling.
        
        Args:
            center_idx: Center word index
            context_idx: Positive context word index
        
        Returns:
            loss: Scalar loss value
            cache: Dictionary with intermediate values for backward pass
        """
        # Get center word embedding
        v_center = self.W_in[center_idx]
        
        # Positive sample
        v_positive = self.W_out[context_idx]
        score_positive = np.dot(v_center, v_positive)
        sigmoid_positive = 1 / (1 + np.exp(-score_positive))
        
        # Negative samples
        negative_indices = self.sample_negatives(context_idx)
        v_negatives = self.W_out[negative_indices]
        scores_negative = v_negatives @ v_center
        sigmoid_negative = 1 / (1 + np.exp(-scores_negative))
        
        # Loss
        loss_positive = -np.log(sigmoid_positive + 1e-10)
        loss_negative = -np.sum(np.log(1 - sigmoid_negative + 1e-10))
        loss = loss_positive + loss_negative
        
        # Cache for backward pass
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
        """
        Backward pass - compute gradients.
        
        Args:
            cache: Dictionary from forward pass
        
        Returns:
            dW_in: Gradient for input embeddings
            dW_out: Gradient for output embeddings
        """
        center_idx = cache['center_idx']
        context_idx = cache['context_idx']
        negative_indices = cache['negative_indices']
        v_center = cache['v_center']
        v_positive = cache['v_positive']
        v_negatives = cache['v_negatives']
        sigmoid_positive = cache['sigmoid_positive']
        sigmoid_negative = cache['sigmoid_negative']
        
        # Gradients w.r.t. scores
        dscore_positive = sigmoid_positive - 1
        dscore_negative = sigmoid_negative
        
        # Gradient w.r.t. v_center
        dv_center_positive = v_positive * dscore_positive
        dv_center_negative = v_negatives.T @ dscore_negative
        dv_center = dv_center_positive + dv_center_negative
        
        # Gradient w.r.t. W_in (only update center word row)
        dW_in = np.zeros_like(self.W_in)
        dW_in[center_idx] = dv_center
        
        # Gradient w.r.t. W_out
        dW_out = np.zeros_like(self.W_out)
        dW_out[context_idx] = v_center * dscore_positive
        
        for i, neg_idx in enumerate(negative_indices):
            dW_out[neg_idx] += v_center * dscore_negative[i]
        
        return dW_in, dW_out
    
    
    def get_embeddings(self, use_output: bool = False) -> np.ndarray:
        """
        Get word embeddings.
        
        Args:
            use_output: If True, return output embeddings (W_out)
                       If False, return input embeddings (W_in)
        
        Returns:
            Embedding matrix (vocab_size, embed_dim)
        """
        if use_output:
            return self.W_out.copy()
        return self.W_in.copy()