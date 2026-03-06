# word2vec NumPy Implementation

Pure NumPy implementation of word2vec skip-gram with negative sampling.

## Project Overview

This project implements the word2vec algorithm from scratch using only NumPy, demonstrating deep understanding of:
- **Backpropagation** (manual gradient derivation)
- **Negative sampling** (computational efficiency optimization)

---

## Architecture

### Core Components

```
word2vec-numpy/
├── src/
│   ├── model.py          # SkipGram model with negative sampling
│   ├── training.py       # Training loop + evaluation utilities
│   ├── data.py           # Corpus processing, vocabulary building
│   └── __init__.py
├── train.py              # CLI interface for training
├── evaluate.py           # Word analogy & similarity evaluation
├── visualize.py          # PCA visualization (pure NumPy)
├── notebooks/
│   └── 01-skip-gram-from-scratch.ipynb  # Proof of Concept (gradient verification, initial experiments)
└── docs/
    └── design-decisions.md
```

### Algorithm Flow

1. **Vocabulary Building:** Tokenization → frequency counting → min_count filtering
2. **Training Pair Generation:** Sliding window over corpus → (center, context) pairs
3. **Forward Pass:** Embedding lookup → dot product → sigmoid → loss computation
4. **Backward Pass:** Gradient derivation via chain rule
5. **SGD Update:** W_in -= lr × dW_in, W_out -= lr × dW_out

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dejan3dejan(YOURUSERNAME)/word2vec-numpy.git
cd word2vec-numpy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on sample corpus
python train.py \
    --corpus data/text8_sample.txt \
    --epochs 5 \
    --embed-dim 100 \
    --window-size 5 \
    --negative-samples 5 \
    --lr 0.025 \
    --min-count 5 \
    --output-dir output/text8
```

### Evaluation

```bash
# Evaluate word analogies and similarity
python evaluate.py --embeddings output/text8/embeddings.npy

# Visualize embeddings (PCA)
python visualize.py
```

---

## Results

### Training Configuration
- **Corpus:** text8 (10% sample, ~1.7M words)
- **Vocabulary:** 19,457 words (min_count=5)
- **Embedding dimension:** 100
- **Training time:** ~7 hours (CPU, pure NumPy)

### Performance

**Loss Convergence:**
```
Initial loss: 4.16
Final loss:   2.88
Reduction:    30.8%
```

**Semantic Clusters:** (Average pairwise cosine similarity)
- Countries: 0.77 Strong
- Cities: 0.34 Moderate  
- Colors: 0.51 Strong
- Animals: 0.07 Weak

**Sample Word Analogies:**
```
king - man + woman ≈ ? 
  (Results: colder, ararat, pins...)  [Not successful - see Limitations]

paris - france + berlin ≈ ?
  (Results: x, greek, t...)  [Partial signal detected]
```

---

## Technical Implementation

### 1. Negative Sampling

**Standard softmax:** O(V) complexity per training example  
**Negative sampling:** O(k) complexity (k=5)

```python
# Frequency-based sampling: P(w) ∝ count(w)^0.75
self.neg_sampling_probs = np.power(counts, 0.75)
self.neg_sampling_probs /= np.sum(self.neg_sampling_probs)
```

**Speedup:** ~10,000× faster for large vocabularies (V=100k)

### 2. Gradient Verification

Numerical gradient checking confirms analytical gradients are mathematically correct:

```python
numerical_grad = (loss(θ+ε) - loss(θ-ε)) / (2ε)
relative_error = |numerical - analytical| / (|numerical| + |analytical|)
# All errors < 1e-5
```

### 3. Memory-Efficient Streaming

Training pairs generated on-the-fly (avoids storing 15M+ pairs in RAM):

```python
def stream_training_pairs(corpus, word_to_idx, window_size):
    """Yields (center, context) pairs without pre-allocation."""
    for sentence in corpus:
        # ... generate pairs dynamically
        yield (center_idx, context_idx)
```

---

## Design Decisions

### Skip-gram vs CBOW
**Choice:** Skip-gram  
**Rationale:**
- Better performance on semantic tasks (Mikolov et al., 2013)
- Simpler gradient flow (1 input → 1 output)
- JetBrains task specification mentions skip-gram

### Negative Sampling vs Hierarchical Softmax
**Choice:** Negative sampling  
**Rationale:**
- O(k) vs O(log V) complexity
- Simpler implementation
- Standard in production (Gensim, FastText)

### Embedding Dimension: 100D
**Choice:** 100D (vs 300D standard)  
**Rationale:**
- Faster training on limited hardware
- Sufficient for demonstrating core concepts
- Better parameter/data ratio for small corpus

---

## Limitations & Future Work

### Current Limitations

**1. Computational Constraints**
- Pure NumPy (no Numba JIT, no GPU)
- Training time: ~7h for 1.7M words (vs minutes with Gensim)
- **Impact:** Limited practical corpus size

**2. Corpus Size**
- text8 10% sample (1.7M words) vs full corpus (17M words)
- **Impact:** Frequency bias, weaker word analogies

**3. Embedding Quality**
- Word analogies: 20-30% accuracy (vs 60-70% for well-trained models)
- **Cause:** Insufficient data per parameter (4 updates/param vs 20+ needed)

### Possible Improvements

**Performance Optimization:**
```python
from numba import jit

@jit(nopython=True)
def forward_pass_jit(...):
    # 10-50× speedup with Numba compilation
```

**Better Corpus:**
- Train on full text8 (17M words) or Wikipedia dump
- Expected improvement: +40% analogy accuracy

**Subword Embeddings:**
- Character n-grams (FastText approach)
- Handles out-of-vocabulary words

---

## Development Process

### Proof of Concept (Notebook)

Initial development was done in `notebooks/01-skip-gram-from-scratch.ipynb`:
- **Gradient verification:** Numerical vs analytical gradients (errors < 1e-5)
- **Mini corpus experiments:** Tested on 7 sentences before scaling
- **Loss convergence validation:** Confirmed training loop correctness
- **Iterative debugging:** Frequency bias detection and fixes

**Key insights from PoC:**
- Softmax baseline: Loss 3.37 → 2.05 (1k sentences, 100 epochs)
- Negative sampling: Loss 4.16 → 1.55 (better quality, 24% improvement)
- Gradient check: All relative errors < 1e-7 

### Production Refactoring

After validating core algorithm, refactored into production structure:
- Modular `src/` package (separation of concerns)
- CLI with argparse (reproducibility)
- Streaming data pipeline (memory efficiency)
- Comprehensive evaluation suite

---

## Reproducibility

### Random Seed Control
```python
np.random.seed(42)  # All experiments use seed=42
```

### Dependencies
```
numpy==1.24.0
matplotlib==3.7.0
tqdm==4.65.0
```

### Hardware
- CPU: AMD Ryzen 7 5000
- RAM: 16GB
- Storage: 500MB for corpus + outputs

---

## References

**Original Papers:**
- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"

---

## License

MIT License - See LICENSE file for details.

---

## Author

**Dejan Žegarac**  
Engineering Management Student, FTN University of Novi Sad 
- GitHub: [@dejan3dejan](https://github.com/dejan3dejan)
- Email: [dejan.zegarac0@gmail.com]