# Custom Transformer-Based Neural Machine Translation (English–Nepali)

This repository contains a **fully custom implementation of a Transformer-based neural machine translation system** for English–Nepali, built from scratch in PyTorch. Unlike using pre-built libraries, we developed every core component ourselves—including embeddings, positional encoding, multi-head attention, feed-forward layers, normalization, and the full encoder-decoder stack. The project is designed for both research flexibility and practical experimentation, with a focus on transparency, extensibility, and robust evaluation.

---

## What We Developed & Modified

### 1. **Custom Transformer Variants**
- **Vanilla Transformer:**  
  - Standard dense attention (softmax), as in Vaswani et al.
- **Sparse Transformers:**  
  - Block and strided sparse attention patterns for efficient long-sequence modeling.
- **Adaptively Sparse Transformers:**  
  - Implements [entmax15](https://arxiv.org/abs/1905.05702) and α-entmax ([learnable α per head](https://arxiv.org/abs/1909.00015)), allowing the model to learn the optimal sparsity pattern for each attention head.
- **All modules implemented from scratch:**  
  - InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock (with optional cross-head attention and sparse/entmax modes), FeedForward, LayerNormalization, Encoder/Decoder layers, and the final ProjectionLayer.
- **Pre-norm residual connections** and careful tensor shape management for stability and clarity.
- **Flexible configuration:**  
  - Easily adjust model depth, width, number of heads, dropout, sequence length, and **attention type** via `config.py`.

---

## How to Select Transformer Type

In `config.py`, set the following flags to choose your model:

```python
# For vanilla transformer (softmax attention)
'use_sparse': False,
'use_adaptive_sparse': False,
'attn_type': "softmax",

# For sparse transformer (block/strided)
'use_sparse': True,
'use_adaptive_sparse': False,
# (set block/stride params as needed)

# For adaptively sparse transformer (entmax15 or entmax_alpha)
'use_sparse': False,
'use_adaptive_sparse': True,
'attn_type': "entmax15",      # or "entmax_alpha"
```

---

## Data Handling & Preprocessing
- **Custom Dataset class (`dataset.py`):**  
  - Handles tokenization, padding, truncation, and special token management for both source and target languages.
  - Supports dynamic sequence length and batch size, and can easily be set to use only a fraction of the dataset for prototyping or debugging.
- **Tokenizer integration:**  
  - Plug in your own tokenizers; code is agnostic to tokenizer implementation.

### 3. **Training & Validation Pipeline**
- **Efficient training loop (`train.py`):**  
  - Supports multi-GPU training, mixed precision, and gradient accumulation.
  - Automatic checkpointing and resume-from-latest functionality.
  - Batch iterator with progress bar and loss tracking.
- **Validation and metrics:**  
  - Computes BLEU (using corpus BLEU for multi-sentence reliability), Character Error Rate (CER), and Word Error Rate (WER).
  - Handles empty predictions gracefully and logs warnings if the model outputs degenerate results.
  - All metrics are logged to both TensorBoard and a plain text file (`metrics_log.txt`) for easy tracking and analysis.

### 4. **Experiment Management**
- **Configurable experiments:**  
  - All hyperparameters and file paths are managed in `config.py` for reproducibility.
- **Logging:**  
  - Console, TensorBoard, and file logging for all key metrics and training progress.
- **Prototyping support:**  
  - Easily restrict training to a small subset of data for rapid iteration and debugging.

---

## Usage

1. **Install dependencies:**
    ```bash
    pip install torch nltk torchmetrics datasets tqdm
    ```

2. **Configure your experiment:**
    - Edit `config.py` to set batch size, sequence length, number of epochs, model size, etc.

3. **Train the model:**
    ```bash
    python train.py
    ```

4. **Monitor training:**
    - Use TensorBoard:  
      ```bash
      tensorboard --logdir runs/
      ```
    - Or check `metrics_log.txt` for validation metrics.

---

## File Structure

- `model.py` — All custom Transformer modules and attention mechanisms.
- `train.py` — Training loop, validation, metric logging, and checkpointing.
- `dataset.py` — Custom dataset and preprocessing utilities.
- `config.py` — Experiment configuration.
- `metrics_log.txt` — Validation metrics log (auto-generated).

---

## Notes & Best Practices

- **Prototyping:**  
  To debug or iterate quickly, set your data loader to use only a small subset of the dataset (see comments in `train.py` and `dataset.py`).
- **BLEU Calculation:**  
  BLEU is computed using corpus BLEU for reliable evaluation across batches.
- **Extensibility:**  
  The codebase is modular and well-documented, making it easy to adapt for other language pairs, research ideas, or architectural experiments.
- **Resource Management:**  
  Supports multi-GPU and mixed-precision training. Batch size and sequence length can be tuned for your hardware.

---

## References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- PyTorch documentation: https://pytorch.org/
- NLTK BLEU: https://www.nltk.org/_modules/nltk/translate/bleu_score.html

---

**This project is a foundation for research, learning, and practical neural machine translation. Happy translating!**