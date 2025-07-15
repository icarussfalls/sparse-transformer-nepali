# Adaptive Sparse Transformers for Neural Machine Translation

## Overview
Implementation of adaptive sparse transformers for English-Nepali translation, featuring novel sparse attention mechanisms and visualization tools. The model achieves significant improvements over vanilla transformers through adaptive sparsification techniques.

## Key Results
- **BLEU Scores** (at epoch 14):
  ```
  BLEU-1: 41.78%
  BLEU-2: 29.45%
  BLEU-3: 21.34%
  BLEU-4: 14.61%
  ```
- Successful implementation of **Entmax-α** and **Sparsemax** attention
- Comprehensive attention visualization and sparsity analysis tools

## Features

### Model Variants
- **Vanilla Transformer**: Baseline model with standard attention
- **Fixed Sparse Transformer**: Implementation with predefined sparse patterns
- **Adaptive Sparse Transformer**: Dynamic sparse attention using:
  - Entmax-α (α=1.5)
  - Sparsemax attention
  - Adaptive mechanisms for efficient computation

### Architecture
```python
{
    'd_model': 512,          # Model dimension
    'd_ff': 1024,           # Feed-forward dimension
    'N': 4,                 # Number of layers
    'h': 4,                 # Number of attention heads
    'dropout': 0.1,         # Dropout rate
    'batch_size': 64        # Training batch size
}
```

### Visualization Tools
- Attention heatmaps for encoder/decoder layers
- Sparsity pattern analysis
- Cross-attention flow visualization
- Layer-wise attention distribution plots

## Installation

```bash
# Clone repository
git clone https://github.com/icarussfalls/sparse-transformer-nepali.git
cd sparse-transformer-nepali

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

### Visualization
```bash
python visualization.py
```

### Configuration
Modify `config.py` to adjust model parameters:
```python
config = {
    'use_adaptive_sparse': True,
    'attn_type': "sparsemax",  # or "entmax15"
    'visualize': True
}
```

## Project Structure
```
.
├── model.py                 # Vanilla transformer implementation
├── adaptive_sparse_model.py # Adaptive sparse transformer
├── train.py                # Training pipeline
├── visualization.py        # Attention visualization tools
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

## Results

### Sparsity Analysis
- Natural sparsification through Entmax-α
- Automatic learning of important attention connections
- Reduced computational complexity in higher layers

### Performance
- Consistent improvement over vanilla transformers
- Better handling of long-range dependencies
- Efficient attention allocation for morphologically rich languages

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

## License
MIT

## Citation
```bibtex
@article{correia2019adaptively,
  title={Adaptively Sparse Transformers},
  author={Correia, Gonçalo M and Niculae, Vlad and Martins, André FT},
  journal={arXiv preprint arXiv:1909.00015},
  year={2019}
}
```

## Acknowledgments
- Based on the Adaptively Sparse Transformers paper
- Nepali-English parallel corpus from AI4Bharat