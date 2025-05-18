# English-Italian Transformer Translation

This repository contains a **Transformer-based sequence-to-sequence model** for English-to-Italian translation, implemented from scratch in PyTorch. The project covers the full pipeline: data loading, tokenization, model definition, training, validation, and inference (with both greedy and beam search decoding).

---

## Features

- **Custom Transformer model** (encoder-decoder) in PyTorch
- **Custom tokenizers** trained on your parallel corpus (using HuggingFace [tokenizers](https://github.com/huggingface/tokenizers))
- **Training and validation** with metrics: BLEU, Word Error Rate (WER), Character Error Rate (CER)
- **Greedy and beam search decoding** for inference
- **Configurable hyperparameters** via `config.py`
- **Supports CPU, CUDA, and Apple M1 (MPS) devices**
- **Automatic checkpointing and TensorBoard logging**
- **Easy extensibility for other language pairs or datasets**

---

## Project Structure

```
.
├── config.py           # Configuration and checkpoint utilities
├── dataset.py          # Bilingual dataset and preprocessing
├── ds_raw.jsonl        # Example of raw parallel data (en/it)
├── model.py            # Transformer model and embeddings
├── train.py            # Training and validation loop
├── translate.py        # Inference script (greedy & beam search)
├── tokenizer_en.json   # Trained English tokenizer
├── tokenizer_it.json   # Trained Italian tokenizer
├── weights/            # Model checkpoints
└── runs/               # TensorBoard logs
```

---

## How It Works

### Data Loading & Tokenization

- Loads parallel English-Italian data (e.g., from OPUS Books).
- Trains or loads a **WordLevel tokenizer** for each language.
- Tokenizes and pads sentences to a fixed sequence length.

### Model

- Implements a **Transformer encoder-decoder** architecture.
- Embedding layers, positional encoding, multi-head attention, and feed-forward blocks.
- Configurable model size (`d_model`), sequence length, and vocabulary size.

### Training

- Uses cross-entropy loss with label smoothing.
- Supports multi-GPU training with `nn.DataParallel`.
- Saves checkpoints after each epoch.
- Logs training/validation loss and metrics to TensorBoard.

### Validation

- Computes BLEU, WER, and CER on validation set.
- Prints sample translations for qualitative inspection.

### Inference

- `translate.py` script for translating sentences using greedy or beam search decoding.
- Supports running on CPU, CUDA, or Apple M1 (MPS).

---

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/transformers-en-it.git
    cd transformers-en-it
    ```

2. **Install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    Required packages include:
    - torch
    - datasets
    - tokenizers
    - torchmetrics
    - tensorboard
    - tqdm

3. **Prepare data:**
    - Download the OPUS Books English-Italian dataset or another parallel corpus.
    - Preprocess and format as `ds_raw.jsonl` (see example in repo).

4. **Train tokenizers:**
    - The code will automatically train and save `tokenizer_en.json` and `tokenizer_it.json` if they do not exist.

---

## Training

Edit `config.py` to adjust hyperparameters as needed.

Start training:
```bash
python train.py
```
- Model checkpoints will be saved in `weights/`.
- TensorBoard logs will be saved in `runs/`.

Monitor training with:
```bash
tensorboard --logdir runs/
```

---

## Inference

Translate a sentence using greedy or beam search:
```bash
# Greedy decoding
python translate.py "I am not a very good student."

# Beam search (beam width 5)
python translate.py "I am not a very good student." --beam 5
```

---

## Configuration

All hyperparameters and paths are set in `config.py`:
- `batch_size`, `num_epochs`, `lr`, `seq_len`, `d_model`, etc.
- `tokenizer_file`: Path to tokenizer files
- `model_folder`, `experiment_name`: Output directories

---

## Tips for Better Results

- Use as much parallel data as possible.
- Train your tokenizer on the full dataset with a vocab size of 16k–32k.
- For best results, consider fine-tuning a pretrained model (e.g., MarianMT, mBART).
- Monitor BLEU and validation loss in TensorBoard.
- Try to overfit a tiny subset (10–20 pairs) to debug your pipeline.

---

## License

This project is for research and educational purposes.  
Please check the OPUS Books dataset license for data usage terms.

---

## Acknowledgements

- [OPUS Project](https://opus.nlpl.eu/)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [PyTorch](https://pytorch.org/)

---

## Contact

For questions or contributions, open an issue or pull request on GitHub.

