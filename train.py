from model import build_transformer
from sparse_model import build_sparse_transformer
from adaptive_sparse_model import build_adaptive_sparse_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weight_file_path
from visualization import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import re

# Huggingface datasets and tokenizers
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sentencepiece as spm
import numpy as np

def get_or_build_tokenizer(config, ds, lang):
    """Build or load SentencePiece tokenizer"""
    tokenizer_path = Path(f"tokenizer_{lang}.model")
    
    if not tokenizer_path.exists():
        print(f"Building SentencePiece tokenizer for {lang}...")
        
        # Create temporary text file for training
        temp_file = f"temp_{lang}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for item in ds:
                text = item[lang].strip()
                if text:  # Only add non-empty lines
                    f.write(text + '\n')
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=f"tokenizer_{lang}",
            vocab_size=16000,  # Smaller vocab for faster training
            character_coverage=0.9995,  # Good for multilingual
            model_type='bpe',
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=3,
            pad_piece='[PAD]',
            eos_piece='[EOS]',
            unk_piece='[UNK]',
            bos_piece='[SOS]',
            user_defined_symbols=['[MASK]'],
            max_sentence_length=4192,
            shuffle_input_sentence=True
        )
        
        # Clean up temp file
        os.remove(temp_file)
        print(f"SentencePiece tokenizer saved to {tokenizer_path}")
    
    # Load the tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    
    return SentencePieceTokenizer(sp)

class SentencePieceTokenizer:
    """Wrapper to make SentencePiece compatible with existing code"""
    
    def __init__(self, sp_model):
        self.sp = sp_model
        self._vocab_size = self.sp.get_piece_size()
        
        # Define special tokens
        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        
    def encode(self, text):
        """Encode text to token IDs"""
        ids = self.sp.encode_as_ids(text)
        return TokenizedResult(ids)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        # Handle numpy arrays
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        # Handle torch tensors
        elif hasattr(ids, 'cpu'):
            ids = ids.cpu().numpy().tolist()
        # Ensure it's a list
        if not isinstance(ids, list):
            ids = [ids]
        return self.sp.decode_ids(ids)
    
    def token_to_id(self, token):
        """Get token ID from token string"""
        if token == '[PAD]':
            return self.pad_id
        elif token == '[EOS]':
            return self.eos_id
        elif token == '[UNK]':
            return self.unk_id
        elif token == '[SOS]':
            return self.bos_id
        else:
            return self.sp.piece_to_id(token)
    
    def id_to_token(self, id):
        """Get token string from ID"""
        return self.sp.id_to_piece(id)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self._vocab_size

class TokenizedResult:
    """Wrapper for tokenized result to match HuggingFace tokenizer interface"""
    
    def __init__(self, ids):
        self.ids = ids

class BilingualDataset(Dataset):
    """Dataset for bilingual translation compatible with SentencePiece tokenizers"""
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Special tokens
        self.sos_token = tokenizer_tgt.token_to_id('[SOS]')
        self.eos_token = tokenizer_tgt.token_to_id('[EOS]')
        self.pad_token = tokenizer_tgt.token_to_id('[PAD]')
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]
        
        # Tokenize source and target
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Add SOS and EOS tokens to decoder input and label
        dec_input_tokens = [self.sos_token] + dec_input_tokens
        label = dec_input_tokens + [self.eos_token]
        
        # Pad or truncate sequences
        enc_input_tokens = self._pad_or_truncate(enc_input_tokens, self.seq_len)
        dec_input_tokens = self._pad_or_truncate(dec_input_tokens, self.seq_len)
        label = self._pad_or_truncate(label, self.seq_len)
        
        # Create attention masks
        encoder_mask = (torch.tensor(enc_input_tokens) != self.pad_token).unsqueeze(0).int()
        decoder_mask = self._create_decoder_mask(torch.tensor(dec_input_tokens))
        
        return {
            'encoder_input': torch.tensor(enc_input_tokens, dtype=torch.long),
            'decoder_input': torch.tensor(dec_input_tokens, dtype=torch.long),
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
    
    def _pad_or_truncate(self, tokens, max_len):
        """Pad or truncate token sequence to max_len"""
        if len(tokens) > max_len:
            return tokens[:max_len]
        else:
            return tokens + [self.pad_token] * (max_len - len(tokens))
    
    def _create_decoder_mask(self, decoder_input):
        """Create causal mask for decoder"""
        seq_len = decoder_input.size(0)
        decoder_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).int()
        decoder_mask = decoder_mask == 0
        return decoder_mask & (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)

def get_ds(config):
    # Load dataset
    ds_all = load_dataset(f"{config['data_source']}", "default", split='train')

    # Shuffle and select a random subset
    total_len = len(ds_all)
    subset_size = int(0.1 * total_len)
    indices = torch.randperm(total_len).tolist()[:subset_size]
    ds_raw = ds_all.select(indices)
    print(f"Using {subset_size} random samples out of {total_len}")

    # Build SentencePiece tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split into train/val
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 
                               config['lang_src'], config['lang_tgt'], 
                               seq_len=config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, 
                             config['lang_src'], config['lang_tgt'], 
                             seq_len=config['seq_len'])

    # Find max lengths (optional - for analysis)
    max_len_src = 0
    max_len_tgt = 0
    
    for item in list(ds_raw)[:1000]:  # Sample first 1000 for speed
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def causal_mask(size):
    """Create causal mask for decoder"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

# Keep your existing model functions and validation functions unchanged
# ... (rest of your code remains the same)

