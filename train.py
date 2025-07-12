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

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import re

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sentencepiece as spm

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


# simple greedy decode
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]") 

    # precomputing the encoder output and reusing it for each step
    # encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model(source, decoder_input, source_mask, decoder_mask)

        # get next token
        prob = out[:, -1]  # (batch, vocab_size)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ],
            dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # code from huggingface
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # the data only has train split so
    ds_all = load_dataset(f"{config['data_source']}", "default", split='train')

    # Shuffle and select a random 10% subset
    total_len = len(ds_all)
    subset_size = int(0.1 * total_len)
    indices = torch.randperm(total_len).tolist()[:subset_size]
    ds_raw = ds_all.select(indices)
    print(f"Using {subset_size} random samples out of {total_len}")

    ds_all = ds_raw # setting to only 10% of the data to train faster

    # build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_all, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_all, config['lang_tgt'])

    # now 90% for training and remaning for validation
    train_ds_size = int(len(ds_all) * 0.9)
    val_ds_size = len(ds_all) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_all, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])

    # find the minimum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_all:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentences {max_len_tgt}')
    
    # we find the max length to decide seq_len that is large enough to cover most sentences, but not so large that memory is wasted on padding

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    # (src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int = None
    if config['use_sparse']:
        print('building sparse transformers')
        model = build_sparse_transformer(vocab_src_len, vocab_tgt_len, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model = config['d_model'], N = config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'], block_size=config['sparse_block_size'], stride=config['sparse_stride'])
        return model
    if config['use_adaptive_sparse']:
        print('building adaptive sparse model')
        model = build_adaptive_sparse_transformer(vocab_src_len, vocab_tgt_len, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model = config['d_model'], N = config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'], attn_type=config['attn_type'])
        return model
    model = build_transformer(vocab_src_len, vocab_tgt_len, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model = config['d_model'], N = config['N'], h=config['h'], dropout=config['dropout'], d_ff=config['d_ff'])
    print('building vanilla transformers')
    return model


def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, seq_len, device, print_msg, global_step, writer, loss_fn):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_dataloader, desc="Validation")
        for batch in val_bar:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            with autocast():
                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            total_loss += loss.item()
            total_tokens += label.ne(tokenizer_tgt.token_to_id('[PAD]')).sum().item()
            
            val_bar.set_postfix({'val_loss': f"{total_loss/total_tokens:.4f}"})
    
    # Log validation metrics
    avg_loss = total_loss / total_tokens
    writer.add_scalar('val/loss', avg_loss, global_step)
    
    # Generate a sample translation
    if print_msg:
        translate_sample(model, tokenizer_src, tokenizer_tgt, device, print_msg)
    
    model.train()
    return avg_loss
def train_model(config, model=None, train_dataloader=None, val_dataloader=None, tokenizer_src=None, tokenizer_tgt=None):
    # Get model and data if not provided (for non-distributed training)
    if model is None or train_dataloader is None:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    device = torch.device(f"cuda:{config['gpu']}" if config['gpu'] is not None else "cuda")
    model = model.to(device)
    
    # lets make sure the weights folder exists
    Path(f"{config['data_source']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # tensorboard part
    writer = SummaryWriter(config['experiment_name'])
    
    # Get the model filename for preloading
    model_filename = None
    if config['preload'] == 'latest':
        model_filename = latest_weight_file_path(config)
    elif config['preload']:  # If preload is a specific filename
        model_filename = get_weights_file_path(config, config['preload'])
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    def get_lr_scheduler(optimizer, d_model, warmup_steps=4000):
        """
        Implements the learning rate schedule from the Transformer paper:
        lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        """
        def lr_lambda(current_step):
            # Warmup + decay schedule
            current_step = max(1, current_step)  # Prevent division by zero
            factor = d_model ** (-0.5)
            return factor * min(
                current_step ** (-0.5),
                current_step * warmup_steps ** (-1.5)
            )
        
        return LambdaLR(optimizer, lr_lambda)
    scheduler = get_lr_scheduler(optimizer, config['d_model'])

    # Initialize epoch and global_step
    initial_epoch = 0
    global_step = 0

    # Load checkpoint if exists
    if model_filename and os.path.exists(model_filename):
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        initial_epoch = state['epoch'] + 1
        global_step = state.get('global_step', 0)
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    scaler = GradScaler() if device == "cuda" else None

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        accumulated_loss = 0  # Initialize here
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        
        # Initialize epoch statistics
        epoch_stats = {
            'loss': 0.0,
            'steps': 0,
            'lr': scheduler.get_last_lr()[0]
        }
        
        for i, batch in enumerate(batch_iterator):
            # Move to GPU and clear previous gradients
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            # Forward pass with memory optimization
            if scaler is not None:
                with autocast():
                    proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    loss = loss / config['gradient_accumulation_steps']

                # Backward pass
                scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                # Update weights if we've accumulated enough steps
                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    epoch_stats['steps'] += 1  # Increment before division
                    epoch_stats['loss'] += accumulated_loss

                    # Unscale gradients for any gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    # Update statistics
                    epoch_stats['loss'] += accumulated_loss
                    epoch_stats['steps'] += 1
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Update progress bar
                    batch_iterator.set_postfix({
                        'loss': f"{accumulated_loss:.4f}",
                        'avg_loss': f"{epoch_stats['loss']/epoch_stats['steps']:.4f}",
                        'lr': f"{current_lr:.6f}"
                    })
                    
                    # Log to TensorBoard
                    writer.add_scalar('train/loss', accumulated_loss, global_step)
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
                    
                    accumulated_loss = 0
                    global_step += 1
        
        # End of epoch logging
        avg_loss = epoch_stats['loss'] / max(epoch_stats['steps'], 1)  # Prevent division by zero
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {epoch_stats['lr']:.6f}")
        print_gpu_memory()  # Add memory monitoring
        
        # Run validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            val_loss = run_validation(
                model, val_dataloader, tokenizer_src, tokenizer_tgt,
                config['seq_len'], device, 
                lambda msg: batch_iterator.write(msg),
                global_step, writer, loss_fn  # Added loss_fn here
            )
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save model checkpoint
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'train_loss': avg_loss,
                'val_loss': val_loss
            }, model_filename)
            print(f"Saved checkpoint: {model_filename}")

def print_gpu_memory():
    """Print GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved: {reserved:.2f} GB")

def translate_sample(model, tokenizer_src, tokenizer_tgt, device, print_msg):
    sample_text = "This is a test sentence."
    print_msg("\nSample Translation:")
    print_msg(f"Source: {sample_text}")
    
    # Tokenize and translate
    model.eval()
    with torch.no_grad():
        # Implement your translation logic here
        translated = greedy_decode(model, sample_text, tokenizer_src, tokenizer_tgt, device)
    
    print_msg(f"Translation: {translated}\n")
    model.train()

