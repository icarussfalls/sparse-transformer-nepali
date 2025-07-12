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

import logging
# Suppress SentencePiece verbose logs
os.environ['SENTENCEPIECE_MINLOGLEVEL'] = '2'
logging.getLogger('sentencepiece').setLevel(logging.ERROR)


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
    """Build or load SentencePiece tokenizer - distributed training safe"""
    tokenizer_path = Path(f"tokenizer_{lang}.model")
    
    # Check if we're in distributed mode
    is_distributed = torch.distributed.is_initialized()
    is_main_process = not is_distributed or torch.distributed.get_rank() == 0
    
    # Only main process builds the tokenizer
    if is_main_process and not tokenizer_path.exists():
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
            vocab_size=16000,
            character_coverage=0.9995,
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
            shuffle_input_sentence=True,
            minloglevel=2  # Suppress verbose logs
        )
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"SentencePiece tokenizer saved to {tokenizer_path}")
    
    # Wait for main process to finish building tokenizer
    if is_distributed:
        torch.distributed.barrier()
    
    # All processes load the tokenizer
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
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

def get_ds(config):
    # Load dataset
    ds_all = load_dataset(f"{config['data_source']}", "default", split='train')
    
    # Shuffle and select subset
    total_len = len(ds_all)
    subset_size = int(0.1 * total_len)
    indices = torch.randperm(total_len).tolist()[:subset_size]
    ds_raw = ds_all.select(indices)
    print(f"Using {subset_size} random samples out of {total_len}")
    
    # FIRST: Split into train/val BEFORE building tokenizers
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # SECOND: Build tokenizers ONLY on training data
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])  # Only train data
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])  # Only train data

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

    # Create DataLoaders with appropriate batch sizes
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=config['val_batch_size'],  # Use validation batch size
        shuffle=False,  # No need to shuffle validation
        num_workers=4,
        pin_memory=True
    )
    
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
    
    # Store all predictions and references for BLEU calculation
    all_predictions = []
    all_references = []
    sample_translations = []
    
    with torch.no_grad():
        # Only show progress bar on main process
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        if is_main_process:
            val_bar = tqdm(val_dataloader, desc="Validation")
        else:
            val_bar = val_dataloader
            
        for batch_idx, batch in enumerate(val_bar):
            # Handle batch format
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            
            # Forward pass (no autocast for validation)
            proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            
            total_loss += loss.item()
            total_tokens += label.ne(tokenizer_tgt.token_to_id('[PAD]')).sum().item()
            
            # Get predictions for BLEU calculation
            pred_tokens = torch.argmax(proj_output, dim=-1)
            
            # Get special token IDs
            pad_id = tokenizer_tgt.token_to_id('[PAD]')
            sos_id = tokenizer_tgt.token_to_id('[SOS]')
            eos_id = tokenizer_tgt.token_to_id('[EOS]')
            
            # Process each sample in the batch
            for i in range(encoder_input.size(0)):
                try:
                    # Get reference and prediction tokens
                    ref_tokens = label[i].cpu().numpy()
                    pred_tokens_i = pred_tokens[i].cpu().numpy()
                    
                    # Remove special tokens
                    ref_tokens = ref_tokens[ref_tokens != pad_id]
                    ref_tokens = ref_tokens[ref_tokens != sos_id] 
                    ref_tokens = ref_tokens[ref_tokens != eos_id]
                    
                    pred_tokens_i = pred_tokens_i[pred_tokens_i != pad_id]
                    pred_tokens_i = pred_tokens_i[pred_tokens_i != sos_id]
                    pred_tokens_i = pred_tokens_i[pred_tokens_i != eos_id]
                    
                    if len(ref_tokens) > 0 and len(pred_tokens_i) > 0:
                        # Decode tokens to text
                        ref_text = tokenizer_tgt.decode(ref_tokens)
                        pred_text = tokenizer_tgt.decode(pred_tokens_i)
                        
                        # Split into words for BLEU calculation
                        ref_words = ref_text.split()
                        pred_words = pred_text.split()
                        
                        if len(ref_words) > 0 and len(pred_words) > 0:
                            all_references.append([ref_words])
                            all_predictions.append(pred_words)
                            
                            # Store sample translations (first 3 examples)
                            if len(sample_translations) < 3:
                                src_tokens = encoder_input[i].cpu().numpy()
                                src_tokens = src_tokens[src_tokens != tokenizer_src.token_to_id('[PAD]')]
                                src_text = tokenizer_src.decode(src_tokens)
                                sample_translations.append((src_text, pred_text, ref_text))
                                
                except Exception as e:
                    # Skip this sample if decoding fails
                    continue
            
            # Update progress bar only on main process
            if is_main_process:
                val_bar.set_postfix({
                    'val_loss': f"{total_loss/max(total_tokens, 1):.4f}",
                    'samples': len(all_predictions)
                })
    
    # Calculate corpus-level BLEU scores
    bleu_scores = {}
    main_bleu = 0
    
    if len(all_predictions) > 0 and len(all_references) > 0:
        smoothing = SmoothingFunction().method1
        try:
            bleu_scores['BLEU-1'] = corpus_bleu(all_references, all_predictions, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_scores['BLEU-2'] = corpus_bleu(all_references, all_predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_scores['BLEU-3'] = corpus_bleu(all_references, all_predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_scores['BLEU-4'] = corpus_bleu(all_references, all_predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            main_bleu = bleu_scores['BLEU-4']
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            bleu_scores = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
            main_bleu = 0
    else:
        bleu_scores = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
        main_bleu = 0
    
    # Print validation summary (only on main process)
    if is_main_process:
        print("\n" + "="*70)
        print("Validation Summary:")
        print(f"Average Loss: {total_loss/max(total_tokens, 1):.4f}")
        print(f"Samples evaluated: {len(all_predictions)}")
        
        # Print BLEU scores
        print("\nCorpus BLEU Scores:")
        for metric, score in bleu_scores.items():
            print(f"  {metric}: {score*100:.2f}%")
        
        print(f"\n>>> Main Metric: Corpus BLEU-4 = {main_bleu*100:.2f}% <<<")
        
        print("\nSample Translations:")
        for i, (src, pred, ref) in enumerate(sample_translations, 1):
            print(f"\nExample {i}:")
            print(f"  Source:     {src}")
            print(f"  Generated:  {pred}")
            print(f"  Reference:  {ref}")
        print("="*70 + "\n")
    
    # Log to tensorboard (only if writer exists)
    if writer:
        writer.add_scalar('val/loss', total_loss/max(total_tokens, 1), global_step)
        writer.add_scalar('val/BLEU-4', main_bleu*100, global_step)
        for metric, score in bleu_scores.items():
            writer.add_scalar(f'val/{metric}', score*100, global_step)
    
    model.train()
    return total_loss/max(total_tokens, 1)
def train_model(config, model=None, train_dataloader=None, val_dataloader=None, tokenizer_src=None, tokenizer_tgt=None):
    # Get model and data if not provided (for non-distributed training)
    if model is None or train_dataloader is None:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    device = torch.device(f"cuda:{config['gpu']}" if config['gpu'] is not None else "cuda")
    model = model.to(device)
    
    # Check if we're in distributed mode
    is_distributed = torch.distributed.is_initialized()
    is_main_process = not is_distributed or torch.distributed.get_rank() == 0
    
    # Only main process creates directories and initializes tensorboard
    if is_main_process:
        Path(f"{config['data_source']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(config['experiment_name'])
    else:
        writer = None
    
    # Get the model filename for preloading
    model_filename = None
    if config['preload'] == 'latest':
        model_filename = latest_weight_file_path(config)
    elif config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'], 
        steps_per_epoch=len(train_dataloader) // config['gradient_accumulation_steps'],
        epochs=config['num_epochs'],
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Initialize epoch and global_step
    initial_epoch = 0
    global_step = 0

    # Load checkpoint if exists (all processes need to load the same weights)
    if model_filename and os.path.exists(model_filename):
        if is_main_process:
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
        
        if is_main_process:
            print(f"Resuming from epoch {initial_epoch}, global step {global_step}")
    else:
        if is_main_process:
            print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    scaler = GradScaler() if device.type == "cuda" else None

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        accumulated_loss = 0
        
        # Only show progress bar on main process
        if is_main_process:
            batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        else:
            batch_iterator = train_dataloader
        
        for i, batch in enumerate(batch_iterator):
            # Move to GPU
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    loss = loss / config['gradient_accumulation_steps']

                scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Only update progress bar on main process
                    if is_main_process:
                        batch_iterator.set_postfix({
                            'loss': f"{accumulated_loss:.4f}",
                            'avg_loss': f"{epoch_loss/num_batches:.4f}",
                            'lr': f"{current_lr:.6f}"
                        })
                    
                    accumulated_loss = 0
                    global_step += 1
            else:
                # CPU training path
                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss = loss / config['gradient_accumulation_steps']
                
                loss.backward()
                accumulated_loss += loss.item()
                
                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    current_lr = scheduler.get_last_lr()[0]
                    
                    if is_main_process:
                        batch_iterator.set_postfix({
                            'loss': f"{accumulated_loss:.4f}",
                            'avg_loss': f"{epoch_loss/num_batches:.4f}",
                            'lr': f"{current_lr:.6f}"
                        })
                    
                    accumulated_loss = 0
                    global_step += 1
        
        # End of epoch logging - only on main process
        if is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch} Summary:")
            print(f"Average Training Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print_gpu_memory()
            
            # Log to tensorboard
            if writer:
                writer.add_scalar('train/loss', avg_loss, epoch)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    
        # Run validation - only on main process
        if epoch % 2 == 0 and is_main_process:
            val_loss = run_validation(
                model, val_dataloader, tokenizer_src, tokenizer_tgt,
                config['seq_len'], device, 
                lambda msg: print(msg),
                global_step, writer, loss_fn
            )
            print(f"Validation Loss: {val_loss:.4f}")
            
            # ONLY run visualizations every 5-10 epochs, not every validation!
            if config.get('visualize', False) and epoch % 10 == 0:  # Every 10 epochs only
                try:
                    print("Generating visualizations...")
                    log_visualizations(
                        model, 
                        tokenizer_src, 
                        tokenizer_tgt, 
                        global_step, 
                        save_dir=f"visualizations/{config['experiment_name']}"
                    )
                    print("Visualizations saved!")
                except Exception as e:
                    print(f"Error generating visualizations: {e}")
            
            # Save model checkpoint - only main process
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

    # Close tensorboard writer on main process
    if is_main_process and writer:
        writer.close()
def print_gpu_memory():
    """Print GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved: {reserved:.2f} GB")

def translate_sample(model, tokenizer_src, tokenizer_tgt, device, print_msg, max_len=300):  # Add max_len parameter
    """Generate a sample translation during validation"""
    sample_text = "This is a test sentence."
    print_msg("\nSample Translation:")
    print_msg(f"Source: {sample_text}")
    
    # Tokenize the source text
    source_tokens = tokenizer_src.encode(sample_text)
    source = torch.tensor(source_tokens.ids).unsqueeze(0)  # Add batch dimension
    source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0)  # Create attention mask
    
    # Move tensors to device
    source = source.to(device)
    source_mask = source_mask.to(device)
    
    # Tokenize and translate
    model.eval()
    with torch.no_grad():
        translated = greedy_decode(
            model, 
            source,  # Pass tokenized source
            source_mask,  # Pass source mask
            tokenizer_src, 
            tokenizer_tgt, 
            max_len=max_len,
            device=device
        )
        
        # Convert token IDs to text
        translated_tokens = translated.cpu().numpy()
        translated_text = tokenizer_tgt.decode(translated_tokens)
    
    print_msg(f"Translation: {translated_text}\n")
    model.train()

