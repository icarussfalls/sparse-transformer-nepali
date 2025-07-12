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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    # model at eval mode
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get a console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)

    except:
        console_width = 80 # use 80 as default if console width can't be get
    
    with torch.no_grad(): # dont train here, no gradient calc
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "batch size must be one for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print the source, target, and model output
            print_msg('-'*console_width)
            print_msg(f"{f'source: ' :>12}{source_text}")
            print_msg(f"{f'target: ' :>12}{target_text}")
            print_msg(f"{f'predicted: ' :>12}{model_out_text}")

            if count == num_examples:
                break

    if writer:
        # evaluate the character error rate
        # compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # compute the BLEU metric
        # Tokenize by splitting on whitespace
        # metric = torchmetrics.BLEUScore()
        # if bleu doesn't works below, use this bleu = metric(predicted, [[ref] for ref in expected])
        
        # bleu = metric(predicted, expected)
        if not predicted or all(len(p.strip()) == 0 for p in predicted):
            bleu = 0.0
            cer = 1.0
            wer = 1.0
            print_msg("Warning: Empty prediction(s), metrics set to worst values.")
        else:
            # Normalization
            def normalize(text):
                return text.replace("“", "").replace("”", "").replace(",", "").replace(".", "").replace("।", "").strip()

            pred_norm = [normalize(p) for p in predicted]
            ref_norm = [normalize(r) for r in expected]

            # Tokenize each sentence
            pred_tok = [p.split() for p in pred_norm]
            ref_tok = [[r.split()] for r in ref_norm]  # Each reference must be a list of references

            smoothie = SmoothingFunction().method4
            bleu = corpus_bleu(ref_tok, pred_tok, smoothing_function=smoothie)

            # CER & WER
            metric = torchmetrics.CharErrorRate()
            cer = metric(pred_norm, ref_norm)
            metric = torchmetrics.WordErrorRate()
            wer = metric(pred_norm, ref_norm)

        # Logging and printing as before
        writer.add_scalar('validation bleu', bleu, global_step)
        writer.add_scalar('validation cer', cer, global_step)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        print_msg(f"{f'BLEU: ' :>12}{bleu:.4f}")
        print_msg(f"{f'CER: ' :>12}{cer:.4f}")
        print_msg(f"{f'WER: ' :>12}{wer:.4f}")
        with open("metrics_log.txt", "a") as f:
            f.write(f"Step {global_step}: BLEU={bleu:.4f}, CER={cer:.4f}, WER={wer:.4f}\n")


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

    # if loading from checkpoint, also load scheduler state
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.' if it exists
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
    
    else:
        print(' No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    scaler = GradScaler() if device == "cuda" else None

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            # --- Mixed Precision Training ---
            if scaler is not None:
                with autocast():
                    proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
            else:
                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            global_step += 1
            # break # break in a step lol to check

        # run validation at the end of every epochs
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Add visualization for sparse models logging every N epochs
        if config['use_adaptive_sparse'] and config['visualize'] and (epoch + 1) % 2 == 0:
            print("Generating attention visualizations...")
            log_visualizations(model, tokenizer_src, tokenizer_tgt, global_step)

        # Save the model at the end of every N epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == config['num_epochs']:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step
            }, model_filename)

