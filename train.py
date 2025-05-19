from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weight_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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


def get_ds(config):
    # the data only has train split so
    ds_all = load_dataset(f"{config['data_source']}", "default", split='train')

    # lets use only 20% of the data
    subset_size = int(0.2 * len(ds_all))
    ds_raw = ds_all.select(range(subset_size))
    print(f"Using {subset_size} samples out of {len(ds_all)}")
    print(ds_raw[0])

    # build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # now 90% for training and remaning for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len=config['seq_len'])

    # find the minimum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
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
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model = config['d_model'])
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
        metric = torchmetrics.BLEUScore()
        # if bleu doesn't works below, use this bleu = metric(predicted, [[ref] for ref in expected])
        # bleu = metric(predicted, expected)
        predicted_tok = [pred.split() for pred in predicted]
        expected_tok = [[ref.split()] for ref in expected]

        bleu = metric(predicted_tok, expected_tok)
        writer.add_scalar('validation bleu', bleu, global_step)
        writer.flush()

        # print the scores to your console/logs
        print_msg(f"{f'BLEU: ' :>12}{bleu:.4f}")
        print_msg(f"{f'CER: ' :>12}{cer:.4f}")
        print_msg(f"{f'WER: ' :>12}{wer:.4f}")
        
        # --- Log metrics to a file ---
        with open("metrics_log.txt", "a") as f:
            f.write(f"Step {global_step}: BLEU={bleu:.4f}, CER={cer:.4f}, WER={wer:.4f}\n")


def train_model(config):
    # lets define the device first
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("using device", device)
    if (device == "cuda"):
        print(f'Device name: {torch.cuda.get_device_name(device.index)}')
        print(f'Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB')
    elif (device == "mps"):
        print(f'Device name: <mps>')
    else:
        print('Note: if you have a GPU, consider using it for training')


    # lets make sure the weights folder exists
    Path(f"{config['data_source']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # added gpu parallel support
    if torch.cuda.device_count() > 1 and device == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # tensorboard part
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    # if model is specified before training, then need to load that
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weight_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}') 
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
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
            else:
                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss.backward()
                optimizer.step()

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            global_step += 1
            # break # break in a step lol to check

        # run validation at the end of every epochs
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every N epochs (e.g., every 5)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == config['num_epochs']:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

