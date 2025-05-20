from pathlib import Path
from config import get_config, latest_weight_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
import torch
import sys

def translate(sentence: str):
    # define the device tokenizer and the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device ", device)

    config = get_config()
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)

    # load the pretrained weights
    model_filename = latest_weight_file_path(config)
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' if it exists
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state['model_state_dict'], strict=False) # this returns the error since the model was trained on dataparallel on cuda

    # if the sentence is the number, use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]['tgt_text']
    
    # seq_len = config['seq_len']
    seq_len = 256  # hardcoding exactly with what the model was initialized with

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # same process we used in validation
        source = tokenizer_src.encode(sentence)
        
        token_ids = source.ids[:seq_len - 2]  # truncate long input
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(token_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(token_ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        
        # init sos in decoder input
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':12}{id}")
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'Target: ':>12}{label}")
        print(f"{f'Predicted: ':>12}", end='')

        # lets generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target, can use causal_mask here too
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)

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

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break in the end of the sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    return tokenizer_tgt.decode(decoder_input.squeeze(0).tolist())

# read sentence from argument
result = translate(sys.argv[1] if len(sys.argv) > 1 else "She went out")

