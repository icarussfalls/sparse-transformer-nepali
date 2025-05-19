import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ds is the dataset here
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # start of the sentence token
        # end of the sentence token
        # pad token is used for padding
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # truncating the sentences for faster batch sizes
        enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
        dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

        # add sos, eos, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # we will add <s> and </s>
        # only add <s> on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # need to make sure number of paddings tokens is not negative. If thats the case, then the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token, # <s> (start of the sentence)
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, # </s> (end of sentence)
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # add <s> on the decoder
        # why only <s> ?, during training, decoder is fed the ground-truth sequence shifted right by one position (starts with <s>), and the model is trained to predict the next token at each step
        decoder_input = torch.cat(
            [
                self.sos_token, # <s> (start of the sentence)
                torch.tensor(dec_input_tokens, dtype=torch.int64), # tokenized target sentence
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # label, we add only </s> token
        # <eos> or </s> is added to the label so the model knows when to stop generating
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), # tokenized target sentence
                self.eos_token, # </s> (end of the sentence)
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # double check the size of tensor to make sure they are all seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # if idx < 5:
        #     print(f"IDX: {idx}")
        #     print(f"SRC: {src_text}")
        #     print(f"TGT: {tgt_text}")
        #     print("-" * 40)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token) & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0