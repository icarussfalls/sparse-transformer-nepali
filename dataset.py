import torch
from torch.utils.data import Dataset

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
        
        # Special tokens for SentencePiece tokenizers
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
        
        # Add SOS and EOS tokens
        dec_input_tokens = [self.sos_token] + dec_input_tokens
        label = dec_input_tokens + [self.eos_token]
        
        # Pad or truncate sequences
        enc_input_tokens = self._pad_or_truncate(enc_input_tokens, self.seq_len)
        dec_input_tokens = self._pad_or_truncate(dec_input_tokens, self.seq_len)
        label = self._pad_or_truncate(label, self.seq_len)
        
        # Convert to tensors
        enc_input_tensor = torch.tensor(enc_input_tokens, dtype=torch.long)
        dec_input_tensor = torch.tensor(dec_input_tokens, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Create masks with correct dimensions
        encoder_mask = (enc_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # (1, 1, seq_len)
        decoder_mask = self._create_causal_mask(dec_input_tensor)  # (1, seq_len, seq_len)
        
        return {
            'encoder_input': enc_input_tensor,
            'decoder_input': dec_input_tensor,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label_tensor,
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
    
    def _pad_or_truncate(self, tokens, max_len):
        """Pad or truncate token sequence to max_len"""
        if len(tokens) > max_len:
            return tokens[:max_len]
        else:
            return tokens + [self.pad_token] * (max_len - len(tokens))
    
    def _create_causal_mask(self, decoder_input):
        """Create causal mask for decoder with correct dimensions"""
        seq_len = decoder_input.size(0)
        
        # Create causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
        
        # Create padding mask
        pad_mask = (decoder_input != self.pad_token).unsqueeze(0)  # (1, seq_len)
        
        # Combine masks: causal AND not padding
        combined_mask = causal_mask.unsqueeze(0) & pad_mask.unsqueeze(1)  # (1, seq_len, seq_len)
        
        return combined_mask.int()

def causal_mask(size):
    """Create causal mask for decoder"""
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0