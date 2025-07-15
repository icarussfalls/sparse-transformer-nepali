import torch

def create_sparse_mask(seq_len, block_size, stride, causal=False, device = 'cpu'):
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # block-local attention --> fixed pattern
        block_start = ( i // block_size) * block_size
        block_end = min(seq_len, block_start+block_size)
        mask[i, block_start:block_end] = True

        # strided attention
        mask[i, ::stride] = True

        # causal - needed for our task in the decoder
        if causal:
            mask[i, i+1:] = False

    return mask.unsqueeze(0).unsqueeze(0).to(device) 

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Returns the appropriate model based on configuration flags.
    """
    if config.get('use_sparse', False):
        print('Building sparse transformer')
        from sparse_model import build_sparse_transformer
        model = build_sparse_transformer(
            vocab_src_len, vocab_tgt_len, 
            src_seq_len=config['seq_len'], 
            tgt_seq_len=config['seq_len'], 
            d_model=config['d_model'], 
            N=config['N'], 
            h=config['h'], 
            dropout=config['dropout'], 
            d_ff=config['d_ff'], 
            block_size=config['sparse_block_size'], 
            stride=config['sparse_stride']
        )
        return model
    
    if config.get('use_adaptive_sparse', False):
        print('Building adaptive sparse transformer')
        from adaptive_sparse_model import build_adaptive_sparse_transformer
        model = build_adaptive_sparse_transformer(
            vocab_src_len, vocab_tgt_len, 
            src_seq_len=config['seq_len'], 
            tgt_seq_len=config['seq_len'], 
            d_model=config['d_model'], 
            N=config['N'], 
            h=config['h'], 
            dropout=config['dropout'], 
            d_ff=config['d_ff'], 
            attn_type=config['attn_type']
        )
        return model
    
    print('Building vanilla transformer')
    from model import build_transformer
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, 
        src_seq_len=config['seq_len'], 
        tgt_seq_len=config['seq_len'], 
        d_model=config['d_model'], 
        N=config['N'], 
        h=config['h'], 
        dropout=config['dropout'], 
        d_ff=config['d_ff']
    )
    return model

def translate(model, src_text, tokenizer_src, tokenizer_tgt, device, max_len=512):
    """
    Translates a source text using the provided model.
    """
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer_src.encode(src_text).ids
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)
        src_mask = (src != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).to(device)
        
        # Forward pass through encoder
        encoder_output = model.encode(src, src_mask)
        
        # Initialize with SOS token
        tgt = torch.tensor([[tokenizer_tgt.token_to_id("[SOS]")]], dtype=torch.long).to(device)
        
        for i in range(max_len):
            tgt_mask = model.get_tgt_mask(tgt.size(1)).to(device)
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            proj_output = model.project(decoder_output)
            
            # Get the next token
            next_token = torch.argmax(proj_output[:, -1], dim=1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end of sentence token reached
            if next_token.item() == tokenizer_tgt.token_to_id("[EOS]"):
                break
        
        # Convert tokens to text
        output_text = tokenizer_tgt.decode(tgt[0].cpu().numpy())
        return output_text