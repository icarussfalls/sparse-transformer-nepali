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
