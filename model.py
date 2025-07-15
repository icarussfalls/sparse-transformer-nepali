import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import get_config

# input embeddings are created to convert the original sentences into a vector of 512 dimension
# vocab size is the number of unique tokens
# d_model is the size of the embedding vector (dimensionality)
# self.embeddding initializes the embedding layer and maps each token in the vocabulary to  d_model-dimenstion vector

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size:int ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model):
        # multiply by sqrt(d_model) to scale embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(1000.0) / d_model))
        # apply sin to even indices
        pe[:, 0::2] = torch.sin
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (1000 ** (2i / d_model)))
        # add a batch dimension to the position encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the position encoding as a buffer
        self.register_buffer('pe', pe) # buffer registered so as not to compute it again

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# d_k is the dimension of the vector processed by the each head == d_model // h
# w_q, w_k, w_v, w_o are linear layers that project the input vectors to queries, keys, values, and outputs resp.

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # embedding vector size
        self.h = h # number of heads
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h # dimension of the vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        
        # Store last attention weights for visualization
        self.last_attention_weights = None

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # Calculate attention scores
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Store attention weights for visualization (only during eval)
        if not self.training:
            self.last_attention_weights = attn_weights.detach()
            
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ value
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)

# this below adds cross-attention between head treating head as a token
# class MultiHeadAttentionBlock(nn.Module):
#     def __init__(self, d_model, num_heads, dropout):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

#         self.d_model = d_model
#         self.h = num_heads
#         self.d_k = d_model // num_heads

#         # Linear layers for Q, K, V
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)

#         # Output projection
#         self.w_o = nn.Linear(d_model, d_model)

#         # Dropout for attention weights
#         self.dropout = nn.Dropout(dropout)

#         # Cross-head attention: shared layers
#         self.inter_head_q = nn.Linear(self.d_k, self.d_k)
#         self.inter_head_k = nn.Linear(self.d_k, self.d_k)
#         self.inter_head_v = nn.Linear(self.d_k, self.d_k)
#         self.inter_head_out = nn.Linear(self.d_k, self.d_k)

#         # LayerNorm for residuals
#         self.norm1 = LayerNormalization(d_model)


#     def forward(self, q_input, k_input, v_input, mask=None):
#         residual = q_input
#         B, L, _ = q_input.size()
#         Bk, Lk, _ = k_input.size()  # Sequence length of key (encoder output)
#         Bv, Lv, _ = v_input.size()  # Sequence length of key (encoder output)

#         # 1. Linear projections + reshape to (B, h, L, d_k)
#         Q = self.w_q(q_input).view(B, L, self.h, self.d_k).transpose(1, 2)
#         K = self.w_k(k_input).view(Bk, Lk, self.h, self.d_k).transpose(1, 2)
#         V = self.w_v(v_input).view(Bv, Lv, self.h, self.d_k).transpose(1, 2)

#         # 2. Scaled dot-product attention per head
#         attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
#         if mask is not None:
#             attn_scores.masked_fill_(mask == 0, torch.finfo(attn_scores.dtype).min)
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)
#         attn_out = attn_probs @ V  # (B, h, L, d_k)

#         # 3. Inter-head attention: treat each head as a "token"
#         # Transpose to (B, L, h, d_k) for head-wise mixing
#         heads = attn_out.transpose(1, 2)  # (B, L, h, d_k)

#         Qh = self.inter_head_q(heads)
#         Kh = self.inter_head_k(heads)
#         Vh = self.inter_head_v(heads)

#         # Attention across heads: (B, L, h, d_k) × (B, L, d_k, h) → (B, L, h, h)
#         inter_scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(self.d_k)
#         inter_probs = F.softmax(inter_scores, dim=-1)
#         inter_probs = self.dropout(inter_probs)
#         mixed_heads = inter_probs @ Vh  # (B, L, h, d_k)

#         # Reshape back to (B, h, L, d_k)
#         mixed_heads = mixed_heads.transpose(1, 2)

#         # 4. Concatenate heads and final projection
#         concat = mixed_heads.transpose(1, 2).contiguous().view(B, L, self.d_model)
#         output = self.w_o(concat)

#         # 5. Residual + LayerNorm (Post-Norm)
#         out = self.norm1(output + residual)
#         return out

# alpha is the learnable parameter initialized to one of shape (features,) which scales the normalized outputs
# bias is the learnable parameter initialized to zeros of shape (features,) which shifts the normalized outputs
# eps is to prevent dividing by zero when std is very small
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # keep the dimension for broadcasting 
        mean = x.mean(dim=-1, keepdim=True)            # shape: (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # shape: (batch, seq_len, 1)
        norm = (x - mean) / torch.sqrt(var + self.eps)     # normalize
        return self.alpha * norm + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# this is like the one layer
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# stack of layers, the whole vertical stack in the paper is this class  
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    
class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        # add and norm not applied here at output which is done in decoder class
        return x
    
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# our code is improved variant on the original google paper, in original paper, normalization is applied after the residual connection, but here, "pre-norm" before the sublayer, then residual added

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # def encode(self, src, src_mask):
    #     # (batch, seq_len, d_model)
    #     src = self.src_embed(src)
    #     src = self.src_pos(src)
    #     return self.encoder(src, src_mask)

    # def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
    #     # (batch, seq_len, d_model)
    #     tgt = self.tgt_embed(tgt)
    #     tgt = self.tgt_pos(tgt)
    #     return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    # def project(self, x):
    #     # (batch, seq_len, vocab_size)
    #     return self.projection_layer(x)
    
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # Encode
        src = self.src_embed(encoder_input)
        src = self.src_pos(src)
        encoder_output = self.encoder(src, encoder_mask)
        # Decode
        tgt = self.tgt_embed(decoder_input)
        tgt = self.tgt_pos(tgt)
        decoder_output = self.decoder(tgt, encoder_output, encoder_mask, decoder_mask)
        # Project
        proj_output = self.projection_layer(decoder_output)
        return proj_output
        
    def translate_with_attention(self, src, src_mask, tokenizer_src, tokenizer_tgt, max_len=100):
        """
        Translate text and return attention weights for visualization
        """
        # Initialize storage for attention weights
        encoder_attention_maps = []
        decoder_self_attention_maps = []
        decoder_cross_attention_maps = []
        
        # Encode source sequence and collect encoder attention
        src_embedded = self.src_pos(self.src_embed(src))
        
        # Run through encoder collecting attention weights
        encoder_output = src_embedded
        for layer in self.encoder.layers:
            encoder_output = layer(encoder_output, src_mask)
            # Store encoder self-attention weights
            encoder_attention_maps.append(layer.self_attention_block.last_attention_weights)
        
        encoder_output = self.encoder.norm(encoder_output)
        
        # Initialize translation with SOS token
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')
        
        # Start with SOS token
        decoder_input = torch.ones(1, 1).fill_(sos_idx).type_as(src).to(src.device)
        
        # Greedy decode
        for i in range(max_len):
            # Create mask for target
            decoder_mask = torch.triu(
                torch.ones((1, decoder_input.size(1), decoder_input.size(1))), 
                diagonal=1
            ).type(torch.bool).to(src.device)
            
            # Decode step
            decoder_embedded = self.tgt_pos(self.tgt_embed(decoder_input))
            decoder_output = decoder_embedded
            
            # Store attention maps for this step
            step_self_attention = []
            step_cross_attention = []
            
            for layer in self.decoder.layers:
                # Get self-attention weights
                _ = layer.self_attention_block(
                    decoder_output, decoder_output, decoder_output, decoder_mask
                )
                step_self_attention.append(layer.self_attention_block.last_attention_weights)
                
                # Get cross-attention weights
                _ = layer.cross_attention_block(
                    decoder_output, encoder_output, encoder_output, src_mask
                )
                step_cross_attention.append(layer.cross_attention_block.last_attention_weights)
                
                # Full forward pass
                decoder_output = layer(decoder_output, encoder_output, src_mask, decoder_mask)
            
            decoder_output = self.decoder.norm(decoder_output)
            
            # Store attention maps
            decoder_self_attention_maps.append(step_self_attention)
            decoder_cross_attention_maps.append(step_cross_attention)
            
            # Project and get next token
            proj_output = self.projection_layer(decoder_output)
            next_word = proj_output[:, -1].argmax(dim=1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_word], dim=1)
            
            # Stop if EOS token
            if next_word.item() == eos_idx:
                break
        
        # Organize attention maps
        attention_maps = {
            'encoder': encoder_attention_maps,
            'decoder_self': decoder_self_attention_maps,
            'decoder_cross': decoder_cross_attention_maps
        }
        
        return decoder_input, attention_maps

# lets build the full transformers now

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int, h: int, dropout: float, d_ff: int) -> Transformer:    
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformers
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # init the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # xavier uniform (glorot initialization) initializes the weights of neural networks to keep the scale of gradients roughly the same in all layers, help prevent vanishing/exploding gradients
    return transformer
