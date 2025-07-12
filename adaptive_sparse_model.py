import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import get_config
from entmax import entmax15, entmax_bisect

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
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (1000 ** (2i / d_model)))
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
    def __init__(self, d_model: int, h: int, dropout: float, attn_type: str = "entmax15") -> None:
        """
        attn_type: "softmax", "entmax15", or "entmax_alpha"
        """
        super().__init__()

        self.d_model = d_model # embedding vector size
        self.h = h # number of heads
        # making sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimension of the vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)
        self.attn_type = attn_type

        # For alpha-entmax: one learnable alpha per head, initialized to 1.5 (entmax15)
        if attn_type == "entmax_alpha":
            self.alpha = nn.Parameter(torch.ones(h) * 1.5)

    def forward(self, q, k, v, mask):
        B, L, _ = q.size()
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # need to accomodate h here
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, h, L, L)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, torch.finfo(attn_scores.dtype).min)

        # adaptive sparse transformers
        if self.attn_type == "softmax":
            attn_probs = F.softmax(attn_scores, dim=-1)
        elif self.attn_type == "entmax15":
            attn_probs = entmax15(attn_scores, dim=-1)
        elif self.attn_type == "entmax_alpha":
            # alpha entmax, we are using different alpha for each head
            attn_probs = []
            for head in range(self.h):
                # entmax_bisects needs (B, L, L)
                attn_probs.append(entmax_bisect(attn_scores[:, head], alpha=self.alpha[head], dim=-1))
            attn_probs = torch.stack(attn_probs, dim=1) # (B, h, L, L)

        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")

        attn_probs = self.dropout(attn_probs)
        x = attn_probs @ value  # (B, h, L, d_k)
        x = x.transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
    
        # multiply by Wo
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)


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
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.gradient_checkpointing = False

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(encoder_input, decoder_input, encoder_mask, decoder_mask)
        return self._forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

    def _forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_input = self.src_pos(self.src_embed(encoder_input))
        decoder_input = self.tgt_pos(self.tgt_embed(decoder_input))
        
        encoder_output = self.encoder(encoder_input, encoder_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask)
        proj_output = self.projection_layer(decoder_output)
        return proj_output

    def _forward_with_checkpointing(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # Use torch.utils.checkpoint for memory-efficient forward pass
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        encoder_input = self.src_pos(self.src_embed(encoder_input))
        decoder_input = self.tgt_pos(self.tgt_embed(decoder_input))
        
        encoder_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.encoder),
            encoder_input, encoder_mask
        )
        
        decoder_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.decoder),
            decoder_input, encoder_output, encoder_mask, decoder_mask
        )
        
        proj_output = self.projection_layer(decoder_output)
        return proj_output

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False

# lets build the full transformers now
def build_adaptive_sparse_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int,
    N: int,
    h: int,
    dropout: float,
    d_ff: int,
    attn_type: str = "entmax_alpha", 
) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, attn_type=attn_type)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, attn_type=attn_type)
        # Cross-attention is usually dense (softmax) in the paper
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout, attn_type="softmax")
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
    return transformer

if __name__ == "__main__":
    # Example config
    src_vocab_size = 100
    tgt_vocab_size = 120
    src_seq_len = 16
    tgt_seq_len = 16
    d_model = 32
    N = 2
    h = 4
    dropout = 0.1
    d_ff = 64
    attn_type = "entmax_alpha"  # or "entmax15" or "softmax"

    # Build the model
    model = build_adaptive_sparse_transformer(
        src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len,
        d_model, N, h, dropout, d_ff, attn_type=attn_type
    )

    # Dummy batch
    batch_size = 8
    encoder_input = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    decoder_input = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    encoder_mask = torch.ones((batch_size, 1, 1, src_seq_len), dtype=torch.bool)
    decoder_mask = torch.ones((batch_size, 1, tgt_seq_len, tgt_seq_len), dtype=torch.bool)

    # Forward pass
    output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
    print("Output shape:", output.shape)  # Should be (batch_size, tgt_seq_len, tgt_vocab_size)