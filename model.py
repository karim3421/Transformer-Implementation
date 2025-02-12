import torch
import torch.nn as nn
import math

# Embedding Layer 
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def  forward(self, x):
        return math.sqrt(self.d_model) * self.embedding(x) # scale the embedding

# Positional Encoding    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) # nemerator of the positional encoding
        div = torch.exp(torch.arange(0, d_model, 2).float() + (-math.log(10000.0) / d_model)) # denominator of the positional encoding

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1] , :]).requires_grad_(False)  
        return self.dropout(x)
    
# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))   # multiply
        self.bias = nn.Parameter(torch.zeros(1))    # add

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
# Feed Forward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout:float):
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.Linear_2(self.dropout(torch.relu(self.Linear_1(x))))

#MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h  
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)  

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score


    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # print(f"{query.shape} , BEFORE <<<")

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # print(f"{query.shape} , AFTER >>>")

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # print(f"{x.shape} , FINAL !!!")

        return self.w_o(x)

# Residual Connection 
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Encoder Block 
class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))

    def forward(self, x, mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x

        

# Encoder
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, trg_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)

        return x

# Decoder 
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

# Linear layer 
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)

# Transformer (All Together Now !)
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, trg_embed: InputEmbedding, src_pos: PositionalEncoding, trg_pos: PositionalEncoding, project: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.project = project

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def linear(self, x):
        return self.project(x)
    
def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int, dropout: float = 0.2, h: int = 8, n: int = 6, d_ff: int = 2048):
    # create embedding 
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trg_embed = InputEmbedding(d_model, trg_vocab_size)

    # create positional embedding 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # create encoder block
    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # create decoder block
    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create Projection layer 
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)
    
    # initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
