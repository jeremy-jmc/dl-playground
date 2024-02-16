# https://github.com/harvardnlp/annotated-transformer
# https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# TODO: copy.deepcopy() vs torch.clone()
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# -----------------------------------------------------------------------------
# Main architecture
# -----------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# -----------------------------------------------------------------------------
# Attention and Multi-Headed Attention
# -----------------------------------------------------------------------------

def subsequent_mask(size):
    "Mask out subsequent positions."
    # las mascara sirve para que el decoder no vea el futuro, solo el pasado y el presente
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # print('\tforward on MultiHeadedAttention')
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # print(query.shape, key.shape, value.shape)
        # # ! alter the behaviour of summary function: print([lin(x).shape for lin, x in zip(self.linears, (query, key, value))])
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # print(query.shape, key.shape, value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        # print(x.shape, self.attn.shape)
        
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # print(x.shape)
        del query
        del key
        del value
        return self.linears[-1](x)


# -----------------------------------------------------------------------------
# Position-wise FFN, Embeddings, Positional Encoding
# -----------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_subsequent_mask():
    plt.figure(figsize=(8, 8))
    plt.imshow(subsequent_mask(20)[0], cmap='gray')
    plt.colorbar()
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 20, 1))
    plt.xlabel('Window sequence position')
    plt.ylabel('Masking')
    plt.title('Subsequent Mask')
    plt.show()


def test_mha():
    heads = 8
    d_model = 512
    dropout_prob = 0.1
    seq_len = 32
    batch_size = 16
    print('\theads:', heads)
    print('\td_model:', d_model)
    print('\tseq_len:', seq_len)
    print('\tbatch_size:', batch_size)
    print('\td_k:', d_model // heads)
    mha = MultiHeadedAttention(heads, d_model, dropout_prob)

    # data = torch.randn(batch_size, seq_len, d_model)
    # attntn = mha(data, data, data, subsequent_mask(seq_len))

    summary(mha, 
            # query=qkv, key=qkv, value=qkv,
            input_size=[(batch_size, seq_len, d_model), (batch_size, seq_len, d_model), (batch_size, seq_len, d_model)], 
            verbose=2, 
            col_names=["input_size", "output_size", "params_percent", "num_params", "trainable", "mult_adds",], 
            col_width=20,
            device=device)


def test_encoder():
    N = 1
    d_model = 512
    d_ff = 2048
    h = 8
    dropout = 0.1
    seq_len = 32
    batch_size = 16
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    mask = subsequent_mask(seq_len)
    data = torch.randn(batch_size, seq_len, d_model)
    
    # enc(data, mask)
    summary(enc,
            input_size=[(batch_size, seq_len, d_model), (1, seq_len, seq_len)], 
            verbose=1,
            col_names=["input_size", "output_size", "params_percent", "num_params", "trainable", "mult_adds",], 
            col_width=20,
            device=device)


def test_model():
    N = 6
    d_model = 512
    d_ff = 2048
    h = 8
    dropout = 0.1
    seq_len = 32
    batch_size = 16
    model = make_model(10, 10, N)
    # print(model)

    data = torch.randint(0, 10, (batch_size, seq_len))
    mask = subsequent_mask(seq_len)
    out = model(data, data, mask, mask)
    print(f'> out.shape: {out.shape}')
    _, next_word = torch.max(out, dim=1)
    print(f'> next_word.shape: {next_word.shape}')
    print()

    summary(model,
            # input_size=[(batch_size, seq_len), (batch_size, seq_len)],
            input_data=[data, data, mask, mask],
            verbose=1,
            col_names=["input_size", "output_size", "params_percent", "num_params", "trainable", "mult_adds",], 
            col_width=20,
            device=device)

# TODO: from Part 2
if __name__ == '__main__':
    """
    d / d_model
        model size / hidden state dimension / positional encoding size
    h
        number of heads in the multi-head attention
    seq_len / L
        length of the input sequence
    N
        number of attention layers
    """
    test_subsequent_mask()
    test_mha()
    test_encoder()
    test_model()
