
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchinfo import summary
import vector_quantize_pytorch as vq
from torchviz import make_dot

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

"""
---
title: Multi-Headed Attention (MHA)
summary: >
  This implements the Multi-Headed Attention used in transformers
  using PyTorch with explanations.
---

# Multi-Headed Attention (MHA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/basic/autoregressive_experiment.ipynb)

This is a tutorial/implementation of multi-headed attention
from paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
in [PyTorch](https://pytorch.org/).
The implementation is inspired from [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

Here is the [training code](basic/autoregressive_experiment.html) that uses a basic transformer
with MHA for NLP auto-regression.

[Here is an experiment implementation](basic/autoregressive_experiment.html) that trains a simple transformer.
"""

import math
from typing import Optional, List

import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)


class WrapperModel(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # Create dummy inputs for torchinfo.summary
        dummy_input = {
            'query': torch.randn((10, 64, 512)),  # Replace with your desired input size
            'key': torch.randn((10, 64, 512)),    # Replace with your desired input size
            'value': torch.randn((10, 64, 512)),  # Replace with your desired input size
        }

        return super().forward(**dummy_input)

if __name__ == '__main__':
    heads = 8
    d_model = 512
    dropout_prob = 0.1
    seq_len = 32
    batch_size = 64

    print('heads:', heads)
    print('d_model:', d_model)
    print('seq_len:', seq_len)
    print('batch_size:', batch_size)

    pmha = PrepareForMultiHeadAttention(d_model, heads, d_model // heads, True)
    summary(pmha, input_size=(seq_len, batch_size, d_model), 
            verbose=2, 
            col_names=["input_size", "output_size", "params_percent", "num_params", "trainable", "mult_adds",], 
            col_width=20,
            device=device)
    # make_dot(y.mean(), params=dict(pmha.named_parameters()))

    mha = MultiHeadAttention(heads, d_model, dropout_prob, True)
    qkv = torch.randn(seq_len, batch_size, d_model)

    # https://stackoverflow.com/questions/60480686/pytorch-model-summary-forward-func-has-more-than-one-argument
    # https://github.com/sksq96/pytorch-summary#multiple-inputs
    summary(mha, 
            # query=qkv, key=qkv, value=qkv,
            input_size=[(seq_len, batch_size, d_model), (seq_len, batch_size, d_model), (seq_len, batch_size, d_model)], 
            verbose=2, 
            col_names=["input_size", "output_size", "params_percent", "num_params", "trainable", "mult_adds",], 
            col_width=20,
            device=device)
    
# interesante implementacion, pero no me cuadra con lo que he leido del concepto de MultiHeadAttention
# TODO: comparar con la implementacion de Harvard
