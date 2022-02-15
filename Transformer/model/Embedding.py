import math

import torch
from torch import nn


class TransformerEmbedding(nn.Module):
    def __init__(self, num_vocab, dim, dropout_rate, PAD, max_seq_len=5000, embedding_matrix=None, is_frozen=True):
        super(TransformerEmbedding, self).__init__()
        self.dim = dim
        self.embedding = nn.Embedding(num_embeddings=num_vocab, embedding_dim=dim, padding_idx=PAD)
        # the entries at padding_idx do not contribute to the gradient
        # for newly constructed embeddings, the default value of padding_idx is 0 and can be updated to another index
        if embedding_matrix is not None:
            assert embedding_matrix.size(0) != num_vocab or embedding_matrix.size(1) != dim
            self.embedding.from_pretrained(embeddings=embedding_matrix, freeze=is_frozen)
            # if freeze is True: the embeddings will not be updated, otherwise, will be updated.
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('PE', self.PositionalEncoding(max_seq_len=max_seq_len, dim=dim))
        self.layer_norm = nn.LayerNorm(dim)
        # by register_buffer (a method of nn.Module),
        # the input weights will not be part of the gradient computation thus not updated

    @staticmethod
    # implements:
    # PE(pos, 2i) = sin(pos/10000^(2i/dim))
    # PE(pos, 2i+1) = cos(pos/10000^(2i/dim))
    def PositionalEncoding(max_seq_len, dim):
        pe = torch.zeros((max_seq_len, dim))
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        # returns: value: [0, 1, ..., max_seq_len-1] ; shape: (max_seq_len, 1)
        div_term = torch.exp(-torch.arange(0, dim, 2) / dim * math.log(10000.0))
        # though called div_term, this actually computes the following, which later will serve as a multiplier:
        # 1/10000^(2i/dim) = exp(log(1/10000^(2i/dim))) = exp(-(2i/dim)·log10000)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # i:j:k means from position i to j, take indices every k steps.
        return pe

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len)
        seq_len = inputs.size(1)

        return self.layer_norm(self.dropout(self.embedding(inputs) * math.sqrt(self.dim)+self.PE[:seq_len]))
