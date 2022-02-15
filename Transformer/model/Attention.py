import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=None):
        super(Attention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads

        self.sub_dim = dim // num_heads
        self.W_K = nn.Linear(in_features=dim, out_features=dim)
        self.W_Q = nn.Linear(in_features=dim, out_features=dim)
        self.W_V = nn.Linear(in_features=dim, out_features=dim)
        self.W_out = nn.Linear(in_features=dim, out_features=dim)
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask=None):
        # inputs is of shape (batch_size, seq_len, dim)
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, Q.size(1), self.num_heads, self.sub_dim).transpose(1, 2)
        # shape: (batch_size, seq_len, num_heads, sub_dim) ---> (batch_size, num_heads, seq_len_q, sub_dim)
        K = self.W_K(K).view(batch_size, K.size(1), self.num_heads, self.sub_dim).transpose(1, 2)
        # shape: (batch_size, num_heads, seq_len_kv, sub_dim)
        V = self.W_V(V).view(batch_size, V.size(1), self.num_heads, self.sub_dim).transpose(1, 2)
        # shape: (batch_size, num_heads, seq_len_kv, sub_dim)
        att_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.sub_dim)

        if mask is not None:
            # att_score.masked_fill(mask == self.PAD, -1e9)
            att_score = att_score.masked_fill(mask, -1e9)
        att_score = F.softmax(att_score, dim=-1)
        # (batch_size, num_heads, seq_len_q, seq_len_kv)

        if mask is not None:
            # att_score.masked_fill(mask == self.PAD, 0)
            att_score = att_score.masked_fill(mask, 0)
        if self.dropout_rate is not None:
            att_score = self.dropout(att_score)

        outputs = torch.matmul(att_score, V).transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.sub_dim)
        # shape: (batch_size, num_heads, seq_len_q, sub_dim) ---> (batch_size, seq_len_q, num_heads, sub_dim)
        return self.W_out(outputs)
        # shape: (batch_size, seq_len_q, dim)
