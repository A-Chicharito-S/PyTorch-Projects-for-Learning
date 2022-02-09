import torch
from torch import nn
from torch.nn import functional as F


# implements: max(0, x·W_1 + b_1)·W_2 + b_2
class FeedForward(nn.Module):
    def __init__(self, dim, inner_layer_dim, dropout_rate):  # in paper: dim=512, inner_layer_dim=2048
        super(FeedForward, self).__init__()
        self.W_1 = nn.Linear(in_features=dim, out_features=inner_layer_dim)
        self.W_2 = nn.Linear(in_features=inner_layer_dim, out_features=dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        return self.W_2(self.dropout(F.relu(self.W_1(inputs))))
