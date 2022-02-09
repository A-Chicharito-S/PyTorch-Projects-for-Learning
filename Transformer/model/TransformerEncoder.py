from torch import nn
from copy import deepcopy


class TransformerBaseEncoderBlock(nn.Module):
    # the input is of size: (batch_size, seq_len, dim)
    def __init__(self, dim, dropout_rate, Attention, FeedForward):
        super(TransformerBaseEncoderBlock, self).__init__()
        self.att = deepcopy(Attention)
        # for reusable components, we pass it in as parameter and use deepcopy()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        # normalized_shape:
        # If a single integer is used, it is treated as a singleton list, and this module will normalize over the last
        # dimension which is expected to be of that specific size.
        self.feedforward = deepcopy(FeedForward)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, pad_mask):
        inputs = self.layer_norm(inputs + self.att(inputs, inputs, inputs, pad_mask))
        # att: Q=inputs, K=inputs, V=inputs
        return self.layer_norm(inputs + self.dropout(self.feedforward(inputs)))


class TransformerBaseEncoder(nn.Module):
    def __init__(self, num_encoder_block, encoder_block):
        super(TransformerBaseEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([deepcopy(encoder_block) for _ in range(num_encoder_block)])
        # nn.ModuleList receives a list containing modules and stores them

    def forward(self, inputs, mask):
        # inputs are the embedded source sentences
        for block in self.encoder_blocks:
            inputs = block(inputs, mask)
        return inputs
