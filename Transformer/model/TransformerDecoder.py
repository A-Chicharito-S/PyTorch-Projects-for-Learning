from torch import nn
from copy import deepcopy


class TransformerBaseDecoderBlock(nn.Module):
    def __init__(self, dim, dropout_rate, Attention, FeedForward):
        super(TransformerBaseDecoderBlock, self).__init__()
        self.masked_att = deepcopy(Attention)
        self.att = deepcopy(Attention)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.feedforward = deepcopy(FeedForward)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_outputs, inputs, pad_mask, upper_triangle_mask):

        inputs = self.layer_norm(inputs + self.masked_att(inputs, inputs, inputs, upper_triangle_mask))
        # masked_att: Q=inputs, K=inputs, V=inputs, mask=upper_triangle_mask

        inputs = self.layer_norm(inputs + self.att(inputs, encoder_outputs, encoder_outputs, pad_mask))
        # att: Q=inputs, K=encoder_outputs, V=encoder_outputs, mask=pad_mask

        return self.layer_norm(inputs + self.dropout(self.feedforward(inputs)))


class TransformerBaseDecoder(nn.Module):
    def __init__(self, num_decoder_block, decoder_block):
        super(TransformerBaseDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([deepcopy(decoder_block) for _ in range(num_decoder_block)])

    def forward(self, encoder_outputs, inputs, pad_mask, upper_triangle_mask):
        # inputs are the embedded target sentences (training) / the newest embedded decoded word (test)
        for decoder_block in self.decoder_blocks:
            inputs = decoder_block(encoder_outputs, inputs, pad_mask, upper_triangle_mask)
        return inputs
