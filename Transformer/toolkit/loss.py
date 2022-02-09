import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
np.set_printoptions(threshold=np.inf)


class LabelSmoothing(nn.Module):

    def __init__(self, PAD, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.PAD = PAD

    def forward(self, inputs, targets):
        # inputs = F.log_softmax(inputs, dim=-1)
        # inputs: (batch_size, seq_len_decoder, d_vocab)

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        num_vocab = inputs.size(-1)
        mask = (targets == self.PAD).view(-1)
        # (batch_size*seq_len, )

        index = targets.unsqueeze(-1).long()  # (batch_size, seq_len, 1)
        targets = torch.zeros(batch_size, seq_len, num_vocab).to(targets.device).scatter(-1, index, 1)
        # give indices (ground truth) the probability of 1
        targets = targets * (1 - self.smoothing) + self.smoothing / num_vocab

        # split a fraction (smoothing) to other words in the vocab
        loss = self.criterion(inputs.view(-1, num_vocab),
                              targets.view(-1, num_vocab).detach()).sum(dim=-1)
        # (batch_size*seq_len, num_vocab) ---> (batch_size*seq_len, )
        return loss.masked_fill(mask, 0.).sum() / batch_size
    # dividing batch_size makes it comparable for different batch_sizes in one epoch
    # the elements in the distribution where the targets are padded should be ignored (thus set to 0)




