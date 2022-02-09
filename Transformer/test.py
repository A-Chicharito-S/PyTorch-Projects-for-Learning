from toolkit.framework import framework
import torch
import numpy as np
import random

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

framework = framework(seq_len=10, num_vocab_src=11, num_vocab_tgt=11, num_epoch=10,
                      num_encoder_block=2, num_decoder_block=2, num_heads=8,
                      dim=512, inner_layer_dim=2048, dropout_rate=0.1,
                      PAD=0, BOS=1, EOS=2)
framework.train_and_test('beam_search')
# [ 1  5 10  5  4  3  6  0  0  2]
# beam search: [5, 10, 5, 4, 6, 3, 5]
# greedy search: [[ (1),  5, 10, 4, 5, 6, 3, 5,  (2)]]