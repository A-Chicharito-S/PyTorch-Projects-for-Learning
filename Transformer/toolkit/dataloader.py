import numpy as np
import random
from copy import deepcopy


class FakeDataLoader(object):
    def __init__(self, seq_len, num_vocab, PAD=0, BOS=1, EOS=2):
        self.seq_len = seq_len
        self.num_vocab = num_vocab
        self.PAD = PAD
        self.BOS = BOS
        self.EOS = EOS
        self.available_tokens = [x for x in range(num_vocab) if x != PAD and x != BOS and x != EOS]

    def create_one_sentence(self):
        length = min(random.randint(1, self.seq_len-2) + 3, self.seq_len-2)
        sentence = np.random.choice(self.available_tokens, length).tolist()
        sentence = [self.BOS] + sentence + [self.PAD]*(self.seq_len-2-length) + [self.EOS]
        return sentence

    def create_one_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.create_one_sentence())  # batch_size, seq_len
        target_input = np.array(deepcopy(batch))[:, :-1]  # remove the EOS token
        target_output = np.array(deepcopy(batch))[:, 1:]  # remove the BOS token
        return np.array(batch), target_input, target_output



