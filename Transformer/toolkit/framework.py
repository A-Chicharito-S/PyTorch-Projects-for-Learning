from toolkit.dataloader import FakeDataLoader
from model.Transformer import make_model
from toolkit.loss import LabelSmoothing
from toolkit.optimizer import WarmUpOpt
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)


class framework(object):
    def __init__(self, seq_len, num_vocab_src, num_vocab_tgt, num_epoch, num_encoder_block, num_decoder_block, num_heads,
                 dim, inner_layer_dim, dropout_rate, PAD=0, BOS=1, EOS=2):
        self.seq_len = seq_len
        self.PAD = PAD
        self.BOS = BOS
        self.EOS = EOS
        self.fake_dataloader = FakeDataLoader(seq_len, num_vocab_src, PAD, BOS, EOS)
        self.num_epoch = num_epoch
        self.model = make_model(num_encoder_block=num_encoder_block, num_decoder_block=num_decoder_block,
                                num_heads=num_heads, num_vocab_src=num_vocab_src, num_vocab_tgt=num_vocab_tgt,
                                dim=dim, inner_layer_dim=inner_layer_dim, dropout_rate=dropout_rate,
                                PAD=PAD, BOS=BOS, EOS=EOS)
        self.criterion = LabelSmoothing(PAD=PAD)
        self.optimizer = WarmUpOpt(torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), dim=dim)
        self.gradient_clipper = 1000
        self.save_path = 'transformer_base'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda(self.device)
        # this is very important

    def run(self, num_batch, batch_size, mode='train'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_token = 0
        for i in range(num_batch):  # here set 5 batches in every epoch
            inputs, target_inputs, target_outputs = self.fake_dataloader.create_one_batch(batch_size=batch_size)
            # batch_size, seq_len

            num_tokens = (inputs != self.PAD).sum()
            total_token += num_tokens
            inputs = torch.from_numpy(inputs).to(self.device)
            target_inputs = torch.from_numpy(target_inputs).to(self.device)
            target_outputs = torch.from_numpy(target_outputs).to(self.device)

            outputs = self.model(inputs, target_inputs)
            # batch_size, seq_len, vocab

            loss = self.criterion(outputs, target_outputs)

            total_loss += loss.item()*num_batch

            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipper)
                self.optimizer.step()
        if mode != 'train':
            print('loss per token for {} is {}'.format(mode, total_loss / total_token))
            return total_loss

    def train_and_test(self, decoding_method='greedy'):
        best_loss = np.Inf

        for epoch in range(self.num_epoch):
            print('training in epoch {}'.format(epoch+1))
            self.run(num_batch=20, batch_size=30, mode='train')
            with torch.no_grad():  # this is a little bit redundant, since inside run it's set to .eval()
                loss = self.run(num_batch=20, batch_size=30, mode='eval')
                if best_loss > loss:
                    torch.save(self.model, self.save_path)
                    print('model saved!')
                    best_loss = loss
        print('now testing...')
        self.test(decoding_method=decoding_method)
        return 0

    def test(self, num_batch=1, batch_size=1, decoding_method='greedy'):
        # Model class must be defined somewhere
        model = torch.load(self.save_path)
        model.eval()

        for i in range(num_batch):  # here set 5 batches in every epoch
            inputs, target_inputs, target_outputs = self.fake_dataloader.create_one_batch(batch_size=batch_size)
            print('the original 1-st sentence: {}'.format(inputs[0]))  # batch_size, seq_len
            inputs = torch.from_numpy(inputs).to(self.device)
            # target_inputs = torch.from_numpy(target_inputs).to(self.device)
            # target_outputs = torch.from_numpy(target_outputs).to(self.device)

            outputs = self.model.predict(inputs, max_generating_len=self.seq_len, decoding_method=decoding_method)
            print('the predicted 1-st sentence: {}'.format(outputs[0]))

        return 0




