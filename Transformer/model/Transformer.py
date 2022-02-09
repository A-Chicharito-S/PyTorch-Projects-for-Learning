import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from model.Embedding import TransformerEmbedding
from model.Attention import Attention
from model.FeedForward import FeedForward
from model.TransformerEncoder import TransformerBaseEncoderBlock, TransformerBaseEncoder
from model.TransformerDecoder import TransformerBaseDecoderBlock, TransformerBaseDecoder


class TransformerBase(nn.Module):
    def __init__(self, num_encoder_block, num_decoder_block, num_heads, num_vocab_src, num_vocab_tgt,
                 dim, inner_layer_dim, dropout_rate, PAD, BOS, EOS):
        super(TransformerBase, self).__init__()
        self.PAD = PAD
        self.BOS = BOS
        self.EOS = EOS
        self.num_vocab_tgt = num_vocab_tgt

        attention = Attention(dim=dim, num_heads=num_heads)
        feedforward = FeedForward(dim=dim, inner_layer_dim=inner_layer_dim, dropout_rate=dropout_rate)
        encoder_block = TransformerBaseEncoderBlock(dim=dim, dropout_rate=dropout_rate, Attention=attention,
                                                    FeedForward=feedforward)
        decoder_block = TransformerBaseDecoderBlock(dim=dim, dropout_rate=dropout_rate, Attention=attention,
                                                    FeedForward=feedforward)

        self.embedding_src = TransformerEmbedding(num_vocab=num_vocab_src, dim=dim, dropout_rate=dropout_rate, PAD=PAD)
        self.embedding_tgt = TransformerEmbedding(num_vocab=num_vocab_tgt, dim=dim, dropout_rate=dropout_rate, PAD=PAD)

        self.encoder = TransformerBaseEncoder(num_encoder_block=num_encoder_block, encoder_block=encoder_block)
        self.decoder = TransformerBaseDecoder(num_decoder_block=num_decoder_block, decoder_block=decoder_block)
        self.projection = nn.Linear(in_features=dim, out_features=num_vocab_tgt)

    def create_pad_mask(self, x):
        return (x == self.PAD).to(x.device)

    # padded places are True, real words are False

    @staticmethod
    def create_upper_triangle_mask(x):
        seq_len = x.size(-1)  # x shape: (batch_size, seq_len)
        return torch.ones((seq_len, seq_len)).triu(1).bool().to(x.device)

    # unseen words are True, available words are False

    # inputs shape: (batch_size, seq_len)
    def forward(self, inputs, targets):
        inputs_mask = self.create_pad_mask(inputs).unsqueeze(1).unsqueeze(2)
        # of shape: (batch_size, 1, 1, seq_len_inputs) <---> in Attention,
        # when masking, (Q·K.transpose) is of shape: (batch_size, num_heads, seq_len_q, seq_len_kv)
        # where the seq_len_kv should equal inputs' seq_len
        inputs = self.embedding_src(inputs)
        from_encoder = self.encoder(inputs, inputs_mask)
        # shape: (batch_size, seq_len, dim)
        outputs = self.teacher_forcing(inputs_mask=inputs_mask, from_encoder=from_encoder, targets=targets)
        return outputs

    # the outputs are the log probabilities of shape: (batch_size, seq_len, vocab)

    def teacher_forcing(self, inputs_mask, from_encoder, targets):
        # batch_size, seq_len
        # the following masks are all created for targets, thus the key word 'target' is omitted for brevity


        # pad_mask = self.create_pad_mask(targets).unsqueeze(1).unsqueeze(2)
        # # shape: (batch_size, 1, seq_len)
        # #### this mask is the mask of inputs????
        # # print(pad_mask.shape)
        # upper_triangle_mask = self.create_upper_triangle_mask(targets).unsqueeze(0).unsqueeze(1)
        # # print(upper_triangle_mask.shape)  # 1, 1, seq_len, seq_len
        # upper_triangle_mask = upper_triangle_mask + pad_mask

        pad_mask = self.create_pad_mask(targets).unsqueeze(1)
        # shape: (batch_size, 1, seq_len)
        # print(pad_mask.shape)
        upper_triangle_mask = self.create_upper_triangle_mask(targets).unsqueeze(0)
        # print(upper_triangle_mask.shape)  # 1, seq_len, seq_len
        upper_triangle_mask = upper_triangle_mask + pad_mask  # batch_size, seq_len, seq_len

        # now the dtype is int, should convert to bool later
        # the upper_triangle_mask should not only mask the unseen words but also the paddings
        targets = self.embedding_tgt(targets)
        outputs = self.decoder(from_encoder, targets, inputs_mask, upper_triangle_mask.unsqueeze(1).bool())
        #### this is the inputs_mask, not the pad_mask for the targets
        outputs = F.log_softmax(self.projection(outputs), dim=-1)
        # shape: (batch_size, seq_len, dim ---> vocab)
        # outputs = torch.log(F.softmax(self.projection(outputs), dim=-1)+1e-15)
        # may be a better option to avoid stackoverflow

        # The reason why using log_softmax(): since we are using the nn.KLDivLoss, and from the PyTorch documents:
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor.

        return outputs

    def predict(self, inputs, max_generating_len=512, decoding_method='greedy', beam_size=5):
        inputs_mask = self.create_pad_mask(inputs).unsqueeze(1).unsqueeze(2)
        inputs = self.embedding_src(inputs)
        from_encoder = self.encoder(inputs, inputs_mask)
        sentences = self.greedy(inputs_mask=inputs_mask, from_encoder=from_encoder, max_generating_len=max_generating_len) \
            if decoding_method == 'greedy' else \
            self.beam_search(from_encoder=from_encoder, max_generating_len=max_generating_len, beam_size=beam_size)
        return sentences

    # of shape: (batch_size, generated_seq_len)
    #### this might have a very big problem !!!
    def greedy(self, inputs_mask, from_encoder, max_generating_len):
        batch_size = from_encoder.size(0)
        device = from_encoder.device
        sentences = torch.LongTensor(batch_size).fill_(self.BOS).unsqueeze(-1).to(device)
        EOS_indicator = torch.BoolTensor(batch_size).fill_(False).to(device)
        # when all the sentences in the batch reach EOS token in the generating process, there is no need to continue
        for i in range(max_generating_len):
            upper_triangle_mask = self.create_upper_triangle_mask(sentences).unsqueeze(0).unsqueeze(1)
            embeddings = self.embedding_tgt(sentences)
            # shape: (batch_size, i+1, dim)
            outputs = self.decoder(from_encoder, embeddings, inputs_mask, upper_triangle_mask.bool())
            # shape: (batch_size, i+2, dim)
            prob = F.softmax(self.projection(outputs[:, -1:, :]), dim=-1)  # shape: (batch_size, 1, dim ---> vocab)
            words = prob.max(-1)[1].long()  # shape: (batch_size, 1)
            sentences = torch.cat([sentences, words], dim=-1)  # shape: (batch_size, i+2)
            print('the sentence is {}'.format(sentences))
            EOS_indicator |= EOS_indicator | (words.squeeze(-1) == self.EOS).to(device)
            if EOS_indicator.sum() == batch_size:
                # every sentence in the batch has met the EOS token in this round
                break
        sentences = sentences.tolist()
        for i in range(batch_size):
            if self.EOS in sentences[i]:
                # this is only for the sentences that have met the <EOS> token during the 'maxLen' generating length
                index = sentences[i].index(self.EOS)
                sentences[i] = sentences[i][:index]
            sentences[i] = sentences[i][1:]
            # this is for all sentences since the <BOS> token should be removed
        return sentences

    def greedy_decode(self, inputs_mask, from_encoder, max_generating_len):
        # memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(self.BOS).long().to(from_encoder.device)
        for i in range(max_generating_len - 1):
            upper_triangle_mask = self.create_upper_triangle_mask(ys).unsqueeze(0).unsqueeze(1)
            embeddings = self.embedding_tgt(ys)
            out = self.decoder(from_encoder, embeddings, None, upper_triangle_mask.bool())
            # out = self.model.decode(memory, src_mask,
            #                    Variable(ys),
            #                    Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            # prob = model.generator(out[:, -1])
            prob = F.softmax(self.projection(out[:, -1]), dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys,
                            torch.ones(1, 1).fill_(next_word).to(from_encoder.device).long()], dim=1)
        return ys

    def beam_search(self, from_encoder, max_generating_len, beam_size):
        batch_size = from_encoder.size(0)
        device = from_encoder.device
        from_encoder = from_encoder.repeat(beam_size, 1, 1)  # shape: (batch_size*beam_size, seq_len, dim)
        sentences = torch.LongTensor(batch_size * beam_size).fill_(self.BOS).unsqueeze(-1).to(device)
        # shape: (batch_size*beam_size, 1)
        EOS_indicator = torch.BoolTensor(batch_size, beam_size).fill_(False).to(device)
        beam_prob = torch.zeros(batch_size, beam_size).to(device)
        # shape: (batch_size, beam) ; this actually stores the log of multiplications of
        # generating probability at each time-step
        for i in range(max_generating_len):
            upper_triangle_mask = self.create_upper_triangle_mask(sentences).unsqueeze(0).unsqueeze(1)
            embeddings = self.embedding_tgt(sentences)
            # shape: (batch_size*beam, i+1, dim)
            outputs = self.decoder(from_encoder, embeddings, None, upper_triangle_mask.bool())
            # shape: (batch_size*beam, i+2, dim)
            prob = F.softmax(self.projection(outputs[:, -1:, :]), dim=-1).view(batch_size, beam_size, -1)
            # shape: (batch_size*beam_size, 1, vocab) ---> (batch_size, beam_size, vocab)
            prob_mask = EOS_indicator.unsqueeze(-1).repeat(1, 1, self.num_vocab_tgt)
            prob.masked_fill_(prob_mask, 1)
            # for those beams that have reached the EOS token, the generating process should stop,
            # and the probability of generating any words is 1
            prob = torch.log(prob + 1e-15) + beam_prob.unsqueeze(-1)
            # process like: log(x3) + {log(x1)+log(x2)} = log(x1·x2·x3)
            beam_prob, index = prob.view(batch_size, -1).topk(beam_size, dim=-1)
            # shape: (batch_size, k {out of beam_size*vocab}) ; value, index ; k = beam_size
            ### note that should not reshape into (batch_size, beam, vocab) then use max on the last dim
            ### top-k beams should always chose from beam_size*vocab beams.
            words = index % self.num_vocab_tgt  # the period of words is vocab
            beam_index = index // self.num_vocab_tgt  # shape: (batch_size, beam_size)
            # print('at time-step {}, the words:\n {} and beam_index:\n {} and prob:\n {}'.
            # format(i, words, beam_index, beam_prob))
            # the desired (previous) beam index governs a segment of length vocab, in other words:
            # indices: (0, ..., vocab - 1) ---> all come from the 1-st (previous) beam
            EOS_indicator = EOS_indicator.gather(dim=-1, index=beam_index)
            # choose the top-k beams to construct new ones
            EOS_indicator |= (words == self.EOS).to(device)

            sentences = sentences.view(batch_size, beam_size, -1). \
                gather(dim=1, index=beam_index.unsqueeze(-1).repeat(1, 1, i + 1))
            # .gather(): save the sequences from previous time-step w.r.t. top-k beams

            sentences = torch.cat([sentences, words.unsqueeze(-1)], dim=-1).view(batch_size * beam_size, -1)
            # shape: (batch*size, generated_seq_len)
            if EOS_indicator.sum() == batch_size * beam_size:
                break
        final_index = beam_prob.max(-1)[1]  # (batch_size, beam) ---> (batch_size, )
        # print(beam_prob)
        final_sentences = []
        sentences = sentences.view(batch_size, beam_size, -1)
        # print(sentences.tolist())
        for i in range(batch_size):
            sentence = sentences[i, final_index[i], 1:].tolist()
            # remove the BOS token
            if self.EOS in sentence:
                EOS_index = sentence.index(self.EOS)
                sentence = sentence[:EOS_index]
            final_sentences.append(sentence)
            # shape: (batch_size, generated_seq_len)
        return final_sentences


def make_model(num_encoder_block, num_decoder_block, num_heads, num_vocab_src, num_vocab_tgt,
               dim, inner_layer_dim, dropout_rate, PAD, BOS, EOS):
    model = TransformerBase(num_encoder_block, num_decoder_block, num_heads, num_vocab_src, num_vocab_tgt,
                            dim, inner_layer_dim, dropout_rate, PAD, BOS, EOS)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

