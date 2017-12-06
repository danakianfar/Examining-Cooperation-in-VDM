import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
from torch.autograd import Variable


def concatqa(q, a, padding_idx=0):
    """Concatenate question and answer
    Args:
        q (FloatTensor) of size (batch, n_rounds, q_len)
        a (FloatTensor) of size (batch, n_rounds, a_len)
    return a FloatTensor of size (batch, n_rounds, qa_len)"""
    # (1) flatten 3D tensor to 2D
    batch, n_rounds, q_len = q.size()
    a_len = a.size(2)
    assert batch == a.size(0) and n_rounds == a.size(1)
    q2d = q.view(-1, q_len)
    a2d = a.view(-1, a_len)
    # (2) get actual lengths without padding
    real_q_len = q2d.data.ne(padding_idx).int().sum(-1)
    real_a_len = a2d.data.ne(padding_idx).int().sum(-1)
    lengths = real_a_len + real_q_len

    max_lengths = lengths.max()
    # (3) create the concatenation
    qa2d = q2d.data.new(batch * n_rounds, max_lengths).fill_(padding_idx)
    i = 0
    for lq, la in zip(real_q_len.tolist(), real_a_len.tolist()):
        qa2d[i][:lq].copy_(q2d[i][:lq].data)  # copy question
        qa2d[i][lq:lq+la].copy_(a2d[i][:la].data)
        i += 1
    # (4) return 3D Tensor
    qa = qa2d.view(batch, n_rounds, -1)
    return Variable(qa)


def criterion(vocab_size, padding_idx=0):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocab_size)
    weight[padding_idx] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    return crit


def truncate2d(x, padding_idx=0):
    n = x.data.ne(padding_idx).int().sum(0).max()
    return x[:n].contiguous()


def forward_rnn(input, embeddings, rnn, init_hidden=None):
    """Generic RNN forward pass. Here the input is not sorted by its
    actual lengths. First, we will sort the input in the decreasing order
    by its length, then we pass the sorted input to LSTM, which requires
    the lengths is sorted. Finally, we restore the correct order of the
    input.
    Args:
        input (Variable of LongTensor): (seq_length, batch_size)
        embeddings: nn.Embedding object
        rnn: nn.LSTM
        init_hidden: initial hidden state of rnn, (hidden, cell)
    Returns:
        a Tensor size (rnn_dim, batch_size)"""
    # (1) sort input
    lengths = input.data.ne(0).int().sum(0)
    sorted_lengths, idx = lengths.sort(0, True)
    input = input[:, idx]
    emb = embeddings(input)
    packed_emb = pack(emb, sorted_lengths.tolist())
    if init_hidden is not None:
        init_hidden = (init_hidden[0][:, idx],
                       init_hidden[1][:, idx])
    output, hidden = rnn(packed_emb, init_hidden)
    output = unpack(output)[0]  # seq_length, batch_size, hidden_size

    # (2) restore the correct order
    _, original_idx = idx.sort(0, False)
    return output[:, original_idx, :], (hidden[0][:, original_idx, :],
                                        hidden[1][:, original_idx, :])


class QBot(nn.Module):
    # TODO: add inference by stochastic samping or greedy decoding
    """An instance of QBot
    """
    def __init__(self, opt):
        super(QBot, self).__init__()
        # (1) build word embedding layer, not that it is important
        # to set padding idx = 0 here
        self.embeddings = nn.Embedding(opt.vocab_size, opt.embed_dim,
                                       padding_idx=0)
        self.fact_encoder = nn.LSTM(opt.embed_dim, opt.rnn_dim,
                                    opt.num_layers, dropout=opt.dropout)
        self.q_decoder = nn.LSTM(opt.embed_dim, opt.rnn_dim, opt.num_layers,
                                 dropout=opt.dropout)
        self.q_generator = nn.Linear(opt.rnn_dim, opt.vocab_size)
        # tied weights optionally
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016)
        if opt.share_decoder_embeddings:
            assert opt.embed_dim == opt.rnn_dim
            self.embeddings.weight = self.q_generator.weight

        self.history_encoder = nn.LSTM(opt.rnn_dim, opt.rnn_dim,
                                       opt.num_layers, dropout=opt.dropout)
        self.regressor = nn.Linear(opt.rnn_dim, opt.img_feature_size)

        # quick and dirty
        self.crit = criterion(opt.vocab_size)
        self.use_cuda = opt.cuda

    def enc_caption(self, input):
        _, hidden = forward_rnn(input, self.embeddings, self.fact_encoder)
        return hidden

    def forward_question(self, input, init_hidden=None):
        """Compute question,
        Args:
            input (LongTensor): (seq_length, batch_size)
            init_hidden (FloatTensor): (rnn_dim, batch_size)
        Returns:
            log prob over generated words"""
        output, hidden = forward_rnn(input, self.embeddings, self.q_decoder,
                                     init_hidden)
        logit = self.q_generator(output.view(-1, output.size(2)))
        return F.log_softmax(logit), hidden

    def enc_fact(self, input):
        _, hidden = forward_rnn(input, self.embeddings, self.fact_encoder)
        return hidden[0][-1]  # ignore cell

    def pred_img(self, hist_state):
        return self.regressor(hist_state)

    def forward(self, c, img, q, a):
        """ input is (c, img, q, a)
        Args:
            c (LongTensor) of size (batch_size, c_len)
            img (FloatTensor) of size (batch_size, img_feature_size)
            q (LongTensor) of size (batch_size, n_rounds, q_len)
            a (LongTensor) of size (batch_size, n_rounds, q_len)
        """
        c = c.transpose(0, 1)
        c_h = self.enc_fact(c)

        # run fact encoder in parallel for all rounds
        # (1) concat q and a to get fact
        fact = concatqa(q, a)  # batch_size, n_rounds, qa_len
        batch_size, n_rounds = fact.size(0), fact.size(1)

        # (2) flatten 3D tensor to 2D for parallel forward
        #  => qa_len, batch_size * n_rounds
        fact_2d = fact.view(-1, fact.size(2)).transpose(0, 1)

        # (3) roll in fact_encoder
        fact_2d_h = self.enc_fact(fact_2d)  # batch_size * n_rounds, hidden
        fact_h = fact_2d_h.view(batch_size, n_rounds, -1)

        _, init_state = self.history_encoder(c_h[None, :, :])
        fact_h = fact_h.transpose(0, 1)  # n_rounds, batch_size, hidden
        # generating questions
        loss = 0
        img_loss = 0

        for i in range(n_rounds):
            q_i = q[:, i].transpose(0, 1)  # q_len, batch_size
            q_i = truncate2d(q_i)
            # forward from <bos>
            logp_i, _ = self.forward_question(q_i[:-1], init_state)
            loss_i = self.crit(logp_i, q_i[1:, :].view(-1))
            loss += loss_i
            _, init_state = self.history_encoder(fact_h[i].unsqueeze(0),
                                                 init_state)
            # predict the image
            y = self.pred_img(init_state[0][-1])  # take hidden, top layer
            img_loss_i = torch.norm(img - y, 2)
            img_loss += img_loss_i

        return loss, img_loss

    def generate(self, init_state, bos_idx, eos_idx, max_len=20, greedy=True):
        """Generate question! For simplicity we assume the batch_size is 1
        It'll be a bit slow but we don't really care about the speed now.
        Args:
            init_state (tuple of hidden and cell) from history encoder
            greedy (boolean): if False then use stochastic sampling
            max_len (int): max number of tokens in the question
        """
        q = init_state[0].data.new(max_len, 1).fill_(bos_idx).long()
        q = Variable(q, volatile=True)
        hidden = init_state
        for i in range(max_len-1):
            logp, hidden = self.forward_question(q[[i], :], hidden)
            if greedy:
                _, next_idx = logp.max(1)
                q[i+1] = next_idx
            else:
                # stochastic sampling
                nex_idx = logp.exp().multinomial(1)
                q[i+1] = nex_idx
            if q[i+1, :].data[0] == eos_idx:
                return q[1:i+2, :]  # verify format is correct
        return q[1:]


class ABot(nn.Module):
    # TODO: factorize the code, reusable?
    """An instance of ABot
    """
    def __init__(self, opt):
        super(ABot, self).__init__()
        # (1) build word embedding layer, not that it is important
        # to set padding idx = 0 here
        self.embeddings = nn.Embedding(opt.vocab_size, opt.embed_dim,
                                       padding_idx=0)
        self.fact_encoder = nn.LSTM(opt.embed_dim, opt.rnn_dim,
                                    opt.num_layers, dropout=opt.dropout)
        self.q_encoder = nn.LSTM(opt.embed_dim, opt.rnn_dim, opt.num_layers,
                                 dropout=opt.dropout)
        # Answer Decoder
        self.a_decoder = nn.LSTM(opt.embed_dim, opt.rnn_dim, opt.num_layers,
                                 dropout=opt.dropout)
        self.a_generator = nn.Linear(opt.rnn_dim, opt.vocab_size)
        # tied weights optionally
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016)
        if opt.share_decoder_embeddings:
            assert opt.embed_dim == opt.rnn_dim
            self.embeddings.weight = self.a_generator.weight

        hist_dim = opt.rnn_dim * 2 + opt.img_feature_size
        self.history_encoder = nn.LSTM(hist_dim, opt.rnn_dim,
                                       opt.num_layers, dropout=opt.dropout)

        # quick and dirty
        self.crit = criterion(opt.vocab_size)
        self.use_cuda = opt.cuda

    def enc_fact(self, input):
        _, hidden = forward_rnn(input, self.embeddings, self.fact_encoder)
        return hidden[0][-1]  # ignore cell

    def enc_question(self, input):
        _, hidden = forward_rnn(input, self.embeddings, self.q_encoder)
        return hidden[0][-1]

    def forward_answer(self, input, init_hidden):
        output, hidden = forward_rnn(input, self.embeddings, self.a_decoder,
                                     init_hidden)
        logits = self.a_generator(output.view(-1, output.size(-1)))
        return F.log_softmax(logits), hidden

    def forward(self, c, img, q, a):
        """Similar to QBot, check QBot arguments."""
        # (1) encode caption as F_0
        c = c.transpose(0, 1)  # batch second
        c_h = self.enc_fact(c)

        # (2) encode facts up to round T-1
        fact = concatqa(q, a)  # batch_size, n_rounds, qa_len
        batch_size, n_rounds = fact.size(0), fact.size(1)
        # fact up to T-1
        fact = fact[:, :n_rounds-1, :].contiguous()
        fact_2d = fact.view(-1, fact.size(2)).transpose(0, 1)

        # (2.1) roll in fact_encoder
        fact_2d_h = self.enc_fact(fact_2d)
        # batch_size * (n_rounds - 1), hidden
        fact_h = fact_2d_h.view(batch_size, n_rounds-1, -1)
        fact_h = fact_h.transpose(0, 1)

        # concat caption (F_0) and fact features,
        # => (n_rounds, batch_size, q_dim + fact_dim)
        Fa = torch.cat([c_h[None, :, :], fact_h], 0)

        # (3) encode questions
        q2d = q.view(-1, q.size(2)).transpose(0, 1)
        q2d_h = self.enc_question(q2d)
        q_h = q2d_h.view(batch_size, n_rounds, -1).transpose(0, 1)
        # TODO: check when to stop? DONE!
        hist_state = None
        loss = 0
        for i in range(n_rounds):
            inp_i = torch.cat([img, q_h[i], Fa[i]], -1)
            _, hist_state = self.history_encoder(inp_i[None, :, :], hist_state)
            # generate answer
            a_i = truncate2d(a[:, i].transpose(0, 1))
            logp_i, _ = self.forward_answer(a_i[:-1], hist_state)
            loss_i = self.crit(logp_i, a_i[1:, :].view(-1))
            loss += loss_i
        return loss

    def generate(self, init_state, bos_idx, eos_idx, max_len=20, greedy=True):
        """Generate an answer! For simplicity we assume the batch_size is 1
        It'll be a bit slow but we don't really care about the speed now.
        Args:
            init_state (tuple of hidden and cell) from history encoder
            greedy (boolean): if False then use stochastic sampling
            max_len (int): max number of tokens in the question
        """
        # create a tensor of shape [max_len, 1]
        # filled with BOS tokens and of type long
        a = init_state[0].data.new(max_len, 1).fill_(bos_idx).long()
        a = Variable(a, volatile=True)

        hidden = init_state

        # decode token-by-token
        for i in range(max_len-1):

            # feed BOS token or last output
            logp, hidden = self.forward_answer(a[[i], :], hidden)

            if greedy:
                _, next_idx = logp.max(1)
                a[i+1] = next_idx
            else:
                # stochastic sampling
                next_idx = logp.exp().multinomial(1)
                a[i+1] = next_idx

            # stopping criterion
            if a[i+1, :].data[0] == eos_idx:
                return a[1:i+2, :]
        return a[1:]


class QABots(nn.Module):
    def __init__(self, qbot, abot):
        super(QABots, self).__init__()
        self.qbot = qbot
        self.abot = abot

    def forward(self, c, img, q, a):
        loss_q_text, loss_q_img = self.qbot(c, img, q, a)
        loss_a_text = self.abot(c, img, q, a)
        # jointly optimize
        # loss = loss_a_text + loss_q_img + loss_a_text
        return loss_q_text, loss_q_img, loss_a_text
