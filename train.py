import torch
import argparse
from IO import DataLoader
import models
from torch.autograd import Variable
import math
from random import shuffle


parser = argparse.ArgumentParser()
# Data option
parser.add_argument('-input_img', type=str, default='data/data_img.h5')
parser.add_argument('-input_data', type=str, default='data/visdial_data.h5')
parser.add_argument('-input_json', type=str,
                    default='data/visdial_params.json')
# Model option
parser.add_argument('-embed_dim', type=int, default=300)
parser.add_argument('-rnn_dim', type=int, default=512)
parser.add_argument('-fact_dim', type=int, default=512)
parser.add_argument('-history_dim', type=int, default=512)
parser.add_argument('-num_layers', type=int, default=2,
                    help='Number of nlayers in LSTM')
parser.add_argument('-img_norm', type=int, default=1)
parser.add_argument('-img_feature_size', type=int, default=4096)
parser.add_argument('-max_history_len', type=int, default=40)
# Training option
parser.add_argument('-dropout', type=float, default=0)
parser.add_argument('-batch_size', type=int, default=64,
                    help="""Batch size (number of threads)
                    (Adjust base on GPU memory)""")
parser.add_argument('-lr', type=float, default=0.001,
                    help="learning rate.")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="clip gradient at this value")
parser.add_argument('-report_every', type=int, default=50)
parser.add_argument('-save_model', type=str, default='visdialog.pt',
                    help='trained parameters')
# the following options are extremely useful when the data is small
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and out embeddings for decoder.')
parser.add_argument('-share_qa_embeddings', action='store_true',
                    help='share word embeddings between QBot and ABot')

parser.add_argument('-num_epochs', type=int, default=15, help='Epochs')
parser.add_argument('-cuda', action='store_true',
                    help='enable training with cuda')

opt = parser.parse_args()
# utils prepare the data


def truncate(x, padding_idx=0):
    # cut off all unncessary paddings, this makes generation faster
    # because we don't have to predict pad
    # x (batch, rounds, length)
    masked_pad = x.ne(padding_idx).sum(-1)
    max_length = masked_pad.int().max()
    return x[:, :, :max_length].contiguous()


# build dataloader
loader = DataLoader(opt, ['train', 'val'])
vocab_size = loader.vocab_size
opt.vocab_size = vocab_size
bos_idx = loader.data['word2ind']['<START>']
eos_idx = loader.data['word2ind']['<EOS>']


def pad(input, bos_idx, eos_idx, padding_idx=0):
    """pad <START> and <EOS>
    input (LongTensor) of size batch_size, n_rounds, max_length
    """
    # (1) 3D -> 2D tensor
    batch_size, n_rounds, length = input.size()
    input_2d = input.view(-1, length)
    real_lens = input_2d.ne(padding_idx).sum(-1).int()
    new_len = length + 2
    pad_inp = torch.LongTensor(batch_size * n_rounds, new_len) \
        .fill_(padding_idx)
    pad_inp[:, 0] = bos_idx
    pad_inp[:, 1:length+1] = input_2d  # copy shit
    idx = [i for i in range(pad_inp.size(0))]
    pad_inp[idx, (real_lens + 1).tolist()] = eos_idx
    # (2) 2D -> 3D
    return pad_inp.view(batch_size, n_rounds, -1)


def prepare(batch, eval=False):
    """Batch is a tuple of (c, q, a), each tensor is batch_first"""
    c, img, q, a, *_ = batch
    # truncate
    q = pad(q, bos_idx, eos_idx)
    q = truncate(q)
    batch_size, n_rounds, q_len = q.size()
    a = pad(a, bos_idx, eos_idx)
    a = truncate(a)
    ret = [c, img, q, a]
    if opt.cuda:
        ret = [x.cuda() for x in ret]
    # wrap by Variable
    ret = [Variable(x, volatile=eval) for x in ret]
    return ret


# build models
qbot = models.QBot(opt)
abot = models.ABot(opt)
if opt.share_qa_embeddings:
    qbot.embeddings.weight = abot.embeddings.weight

train_data = loader.batchify('train', opt.batch_size)
valid_data = loader.batchify('val', opt.batch_size)
if opt.cuda:
    qbot.cuda()
    abot.cuda()


def eval(valid_data, bots):
    # switch to eval mode
    bots.eval()
    tot_loss = 0
    nsamples = 0
    for i, batch in enumerate(valid_data):
        c, img, q, a = prepare(batch, True)
        nsamples += c.size(0)
        loss_q_text, loss_q_img, loss_a_text = bots(c, img, q, a)
        tot_loss += loss_q_text + loss_q_img + loss_a_text
    loss = tot_loss.data[0] / nsamples
    # resume training mode
    bots.train()
    return loss


def train_bots(opt):
    if opt.share_qa_embeddings:
        qbot.embeddings.weight = abot.embeddings.weight
    bots = models.QABots(qbot, abot)

    print('Training bots...')
    optimizer = torch.optim.Adam(bots.parameters(), lr=opt.lr)
    for e in range(1, opt.num_epochs + 1):
        nbatch = len(train_data)
        # shuffle training data
        shuffle(train_data)
        for i, batch in enumerate(train_data):
            c, img, q, a, *_ = prepare(batch)
            n_q_words = q[:, :, 1:].data.ne(0).int().sum()
            batch_size = c.size(0)
            bots.zero_grad()
            loss_q_text, loss_q_img, loss_a_text = bots(c, img, q, a)
            ppl_q = math.exp(loss_q_text.data[0] / n_q_words)
            tot_loss = loss_q_text + loss_q_img + loss_a_text
            tot_loss.div(batch_size).backward()
            torch.nn.utils.clip_grad_norm(bots.parameters(), opt.max_grad_norm)
            optimizer.step()
            n_a_words = a[:, :, 1:].data.ne(0).int().sum()
            ppl_a = math.exp(loss_a_text.data[0] / n_a_words)
            if i % opt.report_every == 0 and i > 0:
                msg = 'Epoch %d | update %4d / %d | total loss %.1f | ' \
                        + 'question ppl %.1f | answer ppl %.1f'
                msg = msg % (e, i, nbatch, tot_loss.data[0] / batch_size,
                             ppl_q, ppl_a)
                print(msg)
        print('Evaluate!')
        val_loss = eval(valid_data, bots)
        print('Validation loss: %.3f' % val_loss)
        model_state_dict = bots.state_dict()
        checkpoint = {'opt': opt,
                      'params': model_state_dict}
        torch.save(checkpoint, opt.save_model)


train_bots(opt)
