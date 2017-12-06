import torch
import argparse
from IO import DataLoader
import models
from torch.autograd import Variable
from util import args
import pickle
import time
import numpy as np


def decode_dialog(ind2word, seq):
    return " ".join([ind2word[int(x)] for x in seq])


parser = argparse.ArgumentParser()
parser.add_argument('-save_model', type=str, default='visdialog.pt',
                    help='trained parameters')
parser.add_argument('-cuda', action='store_true',
                    help='enable training with cuda')
parser.add_argument('-batch_size', default=1, type=int,
                    help='batch_size (1 for now)')
parser.add_argument('-decoding', default='greedy', type=str,
                    choices=['greedy', 'sampling'],
                    help='Greedy or softmax sampling')
parser.add_argument('-eval_size', default=10000, type=int,
                    help="""Size of validation set to evaluate.
                    Default is 10K, must be an integer <= 40K""")
parser.add_argument('-intervention', default='none', type=str,
                    choices=['none', 'answer', 'question', 'image',
                             'caption', 'negation'],
                    help='Whether to intervene or do standard decoding.')
parser.add_argument('-round', default=0, type=int,
                    help='Intervene from this specific round')
parser.add_argument('-pnoise', default=1.0, type=float,
                    help="""Percentage of random noise.
                    1 means complete noise""")
opt = parser.parse_args()

# Load checkpoint
print("=> loading checkpoint '{}'".format(opt.save_model))
checkpoint = torch.load(opt.save_model)

# Merge two argparse namespaces with priority to args from this script
opt = {**vars(checkpoint['opt']), **vars(opt)}
opt = args(opt)  # wrapper around a dict to object with fields

print(opt.__dict__)

print("=> loaded checkpoint '{}'".format(opt.save_model))

# Construct dataloader
loader = DataLoader(opt, ['val'])
vocab_size = loader.vocab_size
opt.vocab_size = vocab_size
bos_idx = loader.data['word2ind']['<START>']
eos_idx = loader.data['word2ind']['<EOS>']

# Load models
qbot = models.QBot(opt)
abot = models.ABot(opt)
if opt.share_qa_embeddings:
    qbot.embeddings.weight = abot.embeddings.weight

bots = models.QABots(qbot, abot)

if opt.cuda:
    qbot.cuda()
    abot.cuda()
    bots.cuda()
bots.load_state_dict(checkpoint['params'])

# Params
batch_size = opt.batch_size  # TODO change routine for parallel evaluation
num_rounds = 10
max_decode_len = 20
valid_data = loader.batchify(subset='val',
                             batch_size=batch_size,
                             subset_size=opt.eval_size)

bots.eval()  # eval mode


# not sure what's going on here, but let try
def truncate2d(x, padding_idx=0):
    # x (batch, length)
    masked_pad = x.ne(padding_idx).sum(-1)
    max_length = masked_pad.int().max()
    return x[:, :max_length].contiguous()


intervention = opt.intervention
vocab_tokens = list(loader.data['ind2word'].keys())
print('Running evaluation ')


# negation interventions
yes_idx = loader.data['word2ind']['yes']
no_idx = loader.data['word2ind']['no']


def pertube(x, p):
    """x is a Variable of LongTensor"""
    new_x = torch.from_numpy(
        np.random.choice(a=vocab_tokens, size=tuple(x.size()))).long().cuda()
    mask = torch.FloatTensor(new_x.size()).fill_(p).bernoulli().long().cuda()
    new_x = new_x * mask + (1-mask) * x.data
    return Variable(new_x, volatile=True)


gt_img_features = loader.data['val_img_fv'][:40000, :]
# convert to pytorch
val_img_fv = gt_img_features.float().cuda()


def rank(fv, true_idx):
    d = torch.norm(val_img_fv - fv, 2, 1)
    _, idx = torch.sort(d)
    _, r = torch.sort(idx)
    return (r[true_idx] + 0.5) / 40000


reports = {}
for i in range(num_rounds):
    reports[i] = []

n_negs = 0
# start playing game
for i, batch in enumerate(valid_data):
    t = time.time()
    if (i % 500) == 0 and i > 0:
        print('| Evaluating dialogue at {}/{}'.format(i + 1, opt.eval_size))
        for k in range(num_rounds):
            mpr = (1-torch.FloatTensor(reports[k]).mean()) * 100
            print("|\tround %d | MPR %.2f" % (k, mpr))

        if n_negs > 0:
            print('num negations: %d' % n_negs)
        print('-' * 12)

    caption, img_feats, *_, img_idx = batch
    if opt.cuda:
        caption = caption.cuda()
        img_feats = img_feats.cuda()

    caption = truncate2d(caption)  # should truncate first
    # For storing inference results
    dialogue_idx = img_idx.numpy()[0]
    caption = Variable(caption, volatile=True)
    if intervention == 'caption':
        caption = pertube(caption, opt.pnoise)
    caption = caption.transpose(0, 1)
    img = Variable(img_feats, volatile=True)

    # encode caption
    # qbot also updates history
    q_fact_enc = qbot.enc_fact(caption)
    # print('check caption encoded %.3f' % q_fact_enc.data.norm(2))
    _, q_hist_state = qbot.history_encoder(q_fact_enc[None, :, :])
    # print('initial q', q_hist_state[0].data.norm(2))

    a_fact_enc = abot.enc_fact(caption)
    a_hist_state = None  # empty at first

    for r in range(num_rounds):
        # (1) qbot ask question
        # q_i  (length, 1)
        q_i = qbot.generate(q_hist_state, bos_idx, eos_idx,
                            max_decode_len, opt.decoding == 'greedy')

        if intervention == 'question' and r >= opt.round:
            q_i = pertube(q_i, opt.pnoise)

        # (2) abot answer
        # (2.1) encode the question First
        a_question_enc = abot.enc_question(q_i)

        # Intervention for image
        if intervention == 'image' and r == opt.round:
            # Replace image with random noise
            img.data.uniform_()

        # concatenate img, a_q, Fa to pass to history encoder
        # important: preserve order of concatenation as done in training
        inpt = torch.cat([img, a_question_enc, a_fact_enc], -1)
        _, a_hist_state = abot.history_encoder(inpt[None, :, :], a_hist_state)

        # (2.2) generate answer
        a_i = abot.generate(a_hist_state, bos_idx, eos_idx,
                            max_decode_len, opt.decoding == 'greedy')
        # checking
        # print('q: %s' % loader.tensor2string(q_i.data))
        # print('a: %s' % loader.tensor2string(a_i.data))
        # print('----')

        if intervention == 'answer' and r >= opt.round:
            a_i = pertube(a_i, opt.pnoise)
        if intervention == 'negation' and r >= opt.round:
            if a_i.numel() == 2:
                _tmp = a_i.view(-1)
                _idx = _tmp.data[0]
                if _idx == yes_idx or _idx == no_idx:
                    if _idx == yes_idx:
                        _tmp.data[0] = no_idx
                    else:
                        _tmp.data[0] = yes_idx
                    n_negs += 1

        # (3) both bots encode fact fi
        # concatenate q and a (both are tok1...tokN EOS)
        f_i = torch.cat([q_i[:-1], a_i[:-1]], dim=0)

        q_fact_enc = qbot.enc_fact(f_i)
        # print(f_i)
        _, q_hist_state = qbot.history_encoder(q_fact_enc[None, :, :],
                                               q_hist_state)
        # print('q_hist_state', q_hist_state[0].sum().data[0])

        a_fact_enc = abot.enc_fact(f_i)

        # (4) qbot makes image prediction
        y = qbot.pred_img(q_hist_state[0][-1])
        pr = rank(y.data, dialogue_idx)
        reports[r] += [pr]
    if i % 500 == 0 and i > 0:
        save_file = 'report_{}_{}_{}.pkl'.format(intervention,
                                                 opt.round,
                                                 opt.pnoise)
        with open(save_file, 'wb') as f:
            pickle.dump(reports, f)


# save finals results
save_file = 'report_{}_{}_{}.pkl'.format(intervention,
                                         opt.round,
                                         opt.pnoise)

with open(save_file, 'wb') as f:
    pickle.dump(reports, f)
