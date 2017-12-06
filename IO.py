import h5py
import json
import torch
import numpy as np


class DataLoader(object):
    def __init__(self, args, subsets):
        # handling dictionary
        self.data = {}
        for k, v in json.load(open(args.input_json, 'r')).items():
            self.data[k] = v
        nwords = len(self.data['word2ind'])
        self.data['word2ind']['<START>'] = nwords
        self.data['word2ind']['<EOS>'] = nwords + 1
        self.vocab_size = nwords + 2
        print('Vocabulary size: {:d}'.format(self.vocab_size))
        # create an inverse mapping form id to word
        ind2word = {}
        for w, idx in self.data['word2ind'].items():
            ind2word[idx] = w
        self.data['ind2word'] = ind2word

        # handling text data
        data_file = h5py.File(args.input_data, 'r')
        # handling image feats
        imgFile = h5py.File(args.input_img, 'r')
        for subset in subsets:
            c_feat = np.array(data_file['cap_' + subset]).astype('long')
            self.data[subset + '_cap'] = torch.from_numpy(c_feat)
            q_feat = np.array(data_file['ques_' + subset]).astype('long')
            self.data[subset + '_ques'] = torch.from_numpy(q_feat)
            a_feat = np.array(data_file['ans_' + subset]).astype('long')
            self.data[subset + '_ans'] = torch.from_numpy(a_feat)

            img_pos = np.array(data_file['img_pos_' + subset]).astype('int')
            img_feats = np.array(imgFile['/images_' + subset])
            if args.img_norm == 1:
                print('Normalizing image features..')
                nm = np.sqrt(np.sum(
                    np.multiply(img_feats, img_feats), 1))[:, np.newaxis]
                img_feats /= nm.astype('float')

            self.data[subset + '_img_fv'] = torch.from_numpy(img_feats)
            self.data[subset + '_img_pos'] = torch.from_numpy(img_pos)
        data_file.close()

    def batchify(self, subset, batch_size, subset_size=None):
        data = []
        b_cap = self.data[subset + '_cap'][:subset_size] \
            .split(batch_size, 0)
        b_ques = self.data[subset + '_ques'][:subset_size] \
            .split(batch_size, 0)
        b_ans = self.data[subset + '_ans'][:subset_size] \
            .split(batch_size, 0)
        b_img = self.data[subset + '_img_fv'][:subset_size] \
            .split(batch_size, 0)
        b_idx = self.data[subset + '_img_pos'][:subset_size] \
            .split(batch_size, 0)
        for c, i, q, a, j in zip(b_cap, b_img, b_ques, b_ans, b_idx):
            data.append((c, i, q, a, j))
        return data

    def tensor2string(self, x):
        """utility function to debug"""
        words = [self.data['ind2word'].get(idx, '<PAD>')
                 for idx in x.contiguous().view(-1).tolist()]
        return ' '.join(words)
