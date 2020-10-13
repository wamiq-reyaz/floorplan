import os, sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import pickle


class Flip(object):
    def __init__(self,
                 idx,
                 len_sample,
                 width,
                 p=0.5):
        self.idx = idx
        self.len_sample = len_sample
        self.width = width
        self.p = p


    def __call__(self, x):
        if torch.rand(1) < self.p:
            x[self.idx::self.len_sample] = 2 + self.width - x[self.idx::self.len_sample]

        return x

class Rot90(object):
    def __init__(self,
                 len_sample,
                 p=0.5):
        self.len_sample = len_sample
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            bkup = x[1::self.len_sample]
            x[1::self.len_sample] = x[2::self.len_sample]
            x[2::self.len_sample] = bkup

            if self.len_sample > 3:
                bkup = x[3::self.len_sample]
                x[3::self.len_sample] = x[4::self.len_sample]
                x[4::self.len_sample] = bkup

        return x

class Identity(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

class Rplan(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.drop_dim = drop_dim

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_image_nodoor_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_image_nodoor_xyhw.npy')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size+1], dtype=np.uint8)
        full_seq = np.ones(self.seq_len, dtype=np.uint8) * self.vocab_size
        attention_mask = np.ones(self.seq_len)
        pos_id = np.arange(self.seq_len, dtype=np.uint8)

        with open(path, 'rb') as f:
            tokens = np.load(path)
            if self.drop_dim:
                tokens = tokens[:, [0, 3,4]]
            tokens = tokens.ravel() + 1 # shift original by 1
            tokens = self.transforms(tokens)
            tokens = np.hstack((zero_token, tokens, stop_token))
            length = len(tokens)

            full_seq[:length] = tokens
            attention_mask[length+1:] = 0.0

        return {'seq': torch.tensor(full_seq).long(),
                'attn_mask': torch.tensor(attention_mask),
                'pos_id': torch.tensor(pos_id).long()}


    def __len__(self):
        return len(self.file_names)


class RrplanGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 transforms=None):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.edg_len = edg_len

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_image_nodoor_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_image_nodoor_xyhw.npy')
            print(path)

        horiz_dict_file = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_image_nodoor_edge_dict_v.pkl')

        if not os.path.exists(horiz_dict_file):
            horiz_dict_file = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_image_nodoor_edge_dict_v.pkl')
            print(horiz_dict_file)

        # create the vertex_data first
        with open(path, 'rb') as f:
            tokens = np.load(path) + 1 # shift original by 1
            tokens = tokens[:, :3] # drop h and w
            tokens = self.transforms(tokens)
            length = len(tokens)

        vert_seq = np.ones((self.seq_len, 3), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0, 0, 0)
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        flat_list = []
        with open(horiz_dict_file, 'rb') as f:
            dict_edg = pickle.load(f) # shift original by 1
            # print(dict_edg)
            for sublist in dict_edg.values():
                flat_list += sublist
                flat_list += [-1]
            flat_list = [-2] + flat_list + [-2] # -2 at beginning and end
            length = len(flat_list)

        edg_seq = np.ones(self.edg_len) * -2

        # print(flat_list)
        edg_seq[:length] = np.array(flat_list) + 2
        edg_seq[edg_seq == -2] = 0

        attn_mask = np.zeros(self.edg_len)
        attn_mask[:length] = 1

        pos_id = torch.arange(self.edg_len)

        return {'vert_seq': torch.tensor(vert_seq).long(),
                'edg_seq': torch.tensor(edg_seq).long(),
                'attn_mask': torch.tensor(attn_mask),
                'pos_id': pos_id.long(),
                'vert_attn_mask': torch.tensor(vert_attn_mask)}

    def __len__(self):
        return len(self.file_names)


class RrplanNPZTriples(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.file_names = glob(os.path.join(self.root_dir, '*.npz'))

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.edg_len = edg_len

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        path = self.file_names[idx]

        # create the vertex_data first
        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens['arr_0'] + 1 # shift original by 1
            tokens = self.transforms(tokens)
            length = len(tokens)

        vert_seq = np.ones((self.seq_len, 3), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0, 0, 0)
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        return {'vert_seq': torch.tensor(vert_seq).long(),
                'file_name': path,
                'vert_attn_mask': torch.tensor(vert_attn_mask)}

    def __len__(self):
        return len(self.file_names)

if __name__ == '__main__':
    # dset = Rplan(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train')
    # aa = dset[0]
    # print(aa)
    #
    # from torch.utils.data import DataLoader
    #
    # dloader = DataLoader(dset, batch_size=10)
    # cc = dloader.__iter__()
    # print(cc.__next__()['seq'].shape)



    dset_graph = RrplanGraph(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train')
    aa = dset_graph[0]
    print(aa)