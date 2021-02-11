import os, sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import pickle


class Flip(object):
    def __init__(self,
                 # idx,
                 # len_sample,
                 # width,
                 p=0.5):
        # self.idx = idx
        # self.len_sample = len_sample
        self.p = p


    def __call__(self, x):
        flipped = x.copy()
        if torch.rand(1) < self.p:
            flipped[:, 1] = 64 - x[:, 1]

        return flipped


class Rot90(object):
    def __init__(self,
                 p=0.5):
        self.p = p

    def __call__(self, x):
        rotted = x.copy()
        if torch.rand(1) < self.p:
            rotted[:, [1, 2]] = rotted[:, [2, 1]]
            if rotted.shape[-1] == 5:
                rotted[:, [3,4 ]] = rotted[:, [4, 3]]

        return rotted

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
                 transforms=None,
                 wh=False):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.drop_dim = drop_dim
        self.wh = wh

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
                if self.wh:
                    tokens = self.transforms(tokens[:, [0, 3, 4]])
                else:
                    tokens = self.transforms(tokens[:, :3])
            tokens = tokens.ravel() + 1 # shift original by 1
            # tokens = tokens)
            tokens = np.hstack((zero_token, tokens, stop_token))
            length = len(tokens)

            full_seq[:length] = tokens
            attention_mask[length+1:] = 0.0

        return {'seq': torch.tensor(full_seq).long(),
                'attn_mask': torch.tensor(attention_mask),
                'pos_id': torch.tensor(pos_id).long()}


    def __len__(self):
        return len(self.file_names)

class RplanConditional(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.vocab_size = vocab_size
        self.drop_dim = drop_dim
        self.wh = wh

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        name = self.file_names[idx].strip('\n')
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_image_nodoor_xyhw.npy')
        if not os.path.exists(path):
            print(path)
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_image_nodoor_xyhw.npy')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size+1], dtype=np.uint8)

        enc_seq = np.ones(self.enc_len, dtype=np.uint8) * self.vocab_size
        dec_seq = np.ones(self.dec_len, dtype=np.uint8) * self.vocab_size


        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        enc_pos_id = np.arange(self.enc_len, dtype=np.uint8)
        dec_pos_id = np.arange(self.dec_len, dtype=np.uint8)

        enc_temp = []
        dec_temp = []
        with open(path, 'rb') as f:
            tokens = np.load(path)
            if self.drop_dim:
                if self.wh:
                    tokens = self.transforms(tokens[:, [0, 3, 4]])
                else:
                    tokens = self.transforms(tokens[:, :3])

            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1: # idx has been added by one
                    enc_temp.append(curr_node)
                else:
                    dec_temp.append(curr_node)

        enc_tokens = np.hstack(
                            (zero_token,
                             np.asarray(enc_temp).ravel(),
                             stop_token)
        )
        dec_tokens = np.hstack(
                            (zero_token,
                             np.asarray(dec_temp).ravel(),
                             stop_token)
        )

        elength = len(enc_tokens)
        dlength = len(dec_tokens)

        enc_seq[:elength] = enc_tokens
        dec_seq[:dlength] = dec_tokens

        enc_attn[elength+1:] = 0.0
        dec_attn[dlength+1:] = 0.0

        return {'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn),
                'enc_pos_id': torch.tensor(enc_pos_id).long(),
                'dec_pos_id': torch.tensor(dec_pos_id).long(),
                'base_name': name}


    def __len__(self):
        return len(self.file_names)


class LIFULLConditional(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.vocab_size = vocab_size
        self.drop_dim = drop_dim
        self.wh = wh

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        name = self.file_names[idx].strip('\n')
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_xyhw.npy')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size+1], dtype=np.uint8)

        enc_seq = np.ones(self.enc_len, dtype=np.uint8) * self.vocab_size
        dec_seq = np.ones(self.dec_len, dtype=np.uint8) * self.vocab_size


        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        enc_pos_id = np.arange(self.enc_len, dtype=np.uint8)
        dec_pos_id = np.arange(self.dec_len, dtype=np.uint8)

        enc_temp = []
        dec_temp = []
        with open(path, 'rb') as f:
            tokens = np.load(path)
            if self.drop_dim:
                if self.wh:
                    tokens = self.transforms(tokens[:, [0, 3, 4]])
                else:
                    tokens = self.transforms(tokens[:, :3])

            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1: # idx has been added by one
                    enc_temp.append(curr_node)
                else:
                    dec_temp.append(curr_node)

        enc_tokens = np.hstack(
                            (zero_token,
                             np.asarray(enc_temp).ravel(),
                             stop_token)
        )
        dec_tokens = np.hstack(
                            (zero_token,
                             np.asarray(dec_temp).ravel(),
                             stop_token)
        )

        elength = len(enc_tokens)
        dlength = len(dec_tokens)

        enc_seq[:elength] = enc_tokens
        dec_seq[:dlength] = dec_tokens

        enc_attn[elength+1:] = 0.0
        dec_attn[dlength+1:] = 0.0

        return {'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn),
                'enc_pos_id': torch.tensor(enc_pos_id).long(),
                'dec_pos_id': torch.tensor(dec_pos_id).long(),
                'base_name': name}


    def __len__(self):
        return len(self.file_names)

class RplanConditionalDoors(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 doors='all',
                 lifull=False,
                 ver2=False):
        super().__init__()

        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = enc_len
        self.vocab_size = vocab_size
        self.edg_len = dec_len
        self.dims = 3 if drop_dim else 5
        self.doors = doors
        self.wh = wh
        self.lifull = lifull
        self.ver2 = ver2

        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            self.suffix + '_xyhw.npy')

        if self.doors == 'all':
            suffix = '_doorlist_all.pkl'

        elif self.doors == 'vert':
            suffix = '_doorlist_v.pkl'

        elif self.doors == 'horiz':
            suffix = '_doorlist_h.pkl'

        else:
            raise ValueError('The door type is invalid')

        if self.ver2:
            suffix = '_doorlist_all2.pkl'



        door_file = os.path.join(self.root_dir,
                                 self.file_names[idx].strip('\n') + \
                                 self.suffix + suffix)

        if not os.path.exists(door_file):
            print(door_file)

            door_file = os.path.join(self.root_dir,
                                     self.file_names[0].strip('\n') + \
                                     self.suffix + suffix)

        # create the vertex_data first

        ext_boxes = []
        int_boxes = []
        num_interior = 0
        mapper = dict()

        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1: # idx has been added by one
                    ext_boxes.append(idx)
                else:
                    mapper[idx] = num_interior
                    num_interior += 1
                    int_boxes.append(idx)

        tokens = tokens[int_boxes, :self.dims]
        length = len(tokens)

        # print(ext_boxess, int_boxes)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)
            for sublist in door_list:
                if (sublist[0] in ext_boxes) or (sublist[1] in ext_boxes):
                    continue
                flat_list += [mapper[sublist[0]], mapper[sublist[1]]]
                flat_list += [-1]


            flat_list = [-2] + flat_list + [-2] # -2 at beginning and end
            length = len(flat_list)

        edg_seq = np.ones(self.edg_len) * -2
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

class RplanConditionalWalls(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 doors='all',
                 lifull=False):
        super().__init__()

        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = enc_len
        self.vocab_size = vocab_size
        self.edg_len = dec_len
        self.dims = 3 if drop_dim else 5
        self.doors = doors
        self.wh = wh
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            self.suffix + '_xyhw.npy')

        if self.doors == 'all':
            suffix = 'walllist_all.pkl'

        elif self.doors == 'vert':
            suffix = 'walllist_v.pkl'

        elif self.doors == 'horiz':
            suffix = 'walllist_h.pkl'

        else:
            raise ValueError('The door type is invalid')


        door_file = os.path.join(self.root_dir,
                                 self.file_names[idx].strip('\n') + \
                                 self.suffix + suffix)

        if not os.path.exists(door_file):
            print(door_file)

            door_file = os.path.join(self.root_dir,
                                     self.file_names[0].strip('\n') + \
                                     self.suffix + suffix)

        # create the vertex_data first

        ext_boxes = []
        int_boxes = []
        num_interior = 0
        mapper = dict()

        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1: # idx has been added by one
                    ext_boxes.append(idx)
                else:
                    mapper[idx] = num_interior
                    num_interior += 1
                    int_boxes.append(idx)

        tokens = tokens[int_boxes, :self.dims]
        length = len(tokens)

        # print(ext_boxess, int_boxes)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)
            for sublist in door_list:
                if (sublist[0] in ext_boxes) or (sublist[1] in ext_boxes):
                    continue
                flat_list += [mapper[sublist[0]], mapper[sublist[1]]]
                flat_list += [-1]


            flat_list = [-2] + flat_list + [-2] # -2 at beginning and end
            length = len(flat_list)

        edg_seq = np.ones(self.edg_len) * -2
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

class RplanConditionalGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 doors='all',
                 lifull=False):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split + '.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = enc_len
        self.vocab_size = vocab_size
        self.edg_len = dec_len
        self.dims = 3 if drop_dim else 5
        self.doors = doors
        self.wh = wh
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                                self.file_names[0].strip('\n') + \
                                self.suffix + '_xyhw.npy')

        if self.lifull:
            suffix = f'_edgelist_{self.doors}.pkl'
        else:
            suffix = f'_edge_dict_{self.doors}.pkl'

        door_file = os.path.join(self.root_dir,
                                 self.file_names[idx].strip('\n') + \
                                 self.suffix + suffix)

        if not os.path.exists(door_file):
            print(door_file)

            door_file = os.path.join(self.root_dir,
                                     self.file_names[0].strip('\n') + \
                                     self.suffix + suffix)

        # create the vertex_data first

        ext_boxes = []
        int_boxes = []
        num_interior = 0
        mapper = dict()

        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1:  # idx has been added by one
                    ext_boxes.append(idx)
                else:
                    mapper[idx] = num_interior
                    num_interior += 1
                    int_boxes.append(idx)

        tokens = tokens[int_boxes, :self.dims]
        length = len(tokens)

        # print(ext_boxess, int_boxes)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length + 2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length + 2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)

            for node, sublist in door_list.items():
                if node in ext_boxes:
                    continue
                mapped = []
                for elem in sublist:
                    if elem in ext_boxes:
                        continue
                    else:
                        mapped.append(mapper[elem])

                # print(node, mapped, sublist, mapper)
                flat_list += mapped
                flat_list += [-1]

            # print(flat_list, int_boxes, mapper)

            flat_list = [-2] + flat_list + [-2]  # -2 at beginning and end
            length = len(flat_list)

        edg_seq = np.ones(self.edg_len) * -2
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

class LIFULLConditionalGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 doors='all',
                 lifull=True):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split + '.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = enc_len
        self.vocab_size = vocab_size
        self.edg_len = dec_len
        self.dims = 3 if drop_dim else 5
        self.doors = doors
        self.wh = wh
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                                self.file_names[0].strip('\n') + \
                                self.suffix + '_xyhw.npy')

        if self.lifull:
            suffix = f'_edgelist_{self.doors}.pkl'
        else:
            suffix = f'_edge_dict_{self.doors}.pkl'

        door_file = os.path.join(self.root_dir,
                                 self.file_names[idx].strip('\n') + \
                                 self.suffix + suffix)

        if not os.path.exists(door_file):
            print(door_file)

            door_file = os.path.join(self.root_dir,
                                     self.file_names[0].strip('\n') + \
                                     self.suffix + suffix)

        # create the vertex_data first

        ext_boxes = []
        int_boxes = []
        num_interior = 0
        mapper = dict()

        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            n_nodes = tokens.shape[0]
            for idx in range(n_nodes):
                curr_node = tokens[idx, :]
                if curr_node[0] == 1:  # idx has been added by one
                    ext_boxes.append(idx)
                else:
                    mapper[idx] = num_interior
                    num_interior += 1
                    int_boxes.append(idx)

        tokens = tokens[int_boxes, :self.dims]
        length = len(tokens)

        # print(ext_boxess, int_boxes)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length + 2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length + 2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)
            for ss in door_list:
                sublist = list(ss)
                if (sublist[0] in ext_boxes) or (sublist[1] in ext_boxes):
                    continue
                flat_list += [mapper[sublist[0]], mapper[sublist[1]]]
                flat_list += [-1]

            # print(mapper, flat_list, int_boxes)
            flat_list = [-2] + flat_list + [-2]  # -2 at beginning and end
            length = len(flat_list)

        edg_seq = np.ones(self.edg_len) * -2
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

class LIFULL(Dataset):
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
                            '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_xyhw.npy')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size+1], dtype=np.uint8)
        full_seq = np.ones(self.seq_len, dtype=np.uint8) * self.vocab_size
        attention_mask = np.ones(self.seq_len)
        pos_id = np.arange(self.seq_len, dtype=np.uint8)

        with open(path, 'rb') as f:
            tokens = np.load(path)
            if self.drop_dim:
                tokens = tokens[:, :3]
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

class RplanConstrained(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 doors='all',
                 lifull=False,
                 ):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split + '.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.enc_len = enc_len
        self.vocab_size = vocab_size
        self.dec_len = dec_len
        self.dims = 3 if drop_dim else 5
        self.doors = doors
        self.wh = wh
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
            self.constr_suffix = 'lifull_adj'
        else:
            self.suffix = '_image_nodoor'
            self.constr_suffix = 'rplan_adj'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                                self.file_names[0].strip('\n') + \
                                self.suffix + '_xyhw.npy')


        type_file = os.path.join(self.root_dir,
                                 self.constr_suffix,
                                 self.file_names[idx].strip('\n') + \
                                 '_room_types.npz')

        if not os.path.exists(type_file):
            print(type_file)

            type_file = os.path.join(self.root_dir,
                                 self.constr_suffix,
                                 self.file_names[0].strip('\n') + \
                                 '_room_types.npz')

        bbox_file = os.path.join(self.root_dir,
                                 self.constr_suffix,
                                 self.file_names[idx].strip('\n') + \
                                 '_room_bboxes.npz')

        if not os.path.exists(bbox_file):
            print(bbox_file)

            bbox_file = os.path.join(self.root_dir,
                                     self.constr_suffix,
                                     self.file_names[0].strip('\n') + \
                                     '_room_bboxes.npz')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size + 1], dtype=np.uint8)

        enc_seq = np.ones(self.enc_len, dtype=np.uint8) * self.vocab_size
        dec_seq = np.ones(self.dec_len, dtype=np.uint8) * self.vocab_size

        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        enc_pos_id = np.arange(self.enc_len, dtype=np.uint8)
        dec_pos_id = np.arange(self.dec_len, dtype=np.uint8)

        # adjacency data first
        with open(type_file, 'rb') as fd:
            constr_types = np.load(fd)['arr_0']


        with open(bbox_file, 'rb') as fd:
            constr_bboxes = np.load(fd)['arr_0']
            constr_bboxes = constr_bboxes[:, [2, 3]] ## bboxes are xmin, ymin, w, h

        # print(constr_types[:, None].shape, constr_bboxes.shape)
        constr = np.hstack((constr_types[:, None], constr_bboxes)) + 1
        enc_tokens = np.hstack(
                            (
                                zero_token,
                                constr.ravel(),
                                stop_token
                            ))

        # token/node data
        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            if self.dims == 3:
                if self.wh:
                    tokens = tokens[:, [0, 3, 4]]
                else:
                    tokens = tokens[:, :3]
            tokens = tokens.ravel()
            dec_tokens = np.hstack((zero_token, tokens, stop_token))

        elength = len(enc_tokens)
        dlength = len(dec_tokens)

        enc_seq[:elength] = enc_tokens
        dec_seq[:dlength] = dec_tokens

        enc_attn[elength + 1:] = 0.0
        dec_attn[dlength + 1:] = 0.0

        return {'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn),
                'enc_pos_id': torch.tensor(enc_pos_id).long(),
                'dec_pos_id': torch.tensor(dec_pos_id).long(),
                'base_name': self.file_names[idx].strip('\n')}

    def __len__(self):
        return len(self.file_names)


class RplanConstrainedGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 constr_len=200,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 drop_dim=False,
                 transforms=None,
                 wh=False,
                 edg_type='h',
                 lifull=False,
                 ):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split + '.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.constr_len = constr_len
        self.enc_len = enc_len
        self.vocab_size = vocab_size
        self.dec_len = dec_len
        self.dims = 3
        self.edg_type = edg_type
        self.wh = wh
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
            self.constr_suffix = 'lifull_adj'
        else:
            self.suffix = '_image_nodoor'
            self.constr_suffix = 'rplan_adj'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            print(path)
            path = os.path.join(self.root_dir,
                                self.file_names[0].strip('\n') + \
                                self.suffix + '_xyhw.npy')


        constr_file = os.path.join(self.root_dir,
                                 self.constr_suffix,
                                 self.file_names[idx].strip('\n') + \
                                 '_room_door_edges.npz')

        if not os.path.exists(constr_file):
            print(constr_file)

            constr_file = os.path.join(self.root_dir,
                                 self.constr_suffix,
                                 self.file_names[0].strip('\n') + \
                                 '_room_door_edges.npz')

        adj_path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            self.suffix + f'_edgelist_{self.edg_type}.pkl')

        if not os.path.exists(adj_path):
            print(adj_path)
            adj_path = os.path.join(self.root_dir,
                                self.file_names[0].strip('\n') + \
                                self.suffix +  f'_edgelist_{self.edg_type}.pkl')

        zero_token = np.array(0, dtype=np.uint8)
        stop_token = np.array([self.vocab_size + 1], dtype=np.uint8)

        constr_seq = np.zeros(self.constr_len, dtype=np.uint8)
        enc_seq = np.zeros(self.enc_len, dtype=np.uint8)
        dec_seq = np.ones(self.dec_len, dtype=np.uint8) * -2

        constr_attn = np.ones(self.constr_len)
        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        constr_pos_id = np.arange(self.constr_len, dtype=np.uint8)
        enc_pos_id = np.arange(self.enc_len, dtype=np.uint8)
        dec_pos_id = np.arange(self.dec_len, dtype=np.uint8)

        # adjacency data first
        with open(constr_file, 'rb') as fd:
            constrs = np.load(fd)['arr_0']


        #constraint data
        constrs = constrs.ravel() + 1

        constrs_tokens = np.hstack(
                            (
                                zero_token,
                                constrs.ravel(),
                                stop_token
                            ))

        clength = len(constrs_tokens)

        # token/node data
        with open(path, 'rb') as f:
            tokens = np.load(path)
            tokens = tokens + 1

            if self.dims == 3:
                if self.wh:
                    tokens = tokens[:, [0, 3, 4]]
                else:
                    tokens = tokens[:, :3]

            elength = tokens.shape[0]

        #edg data
        flat_list = []
        with open(adj_path, 'rb') as f:
            edg_list = pickle.load(f)  # shift original by 1
            # print(dict_edg)
            for sublist in edg_list():
                flat_list += list(sublist)
                flat_list += [-1]
            flat_list = [-2] + flat_list + [-2]  # -2 at beginning and end
            dlength = len(flat_list)

        # put into proper sequences
        constr_seq[:clength] = constrs_tokens

        enc_seq = np.ones((self.enc_len, 3), dtype=np.uint8) * (self.vocab_size + 1)
        enc_seq[0, :] = (0, 0, 0)
        enc_seq[2:elength+2, :] = tokens

        dec_seq[:dlength] = np.array(flat_list) + 2
        dec_seq[dec_seq == -2] = 0

        #attention masks
        constr_attn[clength + 1: ] = 0.0
        enc_attn[elength + 2:] = 0.0
        dec_attn[dlength + 1:] = 0.0

        return {'constr_seq': torch.tensor(constr_seq).long(),
                'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'constr_attn':torch.tensor(constr_attn),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn)}

    def __len__(self):
        return len(self.file_names)


class RrplanGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 edg_type='v',
                 transforms=None,
                 wh=False):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.edg_len = edg_len
        self.edg_type = edg_type
        self.wh = wh

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
                            f'_image_nodoor_edge_dict_{self.edg_type}.pkl')

        if not os.path.exists(horiz_dict_file):
            horiz_dict_file = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            f'_image_nodoor_edge_dict_{self.edg_type}.pkl')
            print(horiz_dict_file)

        # create the vertex_data first
        with open(path, 'rb') as f:
            tokens = np.load(path) + 1 # shift original by 1
            if self.wh:
                tokens = tokens[:, [0, 3, 4]]
            else:
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


class LIFULLGraph(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 edg_type='v',
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
        self.edg_type = edg_type

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            '_xyhw.npy')
            print(path)

        horiz_dict_file = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            f'_edgelist_{self.edg_type}.pkl')

        if not os.path.exists(horiz_dict_file):
            horiz_dict_file = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            f'_edgelist_{self.edg_type}.pkl')
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
            edg_list = pickle.load(f) # shift original by 1
            # print(dict_edg)
            for sublist in edg_list():
                flat_list += list(sublist)
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

class RrplanNPZFivers(Dataset):
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

        vert_seq = np.ones((self.seq_len, 5), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0, 0, 0, 0, 0)
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        return {'vert_seq': torch.tensor(vert_seq).long(),
                'file_name': path,
                'vert_attn_mask': torch.tensor(vert_attn_mask)}

    def __len__(self):
        return len(self.file_names)



class RrplanDoors(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 transforms=None,
                 dims=5,
                 doors='all',
                 lifull=False,
                 ver2=False):
        # TODO: implement dims and doors strategy
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.edg_len = edg_len
        if dims not in [3, 5]:
            raise ValueError('Dims can only be 3 or 5')
        self.dims = dims
        self.doors = doors
        self.lifull = lifull
        self.ver2 = ver2
        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'


        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    @classmethod
    def from_args(cls, *args, **kwargs):
        return cls.__init(*args, **kwargs)

    def __getitem__(self, idx):
        nodes_exist = True
        doors_exist = True
        name = self.file_names[idx].strip('\n')
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                             self.suffix + '_xyhw.npy')

            nodes_exist = False

        if self.doors == 'all':
            suffix = '_doorlist_all.pkl'

        elif self.doors == 'vert':
            suffix = '_doorlist_v.pkl'

        elif self.doors == 'horiz':
            suffix = '_doorlist_h.pkl'

        else:
            raise ValueError('The door type is invalid')

        if self.ver2:
            suffix = '_doorlist_all2.pkl'


        door_file = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + suffix)

        if not os.path.exists(door_file):
            doors_exist = False
            door_file = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            self.suffix + suffix)

        # create the vertex_data first
        with open(path, 'rb') as f:
            tokens = np.load(path) + 1 # shift original by 1
            tokens = tokens[:, :self.dims] # drop h and w
            tokens = self.transforms(tokens)
            length = len(tokens)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)
            # print(door_list)
            for sublist in door_list:
                flat_list += list(sublist)
                flat_list += [-1]
            flat_list = [-2] + flat_list + [-2] # -2 at beginning and end
            length = len(flat_list)

        # print(flat_list)
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
                'vert_attn_mask': torch.tensor(vert_attn_mask),
                'doors_exist': doors_exist,
                'nodes_exist': nodes_exist,
                'name': name,
                'flat_list':len(flat_list)}

    def __len__(self):
        return len(self.file_names)

class RrplanWalls(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 seq_len=200,
                 vocab_size=65,
                 edg_len=100,
                 transforms=None,
                 dims=5,
                 doors='all',
                 lifull=False):
        # TODO: implement dims and doors strategy
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.edg_len = edg_len
        if dims not in [3, 5]:
            raise ValueError('Dims can only be 3 or 5')
        self.dims = dims
        self.doors = doors
        self.lifull = lifull

        if self.lifull:
            self.suffix = ''
        else:
            self.suffix = '_image_nodoor'

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()


    def __getitem__(self, idx):
        name = self.file_names[idx].strip('\n')

        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + '_xyhw.npy')
        if not os.path.exists(path):
            path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            self.suffix + '_xyhw.npy')
            print(path)

        if self.doors == 'all':
            suffix = 'walllist_all.pkl'

        elif self.doors == 'vert':
            suffix = 'walllist_v.pkl'

        elif self.doors == 'horiz':
            suffix = 'walllist_h.pkl'

        else:
            raise ValueError('The door type is invalid')

        door_file = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            self.suffix + suffix)

        if not os.path.exists(door_file):
            print(door_file)

            door_file = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') +\
                            self.suffix + suffix)

        # create the vertex_data first
        with open(path, 'rb') as f:
            tokens = np.load(path) + 1 # shift original by 1
            tokens = tokens[:, :self.dims] # drop h and w
            tokens = self.transforms(tokens)
            length = len(tokens)

        vert_seq = np.ones((self.seq_len, self.dims), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0,) * self.dims
        vert_seq[2:length+2, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        # create the door data
        flat_list = []
        with open(door_file, 'rb') as f:
            door_list = pickle.load(f)
            # print(door_list)
            for sublist in door_list:
                flat_list += list(sublist)
                flat_list += [-1]
            flat_list = [-2] + flat_list + [-2] # -2 at beginning and end
            length = len(flat_list)

        # print(flat_list)
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
                'vert_attn_mask': torch.tensor(vert_attn_mask),
                'name': name,
                'flat_list': length}

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



    # dset_graph = RrplanGraph(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train')
    # aa = dset_graph[0]
    # print(aa)
    from tqdm import tqdm
    # dset_doors_v = RrplanDoors(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train', doors='vert')
    #
    # # for ii in tqdm(range(len(dset_doors_v))):
    # #     try:
    # #         _ = dset_doors_v[ii]
    # #     except:
    # #         print(ii)
    #
    # print(dset_doors_v.file_names[46545])

    # dset = RplanConditional(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train')
    # aa = dset[0]
    # print(aa)

    # dset = RplanConditionalGraph(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train', doors='v')
    # aa = dset[0]
    # print(aa)

    # dset = LIFULLConditionalGraph(root_dir='/mnt/iscratch/datasets/lifull_ddg_var', split='train', doors='v')
    # aa = dset[0]
    # print(dset.file_names[0])

    # dset = RplanConstrained(root_dir='/mnt/iscratch/datasets/rplan_ddg_var', split='train', doors='v', lifull=False, enc_len=64, dec_len=170)
    #
    # # idx = 99
    # for idx in range(500):
    #     aa = dset[idx]
    #     # print(aa)
    #     print(dset.file_names[idx])

    dset = RplanConstrainedGraph(root_dir='/mnt/iscratch/datasets/lifull_ddg_var',
                                 split='train', edg_type='v', lifull=True,
                                constr_len=64, enc_len=64, dec_len=150)

    idx = 99
    for idx in range(1):
        aa = dset[idx]
        print(aa)
        print(dset.file_names[idx])