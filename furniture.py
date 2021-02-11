import os, sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import networkx as nx
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

class Furniture(Dataset):
    def __init__(self, root_dir,
                 split=None,
                 pad_start=True,
                 pad_end=True,
                 seq_len=240,
                 vocab_size=65,
                 transforms=None):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = Identity()

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') +\
                            '_xywh.npz')

        zero_path = os.path.join(self.root_dir,
                            self.file_names[0].strip('\n') + \
                            '_xywh.npz')

        if not os.path.exists(path):
            path = zero_path

        zero_token = np.array(0, dtype=np.int32)
        stop_token = np.array([self.vocab_size+1], dtype=np.int32)
        full_seq = np.ones(self.seq_len, dtype=np.int32) * self.vocab_size
        attention_mask = np.ones(self.seq_len)
        pos_id = np.arange(self.seq_len, dtype=np.int32)

        len_tokens = np.ones(100) * -1

        with open(path, 'rb') as f:

            arrays = np.load(path)
            rooms = arrays['room']
            furniture = arrays['furniture']

            room_tokens = rooms[:6] + 1
            # unique_pieces, return_idx = np.unique(furniture[:, :6], axis=0, return_index=True)
            len_tokens_ = furniture[:, 5]
            len_orient = len(len_tokens_)
            len_tokens[:len_orient] = len_tokens_
            
            furniture_tokens = furniture[:, :6].ravel() + 1

            tokens = np.hstack((zero_token, room_tokens, furniture_tokens, stop_token))
            length = len(tokens)

        if length > 264:
            # print(idx)
            arrays = np.load(zero_path)
            rooms = arrays['room']
            furniture = arrays['furniture']

            room_tokens = rooms[:6] + 1
            furniture_tokens = furniture[:, :6].ravel() + 1

            tokens = np.hstack((zero_token, room_tokens, furniture_tokens, stop_token))
            length = len(tokens)


        full_seq[:length] = tokens
        attention_mask[length+1:] = 0.0

        return {'seq': torch.tensor(full_seq).long(),
                'attn_mask': torch.tensor(attention_mask),
                'pos_id': torch.tensor(pos_id).long(),
                'len_tokens': len_tokens,
                'file_name': path}


    def __len__(self):
        return len(self.file_names)


class FurnitureConditional(Dataset):
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
        zero_name = self.file_names[0].strip('\n')

        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            '_xywh.npz')

        zero_path = os.path.join(self.root_dir,
                                 self.file_names[0].strip('\n') + \
                                 '_xywh.npz')


        if not os.path.exists(path):
            path = zero_path
            name = zero_name

        zero_token = np.array(0, dtype=np.int32)
        stop_token = np.array([self.vocab_size+1], dtype=np.int32)

        enc_seq = np.ones(self.enc_len, dtype=np.int32) * self.vocab_size
        dec_seq = np.ones(self.dec_len, dtype=np.int32) * self.vocab_size


        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        enc_pos_id = np.arange(self.enc_len, dtype=np.int32)
        dec_pos_id = np.arange(self.dec_len, dtype=np.int32)

        redo = False
        with open(path, 'rb') as f:

            arrays = np.load(f)
            rooms = arrays['room']
            furniture = arrays['furniture']

            rooms = rooms[:6]
            furniture = furniture[:, :6]

            door_idx = [ii for ii, ss in enumerate(furniture) if ss[0] == 2] # 2 is the door index
            enc_temp = np.vstack((rooms, furniture[door_idx, :])) + 1

            unq, return_idx = np.unique(enc_temp, axis=0, return_index=True)

            enc_temp = enc_temp[np.sort(return_idx), :]
            len_tokens = enc_temp.shape[0]

            if enc_temp.size > 100:
                redo = True

            if door_idx == []:
                last_door_idx = 0
            else:
                last_door_idx = max(door_idx) + 1
            dec_temp = furniture[last_door_idx:, :] + 1

            if dec_temp.size > 264:
                redo = True

        if redo:
            # print('redoing', path)
            with open(zero_path, 'rb') as f:
                name = zero_name
                arrays = np.load(f)
                rooms = arrays['room']
                furniture = arrays['furniture']

                rooms = rooms[:6]
                furniture = furniture[:, :6]

                door_idx = [ii for ii, ss in enumerate(furniture) if ss[0] == 2]  # 2 is the door index
                enc_temp = np.vstack((rooms, furniture[door_idx, :])) + 1

                unq, return_idx = np.unique(enc_temp, axis=0, return_index=True)

                enc_temp = enc_temp[np.sort(return_idx), :]

                len_tokens = enc_temp.shape[0]

                if door_idx == []:
                    last_door_idx = 0
                else:
                    last_door_idx = max(door_idx) + 1
                dec_temp = furniture[last_door_idx:, :] + 1




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
        try:
            enc_seq[:elength] = enc_tokens
            dec_seq[:dlength] = dec_tokens
        except:
            print(enc_tokens)
            print(name)

        enc_attn[elength+1:] = 0.0
        dec_attn[dlength+1:] = 0.0

        return {'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn),
                'enc_pos_id': torch.tensor(enc_pos_id).long(),
                'dec_pos_id': torch.tensor(dec_pos_id).long(),
                'base_name': name,
                'len_tokens': len_tokens}

    def __len__(self):
        return len(self.file_names)



class EdgeTest(Dataset):
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
        zero_name = self.file_names[0].strip('\n')

        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            '_xywh.npz')

        zero_path = os.path.join(self.root_dir,
                                 self.file_names[0].strip('\n') + \
                                 '_xywh.npz')


        if not os.path.exists(path):
            path = zero_path
            name = zero_name

        redo = False
        with open(path, 'rb') as f:

            arrays = np.load(f)
            rooms = arrays['room']
            furniture = arrays['furniture']

            rooms = rooms[:6]
            edges = furniture[:, 6:]
            furniture = furniture[:, :6]
            

            unq, return_idx = np.unique(furniture, axis=0, return_index=True)

            # print(return_idx)
            # print(edges.shape)

            edges = edges[return_idx, :]
            edges = edges[:, return_idx]

        # edges = edges & 32
           

        return {'len': len(edges[edges != 0]),
                'name': name}

    def __len__(self):
        return len(self.file_names)


class NPZ(Dataset):
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
            # tokens[0, 0] +=2
            # tokens[0, -1] += 2
            unq, return_idx = np.unique(tokens, axis=0, return_index=True)
            tokens = tokens[return_idx, :]
            length = len(tokens)



        raster_idx = 1000 * tokens[:, 1] + tokens[:, 2]
        sorted_idx = np.argsort(raster_idx)

        tokens = tokens[sorted_idx, :]

        vert_seq = np.ones((self.seq_len, 6), dtype=np.uint8) * (self.vocab_size + 1)
        vert_seq[0, :] = (0, 0, 0, 0, 0, 0)
        vert_seq[1:length+1, :] = tokens

        vert_attn_mask = np.zeros(self.seq_len)
        vert_attn_mask[:length+2] = 1

        return {'vert_seq': torch.tensor(vert_seq).long(),
                'file_name': path,
                'vert_attn_mask': torch.tensor(vert_attn_mask)}

    def __len__(self):
        return len(self.file_names)

class EdgeTest2(Dataset):
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
        zero_name = self.file_names[0].strip('\n')

        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            '_edges.npz')

        zero_path = os.path.join(self.root_dir,
                                 self.file_names[0].strip('\n') + \
                                 '_edges.npz')


        if not os.path.exists(path):
            path = zero_path
            name = zero_name

        redo = False
        with open(path, 'rb') as f:

            arrays = np.load(f)
            nodes = arrays['n']
            edg = arrays['w']

        return {'nodes': nodes,
                'edg': edg}

    def __len__(self):
        return len(self.file_names)


class FurnitureEdges(Dataset):
    def __init__(self, root_dir, split=None,
                 pad_start=True, pad_end=True,
                 enc_len=200,
                 dec_len=100,
                 vocab_size=65,
                 edg_type='w'):
        super().__init__()
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, split+'.txt'), 'r') as fd:
            self.file_names = fd.readlines()

        self.pad_start = pad_start
        self.pad_end = pad_end
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.vocab_size = vocab_size
        self.edg_type = edg_type

    def __getitem__(self, idx):
        name = self.file_names[idx].strip('\n')
        zero_name = self.file_names[0].strip('\n')

        path = os.path.join(self.root_dir,
                            self.file_names[idx].strip('\n') + \
                            '_edges.npz')

        zero_path = os.path.join(self.root_dir,
                                 self.file_names[0].strip('\n') + \
                                 '_edges.npz')


        if not os.path.exists(path):
            path = zero_path
            name = zero_name

        redo = False
        with open(path, 'rb') as fd:
            arrays = np.load(fd)
            rooms = arrays['r'][:6]
            nodes = arrays['n']
            edg = arrays[self.edg_type]

            adj_mat = nx.from_numpy_matrix(edg)
            edg_list = adj_mat.edges()
            # print(len(edg_list))

            if 3*len(edg_list) > 290:
                print(3*len(edg_list))
                print('redone')
                redo = True
        
        if redo:
            with open(zero_path, 'rb') as fd:
                arrays = np.load(fd)
                rooms = arrays['r'][:6] 
                nodes = arrays['n']
                edg = arrays[self.edg_type]

                adj_mat = nx.from_numpy_matrix(edg)
                edg_list = adj_mat.edges()

        zero_token = np.zeros((1, 6), dtype=np.int32)
        stop_token = np.ones((1, 6), dtype=np.int32) *self.vocab_size+1

        enc_seq = np.ones((self.enc_len, 6), dtype=np.int32) * self.vocab_size + 1
        dec_seq = np.ones(self.dec_len, dtype=np.int32) * -2

        enc_attn = np.ones(self.enc_len)
        dec_attn = np.ones(self.dec_len)

        enc_pos_id = np.arange(self.enc_len, dtype=np.int32)
        dec_pos_id = np.arange(self.dec_len, dtype=np.int32)

        # fill in tokens
        # print(zero_token.shape, rooms.shape, nodes.shape)
        # print(rooms)
        enc_tokens = np.vstack((zero_token,
                              rooms+1,
                              nodes + 1,
                              stop_token))

        flat_list = []
        new_list = list(edg_list)
        dlength = len(flat_list)
        for sublist in new_list:
            flat_list += list(sublist)
            flat_list += [-1]
        flat_list = [-2] + flat_list + [-2]
        dlength = len(flat_list)

        elength = len(enc_tokens)
        enc_seq[:elength] = enc_tokens
        dec_seq[:dlength] = np.array(flat_list) + 2
        dec_seq[dec_seq == -2] = 0

        enc_attn[elength+1:] = 0.0
        dec_attn[dlength+1:] = 0.0

        # print(dec_tokens.dtype)

                    
        return {'enc_seq': torch.tensor(enc_seq).long(),
                'dec_seq': torch.tensor(dec_seq).long(),
                'enc_attn': torch.tensor(enc_attn),
                'dec_attn': torch.tensor(dec_attn),
                'enc_pos_id': torch.tensor(enc_pos_id).long(),
                'dec_pos_id': torch.tensor(dec_pos_id).long(),
                'base_name': name}

    def __len__(self):
        return len(self.file_names)


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot  as plt

    dset = FurnitureEdges(root_dir='../furniture_018',
                     split='train',
                     enc_len=100,
                     dec_len=264,
                     vocab_size=65,
                     edg_type='h')

    # print(len(dset))
    idx = 777

    
    print(dset[idx])
    print(dset.file_names[idx])

    # dset = EdgeTest2(root_dir='../furniture_018',
    #                  split='train',
    #                  enc_len=100,
    #                  dec_len=264,
    #                  vocab_size=65)

    # print(len(dset))
    # idx = 8989
    
    # print(dset[idx])
    # print(dset.file_names[idx])
    
    # dset = Furniture(root_dir='../furniture_018',
    #                  split='train',
    #                  enc_len=100,
    #                  dec_len=264,
    #                  vocab_size=65)

    # dset = Furniture(root_dir='../furniture_018',
    #              split='train',
    #              pad_start=True,
    #              pad_end=True,
    #              seq_len=400,
    #              vocab_size=65,
    #              transforms=None)

    # from torch.utils.data import DataLoader

    # loader = DataLoader(dset, batch_size=64, num_workers=10)
    # print(dset[0])

    # for ii, ss in enumerate(dset):
    #     print(ii)
    #     print(ss['name'], ss['len'])
    
    # print(len(dset))



    # from collections import Counter
    # counter = Counter()
    # for ii, ss in enumerate(tqdm(loader)):

    #     for subsample in ss['len_tokens']:
    #         counter.update(subsample.numpy())
            # if len(subsample) > 1:
            #     for ssubsamples in subsample:
            #         hist.append(subsample.numpy())
            # else:
            #     hist.append(subsample.numpy())

    # print(counter)

    # # for kk, vv in counter.items():
    # del counter[-1]
    # plt.bar(counter.keys(), counter.values())
    # plt.savefig('orient_hist.png')

    # np.save('hist_orient', np.asarray(hist))


    # aa = np.load('hist_orient.npy')
    # aa = aa[aa  > 0]
    # print(np.max(aa))
    # print(len(aa))
    # plt.hist(aa, density=True, bins=30)
    # plt.savefig('hist_orient.png')

    # print(len(aa[aa == 0]))



    # print(dset[21]['name'])
    # print(len(aa))
    # print(aa.shape)
    # print(aa[:100])
 
    # # from tqdm import tqdm
    # # dset = Furniture(root_dir='/mnt/iscratch/datasets/furniture_017',
    # #                  split='train',
    # #                  seq_len=264,
    # #                  vocab_size=65)
    # #
    # # # max = 0
    # for ss in tqdm(dset):
    #     seq = ss['seq']
    #     attn = ss['attn_mask']

    # from tqdm import tqdm

    # dset = FurnitureConditional(root_dir='/mnt/iscratch/datasets/furniture_017',
    #                  split='train',
    #                  enc_len=100,
    #                  dec_len=264,
    #                  vocab_size=65)

    # from torch.utils.data import DataLoader

    # loader = DataLoader(dset, batch_size=64, num_workers=10)

    # for ii, ss in enumerate(tqdm(loader)):
    #     try:
    #         enc_seq = ss['enc_seq']
    #         dec_seq = ss['dec_seq']

    #         enc_attn = ss['enc_attn']
    #         dec_attn = ss['dec_attn']
    #     except:
    #         print(ss['base_name'])


    # max = 0
    # for ii, ss in enumerate(tqdm(dset)):
    #     try:
    #         enc_seq = ss['enc_seq']
    #         dec_seq = ss['dec_seq']
    #
    #         enc_attn = ss['enc_attn']
    #         dec_attn = ss['dec_attn']
    #     except:
    #         print(ss['base_name'])

        # print(enc_seq, dec_seq)

        # if ii == 100:
        #     break



        # print(enc_seq)
        # print(dec_seq)
        # sys.exit()



