import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
import torch.functional as F
from tqdm import tqdm
import argparse

from rplan import RrplanNPZFivers
from gpt2 import GraphGPTModel
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime



def parse_edge_seq(seq):
    curr_node = 0
    edge_list = []

    seq_biased = seq - 2
    for elem in seq_biased:
        if elem == -1:
            curr_node += 1
            continue
        elif elem == -2:
            break
        else:
            edge_list.append((curr_node, elem))

    return edge_list

def parse_vert_seq(seq):
    if isinstance(seq, torch.Tensor):
        try:
            seq = seq.cpu().numpy()
        except:
            seq = seq.numpy()

    # print(seq)
    # first slice the head off
    seq = seq[2:, :]

    #find the max along any axis of id, x, y
    stop_token_idx = np.argmax(seq[:, 0])


    boxes = seq[:stop_token_idx, :] - 1

    return boxes



if __name__ == '__main__':
    BATCH_SIZE = 30
    dset = RrplanNPZFivers(root_dir='./samples/triples_0.8',
                 seq_len=120,
                 edg_len=100,
                 vocab_size=65)

    dloader = DataLoader(dset, batch_size=BATCH_SIZE, num_workers=10)


    # TODO: variable - depends on the model you need to sample from
    enc = GPT2Config(
        vocab_size=65,
        n_positions=120,
        n_ctx=120,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=False,
        is_encoder=True
    )

    dec = GPT2Config(
        vocab_size=65,
        n_positions=100,
        n_ctx=100,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=True,
        is_encoder=False
    )
    model = GraphGPTModel(enc, dec)

    model = model.cuda()

    model_dict = {}

    # TODO: the model to load
    ckpt = torch.load('./models/face_model_eps_m6_mlp_lr_m4/face_model_eps_m6_mlp_lr_m4_39.pth', map_location='cpu')

    try:
        weights = ckpt.state_dict()
    except:
        weights = ckpt

    for k, v in weights.items():
        if 'module' in k:
            model_dict[k.replace('module.', '')] = v

    model.load_state_dict(model_dict, strict=True)
    model.eval()



    bs = BATCH_SIZE

    for jj, data in tqdm(enumerate(dloader)):

        vert_seq = data['vert_seq'].cuda()
        vert_attn_mask = data['vert_attn_mask'].cuda()
        # print(data['file_name'])


        input_ids = torch.zeros(bs, dtype=torch.long).cuda().reshape(bs, 1)
        for ii in range(100):
            position_ids = torch.arange(ii+1, dtype=torch.long).cuda().unsqueeze(0).repeat(bs, 1)
            attn_mask = torch.ones(ii+1, dtype=torch.float).cuda().unsqueeze(0).repeat(bs, 1)
            loss = model(node=vert_seq,
                         edg=input_ids,
                         attention_mask=attn_mask,
                         labels=None,
                         vert_attn_mask=vert_attn_mask
                         )

            logits = loss[1][:, ii, :]
            probs = torch.softmax(logits.squeeze(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # print('in generate probs, next_token', probs.shape, next_token.shape)




            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # print(input_ids.shape)
            # print(input_ids)


        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:] # drop 0
        # print(input_ids.shape)
        samples = [input_ids[ii, :] for ii in range(bs)]

        for jj, ss in enumerate(samples):
            full_name =  data['file_name'][jj]
            base_name = os.path.basename(full_name)
            root_name = os.path.splitext(base_name)[0]

            # TODO: path to save
            save_path = os.path.join('samples', 'triples_0.8', 'edges', 'h', root_name + '.pkl')
            with open(save_path, 'wb') as fd:
                pickle.dump(parse_edge_seq(ss), fd, protocol=pickle.HIGHEST_PROTOCOL)





