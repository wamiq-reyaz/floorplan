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

from rplan import RrplanNPZTriples
from gpt2 import GraphGPTModel
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime

from utils import parse_wall_or_door_seq, parse_vert_seq,\
    parse_edge_seq


if __name__ == '__main__':
    BATCH_SIZE = 30
    dset = RrplanNPZTriples(root_dir='./samples/triples_wh_0.9',
                 seq_len=120,
                 edg_len=100,
                 vocab_size=65)

    dloader = DataLoader(dset, batch_size=BATCH_SIZE, num_workers=10)

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
    ckpt = torch.load('./models/face_modelv_eps_m6_mlp_lr_m4/face_modelv_eps_m6_mlp_lr_m4_39.pth', map_location='cpu')

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

            logits = loss[1][:, ii, :] / 0.9
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

            save_path = os.path.join('samples', 'aatriples_0.8', 'edges', 'v', root_name + '.pkl')
            with open(save_path, 'wb') as fd:
                pickle.dump(parse_edge_seq(ss), fd, protocol=pickle.HIGHEST_PROTOCOL)





