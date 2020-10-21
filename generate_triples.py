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

from rplan import Rplan
from gpt2 import GPT2Model
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime

if __name__ == '__main__':
    dset = Rplan(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='train',
                 seq_len=200,
                 vocab_size=65)

    dloader = DataLoader(dset, batch_size=64, num_workers=10)

    config = GPT2Config(
        vocab_size=65,
        n_positions=120,
        n_ctx=120,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=True,
        is_encoder=False,
        n_types=3
    )
    model = GPT2Model(config)

    model = model.cuda()

    model_dict = {}
    ckpt = torch.load('/home/parawr/Projects/floorplan/models/triples/triples_39.pth', map_location='cpu')

    try:
        weights = ckpt.state_dict()
    except:
        weights = ckpt

    for k, v in weights.items():
        if 'module' in k:
            model_dict[k.replace('module.', '')] = v

    model.load_state_dict(model_dict, strict=True)
    model.eval()

    stats = ED()
    NUM_ROOM_TYPES = rplan_map.shape[0]

    stats.n_rooms = []
    stats.widths = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.heights = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.xlocs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.ylocs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.hadjs = [[] for _ in range(NUM_ROOM_TYPES)]
    stats.vadjs = [[] for _ in range(NUM_ROOM_TYPES)]

    bs = 20
    sample_idx = 1
    temperature = 0.5
    for jj in tqdm(range(500)):
        input_ids = torch.zeros(bs, dtype=torch.long).cuda().reshape(bs, 1)
        for ii in range(120):
            position_ids = torch.arange(ii + 1, dtype=torch.long).cuda().unsqueeze(0).repeat(bs, 1)
            attn_mask = torch.ones(ii + 1, dtype=torch.float).cuda().unsqueeze(0).repeat(bs, 1)
            loss = model(input_ids=input_ids,
                         position_ids=position_ids,
                         attention_mask=attn_mask
                         )

            logits = loss[0][:, ii, :] / temperature
            probs = torch.softmax(logits.squeeze(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # print(input_ids.shape, next_token.shape)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # print(input_ids.shape)
            # print(input_ids)

        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:]  # drop 0
        # print(input_ids.shape)
        samples = [input_ids[ii, :] for ii in range(bs)]

        for curr_sample in samples:
            # print(curr_sample.shape)
            stop_token_idx = np.argmax(curr_sample)
            # print(stop_token_idx)

            # if not valid length
            if stop_token_idx % 3 != 0:
                continue
            else:
                sample_idx += 1
                boxes = curr_sample[:stop_token_idx].reshape((-1, 3)) - 1

            curr_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            file_name = '/home/parawr/Projects/floorplan/samples/triples_0.5/' + curr_time + f'triples_39_temp_{temperature}_{sample_idx:04d}.npz'
            np.savez(file_name, boxes)






