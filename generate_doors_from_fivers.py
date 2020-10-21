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
    input_dir = '../data/results/5_tuples_t_0.8/boxes'
    output_dir = '../data/results/5_tuples_t_0.8/door_edges'
    model_path = '../data/models/doors/GraphGPT-18-Oct_23-02-bs32-lr0.00013660761120233735-enl14-decl8-dim_embed144-9c895fce-584b-424e-96f6-3ea3c634bc39model_doors_eps_m6_mlp_lr_m4_34.pth'
    # output_dir = '../data/results/5_tuples_t_0.8/wall_edges'
	
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')
    # Model
    parser.add_argument('--epochs', default=40, type=int, help='number of total epochs to run')
    parser.add_argument('--dim', default=264, type=int, help='number of dims of transformer')
    parser.add_argument('--seq_len', default=120, type=int, help='the number of vertices')
    parser.add_argument('--edg_len', default=48, type=int, help='how long is the edge length or door length')
    parser.add_argument('--vocab', default=65, type=int, help='quantization levels')
    parser.add_argument('--tuples', default=5, type=int, help='3 or 5 based on initial sampler')
    parser.add_argument('--doors', default='all', type=str, help='h/v/all doors')
    parser.add_argument('--enc_n', default=120, type=int, help='number of encoder tokens')
    parser.add_argument('--enc_layer', default=12, type=int, help='number of encoder layers')
    parser.add_argument('--dec_n', default=48, type=int, help='number of decoder tokens')
    parser.add_argument('--dec_layer', default=12, type=int, help='number of decoder layers')

    # optimizer
    parser.add_argument('--bs', default=64, type=int, help='batch size')

    # Data
    parser.add_argument("--datapath", default='/home/parawr/Projects/floorplan/samples/logged_0.8',
                        type=str, help="Root folder to save data in")

    args = parser.parse_args()

    BATCH_SIZE = args.bs
    dset = RrplanNPZFivers(root_dir=args.datapath,
                 seq_len=args.seq_len,
                 edg_len=args.edg_len,
                 vocab_size=args.vocab)

    dloader = DataLoader(dset, batch_size=BATCH_SIZE, num_workers=10)


    # TODO: variable - depends on the model you need to sample from
    enc = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.enc_n,
        n_ctx=args.enc_n,
        n_embd=args.dim,
        n_layer=args.enc_layer,
        n_head=12,
        is_causal=False,
        is_encoder=True
    )

    dec = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.dec_n,
        n_ctx=args.dec_n,
        n_embd=args.dim,
        n_layer=args.dec_layer,
        n_head=12,
        is_causal=True,
        is_encoder=False
    )
    model = GraphGPTModel(enc, dec)

    model = model.cuda()

    model_dict = {}

    # TODO: the model to load
    ckpt = torch.load(model_path, map_location='cpu')

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
        for ii in range(48):
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
        print(samples)

        for jj, ss in enumerate(samples):
            full_name =  data['file_name'][jj]
            base_name = os.path.basename(full_name)
            root_name = os.path.splitext(base_name)[0]

            # TODO: path to save
            save_path = os.path.join(output_dir, root_name + '.pkl')
            with open(save_path, 'wb') as fd:
                pickle.dump(parse_edge_seq(ss), fd, protocol=pickle.HIGHEST_PROTOCOL)
