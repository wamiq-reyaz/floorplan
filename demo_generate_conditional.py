import os, sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from rplan import RplanConditional, LIFULLConditional
from gpt2 import EncDecGPTModel
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime
from utils import parse_wall_or_door_seq, parse_vert_seq, \
    top_k_top_p_filtering, on_local

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')

    # Model
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
    parser.add_argument('--wh', default=False, type=bool, help='number of decoder layers')
    parser.add_argument('--flipped', default=False, type=bool, help='Whether the decoder/encoder are flipped')

    # optimizer
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=40, type=int, help='number of total epochs to run')
    parser.add_argument('--device', default='cuda:0', type=str, help='device to use')

    # Data
    # parser.add_argument('--input_dir', default='/home/parawr/Projects/floorplan/samples/logged_0.8', type=str, help='input directory (containing samples)')

    # parser.add_argument('--input_dir', default='../data/results/5_tuples_t_0.8/boxes', type=str, help='input directory (containing samples)')
    # parser.add_argument('--output_dir', default='../data/results/5_tuples_t_0.8/door_edges', type=str, help='output directory (will contain samples with added edges)')
    # parser.add_argument('--model_path', default='../data/models/doors/GraphGPT-18-Oct_23-02-bs32-lr0.00013660761120233735-enl14-decl8-dim_embed144-9c895fce-584b-424e-96f6-3ea3c634bc39model_doors_eps_m6_mlp_lr_m4_34.pth', type=str, help='location of the model weights file')
    # parser.add_argument('--model_args_path', default='../data/models/doors/GraphGPT-18-Oct_23-02-bs32-lr0.00013660761120233735-enl14-decl8-dim_embed144-9c895fce-584b-424e-96f6-3ea3c634bc39args.json', type=str, help='location of the arguments the model was trained with (needed to reconstruct the model)')

    parser.add_argument('--input_dir', default='../data/results/5_tuples_t_0.8/boxes', type=str,
                        help='input directory (containing samples)')
    parser.add_argument('--output_dir', default='../data/results/5_tuples_t_0.8/wall_edges', type=str,
                        help='output directory (will contain samples with added edges)')
    parser.add_argument('--model_path', default='../data/models/walls/model_doors_eps_m6_mlp_lr_m4_039.pth', type=str,
                        help='location of the model weights file')
    parser.add_argument('--model_args_path', default='', type=str,
                        help='location of the arguments the model was trained with (needed to reconstruct the model)')
    args = parser.parse_args()

    # load training arguments (these overwrite the defaults, but are overwritten by any parameter set in the command line)
    if args.model_args_path != '':
        with open(args.model_args_path, 'r') as f:
            model_args = json.load(f)
        parser.set_defaults(
            **{arg_name: arg_val for arg_name, arg_val in model_args.items() if arg_name in vars(args).keys()})

    args = parser.parse_args()

    if on_local():
        args.root_dir = './'
        args.datapath = '/mnt/iscratch/datasets/lifull_ddg_var'

    else:  # assume IBEX
        args.root_dir = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '/ibex/scratch/parawr/datasets/rplan_ddg_var'


    val_set = LIFULLConditional(root_dir=args.datapath,
                               split='test',
                               enc_len=args.enc_n,
                               dec_len=args.dec_n,
                               vocab_size=args.vocab,
                               drop_dim=args.tuples == 3,
                               wh=args.wh)

    dloader = DataLoader(val_set, batch_size=args.bs, num_workers=10)

    if args.flipped:
        enc_is_causal=True
        dec_is_causal=False
    else:
        enc_is_causal=False
        dec_is_causal=True

    # TODO: variable - depends on the model you need to sample from
    enc = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.enc_n,
        n_ctx=args.enc_n,
        n_embd=args.dim,
        n_layer=args.enc_layer,
        n_head=12,
        is_causal=enc_is_causal,
        is_encoder=True,
        separate=False
    )

    dec = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.dec_n,
        n_ctx=args.dec_n,
        n_embd=args.dim,
        n_layer=args.dec_layer,
        n_head=12,
        is_causal=dec_is_causal,
        is_encoder=False,
        n_types=args.tuples
    )
    model = EncDecGPTModel(enc, dec)

    device = torch.device(args.device)

    model = model.to(device=device)

    model_dict = {}

    # TODO: the model to load
    ckpt = torch.load(args.model_path, map_location=device)

    try:
        weights = ckpt.state_dict()
    except:
        weights = ckpt

    for k, v in weights.items():
        if 'module' in k:
            model_dict[k.replace('module.', '')] = v

    model.load_state_dict(model_dict, strict=True)
    model.eval()

    for jj, data in tqdm(enumerate(dloader)):
        # bs = 10
        # enc_seq = data['enc_seq'].cuda().repeat(bs, 1)
        # enc_attn = data['enc_attn'].cuda().repeat(bs, 1)
        # enc_pos = data['enc_pos_id'].cuda().repeat(bs, 1)
        # names = data['base_name'] * bs
        # print(names)

        enc_seq = data['enc_seq'].cuda()
        enc_attn = data['enc_attn'].cuda()
        enc_pos = data['enc_pos_id'].cuda()
        names = data['base_name']



        bs = enc_seq.shape[0]
        input_ids = torch.zeros(args.dec_n, dtype=torch.long).to(device=device).unsqueeze(0).repeat(bs, 1)
        attn_mask = torch.zeros(args.dec_n, dtype=torch.float, device=device).unsqueeze(0).repeat(bs, 1)
        # print(input_ids.shape, enc_seq.shape)
        for ii in range(args.dec_n - 1):
            attn_mask[:, :ii+1] = 1
            with torch.no_grad():
                if args.flipped:
                    loss = model(enc_seq=input_ids,
                                dec_seq=enc_seq,
                                enc_attn_mask=attn_mask,
                                dec_attn_mask=enc_attn
                                )
                else:
                    loss = model( enc_seq=enc_seq,
                        dec_seq=input_ids,
                        enc_attn_mask=enc_attn,
                        dec_attn_mask=attn_mask
                        )

                logits = top_k_top_p_filtering(loss[0][:, ii, :], top_p=0.9)
                probs = torch.softmax(logits.squeeze(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids[:, ii+1] = next_token.squeeze()

        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:]  # drop 0
        samples = [input_ids[ii, :] for ii in range(bs)]

        for kk, curr_sample in enumerate(samples):
            # print(kk)
            # print(curr_sample.shape)
            stop_token_idx = np.argmax(curr_sample)
            # print(stop_token_idx)

            # if not valid length
            if stop_token_idx % 5 != 0:
                print('shit')
                continue
            else:
                boxes = curr_sample[:stop_token_idx].reshape((-1, 5)) - 1

            new_name = names[kk].replace('/', '_')
            print(new_name)
            with open(f'cond_lifull3_v2/nodes_0.9/{new_name}_{kk}.npz', 'wb') as fd:
                np.savez(fd, boxes)

        if jj==10:
            break