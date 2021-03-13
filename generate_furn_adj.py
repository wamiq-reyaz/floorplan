import os, sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from furniture import NPZ, FurnitureEdges
from gpt2 import GraphGPTModel
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime
from utils import parse_wall_or_door_seq, parse_vert_seq, top_k_top_p_filtering


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')
    
    # Model
    parser.add_argument('--dim', default=264, type=int, help='number of dims of transformer')
    parser.add_argument('--vocab', default=65, type=int, help='quantization levels')
    parser.add_argument('--adj', default='h', type=str, help='h/v/all doors')
    parser.add_argument('--enc_n', default=120, type=int, help='number of encoder tokens')
    parser.add_argument('--enc_layer', default=12, type=int, help='number of encoder layers')
    parser.add_argument('--dec_n', default=48, type=int, help='number of decoder tokens')
    parser.add_argument('--dec_layer', default=12, type=int, help='number of decoder layers')

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

    parser.add_argument('--input_dir', default='../data/results/5_tuples_t_0.8/boxes', type=str, help='input directory (containing samples)')
    parser.add_argument('--output_dir', default='../data/results/5_tuples_t_0.8/wall_edges', type=str, help='output directory (will contain samples with added edges)')
    parser.add_argument('--model_path', default='../data/models/walls/model_doors_eps_m6_mlp_lr_m4_039.pth', type=str, help='location of the model weights file')
    parser.add_argument('--model_args_path', default='', type=str, help='location of the arguments the model was trained with (needed to reconstruct the model)')
    args = parser.parse_args()

    # load training arguments (these overwrite the defaults, but are overwritten by any parameter set in the command line)
    if args.model_args_path != '':
        with open(args.model_args_path, 'r') as f:
            model_args = json.load(f)
        parser.set_defaults(
            **{arg_name: arg_val for arg_name, arg_val in model_args.items() if arg_name in vars(args).keys()})

    args = parser.parse_args()

    BATCH_SIZE = args.bs
    dset = NPZ(
        root_dir=args.input_dir,
        seq_len=args.enc_n,
        edg_len=args.dec_n,
        vocab_size=args.vocab)
    #
    args.root_dir = '/ibex/scratch/parawr/floorplan/'
    args.datapath = '/mnt/ibex/Projects/furniture_018'

    dset = FurnitureEdges(root_dir=args.datapath,
                             split='val',
                             enc_len=args.enc_n,
                             dec_len=args.dec_n,
                             vocab_size=args.vocab,
                             edg_type=args.adj)
    #
    dloader = DataLoader(dset, batch_size=args.bs, num_workers=10, shuffle=True)


    # TODO: variable - depends on the model you need to sample from
    enc = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.enc_n,
        n_ctx=args.enc_n,
        n_embd=args.dim,
        n_layer=args.enc_layer,
        n_head=12,
        is_causal=False,
        is_encoder=True,
        use_pos_embed=True,
        n_types=6,
        separate=True,
        passthrough=False
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for jj, data in tqdm(enumerate(dloader)):

        # vert_seq = data['vert_seq'].to(device=device)
        # vert_attn_mask = data['vert_attn_mask'].to(device=device)

        vert_seq = data['enc_seq'].cuda()
        vert_attn_mask = data['enc_attn'].cuda()
        dec_seq = data['dec_seq'].cuda()
        print(vert_seq[0], dec_seq[0])
        bs = vert_seq.shape[0]
        input_ids = torch.zeros(bs, dtype=torch.long).to(device=device).reshape(vert_seq.shape[0], 1)
        for ii in range(args.dec_n):
            position_ids = torch.arange(ii+1, dtype=torch.long, device=device).unsqueeze(0).repeat(bs, 1)
            attn_mask = torch.ones(ii+1, dtype=torch.float, device=device).unsqueeze(0).repeat(bs, 1)
            with torch.no_grad():
                loss = model(node=vert_seq,
                             edg=input_ids,
                             attention_mask=attn_mask,
                             labels=None,
                             vert_attn_mask=vert_attn_mask
                             )

                # logits = loss[1][:, ii, :]
                logits = top_k_top_p_filtering(loss[1][:, ii, :], top_p=0.9)
                probs = torch.softmax(logits.squeeze(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # print(next_token.shape, input_ids.shape)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            # print(input_ids.shape)
            # print(input_ids)


        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:] - 1# drop 0 -
        # print(input_ids.shape)
        samples = [input_ids[ii, :] for ii in range(bs)]
        # print(samples)

        for kk, ss in enumerate(samples):
            full_name =  data['base_name'][kk]
            base_name = os.path.basename(full_name)
            root_name = os.path.splitext(base_name)[0]

            print(base_name, ss)
            sys.exit()

            # # TODO: path to save
            save_path = os.path.join(args.output_dir, root_name + '.pkl')

            with open(save_path, 'wb') as fd:
                pickle.dump(parse_wall_or_door_seq(ss), fd, protocol=pickle.HIGHEST_PROTOCOL)
