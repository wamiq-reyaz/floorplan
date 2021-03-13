import torch.nn as nn
import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm
from utils import top_k_top_p_filtering
import argparse
from datetime import datetime as dt
from furniture import Furniture
from gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
import shutil
from glob import glob
import argparse
from utils import on_local
import json
from datetime import datetime
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')
    # Model
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--dim', default=264, type=int, help='number of dims of transformer')
    parser.add_argument('--vocab', default=65, type=int, help='quantization levels')
    parser.add_argument('--tuples', default=5, type=int, help='3 or 5 based on initial sampler')
    parser.add_argument('--enc_n', default=264, type=int, help='number of encoder tokens')
    parser.add_argument('--enc_layer', default=12, type=int, help='number of encoder layers')

    # optimizer
    parser.add_argument('--step', default=25, type=int, help='how many epochs before reducing lr')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='reduction in lr')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device to use')


    # Data
    parser.add_argument("--root_dir", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--datapath", default='.', type=str, help="Root folder to save data in")
    parser.add_argument('--dataset', default='rplan', type=str, help='dataset to train on')
    parser.add_argument('--lifull', default=False, type=bool, help='whether to train on lifull data')

    # Notes
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='', type=str, help="Wandb tags")

    parser.add_argument('--input_dir', default='../data/results/5_tuples_t_0.8/boxes', type=str,
                        help='input directory (containing samples)')
    parser.add_argument('--output_dir', default='../data/results/5_tuples_t_0.8/wall_edges', type=str,
                        help='output directory (will contain samples with added edges)')
    parser.add_argument('--model_path', default='../data/models/walls/model_doors_eps_m6_mlp_lr_m4_039.pth', type=str,
                        help='location of the model weights file')
    parser.add_argument('--model_args_path', default='', type=str,
                        help='location of the arguments the model was trained with (needed to reconstruct the model)')
    args = parser.parse_args()

    if args.model_args_path != '':
        with open(args.model_args_path, 'r') as f:
            model_args = json.load(f)
        parser.set_defaults(
            **{arg_name: arg_val for arg_name, arg_val in model_args.items() if arg_name in vars(args).keys()})

    args = parser.parse_args()

    if on_local():
        args.root_dir = './'
        args.datapath = '/mnt/iscratch/datasets/furniture_017'

    else:  # assume IBEX
        args.root_dir = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '/ibex/scratch/parawr/datasets/furniture_017'

    # dset = Furniture(root_dir=args.datapath,
    #                  split='train',
    #                  seq_len=args.enc_n,
    #                  vocab_size=65,
    #                  )

    # val_set = Furniture(root_dir=args.datapath,
    #                     split='val',
    #                     seq_len=args.enc_n,
    #                     vocab_size=65,
    #                     )

    config = GPT2Config(
            vocab_size=args.vocab,
            n_positions=args.enc_n,
            n_ctx=args.enc_n,
            n_embd=args.dim,
            n_layer=args.enc_layer,
            n_head=12,
            n_types=6,
            is_causal=True,
            is_encoder=False
        )
    model = GPT2Model(config)

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


    bs = 20
    for jj in tqdm(range(500)):
        input_ids = torch.tensor([0], dtype=torch.long).cuda().reshape(1,1).repeat(bs, 1)
        for ii in range(args.enc_n-1):
            position_ids = torch.arange(ii+1, dtype=torch.long).cuda().unsqueeze(0).repeat(bs, 1)
            attn_mask = torch.ones(ii+1, dtype=torch.float).cuda().unsqueeze(0).repeat(bs, 1)
            loss = model(input_ids=input_ids,
                         position_ids=position_ids,
                         attention_mask=attn_mask
            )

            logits = top_k_top_p_filtering(loss[0][:, ii, :], top_p=0.9)
            probs = torch.softmax(logits.squeeze(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # if ii == 4:
            #     next_token[:] = 9


            input_ids = torch.cat([input_ids, next_token], dim=-1)



        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:] # drop 0
        samples = [input_ids[ii, :] for ii in range(bs)]


        for kk, curr_sample in enumerate(samples):
            stop_token_idx = np.argmax(curr_sample)

            # print(curr_sample)
            # print(curr_sample[:stop_token_idx])
            # print(len(curr_sample[:stop_token_idx]))
            # print(stop_token_idx)

            if stop_token_idx % 6 != 0:
                print('failed', stop_token_idx)
                continue
            else:
                boxes = curr_sample[:stop_token_idx].reshape((-1, 6)) - 1

            curr_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
            save_path = os.path.join(args.output_dir, curr_time + str(kk) + '.npz')

            # print(boxes)
            # if kk == 4:
            #     sys.exit()
            np.savez(save_path, boxes)






