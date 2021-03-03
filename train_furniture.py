import os, sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm
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
import wandb
import uuid

PROJECT = 'furniture_nodes_suppl'

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

    # Data
    parser.add_argument("--root_dir", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--datapath", default='.', type=str, help="Root folder to save data in")
    parser.add_argument('--dataset', default='rplan', type=str, help='dataset to train on')
    parser.add_argument('--lifull', default=False, type=bool, help='whether to train on lifull data')

    # Notes
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='', type=str, help="Wandb tags")

    args = parser.parse_args()


    if on_local():
        args.root_dir = './'
        # args.datapath = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '../furniture_018'

    else:  # assume IBEX
        args.root_dir = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '../furniture_018'

    dset = Furniture(root_dir=args.datapath,
        split='train',
        seq_len=args.enc_n,
        vocab_size=65,
    )

    val_set = Furniture(root_dir=args.datapath,
        split='val',
        seq_len=args.enc_n,
        vocab_size=65,
    )



    dloader = DataLoader(dset, batch_size=args.bs, num_workers=10, shuffle=, drop_last=True)



    val_loader = DataLoader(val_set, batch_size=args.bs, num_workers=10, shuffle=True, drop_last=True)
    args.passthrough=False


    enc = GPT2Config(
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


    model = GPT2Model(enc).cuda()

    model = DataParallel(model)

    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-6)



    run_id = "Furniture_GPT-{}-bs{}-lr{}-enl{}-dim_embed{}-{}".format(dt.now().strftime('%d-%h_%H-%M'),
                                                                      args.bs, args.lr, args.enc_layer,
                                                                      args.dim, uuid.uuid4())
    wandb.init(project=PROJECT, name=run_id, config=args, dir=".", save_code=True, notes=args.notes)
    wandb.watch(model)
    global_steps = 1
    val_steps = 1

    ## Basic logging
    SAVE_LOCATION = args.root_dir + f'models/furniture/' + run_id + '/'

    code_dir = SAVE_LOCATION + 'code'
    if not os.path.exists(SAVE_LOCATION):
        os.makedirs(code_dir)

    py_files = glob('./*.py')

    for code_file in py_files:
        shutil.copy(code_file, code_dir)

    argsdict = vars(args)
    args_file = SAVE_LOCATION  + 'args.json'

    with open(args_file, 'w') as fd:
        json.dump(argsdict, fd,
                  indent=4)

    best_nll = np.inf
    for epochs in range(args.epochs):
        model.train()
        for steps, data in tqdm(enumerate(dloader), total=len(dloader)):
            global_steps += 1
            optimizer.zero_grad()
            seq = data['seq']
            attn_mask = data['attn_mask']
            pos_id = data['pos_id']

            loss = model(input_ids=seq,
                         attention_mask=attn_mask,
                         position_ids=pos_id,
                         labels=seq)

            loss[0].mean().backward()

            optimizer.step()

            wandb.log({'loss/train': loss[0].mean()}, step=global_steps)

        torch.save(model.state_dict(), SAVE_LOCATION + f'model_furniture.pth')

        model.eval()
        val_step_size = (global_steps - val_steps) // len(val_loader)
        all_val_stats = []
        with torch.no_grad():
            for steps, data in tqdm(enumerate(val_loader)):
                seq = data['seq']
                attn_mask = data['attn_mask']
                pos_id = data['pos_id']

                loss = model(input_ids=seq,
                             attention_mask=attn_mask,
                             position_ids=pos_id,
                             labels=seq)

                all_val_stats.append(loss[0].mean().item())


            total_nll = np.mean(np.asarray(all_val_stats))
            wandb.log({'loss/val': total_nll}, step=global_steps)
            global_steps += 1

        if total_nll < best_nll:
            best_nll = total_nll
            torch.save(model.state_dict(), SAVE_LOCATION + f'model_furniture_best.pth')




