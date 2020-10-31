import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm
import argparse
from utils import on_local

from rplan import Rplan, Flip, Rot90
from gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
import wandb
import json
import uuid
import uuid, shutil
from glob import glob
from datetime import datetime as dt

PROJECT = 'Triples_hw'


if __name__ == '__main__':
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
    parser.add_argument('--step', default=15, type=int, help='how many epochs before reducing lr')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='reduction in lr')
    parser.add_argument('--bs', default=64, type=int, help='batch size')

    # Data
    parser.add_argument("--root_dir", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--datapath", default='.', type=str, help="Root folder to save data in")
    parser.add_argument('--wh', default=False, type=bool, help='Enable id,w,h as triples dataset')

    # Notes
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='', type=str, help="Wandb tags")

    args = parser.parse_args()

    if on_local():
        args.root_dir = './'
        args.datapath = '/mnt/iscratch/datasets/rplan_ddg_var'

    else:  # assume IBEX
        args.root_dir = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '/ibex/scratch/parawr/datasets/rplan_ddg_var'

    from random import choice
    # args.lr = choice([0.001, 0.0005, 0.0007])

    dset = Rplan(root_dir=args.datapath,
                 split='train',
                 seq_len=120,
                 vocab_size=65,
                 drop_dim=True,
                 wh=args.wh)

    dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=True)

    val_set = Rplan(root_dir=args.datapath,
                 split='val',
                 seq_len=120,
                 vocab_size=65,
                drop_dim=True,
                    wh=args.wh)

    val_loader = DataLoader(val_set, batch_size=64, num_workers=10)

    config = GPT2Config(
        vocab_size=65,
        n_positions=120,
        n_ctx=120,
        n_embd=args.dim,
        n_layer=args.enc_layer,
        n_head=12,
        is_causal=True,
        is_encoder=False,
        n_types=3
    )
    model = GPT2Model(config)

    model = DataParallel(model.cuda())

    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    run_id = "GraphGPTxy-{}-bs{}-lr{}-enl{}-decl{}-dim_embed{}-{}".format(dt.now().strftime('%d-%h_%H-%M'),
                                                                        args.bs, args.lr, args.enc_layer,
                                                                        args.dec_layer,
                                                                        args.dim, uuid.uuid4())
    wandb.init(project=PROJECT, name=run_id, config=args, dir=".", save_code=True, notes=args.notes)
    wandb.watch(model)

    global_steps = 1
    val_steps = 1

    if args.wh:
        save_suffix = 'wh'
    else:
        save_suffix = 'xy'
    SAVE_LOCATION = args.root_dir + f'models/triples_{save_suffix}/' + run_id + '/'

    code_dir = SAVE_LOCATION + 'code'
    if not os.path.exists(SAVE_LOCATION):
        os.makedirs(code_dir)

    py_files = glob('./*.py')

    for code_file in py_files:
        shutil.copy(code_file, code_dir)

    argsdict = vars(args)
    args_file = SAVE_LOCATION + 'args.json'

    with open(args_file, 'w') as fd:
        json.dump(argsdict, fd,
                  indent=4)

    best_nll = np.inf
    for epochs in range(args.epochs):
        model.train()
        for steps, data in tqdm(enumerate(dloader)):
            global_steps += 1
            optimizer.zero_grad()
            seq = data['seq']
            attn_mask = data['attn_mask']
            pos_id = data['pos_id']

            loss = model( input_ids=seq,
                          attention_mask=attn_mask,
                          position_ids=pos_id,
                          labels=seq)

            # print(len(loss))
            # for v in loss:
            #     if isinstance(v, torch.Tensor):
            #         print(v.shape)
            #     else:
            #         for vv in v:
            #             print('\t', vv.shape)
            loss[0].mean().backward()

            optimizer.step()

            wandb.log({'loss/train': loss[0].mean()}, step=global_steps)

        torch.save(model.state_dict(), SAVE_LOCATION+ f'triples_hw3.pth')

        lr_scheduler.step()
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

                # writer.add_scalar('loss/val', loss[0].mean(), global_step=global_steps)
                # val_steps += val_step_size
            total_nll = np.mean(np.asarray(all_val_stats))
            wandb.log({'loss/val': total_nll}, step=global_steps)
            global_steps += 1

        if best_nll >= total_nll:
            torch.save(model.state_dict(), SAVE_LOCATION+ f'triples_hw3_best.pth')

    # writer.close()



