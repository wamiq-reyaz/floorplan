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
from rplan import RrplanGraph, Flip, Rot90,RplanConstrainedGraph
from gpt2 import ConstrGPTModel
from transformers.configuration_gpt2 import GPT2Config
import shutil
from glob import glob
import argparse
from utils import on_local
import json
import wandb
import uuid

PROJECT = 'adj_constr_v2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')
    # Model
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--dim', default=264, type=int, help='number of dims of transformer')
    parser.add_argument('--vocab', default=65, type=int, help='quantization levels')
    parser.add_argument('--tuples', default=3, type=int, help='3 or 5 based on initial sampler')
    parser.add_argument('--constr_n', default=70, type=int, help='length of the constraint sequences')
    parser.add_argument('--constr_layer', default=3, type=int, help='layers of the constraint encoder')
    parser.add_argument('--enc_n', default=80, type=int, help='number of encoder tokens')
    parser.add_argument('--enc_layer', default=12, type=int, help='number of encoder layers')
    parser.add_argument('--dec_n', default=220, type=int, help='number of decoder tokens')
    parser.add_argument('--dec_layer', default=12, type=int, help='number of decoder layers')
    parser.add_argument('--pos_id', default=True, type=bool, help='Whether to use pos_id in encoder')
    parser.add_argument('--id_embed', default=False, type=int, help='Separate embedding for the id')
    parser.add_argument('--adj', default='h', type=str, help='h/v/all doors')
    parser.add_argument('--passthrough', default=False, type=bool, help='Whether to have transfoermer layers in encoder')
    parser.add_argument('--wh', default=False, type=bool, help='Whether to have transfoermer layers in encoder')
    parser.add_argument('--flipped', default=False, type=bool, help='Whether the decoder/encoder are flipped')


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
    args.lifull=False

    if args.lifull:
        args.dataset = 'lifull'
        args.dec_n = 220

    if args.dataset == 'rplan':
        if on_local():
            args.root_dir = './'
            args.datapath = '/mnt/iscratch/datasets/rplan_ddg_var'

        else:  # assume IBEX
            args.root_dir = '/ibex/scratch/parawr/floorplan/'
            args.datapath = '/ibex/scratch/parawr/datasets/rplan_ddg_var'

        dset = RplanConstrainedGraph(root_dir=args.datapath,
                     split='train',
                     constr_len=args.constr_n,
                     enc_len=args.enc_n,
                     dec_len=args.dec_n,
                     drop_dim=args.tuples == 3,
                     vocab_size=args.vocab,
                     edg_type=args.adj,
                     wh=args.wh,
                    )
        val_set = RplanConstrainedGraph(root_dir=args.datapath,
                              split='val',
                              constr_len=args.constr_n,
                              enc_len=args.enc_n,
                              dec_len=args.dec_n,
                              vocab_size=args.vocab,
                              edg_type=args.adj,
                              drop_dim=args.tuples == 3,
                              wh=args.wh
                                        )

    elif args.dataset == 'lifull':
        if on_local():
            args.root_dir = './'
            args.datapath = '/mnt/iscratch/datasets/lifull_ddg_var'

        else:  # assume IBEX
            args.root_dir = '/ibex/scratch/parawr/floorplan/'
            args.datapath = '/ibex/scratch/parawr/datasets/lifull_ddg_var'

        dset = RplanConstrainedGraph(root_dir=args.datapath,
                     split='train',
                     constr_len=args.constr_n,
                     enc_len=args.enc_n,
                     dec_len=args.dec_n,
                     drop_dim=args.tuples == 3,
                     vocab_size=args.vocab,
                     edg_type=args.adj,
                     wh=args.wh,
                     lifull=True
                    )

        val_set = RplanConstrainedGraph(root_dir=args.datapath,
                     split='val',
                     constr_len=args.constr_n,
                     enc_len=args.enc_n,
                     dec_len=args.dec_n,
                     drop_dim=args.tuples == 3,
                     vocab_size=args.vocab,
                     edg_type=args.adj,
                     wh=args.wh,
                     lifull=True
                    )

    dloader = DataLoader(dset, batch_size=args.bs, num_workers=10, shuffle=True)



    val_loader = DataLoader(val_set, batch_size=args.bs, num_workers=10, shuffle=True)


    constr = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.constr_n,
        n_ctx=args.constr_n,
        n_embd=args.dim,
        n_layer=args.constr_layer,
        n_head=12,
        is_causal=False,
        is_encoder=True,
        passthrough=False,
        id_embed=args.id_embed,
        use_pos_emb=True,
        n_types=2,
        separate=False
    )

    enc = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.enc_n,
        n_ctx=args.enc_n,
        n_embd=args.dim,
        n_layer=args.enc_layer,
        n_head=12,
        is_causal=False,
        is_encoder=True,
        passthrough=args.passthrough,
        id_embed=args.id_embed,
        use_pos_emb=args.pos_id,
        n_types=args.tuples,
        separate=True
    )

    dec = GPT2Config(
        vocab_size=args.vocab,
        n_positions=args.dec_n,
        n_ctx=args.dec_n,
        n_embd=args.dim,
        n_layer=args.dec_layer,
        n_head=12,
        is_causal=True,
        is_encoder=False,
        n_types=args.tuples
    )

    model = ConstrGPTModel(constr, enc, dec)

    model = DataParallel(model.cuda())

    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-6)

    run_id = "GraphGPT-{}-bs{}-lr{}-enl{}-decl{}-dim_embed{}-{}".format(dt.now().strftime('%d-%h_%H-%M'),
                                                                      args.bs, args.lr, args.enc_layer, args.dec_layer,
                                                                      args.dim, uuid.uuid4())
    wandb.init(project=PROJECT, name=run_id, config=args, dir=".", save_code=True, notes=args.notes)
    wandb.watch(model)
    global_steps = 1
    val_steps = 1

    ## Basic logging
    SAVE_LOCATION = args.root_dir + f'models/{args.dataset}_{args.tuples}_constrained_{args.adj}/' + run_id + '/'

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
        for steps, data in tqdm(enumerate(dloader)):
            global_steps += 1
            optimizer.zero_grad()


            constr_seq = data['constr_seq'].cuda()
            enc_seq = data['enc_seq'].cuda()
            dec_seq = data['dec_seq'].cuda()
            constr_attn = data['constr_attn'].cuda()
            enc_attn = data['enc_attn'].cuda()
            dec_attn = data['dec_attn'].cuda()

            # print(dec_seq)

            loss = model( constr_seq=constr_seq,
                        enc_seq=enc_seq,
                       dec_seq=dec_seq,
                       constr_attn_mask=constr_attn,
                       enc_attn_mask=enc_attn,
                       dec_attn_mask=dec_attn,
                       labels=dec_seq)

            loss[0].mean().backward()


            optimizer.step()
            # warmup_sched.step()

            # if steps % 100 == 0:
            # writer.add_scalar('loss/train', loss[0].mean(), global_step=global_steps)
            wandb.log({'loss/train': loss[0].mean()}, step=global_steps)

        torch.save(model.state_dict(), SAVE_LOCATION + f'model_constr_adj.pth')

        model.eval()
        val_step_size = (global_steps - val_steps) // len(val_loader)
        all_val_stats = []
        with torch.no_grad():
            for steps, data in tqdm(enumerate(val_loader)):
                constr_seq = data['constr_seq'].cuda()
                enc_seq = data['enc_seq'].cuda()
                dec_seq = data['dec_seq'].cuda()
                constr_attn = data['constr_attn'].cuda()
                enc_attn = data['enc_attn'].cuda()
                dec_attn = data['dec_attn'].cuda()

                # print(dec_seq)
                loss = model(constr_seq=constr_seq,
                             enc_seq=enc_seq,
                             dec_seq=dec_seq,
                             constr_attn_mask=constr_attn,
                             enc_attn_mask=enc_attn,
                             dec_attn_mask=dec_attn,
                             labels=dec_seq)

                all_val_stats.append(loss[0].mean().item())


            total_nll = np.mean(np.asarray(all_val_stats))
            wandb.log({'loss/val': total_nll}, step=global_steps)
            global_steps += 1

        if total_nll < best_nll:
            best_nll = total_nll
            torch.save(model.state_dict(), SAVE_LOCATION + f'model_constr_adj_best.pth')




