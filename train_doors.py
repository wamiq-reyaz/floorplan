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

from rplan import RrplanGraph, Flip, Rot90,RrplanDoors
from gpt2 import GraphGPTModel
from transformers.configuration_gpt2 import GPT2Config
import shutil
from glob import glob

if __name__ == '__main__':

    dset = RrplanDoors(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='train',
                 seq_len=120,
                 edg_len=48,
                 vocab_size=65,
                 dims=5,
                 doors='all')

    dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=True)

    val_set = RrplanDoors(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='val',
                 seq_len=120,
                 edg_len=48,
                 vocab_size=65,
                 dims=5,
                 doors='all')

    val_loader = DataLoader(val_set, batch_size=64, num_workers=10, shuffle=True)

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
        n_positions=48,
        n_ctx=48,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=True,
        is_encoder=False
    )

    # print()
    # sys.exit()
    model = GraphGPTModel(enc, dec)

    model = DataParallel(model.cuda())

    optimizer = Adam(model.parameters(), lr=1e-4, eps=1e-6)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    writer = SummaryWriter(comment='intial_door_model_5_tuple')

    global_steps = 1
    val_steps = 1

    ## Basic logging
    SAVE_LOCATION = f'./models/doors/'

    code_dir = SAVE_LOCATION + 'code'
    if not os.path.exists(SAVE_LOCATION):
        os.makedirs(code_dir)

    py_files = glob('./*.py')

    for code_file in py_files:
        shutil.copy(code_file, code_dir)


    for epochs in range(40):
        model.train()
        for steps, data in tqdm(enumerate(dloader)):
            global_steps += 1
            optimizer.zero_grad()
            vert_seq = data['vert_seq'].cuda()
            edg_seq = data['edg_seq'].cuda()
            attn_mask = data['attn_mask'].cuda()
            pos_id = data['pos_id'].cuda()
            vert_attn_mask = data['vert_attn_mask'].cuda()
            # print(vert_seq.shape)

            loss = model( node=vert_seq,
                          edg=edg_seq,
                          attention_mask=attn_mask,
                          labels=edg_seq,
                          vert_attn_mask=vert_attn_mask)

            # print(len(loss))
            # for v in loss:
            #     if isinstance(v, torch.Tensor):
            #         print(v.shape)
            #     else:
            #         for vv in v:
            #             print('\t', vv.shape)
            # print(loss[1])
            loss[0].mean().backward()

            optimizer.step()

            # if steps % 100 == 0:
            writer.add_scalar('loss/train', loss[0].mean(), global_step=global_steps)

        SAVE_LOCATION = f'./models/doors/'
        torch.save(model.state_dict(), SAVE_LOCATION + f'model_doors_eps_m6_mlp_lr_m4_{epochs}.pth')

        code_dir = SAVE_LOCATION + 'code'
        if not os.path.exists(SAVE_LOCATION):
            os.makedirs(code_dir)

        py_files = glob()




        # torch.save(model.state_dict(), f'face_modelv_eps_m6_mlp_lr_m4_{epochs}.pth')

        lr_scheduler.step()
        model.eval()
        val_step_size = (global_steps - val_steps) // len(val_loader)
        with torch.no_grad():
            for steps, data in tqdm(enumerate(val_loader)):
                vert_seq = data['vert_seq'].cuda()
                edg_seq = data['edg_seq'].cuda()
                attn_mask = data['attn_mask'].cuda()
                pos_id = data['pos_id'].cuda()
                vert_attn_mask = data['vert_attn_mask'].cuda()

                loss = model( node=vert_seq,
                          edg=edg_seq,
                          attention_mask=attn_mask,
                          labels=edg_seq,
                              vert_attn_mask=vert_attn_mask)


                writer.add_scalar('loss/val', loss[0].mean(), global_step=val_steps)
                val_steps += val_step_size


    writer.close()



