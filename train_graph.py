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

from rplan import RrplanGraph, Flip, Rot90
from gpt2 import GraphGPTModel
from transformers.configuration_gpt2 import GPT2Config

if __name__ == '__main__':

    dset = RrplanGraph(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='train',
                 seq_len=120,
                 edg_len=100,
                 vocab_size=65)

    dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=True)

    val_set = RrplanGraph(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='val',
                 seq_len=120,
                 edg_len=100,
                 vocab_size=65)

    val_loader = DataLoader(val_set, batch_size=64, num_workers=10, shuffle=True)

    enc = GPT2Config(
        vocab_size=65,
        n_positions=120,
        n_ctx=120,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=False,
        is_encoder=True,
        id_embed=True
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

    model = DataParallel(model.cuda())

    optimizer = Adam(model.parameters(), lr=1e-4, eps=1e-6)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    writer = SummaryWriter(comment='id_but_otherwise_same_as_12_model')

    global_steps = 1
    val_steps = 1
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

        torch.save(model.state_dict(), f'id_embed_12_modelv_eps_m6_mlp_lr_m4_{epochs}.pth')


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

'triples_hw3_best.pth'

