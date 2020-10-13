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

from rplan import Rplan, Flip, Rot90
from gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config

if __name__ == '__main__':

    dset = Rplan(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='train',
                 seq_len=200,
                 vocab_size=65)

    dloader = DataLoader(dset, batch_size=64, num_workers=10, shuffle=True)

    val_set = Rplan(root_dir='/mnt/iscratch/datasets/rplan_ddg_var',
                 split='val',
                 seq_len=200,
                 vocab_size=65)

    val_loader = DataLoader(val_set, batch_size=64, num_workers=10)

    config = GPT2Config(
        vocab_size=65,
        n_positions=200,
        n_ctx=200,
        n_embd=264,
        n_layer=12,
        n_head=12,
        is_causal=False,
        is_encoder=True
    )
    model = GPT2Model(config)

    model = DataParallel(model.cuda())

    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    writer = SummaryWriter()

    global_steps = 1
    for epochs in range(40):
        model.train()
        for steps, data in tqdm(enumerate(dloader)):
            global_steps += 1
            optimizer.zero_grad()
            seq = data['seq']
            attn_mask = data['attn_mask']
            pos_id = data['pos_id']

            loss = model( input_ids=seq,
                          attention_mask=attn_mask,
                          position_ids=pos_id)#,
                          # labels=seq)

            print(len(loss))
            for v in loss:
                if isinstance(v, torch.Tensor):
                    print(v.shape)
                else:
                    for vv in v:
                        print('\t', vv.shape)
            loss[0].mean().backward()

            optimizer.step()

            if steps % 100 == 0:
                writer.add_scalar('loss/train', loss[0].mean(), global_step=global_steps)

        torch.save(model.state_dict(), f'demoaug_logged_{epochs}.pth')

        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            for steps, data in tqdm(enumerate(val_loader)):
                seq = data['seq']
                attn_mask = data['attn_mask']
                pos_id = data['pos_id']

                loss = model(input_ids=seq,
                             attention_mask=attn_mask,
                             position_ids=pos_id,
                             labels=seq)


                writer.add_scalar('loss/val', loss[0].mean(), global_step=global_steps)


    writer.close()



