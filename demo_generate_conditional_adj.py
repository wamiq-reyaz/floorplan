import os, sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from rplan import RplanConditional
from gpt2 import EncDecGPTModel, GraphGPTModel
from utils import make_rgb_indices, rplan_map

from transformers.configuration_gpt2 import GPT2Config
from easydict import EasyDict as ED
import pickle
from datetime import datetime
from utils import parse_wall_or_door_seq, parse_vert_seq, \
    top_k_top_p_filtering, on_local, parse_edge_seq




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')

    parser = argparse.ArgumentParser(description='Model corrector', conflict_handler='resolve')
    # Model
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--dim', default=264, type=int, help='number of dims of transformer')
    parser.add_argument('--vocab', default=65, type=int, help='quantization levels')
    parser.add_argument('--tuples', default=5, type=int, help='3 or 5 based on initial sampler')
    parser.add_argument('--enc_n', default=120, type=int, help='number of encoder tokens')
    parser.add_argument('--enc_layer', default=12, type=int, help='number of encoder layers')
    parser.add_argument('--dec_n', default=100, type=int, help='number of decoder tokens')
    parser.add_argument('--dec_layer', default=12, type=int, help='number of decoder layers')
    parser.add_argument('--pos_id', default=True, type=bool, help='Whether to use pos_id in encoder')
    parser.add_argument('--id_embed', default=False, type=int, help='Separate embedding for the id')
    parser.add_argument('--adj', default='h', type=str, help='h/v/all doors')
    parser.add_argument('--passthrough', default=False, type=bool,
                        help='Whether to have transfoermer layers in encoder')
    parser.add_argument('--wh', default=False, type=bool, help='Whether to have transfoermer layers in encoder')
    parser.add_argument('--flipped', default=False, type=bool, help='Whether the decoder/encoder are flipped')

    # optimizer
    parser.add_argument('--step', default=25, type=int, help='how many epochs before reducing lr')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--gamma', default=0.1, type=float, help='reduction in lr')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
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
        args.datapath = '/mnt/iscratch/datasets/rplan_ddg_var'

    else:  # assume IBEX
        args.root_dir = '/ibex/scratch/parawr/floorplan/'
        args.datapath = '/ibex/scratch/parawr/datasets/rplan_ddg_var'

    # BATCH_SIZE = args.bs
    BATCH_SIZE = 1

    from natsort import natsorted
    from glob import glob

    boxes = glob(os.path.join(args.input_dir, '*.npz'))
    # exteriors = glob(os.path.join(args.input_dir, 'exterior*'))

    boxes = natsorted(boxes)
    # exteriors = natsorted(exteriors)

    class dataset(torch.utils.data.Dataset):
        def __init__(self,
                     boxes,
                     seq_len=args.enc_n,
                     edg_len=args.dec_n):
            super().__init__()

            self.boxes = boxes

            self.seq_len = seq_len
            self.edg_len = edg_len


        def __len__(self):
            return len(boxes)


        def __getitem__(self, idx):
            # print(type(self.boxes), type(self.boxes[idx]))
            with open(self.boxes[idx], 'rb') as fd:
                boxes = np.load(fd)['arr_0']

            name = os.path.basename(self.boxes[idx])
            base_name = os.path.splitext(name)[0]
            length = boxes.shape[0]

            vert_seq = np.ones((self.seq_len, 5), dtype=np.uint8) * (66)
            vert_seq[0, :] = (0,) * 5
            vert_seq[2:length + 2, :] = boxes + 1

            vert_attn_mask = np.zeros(self.seq_len)
            vert_attn_mask[:length + 2] = 1

            return {'vert_seq': torch.tensor(vert_seq).long(),
                    'vert_attn_mask': torch.tensor(vert_attn_mask),
                    'base_name': base_name}


    val_set = dataset(boxes=boxes)
    aa =val_set[0]

    dloader = DataLoader(val_set, batch_size=args.bs, num_workers=10)

    # sys.exit()

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
        passthrough=args.passthrough,
        id_embed=args.id_embed,
        pos_id=args.pos_id,
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
        is_causal=False,
        is_encoder=False,
        n_types=args.tuples
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

    for jj, data in tqdm(enumerate(dloader)):
        vert_seq = data['vert_seq'].cuda()
        vert_attn_mask = data['vert_attn_mask'].cuda()
        names = data['base_name']

        # print(vert_seq)
        # print(vert_attn_mask)

        bs = vert_seq.shape[0]

        input_ids = torch.zeros(args.dec_n, dtype=torch.long).to(device=device).unsqueeze(0).repeat(bs, 1)
        attn_mask = torch.zeros(args.dec_n, dtype=torch.float, device=device).unsqueeze(0).repeat(bs, 1)
        for ii in range(args.dec_n - 1):
            attn_mask[:, :ii+1] = 1
            with torch.no_grad():
                loss = model(node=vert_seq,
                             edg=input_ids,
                             attention_mask=attn_mask,
                             labels=input_ids,
                             vert_attn_mask=vert_attn_mask)

                logits = top_k_top_p_filtering(loss[1][:, ii, :], top_p=0.9)
                probs = torch.softmax(logits.squeeze(), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids[:, ii+1] = next_token.squeeze()

        input_ids = input_ids.cpu().numpy().squeeze()[:, 1:]  # drop 0
        samples = [input_ids[ii, :]  for ii in range(bs)]

        #
        # print_idx = 0
        # print(vert_seq[print_idx], vert_attn_mask[print_idx])
        # print(samples[print_idx])
        # print(parse_wall_or_door_seq(samples[print_idx]))
        # print(names[0], names[1])
        # sys.exit()

        for kk, curr_sample in enumerate(samples):
            root_name = names[kk]
            save_path = os.path.join('cond_lifull3_v2', 'doors_0.9', root_name + '.pkl')
            with open(save_path, 'wb') as fd:
                pickle.dump(parse_wall_or_door_seq(curr_sample), fd, protocol=pickle.HIGHEST_PROTOCOL)

