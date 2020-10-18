#!/bin/bash --login

python train_doors.py \
        --epochs 40 \
        --dim 264 \
        --seq_len 120 \
        --edg_len 48 \
        --vocab 65 \
        --tuples 5\
        --doors 'all' \
        --enc_n 120
