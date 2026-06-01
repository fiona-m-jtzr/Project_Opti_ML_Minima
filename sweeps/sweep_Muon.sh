#!/bin/bash

python csub.py \
    -n "train-muon-lr0.02-bs512" \
    -g 1 \
    --train \
    --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
    --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
        --optimizer muon \
        --batch_size 512 \
        --patience 50 \
        --lr 0.02 \
        --momentum 0.95 \
        --weight_decay 0.0 \
        --eps 1e-10 \
        --beta1 0.9 \
        --beta2 0.95 \
        --epochs 500 \
        --scheduler none"
