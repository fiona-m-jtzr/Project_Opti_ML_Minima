#!/bin/bash

python csub.py \
    -n "train-muon-vit" \
    -g 1 \
    --train \
    --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
    --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
        --model vit \
        --optimizer muon \
        --lr 0.02 \
        --lr_muon_adam 5e-4 \
        --batch_size 512 \
        --beta1 0.9 \
        --beta2 0.999 \
        --weight_decay 0.0 \
        --momentum 0.95 \
        --epochs 600 \
        --scheduler cosine_warmup \
        --warmup_epochs 15 \
        --patience 300 \
        --augment"


for bs in 64 128 256; do
    python csub.py \
        -n "train-muon-resnet-bs${bs}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --model resnet20 \
            --optimizer muon \
            --lr 0.02 \
            --lr_muon_adamw 1e-3 \
            --batch_size $bs \
            --momentum 0.95 \
            --weight_decay 0.0 \
            --epochs 400 \
            --scheduler cosine \
            --patience 100"
done

