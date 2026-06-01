#!/bin/bash

for rho in 0.01 0.05 0.1 0.2; do
    rho_tag=$(echo $rho | tr '.' 'p')
    python csub.py \
        -n "train-sam-rho${rho_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --optimizer sam \
            --base_optimizer sgd \
            --lr 0.1 \
            --rho $rho \
            --momentum 0.9 \
            --nesterov \
            --weight_decay 1e-4 \
            --batch_size 128 \
            --epochs 400 \
            --scheduler cosine \
            --patience 50"
done