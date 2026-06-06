#!/bin/bash

for rho in 0.01 0.05 0.1; do
    rho_tag=$(echo $rho | tr '.' 'p')
    python csub.py \
        -n "train-sam-adam-vit-rho${rho_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --model vit \
            --optimizer sam \
            --base_optimizer adam \
            --lr 5e-4 \
            --rho $rho \
            --beta1 0.9 \
            --beta2 0.999 \
            --weight_decay 0.0 \
            --batch_size 256 \
            --epochs 400 \
            --scheduler cosine_warmup \
            --warmup_epochs 15 \
            --patience 400 \
            --augment"
done


for rho in 0.01 0.05 0.1; do
    rho_tag=$(echo $rho | tr '.' 'p')
    python csub.py \
        -n "train-sam-sgd-resnet-rho${rho_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --model resnet20 \
            --optimizer sam \
            --base_optimizer sgd \
            --lr 0.1 \
            --rho $rho \
            --momentum 0.9 \
            --nesterov \
            --weight_decay 0.0 \
            --batch_size 128 \
            --epochs 200 \
            --scheduler cosine_warmup \
            --warmup_epochs 5 \
            --patience 200 \
            --augment"
done