#!/bin/bash

for beta2 in 0.99 0.999 0.9999; do
    beta2_tag=$(echo $beta2 | tr '.' 'p')
    python csub.py \
        -n "train-adam-vit-${beta2_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --model vit \
            --optimizer adam \
            --lr 5e-4 \
            --batch_size 256 \
            --beta1 0.9 \
            --beta2 $beta2 \
            --weight_decay 0.0 \
            --epochs 400 \
            --scheduler cosine_warmup \
            --warmup_epochs 15 \
            --patience 400 \
            --augment"
done


for beta2 in 0.99 0.999 0.9999; do
    beta2_tag=$(echo $beta2 | tr '.' 'p')
    python csub.py \
        -n "train-adam-resnet-beta2${beta2_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --model resnet20 \
            --optimizer adam \
            --lr 1e-3 \
            --batch_size 128 \
            --beta1 0.9 \
            --beta2 $beta2 \
            --weight_decay 0.0 \
            --epochs 200 \
            --scheduler cosine_warmup \
            --warmup_epochs 5 \
            --patience 200 \
            --augment"
done