#!/bin/bash

for lr in 0.05 0.1; do
    for bs in 64, 128, 256; do
            python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
                --model resnet20 \
                --optimizer sgd \
                --lr $lr \
                --batch_size $bs \
                --momentum 0.9 \
                --nesterov \
                --weight_decay 0.0 \
                --epochs 600 \
                --scheduler cosine \
                --seed 1
    done
done


for lr in 0.01; do
    for bs in 64 128 256; do
        lr_tag=$(echo $lr | tr '.' 'p')
        python csub.py \
            -n "train-sgd-resnet-lr${lr_tag}-bs${bs}" \
            -g 1 \
            --train \
            --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
            --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
                --model resnet20 \
                --optimizer sgd \
                --lr $lr \
                --batch_size $bs \
                --momentum 0.9 \
                --nesterov \
                --weight_decay 0.0 \
                --epochs 600 \
                --scheduler cosine \
                --seed 1"
    done
done
