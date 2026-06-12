#!/bin/bash

for rho in 0.01 0.05 0.1; do
    python RUN.py \
    --model vit \
    --optimizer sam \
    --base_optimizer adam \
    --lr 5e-4 \
    --rho $rho \
    --beta1 0.9 \
    --beta2 0.999 \
    --weight_decay 0.0 \
    --batch_size 256 \
    --epochs 600 \
    --scheduler cosine_warmup \
    --warmup_epochs 15 \
    --seed 1
done

for rho in 0.01 0.05 0.1; do
    python RUN.py \
    --model resnet20 \
    --optimizer sam \
    --base_optimizer sgd \
    --lr 0.1 \
    --rho $rho \
    --momentum 0.9 \
    --nesterov \
    --weight_decay 0.0 \
    --batch_size 128 \
    --epochs 400 \
    --scheduler cosine \
    --seed 1
done