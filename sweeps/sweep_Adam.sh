#!/bin/bash

for beta2 in 0.99 0.999 0.9999; do
    python RUN.py \
    --model vit \
    --optimizer adam \
    --lr 5e-4 \
    --batch_size 256 \
    --beta1 0.9 \
    --beta2 $beta2 \
    --weight_decay 0.0 \
    --epochs 600 \
    --scheduler cosine_warmup \
    --warmup_epochs 15 \
    --seed 1
done

for beta2 in 0.99 0.999 0.9999; do
    python RUN.py \
    --model resnet20 \
    --optimizer adam \
    --lr 1e-3 \
    --batch_size 128 \
    --beta1 0.9 \
    --beta2 $beta2 \
    --weight_decay 0.0 \
    --epochs 400 \
    --scheduler cosine \
    --seed 1
done