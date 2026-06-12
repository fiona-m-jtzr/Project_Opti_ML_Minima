#!/bin/bash

for lr in 0.01 0.05 0.1; do
    for bs in 64, 128, 256; do
                python RUN.py \
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


for lr in 0.01 0.05 0.1; do
    for bs in 64 128 256; do
                python RUN.py \
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
