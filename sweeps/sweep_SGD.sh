#!/bin/bash

for lr in 0.01 0.1; do
    for bs in 64 128 256; do
        lr_tag=$(echo $lr | tr '.' 'p')
        python csub.py \
            -n "train-sgd-lr${lr_tag}-bs${bs}" \
            -g 1 \
            --train \
            --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
            --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
                    --optimizer sgd \
                    --lr $lr --batch_size $bs \
                    --epochs 200 --scheduler cosine"
    done
done
