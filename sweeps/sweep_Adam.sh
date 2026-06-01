#!/bin/bash

for beta2 in 0.9 0.99 0.999 0.9999; do
    beta2_tag=$(echo $beta2 | tr '.' 'p')
    python csub.py \
        -n "train-adam-beta2${beta2_tag}" \
        -g 1 \
        --train \
        --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
        --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py \
            --optimizer adam \
            --lr 0.001 \
            --batch_size 128 \
            --beta1 0.9 \
            --beta2 $beta2 \
            --weight_decay 1e-4 \
            --epochs 400 \
            --scheduler cosine \
            --patience 50"
done