#!/bin/bash

for lr in 0.01 0.1; do
    for bs in 64 128 256; do
        for rho in 0.01 0.05 0.1 0.2; do
            lr_tag=$(echo $lr | tr '.' 'p')
            rho_tag=$(echo $rho | tr '.' 'p')
            python csub.py \
                -n "train-sam-lr${lr_tag}-bs${bs}-rho${rho_tag}" \
                -g 1 \
                --train \
                --command "source /mloscratch/homes/jetzer/.venv/bin/activate && \
                        cd /mloscratch/homes/jetzer/Project_Opti_ML_Minima && \
                        python RUN.py --optimizer sam \
                        --lr $lr --batch_size $bs \
                        --epochs 200 --scheduler cosine --momentum 0.9\
                        --base_optimizer sgd"
                
        done
    done
done