#!/bin/bash

for lr in 0.01 0.1; do
    for bs in 64 128 256; do
        python csub.py \
            -n "train-${optimizer}-lr${lr}-bs${bs}" \
            -g 1 \
            --train \
            --command "source /mloscratch/homes/jetzer/venv/bin/activate && \
                        cd /mloscratch/homes/jetzer/Project_Opti_ML_Minima && \
                        python RUN.py --optimizer sgd \
                        --lr $lr --batch_size $bs \
                        --epochs 200 --scheduler cosine --nesterov"
    done
done
