#!/bin/bash

for lr in 0.001 0.01; do
    for bs in 64 128 256; do
        for beta2 in 0.9 0.99 0.999 0.9999; do
            lr_tag=$(echo $lr | tr '.' 'p')  
            beta2_tag=$(echo $beta2 | tr '.' 'p') 
            python csub.py \
                -n "train-adam-lr${lr_tag}-bs${bs}-beta2${beta2_tag}" \
                -g 1 \
                --train \
                --venv /mloscratch/homes/jetzer/Project_Opti_ML_Minima/.venv \
                --command "python /mloscratch/homes/jetzer/Project_Opti_ML_Minima/RUN.py --optimizer adam \
                            --lr $lr --batch_size $bs \
                            --epochs 400 --scheduler cosine \
                            --beta1 0.9 --beta2 $beta2"
        done
    done
done