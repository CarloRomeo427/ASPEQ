#!/bin/bash
for seed in 0; do
    for env in HalfCheetah-v5; do
        python main_unified_flops.py --env $env --seed $seed --log-wandb --algo calql \
            --gpu-id 0 --use-offline-data --dataset-quality expert --offline-pretrain-steps 1000000
    done
done