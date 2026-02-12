#!/bin/bash
for seed in 0; do
    for env in HalfCheetah-v5; do
        python main_unified_flops.py --env $env --seed $seed --log-wandb --algo sacfd \
            --gpu-id 1 --use-offline-data --dataset-quality expert
    done
done