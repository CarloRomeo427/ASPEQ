#!/bin/bash
for seed in 0 42 1234; do
    for env in Ant-v5 HalfCheetah-v5; do
        for batch in 0.1 0.2; do
        python main.py --algo faspeq_pct --env $env --use-offline-data --val-pct $batch --dataset-quality medium \
        --log-wandb --seed $seed --gpu-id 0
        done
    done
done