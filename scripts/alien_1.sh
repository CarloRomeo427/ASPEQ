#!/bin/bash
for seed in 0; do
    for env in Ant-v5 HalfCheetah-v5 Walker2d-v5; do
        for batch in 0.1 0.2; do
        python main.py --algo faspeq_pct --env $env --use-offline-data --val-pct $batch --dataset-quality expert --log-wandb
        done
    done
done