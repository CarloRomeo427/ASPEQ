#!/bin/bash
for seed in 42; do
    for env in Humanoid-v5 Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
        for batch in 0.2; do
            for patience in 5000; do
                python main.py --algo faspeq_pct --env $env --use-offline-data --val-pct $batch \
                    --dataset-quality expert --log-wandb --seed $seed --faspeq-pct-use-td --val-patience $patience --gpu-id 0
            done
        done
    done
done