#!/bin/bash
for seed in 0 42 1234; do
    for env in HalfCheetah-v5; do
        python main.py --algo faspeq_exc_online --env $env --seed $seed --epochs 300 --log-wandb --val-patience 10000
    done
done