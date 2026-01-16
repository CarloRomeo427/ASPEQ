#!/bin/bash
for seed in 0 42 1234; do
    for env in Humanoid-v5; do
        python main_a3speq.py --env $env --seed $seed --log-wandb --use-minari
    done
done
