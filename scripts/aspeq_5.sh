#!/bin/bash
for seed in 0 42 1234 5678; do
    for env in Pusher-v5 Reacher-v5 Swimmer-v5 InvertedDoublePendulum-v5 InvertedPendulum-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo aspeq --gpu-id 0
    done
done