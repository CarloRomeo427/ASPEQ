#!/bin/bash
for seed in 0 42 1234; do
    for env in Pusher-v5 Swimmer-v5 Reacher-v5 InvertedPendulum-v5 InvertedDoublePendulum-v5; do
        python main_faspeq_online.py --env $env --seed $seed --epochs 300 --log-wandb
    done
done