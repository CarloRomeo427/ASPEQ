#!/bin/bash
for seed in 0 42 1234; do
    for env in Swimmer-v5 Reacher-v5 Pusher-v5 InvertedPendulum-v5 InvertedDoublePendulum-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --calc-plasticity --algo paspeq --gpu-id 1
    done
done