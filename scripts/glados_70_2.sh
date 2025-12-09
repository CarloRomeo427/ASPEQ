#!/bin/bash
for seed in 0 42 1234 5678 9876; do
    for env in Reacher-v5 Swimmer-v5 Pusher-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo aspeq_o2o --gpu-id 1 --minari-quality expert
    done
done