#!/bin/bash
for seed in 0 42 1234 5678 9876; do
    for env in Hopper-v5 InvertedPendulum-v5 InvertedDoublePendulum-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo aspeq_o2o --gpu-id 1 --minari-quality expert
    done
done