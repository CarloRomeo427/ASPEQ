#!/bin/bash
for seed in 0 42 1234 5678 9876; do
    for env in HalfCheetah-v5 Ant-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo aspeq_o2o --gpu-id 0 --minari-quality expert
    done
done