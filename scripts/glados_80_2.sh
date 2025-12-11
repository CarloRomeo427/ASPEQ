#!/bin/bash
for seed in 5678 9876; do
    for env in Ant-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --calc-plasticity --algo paspeq --gpu-id 0
    done
done