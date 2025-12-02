#!/bin/bash
for seed in 1234; do
    for env in Ant-v5 Humanoid-v5 Hopper-v5 Walker2d-v5 HalfCheetah-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo aspeq --gpu-id 1
done