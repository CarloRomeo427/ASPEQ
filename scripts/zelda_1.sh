#!/bin/bash
for seed in 0 42 1234; do
    for env in Humanoid-v5 Ant-v5 HalfCheetah-v5 Hopper-v5 Walker2d-v5; do
        python main_faspeq_online.py --env $env --seed $seed --epochs 300 --log-wandb
    done
done