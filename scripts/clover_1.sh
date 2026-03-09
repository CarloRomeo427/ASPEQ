#!/bin/bash
for seed in 0; do
    for env in Humanoid-v5 HalfCheetah-v5 Ant-v5 Hopper-v5 Walker2d-v5 InvertedPendulum-v5 InvertedDoublePendulum-v5 Pusher-v5 Reacher-v5 Swimmer-v5; do
        python main.py --env $env --seed $seed --log-wandb 
    done
done
