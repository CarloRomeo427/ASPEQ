#!/bin/bash
for seed in 1234 5678; do
    for env in 'Hopper-v5' 'Humanoid-v5' 'Ant-v5' 'Walker2d-v5' 'HalfCheetah-v5' 'Pusher-v5' 'InvertedPendulum-v5' 'InvertedDoublePendulum-v5' 'Swimmer-v5' 'Reacher-v5'; do
        python main_o2o_auto.py --env $env --gpu-id 0 --seed $seed --exp-name speq_online_auto --log-wandb --calc-plasticity
    done  
done