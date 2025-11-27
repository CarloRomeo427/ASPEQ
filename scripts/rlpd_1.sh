#!/bin/bash
for env in 'Hopper-v5' 'Humanoid-v5' 'HalfCheetah-v5' 'Walker2d-v5' 'Ant-v5' 'Pusher-v5' 'InvertedPendulum-v5' 'InvertedDoublePendulum-v5' 'Swimmer-v5' 'Reacher-v5'; do
    for seed in 0 42; do
        python main_o2o.py --use-minari --offline-epochs 1 --offline-frequency 0 --utd 20 --target-drop-rate 0 \
        --num-q 10 --gpu-id 0 --log-wandb --exp-name rlpd --seed $seed --env $env
    done  
done