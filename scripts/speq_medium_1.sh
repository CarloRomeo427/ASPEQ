#!/bin/bash
for seed in 0 42 1234 5678; do
    for env in 'Hopper-v5' 'InvertedPendulum-v5' 'InvertedDoublePendulum-v5' 'Swimmer-v5' 'Reacher-v5'; do
        python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.001 --env $env --gpu-id 0 --seed $seed --minari-quality medium
    done  
done