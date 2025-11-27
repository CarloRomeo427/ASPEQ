#!/bin/bash
for env in 'Hopper-v5' 'InvertedPendulum-v5' 'InvertedDoublePendulum-v5' 'Swimmer-v5' 'Reacher-v5'; do
    for seed in 0 42; do
        python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.001 --env $env --gpu-id 1 --seed $seed
    done  
done