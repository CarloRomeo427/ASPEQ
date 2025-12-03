#!/bin/bash
for seed in 777 9876 13579 31415 24680 27182; do
    for env in Humanoid-v5 Walker2d-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo speq --gpu-id 0
    done
done