#!/bin/bash
for seed in 777 13579 24680; do
    for env in Hopper-v5; do
        python main.py --env $env --seed $seed --log-wandb --algo rlpd --gpu-id 1 --minari-quality simple --use-minari
    done
done