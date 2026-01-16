#!/bin/bash
for seed in 1234; do
    for env in Hopper-v5; do
        python main.py --env $env --seed $seed --log-wandb --algo rlpd --gpu-id 0 --minari-quality simple --use-minari
    done
done