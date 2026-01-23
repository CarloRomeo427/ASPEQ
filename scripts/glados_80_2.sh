#!/bin/bash
for seed in 0 42 1234; do
    for env in Hopper-v5 Walker2d-v5; do
        python main.py --env $env --seed $seed --log-wandb --algo speq_o2o --gpu-id 0 --use-minari --minari-quality simple
    done
done
