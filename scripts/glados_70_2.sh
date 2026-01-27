#!/bin/bash
for seed in 42 1234; do
    for env in Pusher-v5; do
        for quality in expert; do
            python main.py --algo speq_o2o --env $env --use-offline-data --dataset-quality $quality --seed $seed --log-wandb --gpu-id 1
        done
    done
done
