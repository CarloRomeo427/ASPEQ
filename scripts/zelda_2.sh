#!/bin/bash
for seed in 0 42 1234; do
    for env in Humanoid-v5; do
        for epochs in 10000 25000 50000 100000; do
            python main.py --algo speq_o2o --env $env --seed $seed --epochs 300 --log-wandb --offline-epochs $epochs --use-offline-data --dataset-quality expert
        done
    done
done