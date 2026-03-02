#!/bin/bash
for seed in 0 42 1234; do
    for env in HalfCheetah-v5; do
        for epochs in 10000 25000 50000 100000
        python main.py --algo speq_o2o --env $env --seed $seed --epochs 300 --log-wandb --offline-epochs $epochs
    done
done