#!/bin/bash
for seed in 0 42 1234; do
    for env in InvertedPendulum-v5; do
        for quality in simple; do
            python main.py --algo speq_o2o --env $env --use-offline-data --dataset-quality $quality --seed $seed --log-wandb 
        done
    done
done
