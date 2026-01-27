#!/bin/bash
for seed in 42; do
    for quality in medium simple; do
        python main.py --algo speq_o2o --env halfcheetah --use-offline-data --dataset-quality $quality --seed $seed --log-wandb 
    done
done
