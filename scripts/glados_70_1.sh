#!/bin/bash
for seed in 1234; do
    for quality in diverse play; do
        python main.py --algo speq --env antmaze-large --use-offline-data --dataset-quality $quality --seed $seed --log-wandb --gpu-id 1
    done
done
