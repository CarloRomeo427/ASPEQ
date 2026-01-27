#!/bin/bash
for seed in 0; do
    for quality in diverse play; do
        # python main.py --algo speq_o2o --env halfcheetah --use-offline-data --dataset-quality $quality --seed $seed --log-wandb 
         python main.py --algo iql --env antmaze-large --dataset-quality $quality --use-offline-data --offline-pretrain-steps 10000 --epochs 10 \
         --log-wandb 
    done
done
