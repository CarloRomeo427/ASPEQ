#!/bin/bash
for seed in 24680 27182; do
    for env in InvertedPendulum-v5 InvertedDoublePendulum-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --algo paspeq_o2o --gpu-id 0 \
        --offline-frequency 1000
    done
done