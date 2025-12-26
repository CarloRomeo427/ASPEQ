#!/bin/bash
for seed in 24680 27182; do
    for env in Humanoid-v5 Ant-v5 HalfCheetah-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --algo paspeq_o2o --gpu-id 1 \
        --offline-frequency 1000
    done
done