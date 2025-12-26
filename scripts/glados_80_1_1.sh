#!/bin/bash
for seed in 0 42 1234; do
    for env in Humanoid-v5 Ant-v5 HalfCheetah-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --algo paspeq_o2o --gpu-id 0 --offline-frequency 1000
    done
done