#!/bin/bash
for seed in 0 42 1234; do
    for env in Humanoid-v5 HalfCheetah-v5 Hopper-v5; do
        python main_a3rl.py --env $env --seed $seed  --log-wandb    
    done
done