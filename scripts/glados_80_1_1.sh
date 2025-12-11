#!/bin/bash
for seed in 777 13579 31415 24680 27182; do
    for env in Hopper-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --calc-plasticity --algo paspeq --gpu-id 0
    done
done