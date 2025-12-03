#!/bin/bash
for seed in 777 9876 13579 31415 24680 27182; do
    for env in Reacher-v5 Swimmer-v5 Pusher-v5; do
        python main.py --env $env --seed $seed --log-wandb --calc-plasticity --algo speq --gpu-id 0
    done
done