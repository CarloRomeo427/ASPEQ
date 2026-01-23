#!/bin/bash
for seed in 0 42 1234; do
    for env in Pusher-v5 Swimmer-v5 Reacher-v5; do
        python main.py --env $env --seed $seed --log-wandb --algo speq_o2o --gpu-id 1 --use-minari --minari-quality simple
    done
done
