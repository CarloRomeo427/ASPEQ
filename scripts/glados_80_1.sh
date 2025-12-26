#!/bin/bash
for seed in 777 13579 31415; do
    for env in Swimmer-v5 Reacher-v5 Pusher-v5; do
        python main_p.py --env $env --seed $seed --log-wandb --algo paspeq_o2o --gpu-id 0 \
        --offline-frequency 1000 
    done
done