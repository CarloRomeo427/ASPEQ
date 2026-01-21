#!/bin/bash
for seed in 0 42 1234; do
    for env in HalfCheetah-v5; do
        python main_faspeq_td_val.py --env HalfCheetah-v5 --seed $seed --log-wand --use-minari --minari-quality expert
    done
done