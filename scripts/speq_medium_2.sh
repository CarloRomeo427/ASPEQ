#!/bin/bash
for seed in 0 42 1234 5678; do
    python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.01 --env Ant-v5 --gpu-id 0 --seed $seed --minari-quality medium
    python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.1 --env Humanoid-v5 --gpu-id 0 --seed $seed --minari-quality medium
    python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.005 --env Walker2d-v5 --gpu-id 0 --seed $seed --minari-quality medium
    python main_o2o.py --log-wandb --use-minari --target-drop-rate 0.005 --env HalfCheetah-v5 --gpu-id 0 --seed $seed --minari-quality medium
done