# Offline-to-Online Reinforcement Learning

## Algorithms

| Algorithm | Stabilization Trigger | Stabilization Length | Sampling | Early Stopping |
|-----------|----------------------|---------------------|----------|----------------|
| **SPEQ** | Every 10k steps | Fixed 75k | Online only | No |
| **SPEQ_O2O** | Every 10k steps | Fixed 75k | 50/50 mix | No |
| **FASPEQ_O2O** | Every 10k steps | Up to 75k | 50/50 mix | Policy loss on val |
| **FASPEQ_TD_VAL** | Every 10k steps | Up to 75k | 50/50 mix | TD error on val |
| **IQL** | - | - | Configurable | - |
| **CalQL** | - | - | Configurable | - |
| **RLPD** | - | - | 50/50 mix | - |

## Installation

```bash
pip install gymnasium minari gymnasium-robotics wandb torch numpy
```

## Usage

### MuJoCo (via Minari)

```bash
# IQL on Hopper with expert data
python main.py --algo iql --env Hopper-v5 --use-offline-data --log-wandb

# SPEQ_O2O on HalfCheetah with medium data
python main.py --algo speq_o2o --env HalfCheetah-v5 --use-offline-data --dataset-quality medium

# FASPEQ on Walker2d
python main.py --algo faspeq_o2o --env Walker2d-v5 --use-offline-data
```

### AntMaze (via Gymnasium-Robotics + Minari)

Environment naming: `antmaze-{size}` where size is `umaze`, `medium`, or `large`
Dataset qualities: `v1` (basic), `diverse`, `play`

```bash
# IQL on AntMaze-umaze
python main.py --algo iql --env antmaze-umaze --use-offline-data --log-wandb

# CalQL on AntMaze-medium with diverse dataset
python main.py --algo calql --env antmaze-medium --use-offline-data --dataset-quality diverse

# FASPEQ on AntMaze-large with play dataset
python main.py --algo faspeq_o2o --env antmaze-large --use-offline-data --dataset-quality play
```

### Adroit (via Gymnasium-Robotics + Minari)

Environment naming: `{task}` where task is `pen`, `door`, `hammer`, or `relocate`
Dataset qualities: `human`, `cloned`, `expert`

```bash
# IQL on Door with human demonstrations
python main.py --algo iql --env door --use-offline-data --dataset-quality human --log-wandb

# CalQL on Hammer with cloned data
python main.py --algo calql --env hammer --use-offline-data --dataset-quality cloned

# SPEQ_O2O on Pen with expert data
python main.py --algo speq_o2o --env pen --use-offline-data --dataset-quality expert

# FASPEQ on Relocate with human data
python main.py --algo faspeq_o2o --env relocate --use-offline-data --dataset-quality human
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `iql` | Algorithm |
| `--env` | `Hopper-v5` | Environment |
| `--seed` | `0` | Random seed |
| `--epochs` | `300` | Training epochs |
| `--use-offline-data` | `False` | Load offline dataset |
| `--dataset-quality` | `expert` | Dataset quality (varies by env suite) |
| `--log-wandb` | `False` | Enable WandB |

### FASPEQ-specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--offline-epochs` | `75000` | Max epochs per stabilization |
| `--val-patience` | `10000` | Early stopping patience |
| `--val-check-interval` | `1000` | Validation check frequency |

## Logged Metrics

**Terminal:** epoch, step, EvalReward, time

**WandB:** epoch, policy_loss, mean_q_loss, EvalReward, OfflineEpochs (SPEQ/FASPEQ), OfflineEvalReward (SPEQ/FASPEQ)
