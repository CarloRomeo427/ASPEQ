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

The script automatically resolves short environment names to the latest versions:

### MuJoCo

```bash
# Short names work (auto-resolves to Hopper-v5, mujoco/hopper/expert-v0)
python main.py --algo iql --env hopper --dataset-quality expert --use-offline-data --log-wandb

# Full names also work
python main.py --algo speq_o2o --env HalfCheetah-v5 --dataset-quality medium --use-offline-data

# Available: hopper, halfcheetah, walker2d, ant, swimmer, humanoid, 
#           invertedpendulum, inverteddoublependulum, pusher, reacher
# Dataset qualities: expert, medium, simple (not all combos available)
```

### AntMaze

```bash
# Short: antmaze-{size} (auto-resolves to AntMaze_{Size}-v4, antmaze-{size}-{quality}-v0)
python main.py --algo iql --env antmaze-large --dataset-quality diverse --use-offline-data

python main.py --algo calql --env antmaze-medium --dataset-quality play --use-offline-data

python main.py --algo faspeq_o2o --env antmaze-umaze --use-offline-data

# Sizes: umaze, medium, large
# Dataset qualities: diverse, play (or omit for basic)
```

### Adroit

```bash
# Short: {task} (auto-resolves to AdroitHand{Task}-v1, D4RL/{task}/{quality}-v2)
python main.py --algo iql --env door --dataset-quality human --use-offline-data --log-wandb

python main.py --algo calql --env hammer --dataset-quality cloned --use-offline-data

python main.py --algo speq_o2o --env pen --dataset-quality expert --use-offline-data

# Tasks: pen, door, hammer, relocate
# Dataset qualities: human, cloned, expert
```

## Environment Name Resolution

| User Input | Gymnasium Env | Minari Dataset |
|------------|---------------|----------------|
| `hopper` | `Hopper-v5` | `mujoco/hopper/{quality}-v0` |
| `Hopper-v5` | `Hopper-v5` | `mujoco/hopper/{quality}-v0` |
| `antmaze-large` | `AntMaze_Large-v4` | `antmaze-large-{quality}-v0` |
| `door` | `AdroitHandDoor-v1` | `D4RL/door/{quality}-v2` |

## WandB Run Naming

Runs are named as `{algo}_{env}-{quality}_{seed}`:
- `speq_hopper-expert_1234`
- `iql_antmaze-large-diverse_0`
- `calql_door-human_42`

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | `iql` | Algorithm |
| `--env` | `hopper` | Environment (short name or full) |
| `--seed` | `0` | Random seed |
| `--epochs` | `300` | Training epochs |
| `--use-offline-data` | `False` | Load offline dataset |
| `--dataset-quality` | `expert` | Dataset quality |
| `--log-wandb` | `False` | Enable WandB |
| `--target-drop-rate` | `-1` | Dropout rate (-1 for auto) |

### FASPEQ-specific

| Argument | Default | Description |
|----------|---------|-------------|
| `--offline-epochs` | `75000` | Max epochs per stabilization |
| `--val-patience` | `10000` | Early stopping patience |
| `--val-check-interval` | `1000` | Validation check frequency |

## Logged Metrics

**Terminal:** epoch, step, EvalReward, time

**WandB:** epoch, policy_loss, mean_q_loss, EvalReward, OfflineEpochs (SPEQ/FASPEQ), OfflineEvalReward (SPEQ/FASPEQ)