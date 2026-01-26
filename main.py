"""
Unified Training Script for Offline-to-Online Reinforcement Learning
=====================================================================

Supports:
- MuJoCo environments (Hopper-v5, HalfCheetah-v5, etc.) via Minari
- AntMaze environments via Gymnasium-Robotics + Minari
- Adroit environments via Gymnasium-Robotics + Minari

Requirements:
    pip install minari gymnasium gymnasium-robotics

Usage examples in README.md
"""

import argparse
import time
import numpy as np
import torch
import wandb
import gymnasium as gym
import gymnasium_robotics

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Suite Detection
# ─────────────────────────────────────────────────────────────────────────────

MUJOCO_ENVS = {'hopper', 'halfcheetah', 'walker2d', 'ant', 'swimmer', 'humanoid'}
ADROIT_ENVS = {'pen', 'door', 'hammer', 'relocate'}


def get_env_suite(env_name: str) -> str:
    """Determine which environment suite an environment belongs to."""
    env_lower = env_name.lower()
    base_name = env_lower.split('-')[0]
    
    if base_name in MUJOCO_ENVS:
        return 'mujoco'
    if env_lower.startswith('antmaze'):
        return 'antmaze'
    if base_name in ADROIT_ENVS:
        return 'adroit'
    
    raise ValueError(f"Unknown environment: {env_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Dropout Rate Selection
# ─────────────────────────────────────────────────────────────────────────────

def get_dropout_rate(env_name: str, target_drop_rate: float = -1.0) -> float:
    """
    Get dropout rate based on environment complexity.
    
    Dropout rates scale with state-action space dimensionality:
    - Higher dimensional environments benefit from more regularization
    - Pattern: dropout ≈ f(obs_dim, act_dim)
    
    Args:
        env_name: Environment name
        target_drop_rate: If >= 0, use this value directly. If < 0, auto-select.
    
    Returns:
        Dropout rate for target Q-networks
    """
    # If explicit rate provided, use it
    if target_drop_rate >= 0:
        return target_drop_rate
    
    env_lower = env_name.lower()
    
    # MuJoCo - High dimensional (obs=376, act=17)
    if 'humanoid' in env_lower:
        return 0.1
    
    # Adroit - High-DoF hand manipulation (obs=39-46, act=24-30)
    if any(x in env_lower for x in ['pen', 'door', 'hammer', 'relocate']):
        return 0.05
    
    # AntMaze (obs=29+2, act=8) and Ant (obs=27, act=8) - Medium-high dimensional
    if 'antmaze' in env_lower or 'ant' in env_lower:
        return 0.01
    
    # MuJoCo - Medium dimensional (obs=17, act=6)
    if any(x in env_lower for x in ['walker', 'halfcheetah', 'pusher']):
        return 0.005
    
    # MuJoCo - Low dimensional (obs=8-11, act=2-3)
    if any(x in env_lower for x in ['hopper', 'swimmer', 'reacher', 'pendulum']):
        return 0.001
    
    return 0.005  # Default


# ─────────────────────────────────────────────────────────────────────────────
# Observation Wrapper for Goal-Conditioned Environments
# ─────────────────────────────────────────────────────────────────────────────

class FlattenObsWrapper(gym.ObservationWrapper):
    """Wrapper that flattens goal-conditioned observations."""
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = env.observation_space['observation'].shape[0]
            if 'desired_goal' in env.observation_space.spaces:
                obs_dim += env.observation_space['desired_goal'].shape[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
    
    def observation(self, obs):
        if isinstance(obs, dict):
            flat_obs = obs['observation'].astype(np.float32)
            if 'desired_goal' in obs:
                flat_obs = np.concatenate([flat_obs, obs['desired_goal'].astype(np.float32)])
            return flat_obs
        return obs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Environment Creation
# ─────────────────────────────────────────────────────────────────────────────

def make_env(env_name: str, env_suite: str, quality: str = None):
    """Create environment using Gymnasium API."""
    import minari
    
    # For goal-conditioned envs with offline data, recover from Minari dataset
    if env_suite in ['antmaze', 'adroit'] and quality is not None:
        try:
            dataset_name = get_minari_dataset_name(env_name, quality, env_suite)
            dataset = minari.load_dataset(dataset_name)
            env = dataset.recover_environment()
            env = FlattenObsWrapper(env)
            return env
        except Exception as e:
            print(f"Failed to recover from dataset: {e}, falling back to gym.make")
    
    # Standard path for MuJoCo
    if env_suite == 'mujoco':
        return gym.make(env_name)
    
    # AntMaze
    if env_suite == 'antmaze':
        parts = env_name.lower().replace('antmaze-', '').split('-')
        maze_type = parts[0].capitalize() if parts else 'Umaze'
        gym_name = f'AntMaze_{maze_type}-v4'
        env = gym.make(gym_name)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObsWrapper(env)
        return env
    
    # Adroit
    if env_suite == 'adroit':
        base_name = env_name.lower().split('-')[0]
        task_map = {
            'pen': 'AdroitHandPen-v1',
            'door': 'AdroitHandDoor-v1',
            'hammer': 'AdroitHandHammer-v1',
            'relocate': 'AdroitHandRelocate-v1'
        }
        gym_name = task_map.get(base_name, env_name)
        env = gym.make(gym_name)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObsWrapper(env)
        return env
    
    return gym.make(env_name)


def get_env_info(env):
    """Extract environment dimensions."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    
    if hasattr(env, '_max_episode_steps'):
        max_ep_len = env._max_episode_steps
    elif hasattr(env, 'spec') and env.spec is not None and hasattr(env.spec, 'max_episode_steps'):
        max_ep_len = env.spec.max_episode_steps or 1000
    else:
        max_ep_len = 1000
    
    return obs_dim, act_dim, act_limit, max_ep_len


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading (All via Minari)
# ─────────────────────────────────────────────────────────────────────────────

def get_minari_dataset_name(env_name: str, quality: str, env_suite: str) -> str:
    """Get Minari dataset name for any supported environment."""
    env_lower = env_name.lower()
    
    if env_suite == 'mujoco':
        base_name = env_lower.split('-')[0]
        return f'mujoco/{base_name}/{quality}-v0'
    
    elif env_suite == 'antmaze':
        parts = env_lower.replace('antmaze-', '').split('-')
        maze_type = parts[0] if parts else 'umaze'
        
        # Check if quality is embedded in env_name
        embedded_quality = None
        if len(parts) >= 2 and parts[1] in ['diverse', 'play']:
            embedded_quality = parts[1]
        
        effective_quality = embedded_quality if embedded_quality and quality == 'expert' else quality
        
        if effective_quality in ['v1', 'basic', None, 'expert'] or (maze_type == 'umaze' and effective_quality not in ['diverse', 'play']):
            return f'D4RL/antmaze/{maze_type}-v1'
        else:
            return f'D4RL/antmaze/{maze_type}-{effective_quality}-v1'
    
    elif env_suite == 'adroit':
        base_name = env_lower.split('-')[0]
        return f'D4RL/{base_name}/{quality}-v2'
    
    raise ValueError(f"Cannot determine Minari dataset for: {env_name}")


def load_minari_dataset(agent, env_name: str, quality: str, env_suite: str, compute_mc_returns: bool = False):
    """Load Minari dataset into agent's offline buffer. Works for all environment suites."""
    import minari
    
    dataset_name = get_minari_dataset_name(env_name, quality, env_suite)
    print(f"Loading Minari dataset: {dataset_name}")
    
    try:
        dataset = minari.load_dataset(dataset_name)
    except:
        print(f"Downloading dataset: {dataset_name}")
        minari.download_dataset(dataset_name)
        dataset = minari.load_dataset(dataset_name)
    
    print(f"Dataset: {dataset.total_episodes} episodes, {dataset.total_steps} steps")
    
    gamma = agent.gamma
    total = 0
    
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        T = len(actions)
        
        if compute_mc_returns:
            mc_returns = np.zeros(T)
            mc_returns[-1] = rewards[-1]
            for t in range(T - 2, -1, -1):
                mc_returns[t] = rewards[t] + gamma * mc_returns[t + 1]
        
        for t in range(T):
            # Handle different observation formats
            if isinstance(observations, dict):
                obs = observations['observation'][t].astype(np.float32)
                if 'desired_goal' in observations:
                    obs = np.concatenate([obs, observations['desired_goal'][t].astype(np.float32)])
                next_obs = observations['observation'][t + 1].astype(np.float32)
                if 'desired_goal' in observations:
                    next_obs = np.concatenate([next_obs, observations['desired_goal'][t + 1].astype(np.float32)])
            else:
                obs = observations[t].astype(np.float32)
                next_obs = observations[t + 1].astype(np.float32)
            
            action = actions[t].astype(np.float32)
            reward = float(rewards[t])
            done = bool(terminations[t])
            
            if compute_mc_returns:
                agent.store_data_offline(obs, action, reward, next_obs, done, mc_returns[t])
            else:
                agent.store_data_offline(obs, action, reward, next_obs, done)
            total += 1
    
    print(f"Loaded {total} transitions")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm Registry
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = {
    'iql': 'src.algos.agent_iql.IQLAgent',
    'calql': 'src.algos.agent_calql.CalQLAgent',
    'rlpd': 'src.algos.agent_rlpd.RLPDAgent',
    'speq': 'src.algos.agent_speq.SPEQAgent',
    'speq_o2o': 'src.algos.agent_speq.SPEQAgent',
    'faspeq_o2o': 'src.algos.agent_faspeq.FASPEQAgent',
    'faspeq_td_val': 'src.algos.agent_faspeq.FASPEQAgent',
}


def get_agent_class(algo_name: str):
    """Dynamically import and return the agent class."""
    if algo_name.lower() not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    module_path, class_name = ALGORITHMS[algo_name.lower()].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_algo_config(algo_name: str, args, dropout_rate: float) -> dict:
    """Get algorithm-specific configuration."""
    algo = algo_name.lower()
    
    config = {
        'hidden_sizes': (args.network_width, args.network_width),
        'num_Q': args.num_q,
        'utd_ratio': args.utd_ratio,
        'layer_norm': args.layer_norm,
        'target_drop_rate': dropout_rate if algo in ['speq', 'speq_o2o', 'faspeq_o2o', 'faspeq_td_val'] else 0.0,
    }
    
    if algo == 'iql':
        config.update({
            'iql_tau': args.iql_tau,
            'iql_beta': args.iql_beta,
            'policy_update_delay': 1,
            'o2o': args.use_offline_data,
        })
    
    elif algo == 'calql':
        config.update({
            'cql_alpha': args.cql_alpha,
            'cql_n_actions': args.cql_n_actions,
            'cql_lagrange': args.cql_lagrange,
            'mixing_ratio': args.mixing_ratio,
            'policy_update_delay': 1,
            'o2o': args.use_offline_data,
        })
    
    elif algo == 'rlpd':
        config.update({
            'num_Q': 10,
            'utd_ratio': 20,
            'policy_update_delay': 1,
            'o2o': True,
        })
    
    elif algo == 'speq':
        config.update({
            'policy_update_delay': 20,
            'offline_epochs': 75000,
            'trigger_interval': 10000,
            'o2o': False,
        })
    
    elif algo == 'speq_o2o':
        config.update({
            'policy_update_delay': 20,
            'offline_epochs': 75000,
            'trigger_interval': 10000,
            'o2o': True,
        })
    
    elif algo == 'faspeq_o2o':
        config.update({
            'policy_update_delay': 20,
            'offline_epochs': args.offline_epochs,
            'trigger_interval': 10000,
            'val_check_interval': args.val_check_interval,
            'val_patience': args.val_patience,
            'use_td_val': False,
            'n_val_batches': args.n_val_batches,
            'o2o': True,
        })
    
    elif algo == 'faspeq_td_val':
        config.update({
            'policy_update_delay': 20,
            'offline_epochs': args.offline_epochs,
            'trigger_interval': 10000,
            'val_check_interval': args.val_check_interval,
            'val_patience': args.val_patience,
            'use_td_val': True,
            'n_val_batches': args.n_val_batches,
            'o2o': True,
        })
    
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, env, max_ep_len: int, num_episodes: int = 10) -> float:
    """Evaluate agent and return mean reward."""
    rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_ret = 0
        for _ in range(max_ep_len):
            action = agent.get_test_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                break
        rewards.append(ep_ret)
    
    return np.mean(rewards)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Determine environment suite
    env_suite = get_env_suite(args.env)
    print(f"Environment suite: {env_suite}")
    
    # Environment - pass quality for goal-conditioned envs to recover from dataset
    quality = args.dataset_quality if args.use_offline_data else None
    env = make_env(args.env, env_suite, quality)
    test_env = make_env(args.env, env_suite, quality)
    obs_dim, act_dim, act_limit, max_ep_len = get_env_info(env)
    max_ep_len = min(args.max_ep_len, max_ep_len)
    
    # Total steps
    total_steps = args.steps_per_epoch * args.epochs + 1
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Compute dropout rate
    dropout_rate = get_dropout_rate(args.env, args.target_drop_rate)
    
    # Create agent
    AgentClass = get_agent_class(args.algo)
    algo_config = get_algo_config(args.algo, args, dropout_rate)
    
    agent = AgentClass(
        env_name=args.env,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        device=device,
        start_steps=args.start_steps,
        **algo_config
    )
    
    print(f"Algorithm: {args.algo}, Env: {args.env}, obs_dim: {obs_dim}, act_dim: {act_dim}, dropout: {dropout_rate}")
    
    # Load offline data
    if args.use_offline_data:
        compute_mc = args.algo.lower() == 'calql'
        load_minari_dataset(agent, args.env, args.dataset_quality, env_suite, compute_mc)
        
        # Offline pretraining for IQL/CalQL
        if args.algo.lower() in ['iql', 'calql'] and args.offline_pretrain_steps > 0:
            print(f"Offline pretraining: {args.offline_pretrain_steps} steps")
            agent.train_offline(epochs=args.offline_pretrain_steps, test_env=test_env)
    
    # Initialize (all envs use Gymnasium API now)
    obs, _ = env.reset(seed=args.seed)
    ep_ret, ep_len = 0, 0
    start_time = time.time()
    
    # Training loop
    for t in range(total_steps):
        action = agent.get_exploration_action(obs, env)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        ep_len += 1
        agent.store_data(obs, action, reward, obs_next, terminated and ep_len < max_ep_len)
        
        # Check stabilization trigger (SPEQ/FASPEQ)
        if hasattr(agent, 'check_should_trigger_offline_stabilization'):
            if agent.check_should_trigger_offline_stabilization():
                agent.finetune_offline(test_env=test_env, current_env_step=t+1)
        
        # Training step
        pi_loss, q_loss = agent.train(current_env_step=t+1)
        
        obs = obs_next
        ep_ret += reward
        
        # Episode end
        if done or ep_len >= max_ep_len:
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0
        
        # Epoch logging
        if (t + 1) % args.steps_per_epoch == 0:
            epoch = (t + 1) // args.steps_per_epoch
            eval_reward = evaluate(agent, test_env, max_ep_len)
            
            print(f"Epoch {epoch:4d} | Step {t+1:7d} | EvalReward: {eval_reward:8.2f} | Time: {time.time()-start_time:.0f}s")
            
            wandb.log({
                "epoch": epoch,
                "policy_loss": pi_loss,
                "mean_q_loss": q_loss,
                "EvalReward": eval_reward,
            }, step=t+1)
    
    print("Training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic
    parser.add_argument("--algo", type=str, default='iql',
                        choices=['iql', 'calql', 'rlpd', 'speq', 'speq_o2o', 'faspeq_o2o', 'faspeq_td_val'])
    parser.add_argument("--env", type=str, default='Hopper-v5')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--max-ep-len", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    
    # Network
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--layer-norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--target-drop-rate", type=float, default=-1.0,
                        help="Dropout rate for target Q-networks. -1 for auto-selection based on env.")
    
    # Offline data
    parser.add_argument("--use-offline-data", action="store_true")
    parser.add_argument("--dataset-quality", type=str, default='expert',
                        help="Dataset quality. MuJoCo: expert/medium/simple | "
                             "AntMaze: v1/diverse/play | Adroit: human/cloned/expert")
    
    # IQL
    parser.add_argument("--iql-tau", type=float, default=0.7)
    parser.add_argument("--iql-beta", type=float, default=3.0)
    parser.add_argument("--offline-pretrain-steps", type=int, default=1000000)
    
    # CalQL
    parser.add_argument("--cql-alpha", type=float, default=5.0)
    parser.add_argument("--cql-n-actions", type=int, default=10)
    parser.add_argument("--cql-lagrange", action="store_true")
    parser.add_argument("--mixing-ratio", type=float, default=0.5)
    
    # SPEQ/FASPEQ
    parser.add_argument("--offline-epochs", type=int, default=75000)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--val-patience", type=int, default=10000)
    parser.add_argument("--n-val-batches", type=int, default=5,
                        help="Number of batches from each buffer for validation (FASPEQ)")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    exp_name = f"{args.algo}_{args.env}_{args.seed}"
    
    wandb.init(
        name=exp_name,
        project="SPEQ",
        config=vars(args),
        mode='online' if args.log_wandb else 'disabled'
    )
    
    train(args)
    wandb.finish()
