"""
Unified SPEQ Training Script
Supports Minari datasets for MuJoCo, AntMaze, and Adroit environments.

All datasets are now available through Minari with Gymnasium API:
- MuJoCo: Hopper, HalfCheetah, Walker2d, Ant, Humanoid, etc.
- AntMaze: umaze, medium, large (with play/diverse variants)
- Adroit: pen, door, hammer, relocate (with human/cloned/expert variants)

Requirements:
    pip install minari gymnasium gymnasium-robotics
"""

import gymnasium as gym
import gymnasium_robotics  # Required to register AntMaze and Adroit environments
import numpy as np
import torch
import time
import sys
import wandb
import os
import minari

from src.algos_non_opt.agent import Agent
from src.algos_non_opt.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

# Add this class after the imports in main.py

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


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# MuJoCo locomotion environments
MUJOCO_ENVS = {
    'hopper', 'halfcheetah', 'walker2d', 'ant', 'swimmer', 'reacher', 
    'pusher', 'invertedpendulum', 'inverteddoublependulum', 'humanoid', 'humanoidstandup'
}

# AntMaze environments (now in Gymnasium-Robotics)
ANTMAZE_ENVS = {
    'antmaze-umaze', 'antmaze-medium', 'antmaze-large'
}

# Adroit hand environments (now in Gymnasium-Robotics)
ADROIT_ENVS = {
    'pen', 'door', 'hammer', 'relocate'
}

# Dataset quality mappings per environment type
MUJOCO_QUALITIES = ['expert', 'medium', 'simple']
ANTMAZE_QUALITIES = ['v1', 'diverse', 'play']  # v1 is the basic version
ADROIT_QUALITIES = ['human', 'cloned', 'expert']


def get_env_suite(env_name):
    """Determine which environment suite an environment belongs to."""
    env_lower = env_name.lower()
    base_name = env_lower.split('-')[0]
    
    # Check MuJoCo locomotion (e.g., "Hopper-v5", "HalfCheetah-v5")
    if base_name in MUJOCO_ENVS:
        return 'mujoco'
    
    # Check AntMaze (e.g., "antmaze-umaze", "antmaze-large")
    if env_lower.startswith('antmaze'):
        return 'antmaze'
    
    # Check Adroit (e.g., "pen", "door", "hammer", "relocate")
    if base_name in ADROIT_ENVS:
        return 'adroit'
    
    raise ValueError(f"Unknown environment: {env_name}. "
                     f"Supported: MuJoCo ({MUJOCO_ENVS}), "
                     f"AntMaze ({ANTMAZE_ENVS}), "
                     f"Adroit ({ADROIT_ENVS})")


def get_minari_dataset_name(env_name, quality, env_suite):
    """
    Get Minari dataset name for any supported environment.
    
    Minari dataset naming conventions (from remote server):
    - MuJoCo:   mujoco/{env}/{quality}-v0          (e.g., mujoco/hopper/expert-v0)
    - AntMaze:  D4RL/antmaze/{maze}-{quality}-v1   (e.g., D4RL/antmaze/large-diverse-v1)
    - Adroit:   D4RL/{task}/{quality}-v2           (e.g., D4RL/door/human-v2)
    """
    env_lower = env_name.lower()
    
    if env_suite == 'mujoco':
        base_name = env_lower.split('-')[0]
        return f'mujoco/{base_name}/{quality}-v0'
    
    elif env_suite == 'antmaze':
        # Parse env_name: could be "antmaze-large", "antmaze-large-diverse-v1", etc.
        parts = env_lower.replace('antmaze-', '').split('-')
        maze_type = parts[0] if parts else 'umaze'
        
        # Check if quality is embedded in env_name (e.g., "antmaze-large-diverse-v1")
        embedded_quality = None
        if len(parts) >= 2 and parts[1] in ['diverse', 'play']:
            embedded_quality = parts[1]
        
        # Use embedded quality if present and user didn't override with non-default
        effective_quality = embedded_quality if embedded_quality and quality == 'expert' else quality
        
        # Construct Minari dataset name
        if effective_quality in ['v1', 'basic', None, 'expert'] or (maze_type == 'umaze' and effective_quality not in ['diverse', 'play']):
            return f'D4RL/antmaze/{maze_type}-v1'
        else:
            return f'D4RL/antmaze/{maze_type}-{effective_quality}-v1'
    
    elif env_suite == 'adroit':
        base_name = env_lower.split('-')[0]
        return f'D4RL/{base_name}/{quality}-v2'
    
    raise ValueError(f"Cannot determine Minari dataset name for: {env_name}")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_minari_dataset(agent, env_name, quality, env_suite):
    """
    Load any Minari dataset into agent's offline replay buffer.
    Works for MuJoCo, AntMaze, and Adroit environments.
    """
    dataset_name = get_minari_dataset_name(env_name, quality, env_suite)
    print(f"Loading Minari dataset: {dataset_name}")
    
    try:
        dataset = minari.load_dataset(dataset_name)
    except Exception as e:
        print(f"Dataset not found locally. Attempting to download: {dataset_name}")
        try:
            minari.download_dataset(dataset_name)
            dataset = minari.load_dataset(dataset_name)
        except Exception as e2:
            # List available datasets for debugging
            print(f"Error: {e2}")
            print("\nAvailable Minari datasets:")
            available = minari.list_remote_datasets()
            # Filter relevant datasets
            for ds in sorted(available):
                if any(x in ds.lower() for x in [env_name.lower().split('-')[0], 'antmaze', 'door', 'pen', 'hammer', 'relocate']):
                    print(f"  - {ds}")
            raise RuntimeError(f"Failed to load Minari dataset {dataset_name}: {e2}")
    
    print(f"Dataset info: {dataset.total_episodes} episodes, {dataset.total_steps} steps")
    
    # Check observation space structure
    obs_space = dataset.observation_space
    print(f"Dataset observation space: {obs_space}")
    
    total_transitions = 0
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations
        
        T = len(actions)
        for t in range(T):
            # Handle different observation formats
            if isinstance(observations, dict):
                # Goal-conditioned environments (AntMaze, Adroit)
                # Flatten observation: concat observation + desired_goal
                obs = observations['observation'][t].astype(np.float32)
                if 'desired_goal' in observations:
                    goal = observations['desired_goal'][t].astype(np.float32)
                    obs = np.concatenate([obs, goal])
                
                next_obs = observations['observation'][t + 1].astype(np.float32)
                if 'desired_goal' in observations:
                    next_goal = observations['desired_goal'][t + 1].astype(np.float32)
                    next_obs = np.concatenate([next_obs, next_goal])
            elif isinstance(obs_space, gym.spaces.Dict):
                # Observation space is Dict but episode data might be structured differently
                # Try to access as dict keys
                if hasattr(observations, 'keys') or isinstance(observations, dict):
                    obs = observations['observation'][t].astype(np.float32)
                    if 'desired_goal' in observations:
                        goal = observations['desired_goal'][t].astype(np.float32)
                        obs = np.concatenate([obs, goal])
                    
                    next_obs = observations['observation'][t + 1].astype(np.float32)
                    if 'desired_goal' in observations:
                        next_goal = observations['desired_goal'][t + 1].astype(np.float32)
                        next_obs = np.concatenate([next_obs, next_goal])
                else:
                    # Fall back to flat observations
                    obs = observations[t].astype(np.float32)
                    next_obs = observations[t + 1].astype(np.float32)
            else:
                # Standard observations (MuJoCo locomotion)
                obs = observations[t].astype(np.float32)
                next_obs = observations[t + 1].astype(np.float32)
            
            action = actions[t].astype(np.float32)
            reward = float(rewards[t])
            done = bool(terminations[t])
            
            agent.store_data_offline(obs, action, reward, next_obs, done)
            total_transitions += 1
        
        # Print debug info for first episode only
        if total_transitions <= T:
            print(f"First episode - obs shape: {obs.shape}, action shape: {action.shape}")
    
    print(f"Loaded {total_transitions} transitions")
    return total_transitions


def load_dataset(agent, env_name, quality, env_suite):
    """Unified dataset loading function - all through Minari now."""
    return load_minari_dataset(agent, env_name, quality, env_suite)


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATION
# ══════════════════════════════════════════════════════════════════════════════

def get_gymnasium_env_name(env_name, env_suite):
    """
    Get the Gymnasium environment name for creating the environment.
    
    For AntMaze and Adroit, we can recover the environment from the Minari dataset,
    or use the Gymnasium-Robotics environment names directly.
    """
    env_lower = env_name.lower()
    
    if env_suite == 'mujoco':
        # Standard Gymnasium MuJoCo environments (e.g., Hopper-v5)
        return env_name
    
    elif env_suite == 'antmaze':
        # Use v4 to match Minari datasets (not v5)
        parts = env_lower.replace('antmaze-', '').split('-')
        maze_type = parts[0] if parts else 'umaze'
        maze_type_cap = maze_type.capitalize()
        # Note: actual env name depends on dataset variant
        return f'AntMaze_{maze_type_cap}-v4'
    
    elif env_suite == 'adroit':
        # Gymnasium-Robotics Adroit environments
        # Format: Adroit{Task}-v1 (e.g., AdroitHandDoor-v1)
        base_name = env_lower.split('-')[0]
        task_map = {
            'pen': 'AdroitHandPen-v1',
            'door': 'AdroitHandDoor-v1',
            'hammer': 'AdroitHandHammer-v1',
            'relocate': 'AdroitHandRelocate-v1'
        }
        return task_map.get(base_name, env_name)
    
    return env_name


def make_env(env_name, env_suite, quality=None):
    """
    Create environment using Gymnasium API.
    For AntMaze/Adroit with offline data, recover from dataset to ensure compatibility.
    """
    # For goal-conditioned envs with offline data, recover from dataset
    if env_suite in ['antmaze', 'adroit'] and quality is not None:
        try:
            dataset_name = get_minari_dataset_name(env_name, quality, env_suite)
            print(f"Recovering environment from Minari dataset: {dataset_name}")
            dataset = minari.load_dataset(dataset_name)
            env = dataset.recover_environment()
            # Wrap to flatten observations
            env = FlattenObsWrapper(env)
            return env
        except Exception as e:
            print(f"Failed to recover from dataset: {e}, falling back to gym.make")
    
    # Standard path for MuJoCo or when no quality specified
    gym_env_name = get_gymnasium_env_name(env_name, env_suite)
    print(f"Creating Gymnasium environment: {gym_env_name}")
    
    try:
        env = gym.make(gym_env_name)
        # Wrap goal-conditioned envs
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObsWrapper(env)
    except Exception as e:
        print(f"Error creating {gym_env_name}: {e}")
        raise
    
    return env


def get_env_specs(env, env_suite):
    """Extract observation/action dimensions and limits from environment."""
    obs_space = env.observation_space
    act_space = env.action_space
    
    # Handle goal-conditioned environments (dict observation space)
    if isinstance(obs_space, gym.spaces.Dict):
        # For goal-conditioned envs, we need to decide how to handle observations
        # Option 1: Use only the 'observation' part (matches D4RL convention)
        # Option 2: Concatenate observation + desired_goal
        # 
        # D4RL/Minari datasets for AntMaze use flat observations that already include goal
        # Let's check the actual dimensions
        obs_dim = obs_space['observation'].shape[0]
        goal_dim = obs_space['desired_goal'].shape[0] if 'desired_goal' in obs_space.spaces else 0
        
        print(f"Goal-conditioned env detected:")
        print(f"  observation dim: {obs_dim}")
        print(f"  desired_goal dim: {goal_dim}")
        
        # For D4RL AntMaze datasets, the observations in the dataset already include goal info
        # So we should NOT add goal_dim here - the dataset loader will handle flattening
        # We'll use the combined dimension to match what the dataset provides
        obs_dim = obs_dim + goal_dim
        print(f"  combined obs_dim for agent: {obs_dim}")
    else:
        obs_dim = obs_space.shape[0]
    
    act_dim = act_space.shape[0]
    act_limit = float(act_space.high[0])
    
    # Get max episode length
    if hasattr(env, '_max_episode_steps') and env._max_episode_steps is not None:
        max_ep_len = env._max_episode_steps
    elif hasattr(env, 'spec') and env.spec is not None and hasattr(env.spec, 'max_episode_steps'):
        max_ep_len = env.spec.max_episode_steps or 1000
    else:
        max_ep_len = 1000
    
    return obs_dim, act_dim, act_limit, max_ep_len


def get_obs_dim_from_dataset(env_name, quality, env_suite):
    """
    Get the actual observation dimension from the Minari dataset.
    This is important for goal-conditioned environments where the dataset
    may have different observation structure than the live environment.
    """
    dataset_name = get_minari_dataset_name(env_name, quality, env_suite)
    
    try:
        dataset = minari.load_dataset(dataset_name)
    except:
        try:
            minari.download_dataset(dataset_name)
            dataset = minari.load_dataset(dataset_name)
        except:
            return None  # Will fall back to env-based obs_dim
    
    obs_space = dataset.observation_space
    
    if isinstance(obs_space, gym.spaces.Dict):
        obs_dim = obs_space['observation'].shape[0]
        if 'desired_goal' in obs_space.spaces:
            obs_dim += obs_space['desired_goal'].shape[0]
        return obs_dim
    else:
        return obs_space.shape[0]


def env_reset(env, env_suite, seed=None):
    """Reset environment with Gymnasium API."""
    return env.reset(seed=seed)


def env_step(env, action, env_suite):
    """Step environment with Gymnasium API."""
    return env.step(action)


def flatten_obs(obs, env_suite):
    """Flatten observation for goal-conditioned environments."""
    if isinstance(obs, dict):
        # Goal-conditioned environment
        flat_obs = obs['observation'].astype(np.float32)
        if 'desired_goal' in obs:
            goal = obs['desired_goal'].astype(np.float32)
            flat_obs = np.concatenate([flat_obs, goal])
        return flat_obs
    return obs.astype(np.float32) if hasattr(obs, 'astype') else np.array(obs, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_class_attributes(obj):
    """Prints all attributes of an object along with their values."""
    attributes = vars(obj)
    for attr, value in attributes.items():
        print(f"{attr}: {value}")


def set_dropout(env_name, target_drop_rate=0.0):
    """Set dropout rate based on environment complexity."""
    if target_drop_rate == 0.0:
        return 0.0
    
    env_lower = env_name.lower()
    
    # High-dimensional environments
    if 'humanoid' in env_lower:
        return 0.1
    elif 'ant' in env_lower:
        return 0.01
    elif any(x in env_lower for x in ['walker', 'halfcheetah', 'pusher']):
        return 0.005
    elif any(x in env_lower for x in ['hopper', 'swimmer', 'reacher', 'pendulum']):
        return 0.001
    # D4RL Adroit (high-DoF hand)
    elif any(x in env_lower for x in ['pen', 'door', 'hammer', 'relocate']):
        return 0.01
    
    return 0.005  # Default


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def SPEQ(env_name, seed=0, epochs=300, steps_per_epoch=1000,
         max_ep_len=1000, n_evals_per_epoch=1,
         logger_kwargs=dict(), debug=False,
         hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
         lr=3e-4, gamma=0.99, polyak=0.995,
         alpha=0.2, auto_alpha=True, target_entropy='mbpo',
         start_steps=5000, delay_update_steps='auto',
         utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
         policy_update_delay=20,
         evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
         gpu_id=0, target_drop_rate=0.0, layer_norm=False, 
         offline_epochs=100, use_offline_data=False, dataset_quality='expert',
         val_buffer_prob=0.1, val_buffer_offline_frac=0.1,
         val_check_interval=1000, val_patience=5000,
         adaptive_trigger_expansion_rate=1.1, auto_stab=False
         ):
    """
    SPEQ training with support for multiple environment suites.
    """
    
    if debug:
        hidden_sizes = [2, 2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    
    # Determine environment suite
    env_suite = get_env_suite(env_name)
    print(f"\n{'='*80}")
    print(f"Environment Suite: {env_suite}")
    print(f"Environment: {env_name}")
    print(f"Dataset Quality: {dataset_quality}")
    print(f"{'='*80}\n")

    if epochs == 'mbpo' or epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        epochs = mbpo_epoches.get(env_name, 300)
    total_steps = steps_per_epoch * epochs + 1

    # Set up logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Create environments
    env = make_env(env_name, env_suite, dataset_quality)
    test_env = make_env(env_name, env_suite, dataset_quality)
    bias_eval_env = make_env(env_name, env_suite, dataset_quality) if evaluate_bias else None

    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        return env_seed, test_env_seed, bias_eval_env_seed

    env_seed, test_env_seed, bias_eval_env_seed = seed_all(epoch=0)

    # Get environment specifications
    obs_dim, act_dim, act_limit, env_max_ep_len = get_env_specs(env, env_suite)
    max_ep_len = min(max_ep_len, env_max_ep_len) if env_max_ep_len else max_ep_len
    
    # If using offline data, get obs_dim from dataset to ensure consistency
    # If using offline data, get obs_dim from actual dataset samples
    if use_offline_data:
        dataset_name = get_minari_dataset_name(env_name, dataset_quality, env_suite)
        try:
            dataset = minari.load_dataset(dataset_name)
        except FileNotFoundError:
            print(f"Downloading dataset: {dataset_name}")
            minari.download_dataset(dataset_name)
            dataset = minari.load_dataset(dataset_name)
        for ep in dataset.iterate_episodes():
            obs = ep.observations
            if isinstance(obs, dict):
                sample = obs['observation'][0]
                if 'desired_goal' in obs:
                    sample = np.concatenate([sample, obs['desired_goal'][0]])
                obs_dim = sample.shape[0]
            else:
                obs_dim = obs[0].shape[0]
            print(f"Dataset actual obs_dim: {obs_dim}")
            break
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Action limit: {act_limit}")
    print(f"Max episode length: {max_ep_len}")
    
    start_time = time.time()
    sys.stdout.flush()

    # Initialize agent
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size,
                  lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy,
                  start_steps, delay_update_steps,
                  utd_ratio, num_Q, num_min, q_target_mode,
                  policy_update_delay,
                  target_drop_rate=target_drop_rate,
                  layer_norm=layer_norm,
                  o2o=use_offline_data,
                  val_buffer_prob=val_buffer_prob,
                  val_buffer_offline_frac=val_buffer_offline_frac,
                  val_check_interval=val_check_interval,
                  val_patience=val_patience,
                  adaptive_trigger_expansion_rate=adaptive_trigger_expansion_rate,
                  auto_stab=auto_stab
                  )

    print_class_attributes(agent)

    # Load offline dataset if specified
    offline_samples = 0
    if use_offline_data:
        offline_samples = load_dataset(agent, env_name, dataset_quality, env_suite)
        print(f"Loaded {offline_samples} offline transitions")
        wandb.log({"offline_samples_loaded": offline_samples}, step=0)
        agent.populate_val_buffer_from_offline()
        wandb.log({"val_buffer_size": agent.val_replay_buffer.size}, step=0)

    # Initialize episode
    o, info = env_reset(env, env_suite, seed=env_seed)
    o = flatten_obs(o, env_suite)  # Flatten for goal-conditioned envs
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    total_stabilization_triggers = 0
    total_stabilization_epochs = 0

    for t in range(total_steps):
        a = agent.get_exploration_action(o, env)
        o2_raw, r, terminated, truncated, info = env_step(env, a, env_suite)
        o2 = flatten_obs(o2_raw, env_suite)  # Flatten for goal-conditioned envs
        d = terminated or truncated
        ep_len += 1
        d_store = terminated and not truncated

        agent.store_data(o, a, r, o2, d_store)

        # Check for offline stabilization trigger
        if agent.check_should_trigger_offline_stabilization():
            total_stabilization_triggers += 1
            wandb.log({
                "stabilization_trigger": total_stabilization_triggers,
                "buffer_size_at_trigger": agent.replay_buffer.size,
                "next_trigger_size": agent.next_trigger_size
            }, step=t+1)
            
            if auto_stab:
                epochs_performed = agent.finetune_offline_auto(
                    epochs=offline_epochs, test_env=test_env, current_env_step=t+1
                )
            else:
                epochs_performed = agent.finetune_offline(
                    epochs=offline_epochs, test_env=test_env, current_env_step=t+1
                )
            total_stabilization_epochs += epochs_performed
            
            wandb.log({
                "stabilization_epochs_performed": epochs_performed,
                "total_stabilization_epochs": total_stabilization_epochs,
                "avg_epochs_per_stabilization": total_stabilization_epochs / max(1, total_stabilization_triggers)
            }, step=t+1)

        agent.train(logger, current_env_step=t+1)

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o_raw, info = env_reset(env, env_suite)
            o = flatten_obs(o_raw, env_suite)
            r, ep_ret, ep_len = 0, 0, 0
            terminated, truncated = False, False

        # Epoch logging
        if (t + 1) % 1000 == 0:
            epoch = t // 1000
            current_step = t + 1

            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({
                "EvalReward": np.mean(test_rw),
                "val_buffer_size": agent.val_replay_buffer.size
            }, step=current_step)

            if evaluate_bias:
                normalized_bias_sqr_per_state, normalized_bias_per_state, bias = log_bias_evaluation(
                    bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff
                )
                wandb.log({
                    "normalized_bias_sqr_per_state": np.abs(np.mean(normalized_bias_sqr_per_state)),
                    "normalized_bias_per_state": np.mean(normalized_bias_per_state),
                    "bias": np.mean(bias)
                }, step=current_step)

            if reseed_each_epoch:
                env_seed, test_env_seed, bias_eval_env_seed = seed_all(epoch)
                o_raw, info = env_reset(env, env_suite, seed=env_seed)
                o = flatten_obs(o_raw, env_suite)
                r, ep_ret, ep_len = 0, 0, 0
                terminated, truncated = False, False

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular('TestEpBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBiasSqr', with_min_and_max=True)
            logger.dump_tabular()

            sys.stdout.flush()
    
    print(f"\n{'='*80}\nTRAINING SUMMARY\n{'='*80}")
    print(f"Total stabilization triggers: {total_stabilization_triggers}")
    print(f"Total stabilization epochs: {total_stabilization_epochs}")
    print(f"Final validation buffer size: {agent.val_replay_buffer.size}")
    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING AND MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train SPEQ on MuJoCo (Minari) or D4RL (AntMaze, Adroit) environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ─── Environment & Experiment ─────────────────────────────────────────────────
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "droq", "speq", "aspeq", "speq_o2o", "aspeq_o2o", 
                                 "paspeq_o2o", "redq", "rlpd", "sac_drop"],
                        help="RL algorithm to use")
    parser.add_argument("--env", type=str, default="Hopper-v5",
                        help="Environment name. Examples: "
                             "MuJoCo: Hopper-v5, HalfCheetah-v5, Ant-v5 | "
                             "AntMaze: antmaze-umaze, antmaze-medium, antmaze-large | "
                             "Adroit: pen, door, hammer, relocate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--exp-name", type=str, default="speq", help="Experiment name")
    parser.add_argument("--info", type=str, help="Path to experiment folder")

    # ─── Dataset Configuration ────────────────────────────────────────────────────
    parser.add_argument("--use-offline-data", action="store_true",
                        help="Load offline dataset for O2O training")
    parser.add_argument("--dataset-quality", type=str, default='expert',
                        help="Dataset quality. "
                             "MuJoCo: expert, medium, simple | "
                             "AntMaze: v1 (basic), diverse, play | "
                             "Adroit: human, cloned, expert")

    # ─── Logging & Debug ──────────────────────────────────────────────────────────
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-wandb", action="store_true", help="Log to Weights & Biases")

    # ─── Hardware ─────────────────────────────────────────────────────────────────
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU device ID")

    # ─── Training Parameters ──────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--offline-epochs", type=int, default=75_000,
                        help="Maximum number of offline update epochs")
    parser.add_argument("--utd", type=int, default=1, help="Update-to-data ratio")

    # ─── Validation Parameters ────────────────────────────────────────────────────
    parser.add_argument("--val-buffer-prob", type=float, default=0.1,
                        help="Probability of adding online sample to validation buffer")
    parser.add_argument("--val-buffer-offline-frac", type=float, default=0.1,
                        help="Fraction of offline data to add to validation buffer")
    parser.add_argument("--val-check-interval", type=int, default=1000,
                        help="Steps between validation checks")
    parser.add_argument("--val-patience", type=int, default=5000,
                        help="Steps without improvement for early stopping")

    # ─── Adaptive Triggering ──────────────────────────────────────────────────────
    parser.add_argument("--adaptive-trigger-rate", type=float, default=1.1,
                        help="Buffer growth rate for adaptive triggering")

    # ─── Network & Optimization ───────────────────────────────────────────────────
    parser.add_argument("--network-width", type=int, default=256, help="Hidden units per layer")
    parser.add_argument("--num-q", type=int, default=2, help="Number of Q-networks")
    parser.add_argument("--target-drop-rate", type=float, default=999.0, help="Dropout rate")

    # ─── Boolean Toggles ──────────────────────────────────────────────────────────
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false",
                        help="Disable layer normalization")
    parser.add_argument("--evaluate-bias", action="store_true", help="Evaluate policy bias")

    args = parser.parse_args()

    # Determine environment suite for naming
    env_suite = get_env_suite(args.env)
    
    # Build experiment name
    if args.use_offline_data:
        exp_name_full = f"{args.algo}_{args.env}_{args.dataset_quality}"
    else:
        exp_name_full = f"{args.algo}_{args.env}"

    args.data_dir = './runs/' + str(args.info) + '/'
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)
    auto_dropout = set_dropout(args.env, args.target_drop_rate)

    def set_algo_params(args):
        params = {
            'exp_name': exp_name_full,
            'seed': args.seed,
            'env': args.env,
            'epochs': args.epochs,
            'gpu_id': args.gpu_id,
            'debug': args.debug,
            'evaluate_bias': False,
            'layer_norm': True,
            'hidden_sizes': (256, 256),
            'num_q': 2,
            'offline_epochs': 0,
            'utd_ratio': 1,
            'val_buffer_prob': 0.0,
            'val_buffer_offline_frac': 0.0,
            'val_check_interval': 0,
            'val_patience': 0,
            'adaptive_trigger_rate': 1.0,
            'dropout': 0.0,
            'use_offline_data': False,
            'dataset_quality': args.dataset_quality,
            'auto_stab': False
        }
        
        name_algo = args.algo.lower()
        
        if name_algo == "sac":
            return params
        elif name_algo == "sac_drop":
            params['dropout'] = auto_dropout
            params['use_offline_data'] = args.use_offline_data
            return params
        elif name_algo == "droq":
            params['utd_ratio'] = 20
            return params
        elif name_algo == "redq":
            params['num_q'] = 20
            params['utd_ratio'] = 20
            return params
        elif name_algo == "rlpd":
            params['num_q'] = 10
            params['utd_ratio'] = 20
            params['use_offline_data'] = True
            return params
        elif name_algo == "speq":
            params['offline_epochs'] = 75_000
            params['dropout'] = auto_dropout
            return params
        elif name_algo == "aspeq":
            params['offline_epochs'] = 75_000
            params['val_buffer_prob'] = 0.1
            params['val_buffer_offline_frac'] = 0.1
            params['val_check_interval'] = 1000
            params['val_patience'] = 5000
            params['adaptive_trigger_rate'] = 1.1
            params['dropout'] = auto_dropout
            params['auto_stab'] = True
            return params   
        elif name_algo == "speq_o2o":
            params['offline_epochs'] = 75_000
            params['dropout'] = auto_dropout
            params['use_offline_data'] = True
            return params
        elif name_algo in ["aspeq_o2o", "paspeq_o2o"]:
            params['offline_epochs'] = 75_000
            params['val_buffer_prob'] = 0.1
            params['val_buffer_offline_frac'] = 0.1
            params['val_check_interval'] = 1000
            params['val_patience'] = 5000
            params['adaptive_trigger_rate'] = 1.1
            params['dropout'] = auto_dropout
            params['use_offline_data'] = True
            params['auto_stab'] = True
            return params
        else:
            raise ValueError(f"Unknown algorithm: {args.algo}")

    params = set_algo_params(args)

    # Print configuration
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print(f"{'='*80}")
    print(f"Environment Suite: {env_suite}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Dataset Quality: {args.dataset_quality}")
    print(f"Use Offline Data: {params['use_offline_data']}")
    print(f"{'='*80}\n")

    wandb.init(
        name=f'{exp_name_full}',
        project="SPEQ",
        config=params,
        mode='online' if args.log_wandb else 'disabled',
        save_code=True
    )

    SPEQ(env_name=params['env'], 
         seed=params['seed'], 
         epochs=params['epochs'],
         logger_kwargs=logger_kwargs, 
         debug=params['debug'],
         gpu_id=params['gpu_id'],
         target_drop_rate=params['dropout'],
         layer_norm=params['layer_norm'],
         num_Q=params['num_q'],
         offline_epochs=params['offline_epochs'],
         hidden_sizes=params['hidden_sizes'],
         evaluate_bias=params['evaluate_bias'],
         use_offline_data=params['use_offline_data'],
         dataset_quality=params['dataset_quality'],
         val_buffer_prob=params['val_buffer_prob'],
         val_buffer_offline_frac=params['val_buffer_offline_frac'],
         val_check_interval=params['val_check_interval'],
         val_patience=params['val_patience'],
         adaptive_trigger_expansion_rate=params['adaptive_trigger_rate'],
         utd_ratio=params['utd_ratio'],
         auto_stab=params['auto_stab']
         )