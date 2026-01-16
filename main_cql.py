### MAIN FILE FOR CQL (Conservative Q-Learning) OFFLINE-TO-ONLINE TRAINING
### Based on: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
### Reference implementation: https://github.com/young-geng/CQL
### Adapted to match FASPEQ/IQL codebase structure

import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
import minari

from src.algos.agent_cql import Agent
from src.algos.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger


def print_class_attributes(obj):
    """Prints all attributes of an object along with their values."""
    attributes = vars(obj)
    for attr, value in attributes.items():
        print(f"{attr}: {value}")


def get_minari_dataset_name(env_name, quality='expert'):
    """
    Map Gymnasium environment name to Minari dataset name.
    """
    base_name = env_name.split('-')[0].lower()
    
    env_mapping = {
        'hopper': 'hopper',
        'halfcheetah': 'halfcheetah',
        'walker2d': 'walker2d',
        'ant': 'ant',
        'swimmer': 'swimmer',
        'reacher': 'reacher',
        'pusher': 'pusher',
        'invertedpendulum': 'invertedpendulum',
        'inverteddoublependulum': 'inverteddoublependulum',
        'humanoid': 'humanoid',
        'humanoidstandup': 'humanoidstandup'
    }
    
    minari_env_name = env_mapping.get(base_name, base_name)
    return f'mujoco/{minari_env_name}/{quality}-v0'


def load_minari_dataset(agent, env_name=None, quality='expert', reward_scale=1.0, reward_bias=0.0):
    """
    Load Minari dataset into agent's replay buffer.
    
    Args:
        agent: CQL agent
        env_name: Environment name
        quality: Dataset quality ('expert', 'medium', etc.)
        reward_scale: Scale factor for rewards (default: 1.0)
        reward_bias: Bias to add to rewards (default: 0.0)
    """
    dataset_name = get_minari_dataset_name(env_name, quality)
    
    print(f"Loading Minari dataset: {dataset_name}")
    print(f"Reward scaling: scale={reward_scale}, bias={reward_bias}")
    
    try:
        dataset = minari.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Attempting to download the dataset...")
        try:
            minari.download_dataset(dataset_name)
            dataset = minari.load_dataset(dataset_name)
        except Exception as e2:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {e2}")
    
    print(f"Dataset info:")
    print(f"  Total episodes: {dataset.total_episodes}")
    print(f"  Total steps: {dataset.total_steps}")
    print(f"  Observation space: {dataset.observation_space}")
    print(f"  Action space: {dataset.action_space}")
    
    total_transitions = 0
    
    for episode in dataset.iterate_episodes():
        T = len(episode.actions)
        
        for t in range(T):
            obs = episode.observations[t]
            action = episode.actions[t]
            # Apply reward scaling
            reward = episode.rewards[t] * reward_scale + reward_bias
            next_obs = episode.observations[t + 1]
            done = episode.terminations[t]
            
            agent.store_data_offline(obs, action, reward, next_obs, done)
            total_transitions += 1
    
    print(f"Successfully loaded {total_transitions} transitions from {dataset.total_episodes} episodes")
    
    return total_transitions


def CQL(env_name, seed=0, epochs=300, steps_per_epoch=1000,
        max_ep_len=1000, n_evals_per_epoch=1,
        logger_kwargs=dict(), debug=False,
        hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
        lr=3e-4, gamma=0.99, polyak=0.995,
        alpha=0.2, auto_alpha=True, target_entropy='mbpo',
        start_steps=5000, delay_update_steps='auto',
        utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
        policy_update_delay=1,
        evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
        gpu_id=0, target_drop_rate=0.0, layer_norm=False,
        # CQL specific parameters
        cql_alpha=5.0, cql_n_actions=10, cql_temp=1.0,
        cql_lagrange=False, cql_target_action_gap=1.0,
        cql_clip_diff_min=-np.inf, cql_max_target_backup=False,
        offline_pretrain_steps=1000000,
        # Reward scaling (important for some environments)
        reward_scale=1.0, reward_bias=0.0,
        # Online fine-tuning mixing ratio
        mixing_ratio=0.5,
        # Minari parameters
        use_minari=False, minari_quality='expert'
        ):
    """
    CQL - Conservative Q-Learning for Offline RL.
    
    CQL learns conservative Q-values by adding a regularizer that:
    1. Minimizes Q-values under the learned policy (OOD actions)
    2. Maximizes Q-values under the dataset distribution (in-distribution actions)
    
    CQL specific parameters:
        cql_alpha: Weight for CQL conservative regularizer (default: 5.0)
        cql_n_actions: Number of actions for CQL logsumexp (default: 10)
        cql_temp: Temperature for CQL logsumexp (default: 1.0)
        cql_lagrange: Use Lagrangian CQL (learns alpha) (default: False)
        cql_target_action_gap: Target gap for Lagrange (default: 1.0)
        offline_pretrain_steps: Number of offline pretraining steps (default: 1M)
        reward_scale: Scale factor for rewards (default: 1.0)
        reward_bias: Bias to add to rewards (default: 0.0)
        mixing_ratio: Fraction of offline data in mixed batches (default: 0.5)
    
    Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
    """
    if debug:
        hidden_sizes = [2, 2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100
        offline_pretrain_steps = 1000
        cql_n_actions = 2

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    if epochs == 'mbpo' or epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(args.env)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()

    def seed_all(epoch):
        """Seed all environments and random number generators."""
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        
        return env_seed, test_env_seed, bias_eval_env_seed

    # Initial seeding
    env_seed, test_env_seed, bias_eval_env_seed = seed_all(epoch=0)

    """prepare to init agent"""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()
    start_time = time.time()
    sys.stdout.flush()

    """init CQL agent"""
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size,
                  lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy,
                  start_steps, delay_update_steps,
                  utd_ratio, num_Q, num_min, q_target_mode,
                  policy_update_delay,
                  target_drop_rate=target_drop_rate,
                  layer_norm=layer_norm,
                  o2o=use_minari,
                  cql_alpha=cql_alpha,
                  cql_n_actions=cql_n_actions,
                  cql_temp=cql_temp,
                  cql_lagrange=cql_lagrange,
                  cql_target_action_gap=cql_target_action_gap,
                  cql_clip_diff_min=cql_clip_diff_min,
                  cql_max_target_backup=cql_max_target_backup,
                  mixing_ratio=mixing_ratio
                  )

    print_class_attributes(agent)

    # Load Minari dataset if specified
    offline_samples = 0
    if use_minari:
        offline_samples = load_minari_dataset(
            agent, 
            env_name=env_name, 
            quality=minari_quality,
            reward_scale=reward_scale,
            reward_bias=reward_bias
        )
        print(f"Loaded {offline_samples} offline transitions")
        wandb.log({"offline_samples_loaded": offline_samples}, step=0)
        
        # Run offline pretraining with CQL
        if offline_pretrain_steps > 0:
            print(f"\n{'='*80}")
            print(f"STARTING CQL OFFLINE PRETRAINING ({offline_pretrain_steps} steps)")
            print(f"{'='*80}\n")
            agent.train_offline(
                epochs=offline_pretrain_steps,
                test_env=test_env,
                current_env_step=0,
                log_interval=max(1000, offline_pretrain_steps // 100),
                max_ep_len=max_ep_len
            )

    # Initialize environment with gymnasium interface
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    print(f"\n{'='*80}")
    print(f"STARTING ONLINE FINE-TUNING ({total_steps} steps)")
    print(f"Mixing ratio: {mixing_ratio} (offline fraction in batches)")
    print(f"{'='*80}\n")

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)

        # gymnasium step() returns 5 values
        o2, r, terminated, truncated, info = env.step(a)
        
        d = terminated or truncated
        ep_len += 1
        d_store = terminated and not truncated

        # Store in main (online) buffer
        agent.store_data(o, a, r, o2, d_store)

        # ONLINE TRAINING with CQL
        agent.train(logger, current_env_step=t+1)

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, info = env.reset()
            r, ep_ret, ep_len = 0, 0, 0
            terminated, truncated = False, False

        # EPOCH LOGGING
        if (t + 1) % 1000 == 0:
            epoch = t // 1000
            current_step = t + 1

            # Evaluate agent performance
            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({
                "EvalReward": np.mean(test_rw),
            }, step=current_step)

            # Bias evaluation if enabled
            if evaluate_bias:
                normalized_bias_sqr_per_state, normalized_bias_per_state, bias = log_bias_evaluation(
                    bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff
                )
                wandb.log({
                    "normalized_bias_sqr_per_state": np.abs(np.mean(normalized_bias_sqr_per_state)),
                    "normalized_bias_per_state": np.mean(normalized_bias_per_state),
                    "bias": np.mean(bias)
                }, step=current_step)

            # Reseed if configured
            if reseed_each_epoch:
                env_seed, test_env_seed, bias_eval_env_seed = seed_all(epoch)
                o, info = env.reset(seed=env_seed)
                r, ep_ret, ep_len = 0, 0, 0
                terminated, truncated = False, False

            # Console logging
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
            logger.log_tabular('LossV', average_only=True)  # CQL loss
            logger.log_tabular('VMean', average_only=True)  # CQL diff

            if evaluate_bias:
                logger.log_tabular('TestEpBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBiasSqr', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # ─── Basic Setup ────────────────────────────────────────────────────────────
    parser.add_argument("--env", type=str, default='Hopper-v5', help="Environment name")
    parser.add_argument("--algo", type=str, default='cql', help="Algorithm name")
    parser.add_argument("--info", type=str, default='test', help="Experiment information / run name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with smaller networks")
    parser.add_argument("--log-wandb", action="store_true", help="Enable WandB logging")

    # ─── CQL specific ───────────────────────────────────────────────────────────
    # NOTE: For AntMaze, use --cql-lagrange with target_action_gap
    #       For locomotion, use fixed cql_alpha=5.0 or 10.0
    parser.add_argument("--cql-alpha", type=float, default=5.0, 
                        help="CQL conservative regularizer weight (5.0 or 10.0 typically)")
    parser.add_argument("--cql-n-actions", type=int, default=10, 
                        help="Number of actions for CQL logsumexp")
    parser.add_argument("--cql-temp", type=float, default=1.0, 
                        help="Temperature for CQL logsumexp")
    parser.add_argument("--cql-lagrange", action="store_true", 
                        help="Use Lagrangian CQL (learns alpha)")
    parser.add_argument("--cql-target-action-gap", type=float, default=1.0, 
                        help="Target action gap for Lagrange CQL")
    parser.add_argument("--cql-clip-diff-min", type=float, default=-np.inf,
                        help="Minimum value for CQL diff clipping (use -200 for AntMaze)")
    parser.add_argument("--cql-max-target-backup", action="store_true",
                        help="Use max target backup (more conservative)")
    parser.add_argument("--offline-pretrain-steps", type=int, default=1000000, 
                        help="Number of offline pretraining gradient steps")
    parser.add_argument("--start-steps", type=int, default=0, 
                        help="Random actions before learning (0 for O2O since policy is pretrained)")
    
    # ─── Reward Scaling ─────────────────────────────────────────────────────────
    # AntMaze: reward_scale=10.0, reward_bias=-5.0
    # Others:  reward_scale=1.0,  reward_bias=0.0
    parser.add_argument("--reward-scale", type=float, default=1.0, 
                        help="Scale factor for rewards (use 10.0 for AntMaze)")
    parser.add_argument("--reward-bias", type=float, default=0.0, 
                        help="Bias to add to rewards (use -5.0 for AntMaze)")
    
    # ─── Online Fine-tuning ─────────────────────────────────────────────────────
    parser.add_argument("--mixing-ratio", type=float, default=0.5, 
                        help="Fraction of offline data in mixed batches (0-1)")

    # ─── Network & Optimization ─────────────────────────────────────────────────
    parser.add_argument("--network-width", type=int, default=256, help="Hidden units per layer")
    parser.add_argument("--num-hidden-layers", type=int, default=2, 
                        help="Number of hidden layers (use 3 for AntMaze Q-networks)")
    parser.add_argument("--num-q", type=int, default=2, help="Number of Q-networks")
    parser.add_argument("--utd-ratio", type=int, default=1, help="Update-to-data ratio")
    parser.add_argument("--target-drop-rate", type=float, default=0.0, help="Dropout rate")
    
    # ─── Minari Offline-to-Online ───────────────────────────────────────────────
    parser.add_argument("--use-minari", action="store_true", 
                        help="Load Minari dataset for offline-to-online training")
    parser.add_argument("--minari-quality", type=str, default='expert', 
                        choices=['expert', 'medium', 'simple'], help="Dataset quality")

    # ─── Boolean Toggles ────────────────────────────────────────────────────────
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false", 
                        help="Disable layer normalization")
    parser.add_argument("--evaluate-bias", action="store_true", 
                        help="Evaluate policy bias during training")

    args = parser.parse_args()

    
    if args.use_minari:
        exp_name_full = args.algo + '_%s' % args.env + '_%s' % args.minari_quality
    else:
        exp_name_full = args.algo + '_%s' % args.env

    args.data_dir = './runs/' + str(args.info) + '/'
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = tuple([args.network_width] * args.num_hidden_layers)

    params = {
        'exp_name': exp_name_full,
        'seed': args.seed,
        'env': args.env,
        'epochs': args.epochs,
        'gpu_id': args.gpu_id,
        'debug': args.debug,
        'evaluate_bias': args.evaluate_bias,
        'layer_norm': args.layer_norm,
        'hidden_sizes': hidden_sizes,
        'num_q': args.num_q,
        'utd_ratio': args.utd_ratio,
        'dropout': args.target_drop_rate,
        'use_minari': args.use_minari,
        'minari_quality': args.minari_quality,
        'cql_alpha': args.cql_alpha,
        'cql_n_actions': args.cql_n_actions,
        'cql_temp': args.cql_temp,
        'cql_lagrange': args.cql_lagrange,
        'cql_target_action_gap': args.cql_target_action_gap,
        'cql_clip_diff_min': args.cql_clip_diff_min,
        'cql_max_target_backup': args.cql_max_target_backup,
        'offline_pretrain_steps': args.offline_pretrain_steps,
        'reward_scale': args.reward_scale,
        'reward_bias': args.reward_bias,
        'mixing_ratio': args.mixing_ratio,
        'start_steps': args.start_steps
    }

    wandb.init(
        name=f'{exp_name_full}',
        project="SPEQ",
        config=params,
        mode='online' if args.log_wandb else 'disabled',
        save_code=True
    )

    CQL(env_name=params['env'], 
        seed=params['seed'], 
        epochs=params['epochs'],
        logger_kwargs=logger_kwargs, 
        debug=params['debug'],
        gpu_id=params['gpu_id'],
        target_drop_rate=params['dropout'],
        layer_norm=params['layer_norm'],
        num_Q=params['num_q'],
        hidden_sizes=params['hidden_sizes'],
        evaluate_bias=params['evaluate_bias'],
        use_minari=params['use_minari'],
        minari_quality=params['minari_quality'],
        utd_ratio=params['utd_ratio'],
        cql_alpha=params['cql_alpha'],
        cql_n_actions=params['cql_n_actions'],
        cql_temp=params['cql_temp'],
        cql_lagrange=params['cql_lagrange'],
        cql_target_action_gap=params['cql_target_action_gap'],
        cql_clip_diff_min=params['cql_clip_diff_min'],
        cql_max_target_backup=params['cql_max_target_backup'],
        offline_pretrain_steps=params['offline_pretrain_steps'],
        reward_scale=params['reward_scale'],
        reward_bias=params['reward_bias'],
        mixing_ratio=params['mixing_ratio'],
        start_steps=params['start_steps']
        )
