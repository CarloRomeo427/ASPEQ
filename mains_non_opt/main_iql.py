### MAIN FILE FOR IQL (Implicit Q-Learning) OFFLINE-TO-ONLINE TRAINING
### Adapted from gwthomas/IQL-PyTorch to match FASPEQ codebase structure

import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
import minari

from src.algos_non_opt.agent_iql import Agent
from src.algos_non_opt.core import mbpo_epoches, test_agent
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


def load_minari_dataset(agent, env_name=None, quality='expert'):
    """
    Load Minari dataset into agent's replay buffer.
    """
    dataset_name = get_minari_dataset_name(env_name, quality)
    
    print(f"Loading Minari dataset: {dataset_name}")
    
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
            # Explicit float32 conversion to avoid dtype issues with float64 observations
            obs = episode.observations[t].astype(np.float32)
            action = episode.actions[t].astype(np.float32)
            reward = float(episode.rewards[t])
            next_obs = episode.observations[t + 1].astype(np.float32)
            done = episode.terminations[t]
            
            agent.store_data_offline(obs, action, reward, next_obs, done)
            total_transitions += 1
    
    print(f"Successfully loaded {total_transitions} transitions from {dataset.total_episodes} episodes")
    
    return total_transitions


def IQL(env_name, seed=0, epochs=300, steps_per_epoch=1000,
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
        # IQL-specific parameters
        iql_tau=0.7, iql_beta=3.0, offline_pretrain_steps=1000000,
        # Minari parameters
        use_minari=False, minari_quality='expert'
        ):
    """
    IQL - Implicit Q-Learning for Offline-to-Online RL.
    
    IQL-specific parameters:
        iql_tau: Expectile for value function training (default: 0.7)
                 - Higher values (0.7-0.9) focus more on high-value actions
                 - 0.5 = standard mean, approaching 1.0 = max
        iql_beta: Temperature for advantage-weighted policy extraction (default: 3.0)
                  - Lower values make policy more greedy
                  - Higher values give softer/more exploratory policy
        offline_pretrain_steps: Number of offline pretraining gradient steps (default: 1M)
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

    """init IQL agent"""
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
                  iql_tau=iql_tau,
                  iql_beta=iql_beta
                  )

    print_class_attributes(agent)

    # Load Minari dataset if specified
    offline_samples = 0
    if use_minari:
        offline_samples = load_minari_dataset(
            agent, 
            env_name=env_name, 
            quality=minari_quality
        )
        print(f"Loaded {offline_samples} offline transitions")
        wandb.log({"offline_samples_loaded": offline_samples}, step=0)
        
        # Run offline pretraining
        if offline_pretrain_steps > 0:
            print(f"\n{'='*80}")
            print(f"STARTING OFFLINE PRETRAINING ({offline_pretrain_steps} steps)")
            print(f"{'='*80}\n")
            agent.train_offline(
                epochs=offline_pretrain_steps,
                test_env=test_env,
                current_env_step=0,
                log_interval=max(1000, offline_pretrain_steps // 100)
            )

    # Initialize environment with gymnasium interface
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)

        # gymnasium step() returns 5 values
        o2, r, terminated, truncated, info = env.step(a)
        
        d = terminated or truncated
        ep_len += 1
        d_store = terminated and not truncated

        # Store in main buffer
        agent.store_data(o, a, r, o2, d_store)

        # ONLINE TRAINING
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
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('VMean', average_only=True)

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
    parser.add_argument("--algo", type=str, default='iql', help="Algorithm name")
    parser.add_argument("--info", type=str, default='test', help="Experiment information / run name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with smaller networks")
    parser.add_argument("--log-wandb", action="store_true", help="Enable WandB logging")

    # ─── IQL-specific ───────────────────────────────────────────────────────────
    parser.add_argument("--iql-tau", type=float, default=0.7, help="Expectile for value function (0.5-0.99)")
    parser.add_argument("--iql-beta", type=float, default=3.0, help="Temperature for AWR policy extraction")
    parser.add_argument("--offline-pretrain-steps", type=int, default=1000000, help="Number of offline pretraining gradient steps")
    parser.add_argument("--start-steps", type=int, default=5000, help="Number of random actions before learning starts")

    # ─── Network & Optimization ─────────────────────────────────────────────────
    parser.add_argument("--network-width", type=int, default=256, help="Hidden units per layer")
    parser.add_argument("--num-q", type=int, default=2, help="Number of Q-networks")
    parser.add_argument("--utd-ratio", type=int, default=1, help="Update-to-data ratio")
    parser.add_argument("--target-drop-rate", type=float, default=0.0, help="Dropout rate")
    
    # ─── Minari Offline-to-Online ───────────────────────────────────────────────
    parser.add_argument("--use-minari", action="store_true", help="Load Minari dataset for offline-to-online training")
    parser.add_argument("--minari-quality", type=str, default='expert', choices=['expert', 'medium', 'simple'], help="Dataset quality")

    # ─── Boolean Toggles ────────────────────────────────────────────────────────
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false", help="Disable layer normalization")
    parser.add_argument("--evaluate-bias", action="store_true", help="Evaluate policy bias during training")

    args = parser.parse_args()

    
    if args.use_minari:
        exp_name_full = args.algo + '_%s' % args.env + '_%s' % args.minari_quality
    else:
        exp_name_full = args.algo + '_%s' % args.env

    args.data_dir = './runs/' + str(args.info) + '/'
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)

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
        'iql_tau': args.iql_tau,
        'iql_beta': args.iql_beta,
        'offline_pretrain_steps': args.offline_pretrain_steps,
        'start_steps': args.start_steps
    }

    wandb.init(
        name=f'{exp_name_full}',
        project="SPEQ",
        config=params,
        mode='online' if args.log_wandb else 'disabled',
        save_code=True
    )

    IQL(env_name=params['env'], 
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
        iql_tau=params['iql_tau'],
        iql_beta=params['iql_beta'],
        offline_pretrain_steps=params['offline_pretrain_steps'],
        start_steps=params['start_steps']
        )