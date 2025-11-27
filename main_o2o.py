### MAIN FILE FOR SPEQ OFFLINE-TO-ONLINE TRAINING USING MINARI DATASETS


import gymnasium as gym
import numpy as np
import torch
import time
import sys
import mujoco_py
import wandb
import minari

from src.algos.agent import Agent
from src.algos.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger


def print_class_attributes(obj):
    """
    Prints all attributes of an object along with their values.

    Parameters:
    obj (object): The object whose attributes need to be printed.
    """
    attributes = vars(obj)
    for attr, value in attributes.items():
        print(f"{attr}: {value}")


def get_minari_dataset_name(env_name, quality='expert'):
    """
    Map Gymnasium environment name to Minari dataset name.
    
    Args:
        env_name: Gymnasium environment name (e.g., 'Hopper-v5', 'HalfCheetah-v4')
        quality: Dataset quality level ('expert', 'medium', 'simple')
    
    Returns:
        Minari dataset name (e.g., 'mujoco/hopper/expert-v0')
    """
    # Extract base environment name (remove version suffix)
    base_name = env_name.split('-')[0].lower()
    
    # Map common environment names to Minari naming convention
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


def load_minari_dataset(agent, dataset_name=None, env_name=None, quality='expert'):
    """
    Load Minari dataset into agent's replay buffer.
    
    Args:
        agent: The SPEQ agent with a replay buffer
        dataset_name: Specific Minari dataset name (e.g., 'mujoco/hopper/expert-v0')
        env_name: Gymnasium environment name (used if dataset_name not provided)
        quality: Dataset quality level ('expert', 'medium', 'simple')
    
    Returns:
        Number of transitions loaded
    """
    # Determine dataset name
    if dataset_name is None:
        if env_name is None:
            raise ValueError("Either dataset_name or env_name must be provided")
        dataset_name = get_minari_dataset_name(env_name, quality)
    
    print(f"Loading Minari dataset: {dataset_name}")
    
    # Load the dataset (will download if not present locally)
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
    
    # Extract transitions from episodes and add to replay buffer
    total_transitions = 0
    
    for episode in dataset.iterate_episodes():
        # episode.observations has shape (T+1, obs_dim)
        # episode.actions has shape (T, act_dim)
        T = len(episode.actions)
        
        for t in range(T):
            obs = episode.observations[t]
            action = episode.actions[t]
            reward = episode.rewards[t]
            next_obs = episode.observations[t + 1]
            
            # In Gymnasium, done = terminated OR truncated
            # For replay buffer, we typically want to mark episodes as done
            # only if truly terminated (not truncated by time limit)
            # to avoid improper bootstrapping
            done = episode.terminations[t]
            
            agent.store_data_offline(obs, action, reward, next_obs, done)
            total_transitions += 1
    
    print(f"Successfully loaded {total_transitions} transitions from {dataset.total_episodes} episodes")
    
    return total_transitions


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
         gpu_id=0, target_drop_rate=0.0, layer_norm=False, offline_frequency=1000,
         offline_epochs=100, use_minari=False, minari_dataset=None, minari_quality='expert',
        #  initial_offline_training=True
         ):
    if debug:
        hidden_sizes = [2, 2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

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
        """
        Seed all environments and random number generators.
        In gymnasium, seeding is done through reset() with seed parameter.
        """
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        
        # Seed PyTorch and NumPy
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        
        # Store seeds for use in reset() - gymnasium uses reset(seed=...) instead of env.seed()
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

    

    """init agent and start training"""
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size,
                  lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy,
                  start_steps, delay_update_steps,
                  utd_ratio, num_Q, num_min, q_target_mode,
                  policy_update_delay,
                  target_drop_rate=target_drop_rate,
                  layer_norm=layer_norm,
                  o2o = args.use_minari
                  )

    print_class_attributes(agent)

    # Load Minari dataset if specified
    offline_samples = 0
    if use_minari:
        offline_samples = load_minari_dataset(
            agent, 
            dataset_name=minari_dataset, 
            env_name=env_name, 
            quality=minari_quality
        )
        print(f"Loaded {offline_samples} offline transitions")
        wandb.log({"offline_samples_loaded": offline_samples}, step=0)
        
        # # Optionally: do initial offline training on Minari data
        # if initial_offline_training and offline_samples > start_steps:
        #     print(f"Performing initial offline training on Minari data ({offline_epochs} epochs)...")
        #     agent.finetune_offline(epochs=offline_epochs, test_env=test_env, current_env_step=0)
        #     print("Initial offline training complete")

    # Initialize environment with gymnasium interface
    # reset() now returns (observation, info) tuple
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)

        # gymnasium step() returns 5 values: (obs, reward, terminated, truncated, info)
        o2, r, terminated, truncated, info = env.step(a)
        
        # Episode is done if either terminated or truncated
        d = terminated or truncated

        ep_len += 1
        # In gymnasium, truncated specifically indicates time limit cutoff
        # We only want to mark as 'done' if truly terminated, not truncated
        d_store = terminated and not truncated

        agent.store_data(o, a, r, o2, d_store)

        # OFFLINE TRAINING: Pass current step for proper logging
        if offline_frequency > 0 and (t + 1) % offline_frequency == 0:
            agent.finetune_offline(epochs=offline_epochs, test_env=test_env, current_env_step=t+1)

        # ONLINE TRAINING: Pass current step for proper logging
        agent.train(logger, current_env_step=t+1)

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            # Reset returns (observation, info) in gymnasium
            o, info = env.reset()
            r, ep_ret, ep_len = 0, 0, 0
            terminated, truncated = False, False

        # EPOCH LOGGING: Always at 1000-step intervals based on environment steps
        if (t + 1) % 1000 == 0:
            epoch = t // 1000
            current_step = t + 1

            # Evaluate agent performance
            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({"EvalReward": np.mean(test_rw)}, step=current_step)

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
                
            # Effective rank metrics every 5 epochs
            if epoch % 5 == 0 and args.calc_plasticity:
                rank_metrics = agent.compute_effective_rank_metrics(batch_size=256)
                wandb.log(rank_metrics, step=current_step)

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

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            sys.stdout.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or evaluate SPEQ on MuJoCo environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ─── Environment & Experiment ─────────────────────────────────────────────────
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v5",
        help="Gym environment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="speq_o2o",
        help="Experiment name (used for checkpoints, logs, etc.)"
    )
    parser.add_argument(
        "--info",
        type=str,
        help="Path to experiment folder (for resuming or analysis)"
    )

    # ─── Logging & Debug ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (more verbose logging)"
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log metrics to Weights & Biases"
    )

    # ─── Hardware ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID"
    )

    # ─── Training Parameters ──────────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--offline-frequency",
        type=int,
        default=10_000,
        help="Offline update frequency (in steps)"
    )
    parser.add_argument(
        "--offline-epochs",
        type=int,
        default=75_000,
        help="Number of offline update epochs"
    )
    parser.add_argument(
        "--utd",
        type=int,
        default=1,
        help="Update-to-data ratio"
    )

    # ─── Network & Optimization ────────────────────────────────────────────────────
    parser.add_argument(
        "--network-width",
        type=int,
        default=256,
        help="Hidden units per layer"
    )
    parser.add_argument(
        "--num-q",
        type=int,
        default=2,
        help="Number of Q-networks (for ensembles)"
    )
    parser.add_argument(
        "--target-drop-rate",
        type=float,
        default=1e-4,
        help="Dropout rate for the target value network"
    )
    # ─── Plasticity ─────────────────────────────────────────────────
    parser.add_argument(
        "--calc-plasticity",
        action="store_true",
        help="Calculate plasticity metrics during training"
    )
    # ─── Minari Offline-to-Online ─────────────────────────────────────────────────
    parser.add_argument(
        "--use-minari",
        action="store_true",
        help="Load Minari dataset for offline-to-online training"
    )
    parser.add_argument(
        "--minari-dataset",
        type=str,
        default=None,
        help="Specific Minari dataset name (e.g., 'mujoco/hopper/expert-v0'). If None, auto-generates from env name and quality"
    )
    parser.add_argument(
        "--minari-quality",
        type=str,
        default='expert',
        choices=['expert', 'medium', 'simple'],
        help="Dataset quality level (expert, medium, simple)"
    )
    # parser.add_argument(
    #     "--no-initial-offline-training",
    #     dest="initial_offline_training",
    #     action="store_false",
    #     help="Skip initial offline training phase on Minari data"
    # )
    # parser.set_defaults(initial_offline_training=True)

    # ─── Boolean Toggles ──────────────────────────────────────────────────────────
    # layer_norm defaults to True, but you can turn it off explicitly:
    parser.set_defaults(layer_norm=True)
    parser.add_argument(
        "--no-layer-norm",
        dest="layer_norm",
        action="store_false",
        help="Disable layer normalization in the networks"
    )
    parser.add_argument(
        "--evaluate-bias",
        action="store_true",
        help="Evaluate policy bias during training"
    )

    args = parser.parse_args()

    exp_name_full = args.exp_name + '_%s' % args.env + '_%s' % args.minari_quality 
    args.data_dir = './runs/' + str(args.info) + '/'
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)

    wandb.init(
        name=f'{exp_name_full}',
        project="SPEQ",
        config=args,
        mode='online' if args.log_wandb else 'disabled',
        save_code=True
    )

    SPEQ(args.env, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs, debug=args.debug,
         gpu_id=args.gpu_id,
         target_drop_rate=args.target_drop_rate,
         layer_norm=args.layer_norm,
         offline_frequency=args.offline_frequency, num_Q=args.num_q,
         offline_epochs=args.offline_epochs,
         hidden_sizes=hidden_sizes,
         evaluate_bias=args.evaluate_bias,
         use_minari=args.use_minari,
         minari_dataset=args.minari_dataset,
         minari_quality=args.minari_quality,
        #  initial_offline_training=args.initial_offline_training
         )