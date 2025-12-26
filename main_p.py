### MAIN FILE FOR SPEQ OFFLINE-TO-ONLINE TRAINING WITH ADAPTIVE TRIGGERING


import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
import minari

from src.algos.agent_p import Agent
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


def load_minari_dataset(agent, env_name=None, quality='expert'):
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

def set_dropout(env_name, target_drop_rate=0.0):
    if target_drop_rate == 0.0:
        return 0.0
    else:
        if env_name == "Humanoid-v5":
            return 0.1
        elif env_name == "Ant-v5":
            return 0.01
        elif env_name in ["Walker2d-v5", "HalfCheetah-v5", "Pusher-v5"]:
            return 0.005
        elif env_name in ["Hopper-v5", "Swimmer-v5", "Reacher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]:
            return 0.001


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
         offline_epochs=100, use_minari=False, minari_quality='expert',
         val_buffer_prob=0.1, val_buffer_offline_frac=0.1,
         val_check_interval=1000, val_patience=5000,
         adaptive_trigger_expansion_rate=1.1, auto_stab=False, policy_based=False
         ):
    """
    SPEQ with adaptive offline stabilization triggering.
    
    NEW PARAMETERS:
        val_buffer_prob: Probability of adding online transitions to validation buffer (default: 0.1)
        val_buffer_offline_frac: Fraction of offline data to add to validation buffer (default: 0.1)
        val_check_interval: Steps between validation checks during offline training (default: 1000)
        val_patience: Steps without validation improvement before early stopping (default: 5000)
        adaptive_trigger_expansion_rate: Buffer growth rate for adaptive triggering (default: 1.1)
        policy_based: If True, monitor policy loss instead of Q loss (PASPEQ/APASPEQ) (default: False)
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

    """init agent with new validation parameters"""
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size,
                  lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy,
                  start_steps, delay_update_steps,
                  utd_ratio, num_Q, num_min, q_target_mode,
                  policy_update_delay,
                  target_drop_rate=target_drop_rate,
                  layer_norm=layer_norm,
                  o2o=args.use_minari,
                  val_buffer_prob=val_buffer_prob,
                  val_buffer_offline_frac=val_buffer_offline_frac,
                  val_check_interval=val_check_interval,
                  val_patience=val_patience,
                  adaptive_trigger_expansion_rate=adaptive_trigger_expansion_rate,
                auto_stab=auto_stab
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
        
        # NEW: Populate validation buffer with offline data
        agent.populate_val_buffer_from_offline()
        wandb.log({"val_buffer_size": agent.val_replay_buffer.size}, step=0)

    # Initialize environment with gymnasium interface
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    # Track stabilization statistics
    total_stabilization_triggers = 0
    total_stabilization_epochs = 0

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

        # Store in main buffer and probabilistically in validation buffer
        agent.store_data(o, a, r, o2, d_store)

        # NEW: ADAPTIVE OFFLINE STABILIZATION TRIGGERING
        if agent.check_should_trigger_offline_stabilization():
            total_stabilization_triggers += 1
            
            # Log trigger information
            wandb.log({
                "stabilization_trigger": total_stabilization_triggers,
                "buffer_size_at_trigger": agent.replay_buffer.size,
                "next_trigger_size": agent.next_trigger_size
            }, step=t+1)
            
            # Run offline stabilization - choose method based on algorithm type
            if policy_based:
                # PASPEQ/APASPEQ: Monitor policy loss with fixed policy
                if auto_stab:
                    # APASPEQ: Use validation set
                    epochs_performed = agent.finetune_offline_policy_auto(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
                else:
                    # PASPEQ: Use training buffer
                    epochs_performed = agent.finetune_offline_policy(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
            else:
                # SPEQ/ASPEQ: Monitor Q loss
                if auto_stab:
                    epochs_performed = agent.finetune_offline_auto(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
                else:
                    epochs_performed = agent.finetune_offline(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
            total_stabilization_epochs += epochs_performed
            
            # Log stabilization statistics
            wandb.log({
                "stabilization_epochs_performed": epochs_performed,
                "total_stabilization_epochs": total_stabilization_epochs,
                "avg_epochs_per_stabilization": total_stabilization_epochs / total_stabilization_triggers
            }, step=t+1)

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
            wandb.log({
                "EvalReward": np.mean(test_rw),
                "val_buffer_size": agent.val_replay_buffer.size
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
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total stabilization triggers: {total_stabilization_triggers}")
    print(f"Total stabilization epochs: {total_stabilization_epochs}")
    print(f"Average epochs per stabilization: {total_stabilization_epochs / max(1, total_stabilization_triggers):.1f}")
    print(f"Final validation buffer size: {agent.val_replay_buffer.size}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or evaluate SPEQ on MuJoCo environments with adaptive triggering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ─── Environment & Experiment ─────────────────────────────────────────────────
    parser.add_argument("--algo",type=str, default="sac",choices=["sac", "droq", "speq", "aspeq", "speq_o2o", "aspeq_o2o", "paspeq", "apaspeq", "paspeq_o2o", "apaspeq_o2o", "redq", "rlpd"],help="RL algorithm to use")
    parser.add_argument("--env",type=str,default="Hopper-v5",help="Gym environment name")
    parser.add_argument("--seed",type=int,default=0,help="Random seed for reproducibility")
    parser.add_argument("--exp-name",type=str,default="speq_o2o_adaptive",help="Experiment name (used for checkpoints, logs, etc.)")
    parser.add_argument("--info",type=str,help="Path to experiment folder (for resuming or analysis)")

    # ─── Logging & Debug ──────────────────────────────────────────────────────────
    parser.add_argument("--debug",action="store_true",help="Enable debug mode (more verbose logging)")
    parser.add_argument("--log-wandb",action="store_true",help="Log metrics to Weights & Biases")

    # ─── Hardware ─────────────────────────────────────────────────────────────────
    parser.add_argument("--gpu-id",type=int,default=0,help="CUDA GPU device ID")

    # ─── Training Parameters ──────────────────────────────────────────────────────
    parser.add_argument("--epochs",type=int,default=300,help="Number of training epochs")
    parser.add_argument("--offline-epochs",type=int,default=75_000,help="Maximum number of offline update epochs per stabilization")
    parser.add_argument("--offline-frequency",type=int,default=10_000,help="Number of environment steps between offline stabilizations")
    parser.add_argument("--utd",type=int,default=1,help="Update-to-data ratio")

    # ─── NEW: Validation Parameters ───────────────────────────────────────────────
    parser.add_argument("--val-buffer-prob",type=float,default=0.1,help="Probability of adding online transitions to validation buffer")
    parser.add_argument("--val-buffer-offline-frac",type=float,default=0.1,help="Fraction of offline data to add to validation buffer")
    parser.add_argument("--val-check-interval",type=int,default=1000,help="Steps between validation checks during offline training")
    parser.add_argument("--val-patience",type=int,default=5000,help="Steps without validation improvement before early stopping")

    # ─── NEW: Adaptive Triggering ─────────────────────────────────────────────────
    parser.add_argument("--adaptive-trigger-rate",type=float,default=1.1,help="Buffer growth rate for adaptive triggering (10%% = 1.1)")

    # ─── Network & Optimization ────────────────────────────────────────────────────
    parser.add_argument("--network-width",type=int,default=256,help="Hidden units per layer")
    parser.add_argument("--num-q",type=int,default=2,help="Number of Q-networks (for ensembles)")
    parser.add_argument("--target-drop-rate",type=float,default=999.0,help="Dropout rate for the target value network")
    
    # ─── Plasticity ─────────────────────────────────────────────────
    parser.add_argument("--calc-plasticity",action="store_true",help="Calculate plasticity metrics during training")
    
    # ─── Minari Offline-to-Online ─────────────────────────────────────────────────
    parser.add_argument("--use-minari",action="store_true",help="Load Minari dataset for offline-to-online training")
    parser.add_argument("--minari-quality",type=str,default='expert',choices=['expert', 'medium', 'simple'],help="Dataset quality level (expert, medium, simple)")

    # ─── Boolean Toggles ──────────────────────────────────────────────────────────
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm",dest="layer_norm",action="store_false",help="Disable layer normalization in the networks")
    parser.add_argument("--evaluate-bias",action="store_true",help="Evaluate policy bias during training")

    args = parser.parse_args()

    
    
    if args.use_minari:
        exp_name_full = args.algo + '_%s' % args.env + '_%s' % args.minari_quality
    else:
        exp_name_full = args.algo + '_%s' % args.env

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
            'offline_frequency': 0,
            'utd_ratio': 1,
            'val_buffer_prob': 0.0,
            'val_buffer_offline_frac': 0.0,
            'val_check_interval': 0,
            'val_patience': 0,
            'adaptive_trigger_rate': 1.0,
            'dropout': 0.0,
            'use_minari': False,
            'minari_quality': 'expert',
            'auto_stab': False,
            'policy_based': False
        }
        name_algo = args.algo.lower()
        if name_algo == "sac":
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
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            return params
        elif name_algo == "speq":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = 10000
            params['dropout'] = auto_dropout
            return params
        # elif name_algo == "aspeq":
        #     params['offline_epochs'] = 75_000
        #     params['val_buffer_prob'] = 0.1
        #     params['val_buffer_offline_frac'] = 0.1
        #     params['val_check_interval'] = 1000
        #     params['val_patience'] = 5000
        #     params['adaptive_trigger_rate'] = 1.1
        #     params['dropout'] = auto_dropout
        #     params['auto_stab'] = True
        #     return params   
        elif name_algo == "speq_o2o":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = 10000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            return params
        # elif name_algo == "aspeq_o2o":
        #     params['offline_epochs'] = 75_000
        #     params['val_buffer_prob'] = 0.1
        #     params['val_buffer_offline_frac'] = 0.1
        #     params['val_check_interval'] = 1000
        #     params['val_patience'] = 5000
        #     params['adaptive_trigger_rate'] = 1.1
        #     params['dropout'] = auto_dropout
        #     params['use_minari'] = True
        #     params['minari_quality'] = args.minari_quality
        #     params['auto_stab'] = True
        #     return params
        elif name_algo == "paspeq":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 5000
            params['dropout'] = auto_dropout
            params['policy_based'] = True
            return params
        # elif name_algo == "apaspeq":
        #     params['offline_epochs'] = 75_000
        #     params['val_buffer_prob'] = 0.1
        #     params['val_buffer_offline_frac'] = 0.1
        #     params['val_check_interval'] = 1000
        #     params['val_patience'] = 5000
        #     params['adaptive_trigger_rate'] = 1.1
        #     params['dropout'] = auto_dropout
        #     params['auto_stab'] = True
        #     params['policy_based'] = True
        #     return params
        elif name_algo == "paspeq_o2o":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 10000 #5000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            params['policy_based'] = True
            return params
        # elif name_algo == "apaspeq_o2o":
        #     params['offline_epochs'] = 75_000
        #     params['val_buffer_prob'] = 0.1
        #     params['val_buffer_offline_frac'] = 0.1
        #     params['val_check_interval'] = 1000
        #     params['val_patience'] = 5000
        #     params['adaptive_trigger_rate'] = 1.1
        #     params['dropout'] = auto_dropout
        #     params['use_minari'] = True
        #     params['minari_quality'] = args.minari_quality
        #     params['auto_stab'] = True
        #     params['policy_based'] = True
        #     return params
        else:
            raise ValueError(f"Unknown algorithm: {args.algo}")

    params = set_algo_params(args) 

    

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
         use_minari=params['use_minari'],
         minari_quality=params['minari_quality'],
         val_buffer_prob=params['val_buffer_prob'],
         val_buffer_offline_frac=params['val_buffer_offline_frac'],
         val_check_interval=params['val_check_interval'],
         val_patience=params['val_patience'],
         adaptive_trigger_expansion_rate=params['adaptive_trigger_rate'],
         utd_ratio=params['utd_ratio'], auto_stab=params['auto_stab'],
         policy_based=params['policy_based']
         )