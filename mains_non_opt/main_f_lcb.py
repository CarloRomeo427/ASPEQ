### MAIN FILE FOR FASPEQ OFFLINE-TO-ONLINE TRAINING
### Updated for FASPEQ-LCB Experiment

import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
import minari

from src.algos_non_opt.agent_f_lcb import Agent
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
    
    print(f"Dataset info: {dataset.total_episodes} episodes, {dataset.total_steps} steps")
    total_transitions = 0
    for episode in dataset.iterate_episodes():
        T = len(episode.actions)
        for t in range(T):
            obs = episode.observations[t]
            action = episode.actions[t]
            reward = episode.rewards[t]
            next_obs = episode.observations[t + 1]
            done = episode.terminations[t]
            agent.store_data_offline(obs, action, reward, next_obs, done)
            total_transitions += 1
    
    print(f"Successfully loaded {total_transitions} transitions")
    return total_transitions


def set_dropout(env_name, target_drop_rate=0.0):
    if target_drop_rate == 0.0:
        return 0.0
    else:
        if env_name == "Humanoid-v5": return 0.1
        elif env_name == "Ant-v5": return 0.01
        elif env_name in ["Walker2d-v5", "HalfCheetah-v5", "Pusher-v5"]: return 0.005
        elif env_name in ["Hopper-v5", "Swimmer-v5", "Reacher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]: return 0.001


def FASPEQ(env_name, seed=0, epochs=300, steps_per_epoch=1000,
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
           adaptive_trigger_expansion_rate=1.1, auto_stab=False, 
           policy_based=False, faspeq_mode=False, num_reference_batches=10,
           faspeq_lcb_mode=False # NEW FLAG
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
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        return env_seed, test_env_seed, bias_eval_env_seed

    env_seed, test_env_seed, bias_eval_env_seed = seed_all(epoch=0)

    """prepare to init agent"""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()
    start_time = time.time()
    sys.stdout.flush()

    """init agent with FASPEQ parameters"""
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
                  auto_stab=auto_stab,
                  num_reference_batches=num_reference_batches,
                  faspeq_lcb_mode=faspeq_lcb_mode # Pass LCB mode
                  )

    print_class_attributes(agent)

    # Load Minari dataset if specified
    offline_samples = 0
    if use_minari:
        offline_samples = load_minari_dataset(agent, env_name=env_name, quality=minari_quality)
        wandb.log({"offline_samples_loaded": offline_samples}, step=0)
        # Populate validation buffer with offline data (for non-FASPEQ methods)
        agent.populate_val_buffer_from_offline()

    # Initialize environment
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    total_stabilization_triggers = 0
    total_stabilization_epochs = 0

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)
        o2, r, terminated, truncated, info = env.step(a)
        d = terminated or truncated
        ep_len += 1
        d_store = terminated and not truncated

        agent.store_data(o, a, r, o2, d_store)

        # ADAPTIVE OFFLINE STABILIZATION TRIGGERING
        if agent.check_should_trigger_offline_stabilization():
            total_stabilization_triggers += 1
            wandb.log({
                "stabilization_trigger": total_stabilization_triggers,
                "buffer_size_at_trigger": agent.replay_buffer.size,
                "next_trigger_size": agent.next_trigger_size
            }, step=t+1)
            
            # Run offline stabilization - choose method based on algorithm type
            if faspeq_lcb_mode:
                 # NEW EXPERIMENT: MONITOR LCB
                 epochs_performed = agent.finetune_offline_lcb(
                    epochs=offline_epochs, 
                    test_env=test_env, 
                    current_env_step=t+1
                 )
            elif faspeq_mode:
                # FASPEQ: Use static reference batches and policy loss
                epochs_performed = agent.finetune_offline_policy_faspeq(
                    epochs=offline_epochs, 
                    test_env=test_env, 
                    current_env_step=t+1
                )
            elif policy_based:
                # PASPEQ/APASPEQ
                if auto_stab:
                    epochs_performed = agent.finetune_offline_policy_auto(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
                else:
                    epochs_performed = agent.finetune_offline_policy(
                        epochs=offline_epochs, 
                        test_env=test_env, 
                        current_env_step=t+1
                    )
            else:
                # SPEQ/ASPEQ
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
            wandb.log({
                "stabilization_epochs_performed": epochs_performed,
                "total_stabilization_epochs": total_stabilization_epochs
            }, step=t+1)

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

            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({"EvalReward": np.mean(test_rw)}, step=current_step)

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
                o, info = env.reset(seed=env_seed)
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
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default='Hopper-v5',help="Environment name")
    parser.add_argument("--algo",type=str,default='faspeq_lcb',help="Algorithm (faspeq_lcb, faspeq_o2o, etc.)")
    parser.add_argument("--info",type=str,default='test',help="Experiment information")
    parser.add_argument("--seed",type=int,default=0,help="Random seed")
    parser.add_argument("--epochs",type=int,default=300,help="Training epochs")
    parser.add_argument("--gpu-id",type=int,default=0,help="GPU device ID")
    parser.add_argument("--debug",action="store_true",help="Debug mode")
    parser.add_argument("--log-wandb",action="store_true",help="WandB logging")
    parser.add_argument("--offline-epochs",type=int,default=75000,help="Max epochs per offline phase")
    parser.add_argument("--offline-frequency",type=int,default=10000,help="Trigger stabilization steps")
    parser.add_argument("--val-buffer-prob",type=float,default=0.1,help="Validation probability")
    parser.add_argument("--val-buffer-offline-frac",type=float,default=0.1,help="Validation offline frac")
    parser.add_argument("--val-check-interval",type=int,default=1000,help="Validation check interval")
    parser.add_argument("--val-patience",type=int,default=5000,help="Validation patience")
    parser.add_argument("--adaptive-trigger-rate",type=float,default=1.1,help="Buffer growth rate")
    parser.add_argument("--num-reference-batches",type=int,default=10,help="Num reference batches")
    parser.add_argument("--network-width",type=int,default=256,help="Network width")
    parser.add_argument("--num-q",type=int,default=2,help="Number of Q-networks")
    parser.add_argument("--target-drop-rate",type=float,default=999.0,help="Dropout rate")
    parser.add_argument("--use-minari",action="store_true",help="Load Minari dataset")
    parser.add_argument("--minari-quality",type=str,default='expert',choices=['expert', 'medium', 'simple'],help="Dataset quality")
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm",dest="layer_norm",action="store_false",help="Disable layer norm")
    parser.add_argument("--evaluate-bias",action="store_true",help="Evaluate policy bias")

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
            'policy_based': False,
            'faspeq_mode': False,
            'faspeq_lcb_mode': False,
            'num_reference_batches': 10
        }
        name_algo = args.algo.lower()
        
        # New LCB Algorithm
        if name_algo == "faspeq_lcb":
            params['algo'] = "faspeq_lcb"
            params['faspeq_lcb_mode'] = True
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 5000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            params['num_reference_batches'] = args.num_reference_batches
            return params

        # Existing Algorithms
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
        elif name_algo == "speq_o2o":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = 10000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            return params
        elif name_algo == "paspeq":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 5000
            params['dropout'] = auto_dropout
            params['policy_based'] = True
            return params
        elif name_algo == "paspeq_o2o":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 10000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            params['policy_based'] = True
            return params
        elif name_algo == "faspeq_o2o":
            params['offline_epochs'] = 75_000
            params['offline_frequency'] = args.offline_frequency
            params['val_check_interval'] = 1000
            params['val_patience'] = 10000
            params['dropout'] = auto_dropout
            params['use_minari'] = True
            params['minari_quality'] = args.minari_quality
            params['faspeq_mode'] = True
            params['num_reference_batches'] = args.num_reference_batches
            return params
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

    FASPEQ(env_name=params['env'], 
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
           utd_ratio=params['utd_ratio'], 
           auto_stab=params['auto_stab'],
           policy_based=params['policy_based'],
           faspeq_mode=params['faspeq_mode'],
           faspeq_lcb_mode=params['faspeq_lcb_mode'],
           num_reference_batches=params['num_reference_batches']
           )