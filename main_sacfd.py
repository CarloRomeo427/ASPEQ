import gymnasium as gym
import numpy as np
import torch
import time
import wandb
import minari
import os

from src.algos.agent_sacfd import Agent
from src.algos.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.logx import EpochLogger

def get_minari_dataset_name(env_name, quality='expert'):
    base_name = env_name.split('-')[0].lower()
    env_mapping = {
        'hopper': 'hopper', 'halfcheetah': 'halfcheetah', 'walker2d': 'walker2d', 'ant': 'ant',
        'swimmer': 'swimmer', 'reacher': 'reacher', 'pusher': 'pusher',
        'invertedpendulum': 'invertedpendulum', 'inverteddoublependulum': 'inverteddoublependulum',
        'humanoid': 'humanoid', 'humanoidstandup': 'humanoidstandup'
    }
    minari_env_name = env_mapping.get(base_name, base_name)
    return f'mujoco/{minari_env_name}/{quality}-v0'

def load_minari_dataset(agent, env_name=None, quality='expert'):
    dataset_name = get_minari_dataset_name(env_name, quality)
    print(f"Loading Minari dataset: {dataset_name}")
    try:
        dataset = minari.load_dataset(dataset_name)
    except Exception:
        print(f"Dataset {dataset_name} not found locally. Downloading...")
        try:
            minari.download_dataset(dataset_name)
            dataset = minari.load_dataset(dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    print(f"Dataset info: {dataset.total_episodes} episodes, {dataset.total_steps} steps")
    total = 0
    for episode in dataset.iterate_episodes():
        T = len(episode.actions)
        for t in range(T):
            agent.store_data_offline(
                episode.observations[t], 
                episode.actions[t], 
                episode.rewards[t], 
                episode.observations[t + 1], 
                episode.terminations[t]
            )
            total += 1
    return total

def set_dropout(env_name, target_drop_rate=0.0):
    if target_drop_rate == 0.0: return 0.0
    if env_name == "Humanoid-v5": return 0.1
    elif env_name == "Ant-v5": return 0.01
    elif env_name in ["Walker2d-v5", "HalfCheetah-v5", "Pusher-v5"]: return 0.005
    return 0.001

def SACfD(env_name, seed=0, epochs=300, steps_per_epoch=1000,
          max_ep_len=1000, logger_kwargs=dict(), debug=False,
          hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
          lr=3e-4, gamma=0.99, polyak=0.995,
          alpha=0.2, auto_alpha=True, target_entropy='mbpo',
          start_steps=5000, num_Q=2, gpu_id=0, 
          target_drop_rate=0.0, layer_norm=False, 
          use_minari=True, minari_quality='expert'):

    if debug:
        hidden_sizes = [2, 2]; batch_size = 2; num_Q = 2
        max_ep_len = 100; start_steps = 100; steps_per_epoch = 100

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    
    if epochs == 'mbpo' or epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        epochs = mbpo_epoches.get(env_name, 300)
    
    total_steps = steps_per_epoch * epochs + 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Environment setup
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    
    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0].item()

    # Initialize SACfD Agent
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy, start_steps, 
                  num_Q, target_drop_rate, layer_norm)

    # Initialize Replay Buffer with Offline Data
    if use_minari:
        print("\n--- Initializing Replay Buffer with Offline Data ---")
        loaded_steps = load_minari_dataset(agent, env_name, minari_quality)
        print(f"Replay Buffer Size after initialization: {agent.replay_buffer.size}")
        print("----------------------------------------------------\n")
        
        # Important: If offline data covers the start_steps, we can technically skip random exploration
        # but standard practice often keeps start_steps logic relative to *online* steps 
        # or relies on total buffer size. Agent.get_exploration_action checks buffer.size.
        # If dataset > start_steps, agent will immediately use policy.

    start_time = time.time()
    o, _ = env.reset(seed=seed)
    r, ep_ret, ep_len = 0, 0, 0

    for t in range(total_steps):
        
        # Get action
        a = agent.get_exploration_action(o, env)

        # Step
        o2, r, term, trunc, _ = env.step(a)
        ep_len += 1
        d = term or trunc
        d_store = term and not trunc # Store 'done' only if terminated (not truncated)

        # Store online experience (expands the buffer further)
        agent.store_data(o, a, r, o2, d_store)

        # Update Agent
        agent.train(logger)

        o = o2
        ep_ret += r

        # End of episode handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            r, ep_ret, ep_len = 0, 0, 0

        # Epoch handling and logging
        if (t + 1) % 1000 == 0:
            epoch = t // 1000
            
            # Evaluate
            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({"EvalReward": np.mean(test_rw)}, step=t+1)

            # Log
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Hopper-v5')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--info", type=str, default="sacfd_run")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--minari-quality", type=str, default='expert')
    
    # Defaults for SAC
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--target-drop-rate", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Auto-configure dropout if not specified (legacy support)
    dropout = set_dropout(args.env, args.target_drop_rate) if args.target_drop_rate == 0.0 else args.target_drop_rate

    setup_logger_kwargs(f"sacfd_{args.env}", args.seed, f"./runs/{args.info}/")
    
    wandb.init(
        project="SPEQ",
        name=f"sacfd_{args.env}_{args.minari_quality}",
        config=vars(args),
        mode='online' if args.log_wandb else 'disabled'
    )
    
    SACfD(env_name=args.env, 
          seed=args.seed, 
          epochs=args.epochs,
          gpu_id=args.gpu_id,
          minari_quality=args.minari_quality,
          target_drop_rate=dropout,
          hidden_sizes=(args.network_width, args.network_width),
          logger_kwargs=setup_logger_kwargs(f"sacfd_{args.env}", args.seed, f"./runs/{args.info}/")
          )