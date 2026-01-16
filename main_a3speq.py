### MAIN FILE FOR A3SPEQ
### A3RL Prioritization (every step) + FASPEQ Structure (UTD=1, 2 critics, dropout, offline stabilization)

import gymnasium as gym
import numpy as np
import torch
import time
import wandb
import minari

from src.algos.agent_a3speq import Agent
from src.algos.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger


def get_minari_dataset_name(env_name, quality='expert'):
    base = env_name.split('-')[0].lower()
    mapping = {
        'hopper': 'hopper', 'halfcheetah': 'halfcheetah', 'walker2d': 'walker2d',
        'ant': 'ant', 'swimmer': 'swimmer', 'humanoid': 'humanoid'
    }
    return f"mujoco/{mapping.get(base, base)}/{quality}-v0"


def load_minari_dataset(agent, env_name, quality='expert'):
    dataset_name = get_minari_dataset_name(env_name, quality)
    print(f"Loading Minari dataset: {dataset_name}")
    
    try:
        dataset = minari.load_dataset(dataset_name)
    except:
        print("Downloading dataset...")
        minari.download_dataset(dataset_name)
        dataset = minari.load_dataset(dataset_name)
    
    print(f"Episodes: {dataset.total_episodes}, Steps: {dataset.total_steps}")
    
    count = 0
    for ep in dataset.iterate_episodes():
        for t in range(len(ep.actions)):
            agent.store_data_offline(
                ep.observations[t], ep.actions[t], ep.rewards[t],
                ep.observations[t + 1], ep.terminations[t]
            )
            count += 1
    
    print(f"Loaded {count} transitions")
    return count


def set_dropout(env_name, target_drop_rate=0.0):
    if target_drop_rate == 0.0:
        return 0.0
    rates = {
        "Humanoid-v5": 0.1, "Ant-v5": 0.01,
        "Walker2d-v5": 0.005, "HalfCheetah-v5": 0.005,
        "Hopper-v5": 0.001, "Swimmer-v5": 0.001
    }
    return rates.get(env_name, 0.0)


def A3SPEQ(env_name, seed=0, epochs=300, steps_per_epoch=1000,
           max_ep_len=1000, logger_kwargs=dict(), debug=False,
           hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
           lr=3e-4, gamma=0.99, polyak=0.995,
           alpha=0.2, auto_alpha=True, target_entropy='mbpo',
           start_steps=5000, delay_update_steps='auto',
           utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
           policy_update_delay=20,
           gpu_id=0, target_drop_rate=0.0, layer_norm=False,
           use_minari=False, minari_quality='expert',
           offline_epochs=75000,
           val_check_interval=1000, val_patience=5000,
           num_reference_batches=10,
           # A3RL parameters
           prio_zeta=0.3, adv_xi=0.03,
           num_density_nets=10,
           beta_start=0.4, beta_end=1.0, beta_annealing_steps=100000,
           evaluate_bias=False, reseed_each_epoch=True):
    """
    A3SPEQ: A3RL prioritization + FASPEQ structure.
    
    A3RL elements (updated every gradient step):
    - Density network updates
    - Priority calculation
    - IS-weighted critic loss
    
    FASPEQ elements:
    - UTD=1
    - 2 Q-critics
    - Dropout
    - Offline stabilization phases
    """
    if debug:
        hidden_sizes = (32, 32)
        batch_size = 32
        num_Q = 2
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100
        num_density_nets = 3

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches.get(env_name, 300)
    total_steps = steps_per_epoch * epochs + 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = gym.make(env_name)
    test_env = gym.make(env_name)

    def seed_all(epoch):
        shift = epoch * 9999
        env_seed = (seed + shift) % 999999
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        return env_seed

    env_seed = seed_all(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = min(max_ep_len, env._max_episode_steps)
    act_limit = env.action_space.high[0].item()

    # Initialize agent
    agent = Agent(
        env_name, obs_dim, act_dim, act_limit, device,
        hidden_sizes=hidden_sizes, replay_size=replay_size, batch_size=batch_size,
        lr=lr, gamma=gamma, polyak=polyak,
        alpha=alpha, auto_alpha=auto_alpha, target_entropy=target_entropy,
        start_steps=start_steps, delay_update_steps=delay_update_steps,
        utd_ratio=utd_ratio, num_Q=num_Q, num_min=num_min,
        q_target_mode=q_target_mode, policy_update_delay=policy_update_delay,
        target_drop_rate=target_drop_rate, layer_norm=layer_norm,
        o2o=use_minari,
        val_check_interval=val_check_interval, val_patience=val_patience,
        num_reference_batches=num_reference_batches,
        prio_zeta=prio_zeta, adv_temp_xi=adv_xi,
        num_density_nets=num_density_nets,
        beta_start=beta_start, beta_end=beta_end, beta_annealing_steps=beta_annealing_steps
    )

    # Load offline data
    if use_minari:
        n_offline = load_minari_dataset(agent, env_name, minari_quality)
        wandb.log({"offline_samples_loaded": n_offline}, step=0)

    o, _ = env.reset(seed=env_seed)
    ep_ret, ep_len = 0, 0
    start_time = time.time()

    for t in range(total_steps):
        # Get action
        a = agent.get_exploration_action(o, env)
        
        # Step environment
        o2, r, terminated, truncated, _ = env.step(a)
        ep_len += 1
        d_store = terminated and not truncated
        
        # Store data
        agent.store_data(o, a, r, o2, d_store)
        
        # Check for offline stabilization trigger
        if use_minari and agent.check_should_trigger_offline_stabilization():
            wandb.log({"stabilization_trigger": 1}, step=t+1)
            agent.finetune_offline(
                epochs=offline_epochs,
                test_env=test_env,
                current_env_step=t+1
            )
        
        # Train (A3RL prioritization happens inside)
        agent.train(logger, current_env_step=t+1)
        
        o = o2
        ep_ret += r
        
        if terminated or truncated or ep_len == max_ep_len:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_ret, ep_len = 0, 0
        
        # Epoch logging
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            
            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({"EvalReward": np.mean(test_rw)}, step=t+1)
            
            if reseed_each_epoch:
                env_seed = seed_all(epoch)
                o, _ = env.reset(seed=env_seed)
                ep_ret, ep_len = 0, 0
            
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Q1Vals', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LogPi', average_only=True)
            logger.log_tabular('PreTanh', average_only=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Hopper-v5')
    parser.add_argument("--algo", type=str, default='a3speq')
    parser.add_argument("--info", type=str, default='test')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")

    # FASPEQ structure
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--policy-update-delay", type=int, default=20)
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--target-drop-rate", type=float, default=0.0)
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")

    # Offline stabilization
    parser.add_argument("--offline-epochs", type=int, default=75000)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--val-patience", type=int, default=5000)
    parser.add_argument("--num-reference-batches", type=int, default=10)

    # A3RL parameters
    parser.add_argument("--prio-zeta", type=float, default=0.3)
    parser.add_argument("--adv-xi", type=float, default=0.03)
    parser.add_argument("--num-density-nets", type=int, default=10)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument("--beta-annealing-steps", type=int, default=100000)

    # Minari
    parser.add_argument("--use-minari", action="store_true")
    parser.add_argument("--minari-quality", type=str, default='expert',
                        choices=['expert', 'medium', 'simple', 'random'])

    args = parser.parse_args()

    exp_name = f"{args.algo}_{args.env}_{args.minari_quality}" if args.use_minari else f"{args.algo}_{args.env}"
    args.data_dir = f'./runs/{args.info}/'
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)
    dropout = set_dropout(args.env, args.target_drop_rate)

    wandb.init(
        name=exp_name,
        project="A3SPEQ",
        config=vars(args),
        mode='online' if args.log_wandb else 'disabled'
    )

    A3SPEQ(
        env_name=args.env, seed=args.seed, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        logger_kwargs=logger_kwargs, debug=args.debug, gpu_id=args.gpu_id,
        hidden_sizes=hidden_sizes,
        utd_ratio=args.utd_ratio, num_Q=args.num_q,
        policy_update_delay=args.policy_update_delay,
        target_drop_rate=dropout, layer_norm=args.layer_norm,
        use_minari=args.use_minari, minari_quality=args.minari_quality,
        offline_epochs=args.offline_epochs,
        val_check_interval=args.val_check_interval,
        val_patience=args.val_patience,
        num_reference_batches=args.num_reference_batches,
        prio_zeta=args.prio_zeta, adv_xi=args.adv_xi,
        num_density_nets=args.num_density_nets,
        beta_start=args.beta_start, beta_end=args.beta_end,
        beta_annealing_steps=args.beta_annealing_steps
    )