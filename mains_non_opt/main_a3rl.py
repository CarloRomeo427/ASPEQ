### A3RL Main Training Loop
### Paper: https://arxiv.org/abs/2502.07937v3

import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb
import minari

from src.algos_non_opt.agent_a3rl import Agent
from src.algos_non_opt.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger


def get_minari_dataset_name(env_name, quality='expert'):
    """Map Gymnasium env to Minari dataset name."""
    base = env_name.split('-')[0].lower()
    mapping = {
        'hopper': 'hopper', 'halfcheetah': 'halfcheetah', 'walker2d': 'walker2d',
        'ant': 'ant', 'swimmer': 'swimmer', 'humanoid': 'humanoid'
    }
    return f"mujoco/{mapping.get(base, base)}/{quality}-v0"


def load_minari_dataset(agent, env_name, quality='expert'):
    """Load Minari dataset into agent's offline buffer."""
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
    """Environment-specific dropout rates."""
    if target_drop_rate == 0.0:
        return 0.0
    rates = {
        "Humanoid-v5": 0.1, "Ant-v5": 0.01,
        "Walker2d-v5": 0.005, "HalfCheetah-v5": 0.005,
        "Hopper-v5": 0.001, "Swimmer-v5": 0.001
    }
    return rates.get(env_name, 0.0)


def A3RL(env_name, seed=0, epochs=300, steps_per_epoch=1000,
         max_ep_len=1000, logger_kwargs=dict(), debug=False,
         hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
         lr=3e-4, gamma=0.99, polyak=0.995,
         alpha=0.2, auto_alpha=True, target_entropy='mbpo',
         start_steps=5000, delay_update_steps='auto',
         utd_ratio=1, num_Q=10, num_min=2, q_target_mode='min',
         policy_update_delay=1,
         evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
         gpu_id=0, target_drop_rate=0.0, layer_norm=True,
         use_minari=False, minari_quality='expert',
         gradient_steps_g=20, prio_zeta=0.3, adv_temp_xi=0.03):
    """
    A3RL: Active Advantage-Aligned RL with Offline Data
    
    Following Algorithm 1 from the paper.
    """
    if debug:
        hidden_sizes = [32, 32]
        batch_size = 32
        num_Q = 3
        gradient_steps_g = 2
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches.get(env_name, 300)
    total_steps = steps_per_epoch * epochs + 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = gym.make(env_name)
    test_env = gym.make(env_name)
    bias_eval_env = gym.make(env_name)

    def seed_all(epoch):
        shift = epoch * 9999
        mod = 999999
        env_seed = (seed + shift) % mod
        test_seed = (seed + 10000 + shift) % mod
        bias_seed = (seed + 20000 + shift) % mod
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        return env_seed, test_seed, bias_seed

    env_seed, test_seed, bias_seed = seed_all(0)

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
        gradient_steps_g=gradient_steps_g,
        prio_zeta=prio_zeta,
        adv_temp_xi=adv_temp_xi
    )

    # Load offline data if O2O
    if use_minari:
        n_offline = load_minari_dataset(agent, env_name, minari_quality)
        wandb.log({"offline_samples_loaded": n_offline}, step=0)

    o, _ = env.reset(seed=env_seed)
    ep_ret, ep_len = 0, 0

    # Main loop (Algorithm 1, Lines 5-21)
    for t in range(total_steps):
        # Line 8: Take action
        if use_minari:
            a = agent.get_exploration_action_o2o(o, env, t)
        else:
            a = agent.get_exploration_action(o, env)

        o2, r, terminated, truncated, _ = env.step(a)
        ep_len += 1
        d_store = terminated and not truncated

        # Line 8: Update buffer R
        agent.store_data(o, a, r, o2, d_store)

        # Lines 9-21: Train (agent handles internally)
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
                env_seed, test_seed, bias_seed = seed_all(epoch)
                o, _ = env.reset(seed=env_seed)
                ep_ret, ep_len = 0, 0

            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, n_mc_eval, n_mc_cutoff, bias_seed, max_ep_len)

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t + 1)
            logger.log_tabular('Time', time.time() - time.time())
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
            if evaluate_bias:
                logger.log_tabular('TestEpBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBiasSqr', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='Hopper-v5')
    parser.add_argument("--algo", type=str, default='a3rl_o2o')
    parser.add_argument("--info", type=str, default='a3rl_run')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")

    # A3RL hyperparameters (Table 2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--network-width", type=int, default=256)
    parser.add_argument("--num-q", type=int, default=10)  # E
    parser.add_argument("--num-min", type=int, default=2)  # Z
    parser.add_argument("--gradient-steps", type=int, default=20)  # G
    parser.add_argument("--prio-zeta", type=float, default=0.3)  # ζ
    parser.add_argument("--adv-xi", type=float, default=0.03)  # ξ
    parser.add_argument("--policy-update-delay", type=int, default=1)

    parser.add_argument("--target-drop-rate", type=float, default=0.0)
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")

    parser.add_argument("--use-minari", action="store_true")
    parser.add_argument("--minari-quality", type=str, default='expert',
                        choices=['expert', 'medium', 'simple', 'random'])

    parser.add_argument("--evaluate-bias", action="store_true")

    args = parser.parse_args()

    exp_name = f"{args.algo}_{args.env}_{args.minari_quality}" if args.use_minari else f"{args.algo}_{args.env}"
    args.data_dir = f'./runs/{args.info}/'
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)
    dropout = set_dropout(args.env, args.target_drop_rate)

    wandb.init(
        name=exp_name,
        project="SPEQ",
        config={
            'env': args.env, 'seed': args.seed, 'epochs': args.epochs,
            'num_q': args.num_q, 'num_min': args.num_min,
            'gradient_steps': args.gradient_steps,
            'prio_zeta': args.prio_zeta, 'adv_xi': args.adv_xi,
            'batch_size': args.batch_size, 'use_minari': args.use_minari,
            'minari_quality': args.minari_quality
        },
        mode='online' if args.log_wandb else 'disabled'
    )

    A3RL(
        env_name=args.env, seed=args.seed, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, logger_kwargs=logger_kwargs,
        debug=args.debug, gpu_id=args.gpu_id,
        hidden_sizes=hidden_sizes, batch_size=args.batch_size,
        start_steps=args.start_steps, num_Q=args.num_q, num_min=args.num_min,
        policy_update_delay=args.policy_update_delay,
        target_drop_rate=dropout, layer_norm=args.layer_norm,
        use_minari=args.use_minari, minari_quality=args.minari_quality,
        evaluate_bias=args.evaluate_bias,
        gradient_steps_g=args.gradient_steps,
        prio_zeta=args.prio_zeta, adv_temp_xi=args.adv_xi
    )