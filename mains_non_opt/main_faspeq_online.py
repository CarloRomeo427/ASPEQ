### MAIN FILE FOR FASPEQ PURE ONLINE TRAINING WITH STATIC REFERENCE BATCHES
### No offline dataset - only online learning with periodic stabilization


import gymnasium as gym
import numpy as np
import torch
import time
import sys
import wandb

from src.algos_non_opt.agent_faspeq_online import Agent
from src.algos_non_opt.core import mbpo_epoches, test_agent
from src.utils.run_utils import setup_logger_kwargs
from src.utils.bias_utils import log_bias_evaluation
from src.utils.logx import EpochLogger


def print_class_attributes(obj):
    """Prints all attributes of an object along with their values."""
    attributes = vars(obj)
    for attr, value in attributes.items():
        print(f"{attr}: {value}")


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


def FASPEQ_Online(env_name, seed=0, epochs=300, steps_per_epoch=1000,
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
                  offline_epochs=75000,
                  val_check_interval=1000, val_patience=5000,
                  num_reference_batches=10
                  ):
    """
    FASPEQ Pure Online: Uses static reference batches for policy loss monitoring.
    No offline dataset - only online learning with periodic stabilization every 10k steps.
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

    print("="*80)
    print("FASPEQ ONLINE CONFIGURATION")
    print("="*80)
    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")
    print(f"Total epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
    print(f"Stabilization interval: 10000 steps")
    print(f"Max offline epochs per stabilization: {offline_epochs}")
    print(f"Reference batches: {num_reference_batches}")
    print(f"Val check interval: {val_check_interval}")
    print(f"Val patience: {val_patience}")
    print("="*80)
    sys.stdout.flush()

    """init agent"""
    agent = Agent(env_name, obs_dim, act_dim, act_limit, device,
                  hidden_sizes, replay_size, batch_size,
                  lr, gamma, polyak,
                  alpha, auto_alpha, target_entropy,
                  start_steps, delay_update_steps,
                  utd_ratio, num_Q, num_min, q_target_mode,
                  policy_update_delay,
                  target_drop_rate=target_drop_rate,
                  layer_norm=layer_norm,
                  val_check_interval=val_check_interval,
                  val_patience=val_patience,
                  num_reference_batches=num_reference_batches
                  )

    print_class_attributes(agent)

    # Initialize environment
    o, info = env.reset(seed=env_seed)
    r, ep_ret, ep_len = 0, 0, 0
    terminated, truncated = False, False

    # Track stabilization statistics
    total_stabilization_triggers = 0
    total_stabilization_epochs = 0

    for t in range(total_steps):

        a = agent.get_exploration_action(o, env)

        o2, r, terminated, truncated, info = env.step(a)
        
        d = terminated or truncated

        ep_len += 1
        d_store = terminated and not truncated

        agent.store_data(o, a, r, o2, d_store)

        # OFFLINE STABILIZATION TRIGGERING (every 10k steps)
        if agent.check_should_trigger_offline_stabilization():
            total_stabilization_triggers += 1
            
            wandb.log({
                "stabilization_trigger": total_stabilization_triggers,
                "buffer_size_at_trigger": agent.replay_buffer.size,
                "next_trigger_size": agent.next_trigger_size
            }, step=t+1)
            
            # Run FASPEQ offline stabilization with static reference batches
            epochs_performed = agent.finetune_offline_faspeq(
                epochs=offline_epochs, 
                test_env=test_env, 
                current_env_step=t+1
            )
            total_stabilization_epochs += epochs_performed
            
            wandb.log({
                "stabilization_epochs_performed": epochs_performed,
                "total_stabilization_epochs": total_stabilization_epochs,
                "avg_epochs_per_stabilization": total_stabilization_epochs / total_stabilization_triggers
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

        # EPOCH LOGGING: every 1000 steps
        if (t + 1) % 1000 == 0:
            epoch = t // 1000
            current_step = t + 1

            test_rw = test_agent(agent, test_env, max_ep_len, logger)
            wandb.log({
                "EvalReward": np.mean(test_rw),
                "buffer_size": agent.replay_buffer.size
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
                
            if epoch % 5 == 0 and args.calc_plasticity:
                rank_metrics = agent.compute_effective_rank_metrics(batch_size=256)
                wandb.log(rank_metrics, step=current_step)

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
                logger.log_tabular('TestEpNormBias', with_min_and_max=True)
                logger.log_tabular('TestEpNormBiasSqr', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # ─── Basic Setup ────────────────────────────────────────────────────────────
    parser.add_argument("--env", type=str, default='Hopper-v5', help="Environment name")
    parser.add_argument("--info", type=str, default='faspeq_online', help="Experiment information / run name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-wandb", action="store_true", help="Enable WandB logging")

    # ─── Offline Stabilization ───────────────────────────────────────────────────
    parser.add_argument("--offline-epochs", type=int, default=75000, help="Max epochs per offline stabilization phase")
    parser.add_argument("--val-check-interval", type=int, default=1000, help="Steps between policy loss checks")
    parser.add_argument("--val-patience", type=int, default=5000, help="Steps without improvement before early stopping")
    parser.add_argument("--num-reference-batches", type=int, default=10, help="Number of static reference batches")

    # ─── Network ─────────────────────────────────────────────────────────────────
    parser.add_argument("--network-width", type=int, default=256, help="Hidden units per layer")
    parser.add_argument("--num-q", type=int, default=2, help="Number of Q-networks")
    parser.add_argument("--target-drop-rate", type=float, default=999.0, help="Dropout rate")
    
    # ─── Plasticity ─────────────────────────────────────────────────────────────
    parser.add_argument("--calc-plasticity", action="store_true", help="Calculate plasticity metrics")

    # ─── Boolean Toggles ────────────────────────────────────────────────────────
    parser.set_defaults(layer_norm=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false", help="Disable layer normalization")
    parser.add_argument("--evaluate-bias", action="store_true", help="Evaluate policy bias")

    args = parser.parse_args()

    exp_name_full = f'faspeq_online_{args.env}'

    args.data_dir = './runs/' + str(args.info) + '/'
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    hidden_sizes = (args.network_width, args.network_width)
    auto_dropout = set_dropout(args.env, args.target_drop_rate)

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
        'offline_epochs': args.offline_epochs,
        'val_check_interval': args.val_check_interval,
        'val_patience': args.val_patience,
        'dropout': auto_dropout,
        'num_reference_batches': args.num_reference_batches
    }

    wandb.init(
        name=f'{exp_name_full}',
        project="SPEQ",
        config=params,
        mode='online' if args.log_wandb else 'disabled',
        save_code=True
    )

    FASPEQ_Online(
        env_name=params['env'], 
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
        val_check_interval=params['val_check_interval'],
        val_patience=params['val_patience'],
        num_reference_batches=params['num_reference_batches']
    )