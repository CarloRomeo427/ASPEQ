"""
Analytical FLOP Counter for RL Algorithms
==========================================
Computes FLOPs deterministically from network architecture and update logic,
avoiding FlopCounterMode reliability issues.

FLOPs per linear layer:
  forward  = 2 * batch_size * in_features * out_features  (multiply-add)
  backward ≈ 2x forward  (weight grad + input grad)

Usage: same CLI as main_unified.py
    python main_unified_flops.py --algo faspeq_pct --env hopper \
        --use-offline-data --dataset-quality expert --val-pct 0.1 --log-wandb
"""

import time
import numpy as np
import torch
import wandb

from main import (
    normalize_env_name, get_gymnasium_env_name, get_display_name,
    get_dropout_rate, get_agent_class, get_algo_config,
    make_env, get_env_info, load_minari_dataset, evaluate,
    parse_args,
)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical FLOP computation
# ─────────────────────────────────────────────────────────────────────────────

def mlp_forward_flops(layer_sizes: list, batch_size: int) -> int:
    """
    FLOPs for one forward pass through an MLP.
    layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
    Each linear layer: 2 * batch_size * in * out  (multiply-add)
    Activations, LayerNorm, Dropout are negligible relative to matmuls.
    """
    flops = 0
    for i in range(len(layer_sizes) - 1):
        flops += 2 * batch_size * layer_sizes[i] * layer_sizes[i + 1]
    return flops


class FLOPCounter:
    """
    Analytical FLOP counter for SAC-family algorithms.

    Counts forward and backward pass FLOPs through Q-networks and policy,
    accounting for algorithm-specific differences (UTD, num_Q, CQL sampling,
    offline stabilization, policy update delay).
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, batch_size, num_Q,
                 policy_update_delay, algo_name, cql_n_actions=0):
        self.batch_size = batch_size
        self.num_Q = num_Q
        self.policy_update_delay = policy_update_delay
        self.algo_name = algo_name.lower()
        self.cql_n_actions = cql_n_actions

        # Q-network: (obs+act) → hidden → hidden → 1
        q_layers = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q_fwd = mlp_forward_flops(q_layers, batch_size)
        self.q_bwd = 2 * self.q_fwd  # backward ≈ 2x forward

        # Policy: obs → hidden → hidden → act_dim (mean) + act_dim (log_std)
        # Two output heads share the hidden layers
        pi_hidden = [obs_dim] + list(hidden_sizes)
        pi_head_mean = [hidden_sizes[-1], act_dim]
        pi_head_logstd = [hidden_sizes[-1], act_dim]
        self.pi_fwd = (mlp_forward_flops(pi_hidden, batch_size)
                       + mlp_forward_flops(pi_head_mean, batch_size)
                       + mlp_forward_flops(pi_head_logstd, batch_size))
        self.pi_bwd = 2 * self.pi_fwd

        # CQL extra: evaluate Q on cql_n_actions sampled actions per data point
        if self.cql_n_actions > 0:
            q_layers_cql = [obs_dim + act_dim] + list(hidden_sizes) + [1]
            self.cql_q_fwd = mlp_forward_flops(q_layers_cql, batch_size * cql_n_actions)
        else:
            self.cql_q_fwd = 0

        self.cumulative = 0

    def _q_update_flops(self) -> int:
        """FLOPs for one Q-network update iteration (all Q-nets)."""
        flops = 0
        # Target computation (no grad, still FLOPs):
        #   1 policy forward (next action) + num_Q target-Q forwards
        flops += self.pi_fwd + self.num_Q * self.q_fwd
        # Q prediction: num_Q forwards + num_Q backwards
        flops += self.num_Q * (self.q_fwd + self.q_bwd)
        return flops

    def _policy_update_flops(self) -> int:
        """FLOPs for one policy update."""
        flops = 0
        # Policy forward + backward
        flops += self.pi_fwd + self.pi_bwd
        # Q-nets forward (no grad) for policy loss
        flops += self.num_Q * self.q_fwd
        return flops

    def _cql_extra_flops(self) -> int:
        """Extra FLOPs per update from CQL regularization."""
        if self.cql_n_actions == 0:
            return 0
        # Sample cql_n_actions from policy + cql_n_actions uniform
        # Evaluate all Q-nets on both sets → 2 × num_Q × Q_fwd(batch*n_actions)
        flops = self.pi_fwd  # policy forward for current-policy actions
        flops += 2 * self.num_Q * self.cql_q_fwd
        return flops

    def count_train_step(self, utd_ratio: int):
        """Count FLOPs for one agent.train() call with given UTD."""
        for i_update in range(utd_ratio):
            self.cumulative += self._q_update_flops()

            if 'calql' in self.algo_name:
                self.cumulative += self._cql_extra_flops()

            if ((i_update + 1) % self.policy_update_delay == 0) or (i_update == utd_ratio - 1):
                self.cumulative += self._policy_update_flops()

    def count_offline_epochs(self, n_epochs: int):
        """Count FLOPs for n offline stabilization epochs (Q-only, no policy)."""
        per_epoch = (self.pi_fwd + self.num_Q * self.q_fwd  # target
                     + self.num_Q * (self.q_fwd + self.q_bwd))  # Q update
        self.cumulative += per_epoch * n_epochs

    def count_offline_pretrain(self, n_steps: int):
        """Count FLOPs for IQL/CalQL offline pretraining."""
        per_step = self._q_update_flops() + self._policy_update_flops()
        if 'calql' in self.algo_name:
            per_step += self._cql_extra_flops()
        self.cumulative += per_step * n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Training with FLOP tracking
# ─────────────────────────────────────────────────────────────────────────────

def train_with_flops(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    canonical_name, env_suite = normalize_env_name(args.env)
    gym_env_name = get_gymnasium_env_name(canonical_name, env_suite)
    display_name = get_display_name(canonical_name, env_suite, args.dataset_quality)
    print(f"Environment: {args.env} → {gym_env_name} (suite: {env_suite})")

    quality = args.dataset_quality if args.use_offline_data else None
    env = make_env(canonical_name, env_suite, quality)
    test_env = make_env(canonical_name, env_suite, quality)
    obs_dim, act_dim, act_limit, max_ep_len = get_env_info(env)
    max_ep_len = min(args.max_ep_len, max_ep_len)
    total_steps = args.steps_per_epoch * args.epochs + 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dropout_rate = get_dropout_rate(canonical_name, env_suite, args.target_drop_rate)
    AgentClass = get_agent_class(args.algo)
    algo_config = get_algo_config(args.algo, args, dropout_rate)

    agent = AgentClass(
        env_name=gym_env_name, obs_dim=obs_dim, act_dim=act_dim,
        act_limit=act_limit, device=device, start_steps=args.start_steps,
        **algo_config
    )

    # ── Resolve effective hyperparams ──
    effective_utd = algo_config.get('utd_ratio', args.utd_ratio)
    effective_num_q = algo_config.get('num_Q', args.num_q)
    effective_delay = algo_config.get('policy_update_delay', 20)
    cql_n = args.cql_n_actions if 'calql' in args.algo.lower() else 0

    print(f"Algorithm: {args.algo}, UTD: {effective_utd}, num_Q: {effective_num_q}, "
          f"policy_delay: {effective_delay}, CQL_n: {cql_n}")

    # ── Init analytical FLOP counter ──
    flop_counter = FLOPCounter(
        obs_dim=obs_dim, act_dim=act_dim,
        hidden_sizes=(args.network_width, args.network_width),
        batch_size=256, num_Q=effective_num_q,
        policy_update_delay=effective_delay,
        algo_name=args.algo, cql_n_actions=cql_n,
    )

    print(f"  Q fwd: {flop_counter.q_fwd:.3e}  Q bwd: {flop_counter.q_bwd:.3e}")
    print(f"  π fwd: {flop_counter.pi_fwd:.3e}  π bwd: {flop_counter.pi_bwd:.3e}")
    print(f"  Per Q-update (all nets): {flop_counter._q_update_flops():.3e}")

    # ── Load offline data ──
    if args.use_offline_data:
        compute_mc = args.algo.lower() == 'calql'
        load_minari_dataset(agent, canonical_name, args.dataset_quality, env_suite, compute_mc)

        if args.algo.lower() in ['iql', 'calql'] and args.offline_pretrain_steps > 0:
            print(f"Offline pretraining: {args.offline_pretrain_steps} steps")
            agent.train_offline(epochs=args.offline_pretrain_steps, test_env=test_env)
            flop_counter.count_offline_pretrain(args.offline_pretrain_steps)
            print(f"  Pretrain FLOPs: {flop_counter.cumulative:.3e}")

    obs, _ = env.reset(seed=args.seed)
    ep_ret, ep_len = 0, 0
    start_time = time.time()

    for t in range(total_steps):
        action = agent.get_exploration_action(obs, env)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        ep_len += 1
        agent.store_data(obs, action, reward, obs_next, terminated and ep_len < max_ep_len)

        # ── Stabilization (SPEQ / FASPEQ) ──
        if hasattr(agent, 'check_should_trigger_offline_stabilization'):
            if agent.check_should_trigger_offline_stabilization():
                flops_before = flop_counter.cumulative
                epochs_done = agent.finetune_offline(
                    test_env=test_env, current_env_step=t + 1
                )
                flop_counter.count_offline_epochs(epochs_done)
                stab_flops = flop_counter.cumulative - flops_before
                print(f"  Stabilization @ step {t+1}: {epochs_done} epochs, "
                      f"+{stab_flops:.3e} FLOPs (cum: {flop_counter.cumulative:.3e})")
                wandb.log({"cumulative_flops": flop_counter.cumulative}, step=t + 1)

        # ── Regular training step ──
        pi_loss, q_loss = agent.train(current_env_step=t + 1)

        # Count FLOPs only when agent actually updates (past start_steps)
        if agent.replay_buffer.size > args.start_steps:
            flop_counter.count_train_step(effective_utd)

        obs = obs_next
        ep_ret += reward

        if done or ep_len >= max_ep_len:
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # ── Epoch logging ──
        if (t + 1) % args.steps_per_epoch == 0:
            epoch = (t + 1) // args.steps_per_epoch
            eval_reward = evaluate(agent, test_env, max_ep_len)

            print(f"Epoch {epoch:4d} | Step {t+1:7d} | Reward: {eval_reward:8.2f} "
                  f"| FLOPs: {flop_counter.cumulative:.3e} | Time: {time.time()-start_time:.0f}s")

            wandb.log({
                "epoch": epoch,
                "policy_loss": pi_loss,
                "mean_q_loss": q_loss,
                "EvalReward": eval_reward,
                "cumulative_flops": flop_counter.cumulative,
            }, step=t + 1)

    print(f"Training complete! Total FLOPs: {flop_counter.cumulative:.3e}")
    return flop_counter.cumulative


if __name__ == '__main__':
    args = parse_args()

    canonical_name, env_suite = normalize_env_name(args.env)
    display_name = get_display_name(canonical_name, env_suite, args.dataset_quality)

    if args.algo == 'faspeq_o2o':
        if args.val_patience != 10000:
            exp_name = f"faspeq_o2o_{display_name.capitalize()}_valpat{args.val_patience}"
        else:
            exp_name = f"faspeq_o2o_{display_name.capitalize()}"
    elif args.algo == 'faspeq_pct':
        metric = "td" if args.faspeq_pct_use_td else "pi"
        if args.val_patience != 10000:
            exp_name = f"faspeq_pct{int(args.val_pct*100)}_{metric}_{display_name.capitalize()}_valpat{args.val_patience}"
        else:
            exp_name = f"faspeq_pct{int(args.val_pct*100)}_{metric}_{display_name.capitalize()}"
    elif args.algo == 'faspeq_nosplit':
        metric = "td" if args.faspeq_pct_use_td else "pi"
        exp_name = f"faspeq_nosplit_{metric}_{display_name.capitalize()}"
    elif args.algo == 'faspeq_randearly':
        exp_name = f"faspeq_randearly_mean{int(args.random_early_mean)}_{display_name.capitalize()}"
    else:
        exp_name = f"{args.algo}_{display_name.capitalize()}"

    exp_name = f"{exp_name}_flops"

    wandb.init(
        name=exp_name,
        project="SPEQ",
        config=vars(args),
        mode='online' if args.log_wandb else 'disabled'
    )

    train_with_flops(args)
    wandb.finish()