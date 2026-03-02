"""
FASPEQ Profiler Agent — v2 (Real Early Stopping + Loss Logging)
===============================================================
Runs ACTUAL FASPEQ with patience-based early stopping, but logs
policy loss at every val_check_interval. This preserves real training
dynamics (unlike forced 100k which corrupts subsequent phases).

Uses val_pct-based validation (same as real FASPEQ): a percentage of
the offline buffer is held out for validation, not n_val_batches.

Output: .npy with columns [offline_epoch, online_env_step, policy_loss]
Flushes to disk after every stabilization phase.
"""

import atexit
import os
import numpy as np
import torch
import wandb

from src.algos.agent_speq import SPEQAgent
from src.algos.core import test_agent


class FASPEQProfilerAgent(SPEQAgent):

    def __init__(
        self,
        env_name: str,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        device: torch.device,
        hidden_sizes: tuple = (256, 256),
        replay_size: int = int(1e6),
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        polyak: float = 0.995,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: str = 'mbpo',
        start_steps: int = 5000,
        utd_ratio: int = 1,
        num_Q: int = 2,
        policy_update_delay: int = 20,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = True,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        # Early stopping + profiling
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        val_pct: float = 0.1,
        n_val_batches: int = 0,    # accepted but ignored (use val_pct)
        save_dir: str = '.',
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm,
            o2o=True, offline_epochs=offline_epochs,
            trigger_interval=trigger_interval,
        )

        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.val_pct = val_pct
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self._loss_log: list = []

        # Self-contained trigger
        self._profiler_env_steps = 0
        self._profiler_last_trigger = 0
        self._profiler_trigger_interval = trigger_interval

        self._save_suffix = env_name.replace('/', '_')
        atexit.register(self._atexit_save)

        print(f"  [PROFILER] Real early stopping: patience={val_patience}, "
              f"check_interval={val_check_interval}, val_pct={val_pct}, "
              f"max_epochs={offline_epochs}, save_dir={save_dir}")

    # ------------------------------------------------------------------
    # Self-contained trigger
    # ------------------------------------------------------------------
    def store_data(self, obs, action, reward, next_obs, done):
        super().store_data(obs, action, reward, next_obs, done)
        self._profiler_env_steps += 1

    def check_should_trigger_offline_stabilization(self) -> bool:
        if self._profiler_env_steps < self.start_steps:
            return False
        if self._profiler_env_steps - self._profiler_last_trigger >= self._profiler_trigger_interval:
            self._profiler_last_trigger = self._profiler_env_steps
            print(f"\n  [PROFILER] === Trigger @ env_step {self._profiler_env_steps} ===")
            return True
        return False

    # ------------------------------------------------------------------
    # Validation (val_pct-based, matching real FASPEQ)
    # ------------------------------------------------------------------
    def _prepare_val_set(self):
        """Hold out val_pct of each buffer for validation, like real FASPEQ."""
        self._val_batches = []

        for buf in [self.replay_buffer, self.replay_buffer_offline]:
            if buf.size < self.batch_size:
                continue
            n_val = max(self.batch_size, int(buf.size * self.val_pct))
            n_val = min(n_val, buf.size)
            # Sample validation indices (NOT removed from buffer — non-invasive profiling)
            idx = np.random.choice(buf.size, n_val, replace=False)
            # Split into batches
            n_complete = n_val // self.batch_size
            for i in range(n_complete):
                s, e = i * self.batch_size, (i + 1) * self.batch_size
                bi = idx[s:e]
                self._val_batches.append({
                    'obs':      torch.FloatTensor(buf.obs1_buf[bi].copy()).to(self.device),
                    'obs_next': torch.FloatTensor(buf.obs2_buf[bi].copy()).to(self.device),
                    'acts':     torch.FloatTensor(buf.acts_buf[bi].copy()).to(self.device),
                    'rews':     torch.FloatTensor(buf.rews_buf[bi].copy()).unsqueeze(1).to(self.device),
                    'done':     torch.FloatTensor(buf.done_buf[bi].copy()).unsqueeze(1).to(self.device),
                })

        print(f"  [PROFILER] Validation: {len(self._val_batches)} batches "
              f"(val_pct={self.val_pct})")

    def _evaluate_policy_loss(self) -> float:
        if not self._val_batches:
            return 0.0
        total = 0.0
        for b in self._val_batches:
            with torch.no_grad():
                a, _, _, lp, _, _ = self.policy_net.forward(b['obs'])
                qs = [q(torch.cat([b['obs'], a], 1)) for q in self.q_net_list]
                qm = torch.mean(torch.cat(qs, dim=1), dim=1, keepdim=True)
                total += (self.alpha * lp - qm).mean().item()
        return total / len(self._val_batches)

    # ------------------------------------------------------------------
    # Stabilization: REAL early stopping + logging every check
    # ------------------------------------------------------------------
    def finetune_offline(self, epochs: int = None, test_env=None, current_env_step: int = None) -> int:
        epochs = epochs or self.offline_epochs
        step_tag = current_env_step or self._profiler_env_steps

        self._prepare_val_set()

        init_loss = self._evaluate_policy_loss()
        self._loss_log.append((0, step_tag, init_loss))

        best_loss = init_loss
        no_improve = 0
        epochs_done = 0

        print(f"  [PROFILER] Start: max {epochs} ep, online_step={step_tag}, "
              f"init_loss={init_loss:.4f}")

        for e in range(1, epochs + 1):
            epochs_done = e

            # Q update
            obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            y_q = self.get_sac_q_target(obs_next, rews, done)
            q_preds = [q(torch.cat([obs, acts], 1)) for q in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_exp = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss = self.expectile_loss(q_cat - y_q_exp).mean() * self.num_Q

            for opt in self.q_optimizer_list:
                opt.zero_grad()
            q_loss.backward()
            for opt in self.q_optimizer_list:
                opt.step()
            self.update_target_networks()

            # Check + log every interval
            if e % self.val_check_interval == 0:
                ploss = self._evaluate_policy_loss()
                self._loss_log.append((e, step_tag, ploss))

                improved = ploss < best_loss * 0.999
                if improved:
                    best_loss = ploss
                    no_improve = 0
                else:
                    no_improve += self.val_check_interval

                print(f"  [PROFILER] ep {e} | loss {ploss:.4f} | "
                      f"best {best_loss:.4f} | patience {no_improve}/{self.val_patience}")

                # REAL early stopping
                if no_improve >= self.val_patience:
                    print(f"  [PROFILER] Early stop @ ep {e}")
                    break

            # Periodic eval during stabilization
            if (e % 5000 == 0) and test_env is not None and current_env_step is not None:
                rw = test_agent(self, test_env, 1000, None)
                wandb.log({"OfflineEvalReward": np.mean(rw)}, step=current_env_step)

        final = self._evaluate_policy_loss()
        print(f"  [PROFILER] Done: {epochs_done} ep, final={final:.4f} "
              f"(init={init_loss:.4f}, best={best_loss:.4f})")
        print(f"  [PROFILER] Total entries: {len(self._loss_log)}")

        if current_env_step is not None:
            wandb.log({"OfflineEpochs": epochs_done}, step=current_env_step)

        self._val_batches = []

        # Flush to disk after every phase
        self.save_loss_log(suffix=self._save_suffix)

        return epochs_done

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_loss_log(self, suffix: str = ''):
        if not self._loss_log:
            return
        arr = np.array(self._loss_log, dtype=np.float64)
        tag = suffix or self._save_suffix
        path = os.path.join(self.save_dir, f"faspeq_profiler_losses_{tag}.npy")
        np.save(path, arr)
        print(f"  [PROFILER] Flushed {arr.shape[0]} entries → {path}")

    def _atexit_save(self):
        self.save_loss_log(suffix=self._save_suffix)