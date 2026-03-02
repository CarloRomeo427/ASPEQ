"""
FASPEQ Exclusive Online Agent
==============================
FASPEQ variant that operates WITHOUT any offline dataset.
All stabilization (Q-network finetuning with early stopping) is performed
exclusively on the online replay buffer.

Key differences from standard FASPEQ:
- No offline buffer involvement at any stage
- Validation set: 10% of online replay buffer, removed during stabilization
- Stabilization training: samples from remaining online buffer only
- Early stopping: policy loss evaluated on online validation set only
- After stabilization: validation data restored to online buffer
"""

import numpy as np
import torch
import wandb

from src.algos.agent_faspeq import FASPEQAgent
from src.algos.core import test_agent


class FASPEQExcOnlineAgent(FASPEQAgent):
    """
    FASPEQ agent that uses only online experiences for stabilization.
    
    Inherits triggering logic from SPEQ (every trigger_interval env steps)
    and early stopping from FASPEQ, but all data operations are restricted
    to the online replay buffer.
    
    Args:
        val_pct: Fraction of online buffer used as validation set (default: 0.1)
        val_check_interval: How often to check validation metrics (default: 1000)
        val_patience: Steps without improvement before early stopping (default: 10000)
    """
    
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
        o2o: bool = False,          # False: no offline data
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        val_pct: float = 0.1,
    ):
        # Initialize parent. Note: FASPEQAgent.__init__ hardcodes o2o=True
        # when calling SPEQAgent, so we must override self.o2o after init.
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm,
            o2o=False,  # No offline data
            offline_epochs=offline_epochs,
            trigger_interval=trigger_interval,
            val_check_interval=val_check_interval,
            val_patience=val_patience,
            use_td_val=False,     # Policy loss only
            n_val_batches=0,
            val_pct=val_pct,
            no_split=False,
            random_early=False,
        )
        # FASPEQAgent hardcodes o2o=True to SPEQAgent.__init__, which makes
        # SPEQAgent.train() call sample_data_mix (requires offline buffer).
        # Force o2o=False so train() uses sample_data (online buffer only).
        self.o2o = False
    
    # ------------------------------------------------------------------
    # Override: validation sampling from online buffer only
    # ------------------------------------------------------------------
    def _sample_fixed_validation_batches(self):
        """
        Sample 10% of online replay buffer as validation set.
        Remove sampled transitions from the buffer for the stabilization phase.
        No offline buffer involvement.
        """
        online_batches = []
        
        if self.replay_buffer.size < 2 * self.batch_size:
            print("  [EXC_ONLINE] Online buffer too small for validation split, skipping")
            self.val_batches_online = []
            self.val_batches_offline = []
            return
        
        # Compute validation set size: val_pct of current online buffer
        n_val_transitions = int(self.replay_buffer.size * self.val_pct)
        n_val_transitions = max(self.batch_size, n_val_transitions)
        # Ensure we leave enough data for training
        n_val_transitions = min(n_val_transitions, self.replay_buffer.size - self.batch_size)
        
        # Sample random indices from online buffer
        val_indices = np.random.choice(
            self.replay_buffer.size,
            size=n_val_transitions,
            replace=False
        )
        
        # Store raw data for restoration BEFORE removal
        self._removed_online_data = {
            'obs1': self.replay_buffer.obs1_buf[val_indices].copy(),
            'obs2': self.replay_buffer.obs2_buf[val_indices].copy(),
            'acts': self.replay_buffer.acts_buf[val_indices].copy(),
            'rews': self.replay_buffer.rews_buf[val_indices].copy(),
            'done': self.replay_buffer.done_buf[val_indices].copy(),
        }
        
        # Convert to fixed batches for validation evaluation
        n_complete_batches = n_val_transitions // self.batch_size
        for i in range(n_complete_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_idx = val_indices[start:end]
            
            online_batches.append({
                'obs': torch.FloatTensor(self.replay_buffer.obs1_buf[batch_idx].copy()).to(self.device),
                'obs_next': torch.FloatTensor(self.replay_buffer.obs2_buf[batch_idx].copy()).to(self.device),
                'acts': torch.FloatTensor(self.replay_buffer.acts_buf[batch_idx].copy()).to(self.device),
                'rews': torch.FloatTensor(self.replay_buffer.rews_buf[batch_idx].copy()).unsqueeze(1).to(self.device),
                'done': torch.FloatTensor(self.replay_buffer.done_buf[batch_idx].copy()).unsqueeze(1).to(self.device),
            })
        
        # Remove validation data from training buffer
        self.replay_buffer.remove_indices(val_indices)
        
        self.val_batches_online = online_batches
        self.val_batches_offline = []  # No offline data
        self._removed_offline_data = None
        
        total_val = len(online_batches) * self.batch_size
        print(f"  [EXC_ONLINE] Validation set ({self.val_pct*100:.0f}%): "
              f"{len(online_batches)} batches, {total_val} transitions")
        print(f"  Remaining training buffer: {self.replay_buffer.size}")
    
    # ------------------------------------------------------------------
    # Override: restore only online data
    # ------------------------------------------------------------------
    def _restore_validation_data(self):
        """Restore validation transitions to the online buffer."""
        if self._removed_online_data is not None:
            n_restored = len(self._removed_online_data['obs1'])
            for i in range(n_restored):
                self.replay_buffer.store(
                    self._removed_online_data['obs1'][i],
                    self._removed_online_data['acts'][i],
                    self._removed_online_data['rews'][i],
                    self._removed_online_data['obs2'][i],
                    self._removed_online_data['done'][i]
                )
            print(f"  [EXC_ONLINE] Restored {n_restored} transitions, "
                  f"buffer size: {self.replay_buffer.size}")
        
        # Clear
        self.val_batches_online = None
        self.val_batches_offline = None
        self._removed_online_data = None
        self._removed_offline_data = None
    
    # ------------------------------------------------------------------
    # Override: policy loss evaluation on online validation set only
    # ------------------------------------------------------------------
    def _evaluate_policy_loss_on_fixed_batches(self) -> float:
        """Evaluate policy loss on online-only validation batches."""
        if not self.val_batches_online:
            return 0.0
        
        total_loss = 0.0
        n_batches = 0
        
        for batch in self.val_batches_online:
            with torch.no_grad():
                action, _, _, log_prob, _, _ = self.policy_net.forward(batch['obs'])
                q_values = [q_net(torch.cat([batch['obs'], action], 1))
                            for q_net in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - q_mean).mean()
                total_loss += policy_loss.item()
                n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    # ------------------------------------------------------------------
    # Override: stabilization trains on online buffer only
    # ------------------------------------------------------------------
    def finetune_offline(self, epochs: int = None, test_env=None,
                         current_env_step: int = None) -> int:
        """
        Stabilization phase using only the online replay buffer.
        
        Q-networks are finetuned by sampling from the (remaining) online
        buffer, with early stopping driven by policy loss on the held-out
        online validation set.
        """
        epochs = epochs or self.offline_epochs
        
        # Sample and fix validation set (removes from online buffer)
        self._sample_fixed_validation_batches()
        
        if not self.val_batches_online:
            print("  [EXC_ONLINE] No validation data available, skipping stabilization")
            self._restore_validation_data()
            return 0
        
        # Initial validation policy loss
        initial_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  [EXC_ONLINE] Initial val policy_loss: {initial_loss:.6f}")
        
        best_loss = initial_loss
        steps_without_improvement = 0
        epochs_performed = 0
        
        for e in range(epochs):
            epochs_performed = e + 1
            
            # --- Early stopping check ---
            if e > 0 and e % self.val_check_interval == 0:
                val_loss = self._evaluate_policy_loss_on_fixed_batches()
                
                improved = val_loss < best_loss * 0.999
                status = "improved" if improved else "no improvement"
                print(f"  Epoch {e}: val policy_loss={val_loss:.6f} "
                      f"(best={best_loss:.6f}, {status}, "
                      f"patience={steps_without_improvement}/{self.val_patience})")
                
                if improved:
                    best_loss = val_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += self.val_check_interval
                
                if steps_without_improvement >= self.val_patience:
                    print(f"  Early stop @ epoch {e} "
                          f"(no improvement for {self.val_patience} steps)")
                    break
            
            # --- Q-network update: sample from online buffer only ---
            batch = self.replay_buffer.sample_batch(self.batch_size)
            obs = torch.FloatTensor(batch['obs1']).to(self.device)
            obs_next = torch.FloatTensor(batch['obs2']).to(self.device)
            acts = torch.FloatTensor(batch['acts']).to(self.device)
            rews = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
            done = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
            
            y_q = self.get_sac_q_target(obs_next, rews, done)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss = self.expectile_loss(q_cat - y_q_expanded).mean() * self.num_Q
            
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            self.update_target_networks()
            
            # --- Progress logging ---
            if (e + 1) % 5000 == 0:
                current_val = self._evaluate_policy_loss_on_fixed_batches()
                print(f"  Offline epoch {e+1}/{epochs}, val policy_loss={current_val:.6f}")
                
                if test_env is not None and current_env_step is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)},
                              step=current_env_step)
        
        # Final validation loss
        final_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  [EXC_ONLINE] Final val policy_loss: {final_loss:.6f} "
              f"(started at {initial_loss:.6f})")
        print(f"  Stabilization ran {epochs_performed} epochs")
        
        self._stopping_epochs_history.append(epochs_performed)
        
        if current_env_step is not None:
            wandb.log({"OfflineEpochs": epochs_performed}, step=current_env_step)
        
        # Restore validation data to online buffer
        self._restore_validation_data()
        
        return epochs_performed