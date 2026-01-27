"""
FASPEQ Agent - Fast Adaptive SPEQ
=================================
Same triggering as SPEQ (every 10k env steps), but with early stopping
to reduce stabilization length when validation metrics plateau.

Validation approach:
- At start of each stabilization phase, sample and FIX validation batches
- These batches are REMOVED from training buffers for the stabilization phase
- After stabilization, validation data is RESTORED to the buffers
- Early stopping based on policy loss (FASPEQ_O2O) or TD error (FASPEQ_TD_VAL)

Variants:
- FASPEQ_O2O: Fixed n_val_batches from each buffer, early stopping on policy loss
- FASPEQ_TD_VAL: Fixed n_val_batches from each buffer, early stopping on TD error
- FASPEQ_PCT: Percentage-based validation set (val_pct of online buffer size from each buffer)
"""

import numpy as np
import torch
import wandb

from src.algos.agent_speq import SPEQAgent
from src.algos.core import test_agent


class FASPEQAgent(SPEQAgent):
    """
    FASPEQ Agent with validation data removal from training buffers.
    
    Args:
        val_check_interval: How often to check validation metrics (default: 1000)
        val_patience: Steps without improvement before early stopping (default: 10000)
        use_td_val: If True, use TD error for validation; else use policy loss (default: False)
        n_val_batches: Number of batches from each buffer for validation (default: 5)
        val_pct: If > 0, use percentage-based validation instead of n_val_batches (default: 0.0)
                 The percentage is computed on online buffer size, then same count sampled from offline
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
        o2o: bool = True,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        # FASPEQ-specific
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        use_td_val: bool = False,
        n_val_batches: int = 5,
        val_pct: float = 0.0,  # NEW: percentage-based validation
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm,
            o2o=True, offline_epochs=offline_epochs, trigger_interval=trigger_interval
        )
        
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.use_td_val = use_td_val
        self.n_val_batches = n_val_batches
        self.val_pct = val_pct
        
        # Fixed validation batches (sampled at start of each stabilization)
        # These store the actual data as torch tensors for validation evaluation
        self.val_batches_online = None
        self.val_batches_offline = None
        
        # Store the raw numpy data for restoration after stabilization
        self._removed_online_data = None  # dict with obs1, obs2, acts, rews, done
        self._removed_offline_data = None  # dict with obs1, obs2, acts, rews, done
    
    def _sample_fixed_validation_batches(self):
        """
        Sample and fix validation batches at start of stabilization phase.
        REMOVES sampled data from both online and offline replay buffers.
        Stores raw data for restoration after stabilization.
        
        If val_pct > 0: Sample val_pct% of online buffer size from each buffer
        Else: Sample n_val_batches * batch_size from each buffer
        """
        online_batches = []
        offline_batches = []
        
        # Determine how many transitions to sample
        if self.val_pct > 0:
            # Percentage-based: compute based on online buffer size
            n_val_transitions = int(self.replay_buffer.size * self.val_pct)
            n_val_transitions = max(self.batch_size, n_val_transitions)  # At least one batch
        else:
            # Fixed batch count
            n_val_transitions = self.n_val_batches * self.batch_size
        
        # Sample from online buffer
        if self.replay_buffer.size >= self.batch_size:
            # Don't sample more than available (keep at least batch_size for training)
            n_online_samples = min(n_val_transitions, self.replay_buffer.size - self.batch_size)
            if n_online_samples >= self.batch_size:
                # Sample indices without replacement
                all_online_indices = np.random.choice(
                    self.replay_buffer.size, 
                    size=n_online_samples, 
                    replace=False
                )
                
                # Store raw data for restoration BEFORE removal
                self._removed_online_data = {
                    'obs1': self.replay_buffer.obs1_buf[all_online_indices].copy(),
                    'obs2': self.replay_buffer.obs2_buf[all_online_indices].copy(),
                    'acts': self.replay_buffer.acts_buf[all_online_indices].copy(),
                    'rews': self.replay_buffer.rews_buf[all_online_indices].copy(),
                    'done': self.replay_buffer.done_buf[all_online_indices].copy(),
                }
                
                # Convert to batches for validation evaluation
                n_complete_batches = n_online_samples // self.batch_size
                for i in range(n_complete_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batch_indices = all_online_indices[start_idx:end_idx]
                    
                    online_batches.append({
                        'obs': torch.FloatTensor(self.replay_buffer.obs1_buf[batch_indices].copy()).to(self.device),
                        'obs_next': torch.FloatTensor(self.replay_buffer.obs2_buf[batch_indices].copy()).to(self.device),
                        'acts': torch.FloatTensor(self.replay_buffer.acts_buf[batch_indices].copy()).to(self.device),
                        'rews': torch.FloatTensor(self.replay_buffer.rews_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                        'done': torch.FloatTensor(self.replay_buffer.done_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                    })
                
                # Remove from buffer
                self.replay_buffer.remove_indices(all_online_indices)
        
        # Sample from offline buffer (same count as online for symmetry)
        if self.replay_buffer_offline.size >= self.batch_size:
            # Use same number of transitions as online (or available)
            n_offline_samples = min(n_val_transitions, self.replay_buffer_offline.size - self.batch_size)
            if n_offline_samples >= self.batch_size:
                all_offline_indices = np.random.choice(
                    self.replay_buffer_offline.size,
                    size=n_offline_samples,
                    replace=False
                )
                
                # Store raw data for restoration BEFORE removal
                self._removed_offline_data = {
                    'obs1': self.replay_buffer_offline.obs1_buf[all_offline_indices].copy(),
                    'obs2': self.replay_buffer_offline.obs2_buf[all_offline_indices].copy(),
                    'acts': self.replay_buffer_offline.acts_buf[all_offline_indices].copy(),
                    'rews': self.replay_buffer_offline.rews_buf[all_offline_indices].copy(),
                    'done': self.replay_buffer_offline.done_buf[all_offline_indices].copy(),
                }
                
                # Convert to batches for validation evaluation
                n_complete_batches = n_offline_samples // self.batch_size
                for i in range(n_complete_batches):
                    start_idx = i * self.batch_size
                    end_idx = start_idx + self.batch_size
                    batch_indices = all_offline_indices[start_idx:end_idx]
                    
                    offline_batches.append({
                        'obs': torch.FloatTensor(self.replay_buffer_offline.obs1_buf[batch_indices].copy()).to(self.device),
                        'obs_next': torch.FloatTensor(self.replay_buffer_offline.obs2_buf[batch_indices].copy()).to(self.device),
                        'acts': torch.FloatTensor(self.replay_buffer_offline.acts_buf[batch_indices].copy()).to(self.device),
                        'rews': torch.FloatTensor(self.replay_buffer_offline.rews_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                        'done': torch.FloatTensor(self.replay_buffer_offline.done_buf[batch_indices].copy()).unsqueeze(1).to(self.device),
                    })
                
                # Remove from buffer
                self.replay_buffer_offline.remove_indices(all_offline_indices)
        
        # Store batches for validation
        self.val_batches_online = online_batches
        self.val_batches_offline = offline_batches
        
        total_online = len(online_batches) * self.batch_size
        total_offline = len(offline_batches) * self.batch_size
        
        if self.val_pct > 0:
            print(f"  Validation set ({self.val_pct*100:.1f}%): {len(online_batches)} online + {len(offline_batches)} offline batches")
        else:
            print(f"  Validation set (n_val_batches={self.n_val_batches}): {len(online_batches)} online + {len(offline_batches)} offline batches")
        print(f"  Removed {total_online} online + {total_offline} offline transitions from training")
        print(f"  Remaining: online={self.replay_buffer.size}, offline={self.replay_buffer_offline.size}")
    
    def _restore_validation_data(self):
        """
        Restore validation data to buffers after stabilization phase.
        This ensures the buffers don't progressively shrink across stabilization phases.
        """
        # Restore online data
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
            print(f"  Restored {n_restored} online transitions")
        
        # Restore offline data
        if self._removed_offline_data is not None:
            n_restored = len(self._removed_offline_data['obs1'])
            for i in range(n_restored):
                self.replay_buffer_offline.store(
                    self._removed_offline_data['obs1'][i],
                    self._removed_offline_data['acts'][i],
                    self._removed_offline_data['rews'][i],
                    self._removed_offline_data['obs2'][i],
                    self._removed_offline_data['done'][i]
                )
            print(f"  Restored {n_restored} offline transitions")
        
        print(f"  Final buffer sizes: online={self.replay_buffer.size}, offline={self.replay_buffer_offline.size}")
        
        # Clear stored data to free memory
        self.val_batches_online = None
        self.val_batches_offline = None
        self._removed_online_data = None
        self._removed_offline_data = None
    
    def _evaluate_policy_loss_on_fixed_batches(self) -> float:
        """Evaluate policy loss on fixed validation batches."""
        if not self.val_batches_online and not self.val_batches_offline:
            return 0.0
        
        total_loss = 0.0
        n_batches = 0
        
        all_batches = self.val_batches_online + self.val_batches_offline
        
        for batch in all_batches:
            with torch.no_grad():
                action, _, _, log_prob, _, _ = self.policy_net.forward(batch['obs'])
                q_values = [q_net(torch.cat([batch['obs'], action], 1)) for q_net in self.q_net_list]
                q_mean = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - q_mean).mean()
                total_loss += policy_loss.item()
                n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    def _evaluate_td_loss_on_fixed_batches(self) -> float:
        """Evaluate TD error on fixed validation batches."""
        if not self.val_batches_online and not self.val_batches_offline:
            return 0.0
        
        total_loss = 0.0
        n_batches = 0
        
        all_batches = self.val_batches_online + self.val_batches_offline
        
        for batch in all_batches:
            with torch.no_grad():
                y_q = self.get_sac_q_target(batch['obs_next'], batch['rews'], batch['done'])
                q_preds = [q_net(torch.cat([batch['obs'], batch['acts']], 1)) for q_net in self.q_net_list]
                q_cat = torch.cat(q_preds, dim=1)
                y_q_expanded = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                td_loss = self.mse_criterion(q_cat, y_q_expanded).item() * self.num_Q
                total_loss += td_loss
                n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    def finetune_offline(self, epochs: int = None, test_env=None, current_env_step: int = None) -> int:
        """Stabilization with early stopping based on validation metrics on FIXED batches."""
        epochs = epochs or self.offline_epochs
        
        # Sample and fix validation batches at start of stabilization (also removes from training)
        self._sample_fixed_validation_batches()
        
        # Initial validation loss
        if self.use_td_val:
            initial_loss = self._evaluate_td_loss_on_fixed_batches()
            metric_name = "TD_loss"
        else:
            initial_loss = self._evaluate_policy_loss_on_fixed_batches()
            metric_name = "policy_loss"
        
        print(f"  Initial val {metric_name}: {initial_loss:.6f}")
        
        best_loss = initial_loss
        steps_without_improvement = 0
        epochs_performed = 0
        
        for e in range(epochs):
            epochs_performed = e + 1
            
            # Early stopping check
            if e > 0 and e % self.val_check_interval == 0:
                if self.use_td_val:
                    val_loss = self._evaluate_td_loss_on_fixed_batches()
                else:
                    val_loss = self._evaluate_policy_loss_on_fixed_batches()
                
                # Print validation metrics
                improved = val_loss < best_loss * 0.999
                status = "improved" if improved else "no improvement"
                print(f"  Epoch {e}: val {metric_name}={val_loss:.6f} (best={best_loss:.6f}, {status}, patience={steps_without_improvement}/{self.val_patience})")
                
                if improved:
                    best_loss = val_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += self.val_check_interval
                
                if steps_without_improvement >= self.val_patience:
                    print(f"  Early stop @ epoch {e} (no improvement for {self.val_patience} steps)")
                    break
            
            # Q-network update (always 50/50 symmetric sampling from REMAINING data)
            obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            
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
            
            # Progress logging every 5k epochs
            if (e + 1) % 5000 == 0:
                if self.use_td_val:
                    current_val = self._evaluate_td_loss_on_fixed_batches()
                else:
                    current_val = self._evaluate_policy_loss_on_fixed_batches()
                print(f"  Offline epoch {e+1}/{epochs}, val {metric_name}={current_val:.6f}")
                
                if test_env is not None and current_env_step is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)
        
        # Final validation loss
        if self.use_td_val:
            final_loss = self._evaluate_td_loss_on_fixed_batches()
        else:
            final_loss = self._evaluate_policy_loss_on_fixed_batches()
        print(f"  Final val {metric_name}: {final_loss:.6f} (started at {initial_loss:.6f})")
        
        if current_env_step is not None:
            wandb.log({"OfflineEpochs": epochs_performed}, step=current_env_step)
        
        # RESTORE validation data to buffers after stabilization
        self._restore_validation_data()
        
        return epochs_performed


class FASPEQPctAgent(FASPEQAgent):
    """
    FASPEQ variant that uses percentage-based validation set sizing.
    
    This is a convenience class that sets val_pct by default.
    The validation set size is determined by val_pct of the online buffer size,
    then the same count is sampled from the offline buffer for symmetry.
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
        o2o: bool = True,
        offline_epochs: int = 75000,
        trigger_interval: int = 10000,
        val_check_interval: int = 1000,
        val_patience: int = 10000,
        use_td_val: bool = False,
        val_pct: float = 0.1,  # Default 10% of online buffer
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm,
            o2o=o2o, offline_epochs=offline_epochs, trigger_interval=trigger_interval,
            val_check_interval=val_check_interval, val_patience=val_patience,
            use_td_val=use_td_val, n_val_batches=0, val_pct=val_pct
        )