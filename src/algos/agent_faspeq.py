"""
FASPEQ Agent - Fast Adaptive SPEQ
=================================
Same triggering as SPEQ (every 10k env steps), but with early stopping
to reduce stabilization length when validation metrics plateau.

Validation approach:
- At start of each stabilization phase, sample and FIX 5 online + 5 offline batches
- These batches remain constant throughout the stabilization phase
- Early stopping based on policy loss (FASPEQ_O2O) or TD error (FASPEQ_TD_VAL)

Variants:
- FASPEQ_O2O: Early stopping based on policy loss on fixed validation batches
- FASPEQ_TD_VAL: Early stopping based on TD error on fixed validation batches
"""

import numpy as np
import torch
import wandb

from src.algos.agent_speq import SPEQAgent
from src.algos.core import test_agent


class FASPEQAgent(SPEQAgent):
    
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
        n_val_batches: int = 5,  # Number of batches from each buffer for validation
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
        
        # Fixed validation batches (sampled at start of each stabilization)
        self.val_batches_online = None
        self.val_batches_offline = None
    
    def _sample_fixed_validation_batches(self):
        """Sample and fix validation batches at start of stabilization phase."""
        # Sample n_val_batches from online buffer
        online_batches = []
        if self.replay_buffer.size >= self.batch_size:
            for _ in range(self.n_val_batches):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                online_batches.append({
                    'obs': torch.FloatTensor(batch['obs1']).to(self.device),
                    'obs_next': torch.FloatTensor(batch['obs2']).to(self.device),
                    'acts': torch.FloatTensor(batch['acts']).to(self.device),
                    'rews': torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device),
                    'done': torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device),
                })
        
        # Sample n_val_batches from offline buffer
        offline_batches = []
        if self.replay_buffer_offline.size >= self.batch_size:
            for _ in range(self.n_val_batches):
                batch = self.replay_buffer_offline.sample_batch(self.batch_size)
                offline_batches.append({
                    'obs': torch.FloatTensor(batch['obs1']).to(self.device),
                    'obs_next': torch.FloatTensor(batch['obs2']).to(self.device),
                    'acts': torch.FloatTensor(batch['acts']).to(self.device),
                    'rews': torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device),
                    'done': torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device),
                })
        
        self.val_batches_online = online_batches
        self.val_batches_offline = offline_batches
        
        total_batches = len(online_batches) + len(offline_batches)
        print(f"  Fixed validation batches: {len(online_batches)} online + {len(offline_batches)} offline = {total_batches} total")
    
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
        
        # Sample and fix validation batches at start of stabilization
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
            
            # Q-network update (always 50/50 symmetric sampling)
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
        
        # Clear validation batches to free memory
        self.val_batches_online = None
        self.val_batches_offline = None
        
        return epochs_performed
