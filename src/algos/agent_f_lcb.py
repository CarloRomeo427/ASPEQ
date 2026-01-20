### AGENT CODE FOR FASPEQ-LCB EXPERIMENT
### Implements explicit Variance/LCB Monitoring for Early Stopping

import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict, soft_update_policy
import wandb
from src.algos.core import test_agent

def get_probabilistic_num_min(num_mins):
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins + 1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins

class Agent(object):
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
                 policy_update_delay=20, 
                 target_drop_rate=0.0, layer_norm=False, o2o=False,
                 val_buffer_prob=0.1, val_buffer_offline_frac=0.1,
                 val_check_interval=1000, val_patience=5000,
                 adaptive_trigger_expansion_rate=1.1, auto_stab=False,
                 num_reference_batches=10,
                 # Flag to enable LCB mode stored in agent if needed, though mostly handled in main/methods
                 faspeq_lcb_mode=False
                 ):
        
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, (256, 256), action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                            layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                                   layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        self.auto_alpha = auto_alpha

        self.o2o = o2o

        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = - act_dim
            if target_entropy == 'mbpo':
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2
                try:
                    self.target_entropy = mbpo_target_entropy_dict[env_name]
                except:
                    self.target_entropy = -2
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        
        # Main replay buffers
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.replay_buffer_offline = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        # Validation replay buffer (kept for compatibility with other modes)
        self.val_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size * 0.2))
        
        self.mse_criterion = nn.MSELoss()
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.device = device
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm
        
        # Validation and adaptive triggering parameters
        self.val_buffer_prob = val_buffer_prob
        self.val_buffer_offline_frac = val_buffer_offline_frac
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.adaptive_trigger_expansion_rate = adaptive_trigger_expansion_rate
        
        # Adaptive triggering state
        self.last_trigger_buffer_size = 0
        if auto_stab:
            self.next_trigger_size = int(start_steps * adaptive_trigger_expansion_rate)
        else:
            self.next_trigger_size = 10000
        self.auto_stab = auto_stab
        
        # Reference batches logic
        self.num_reference_batches = num_reference_batches
        self.static_reference_batches = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None
        
        self.faspeq_lcb_mode = faspeq_lcb_mode

    def __get_current_num_data(self):
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)
        if np.random.rand() < self.val_buffer_prob:
            self.val_replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        self.replay_buffer_offline.store(o, a, r, o2, d)

    def populate_val_buffer_from_offline(self):
        if self.replay_buffer_offline.size == 0:
            return
        
        num_samples = int(self.replay_buffer_offline.size * self.val_buffer_offline_frac)
        print(f"Adding {num_samples} offline samples to validation buffer...")
        
        all_data = self.replay_buffer_offline.sample_all()
        indices = np.random.choice(self.replay_buffer_offline.size, 
                                   size=min(num_samples, self.replay_buffer_offline.size), 
                                   replace=False)
        
        for idx in indices:
            self.val_replay_buffer.store(
                all_data['obs1'][idx],
                all_data['acts'][idx],
                all_data['rews'][idx],
                all_data['obs2'][idx],
                all_data['done'][idx]
            )
        print(f"Validation buffer now contains {self.val_replay_buffer.size} samples")

    def check_should_trigger_offline_stabilization(self):
        current_size = self.replay_buffer.size

        if self.auto_stab:
            if self.val_replay_buffer.size < 1000:
                return False
            if current_size >= self.next_trigger_size:
                self.last_trigger_buffer_size = current_size
                self.next_trigger_size = int(current_size * self.adaptive_trigger_expansion_rate)
                
                print(f"\n{'='*80}")
                print(f"OFFLINE STABILIZATION TRIGGERED (Adaptive)")
                print(f"Current buffer size: {current_size}")
                print(f"Next trigger at: {self.next_trigger_size}")
                print(f"{'='*80}\n")
                return True
        else:
            if current_size >= self.next_trigger_size:
                self.last_trigger_buffer_size = current_size
                self.next_trigger_size = self.last_trigger_buffer_size + 10000
                
                print(f"\n{'='*80}")
                print(f"OFFLINE STABILIZATION TRIGGERED (Fixed Interval)")
                print(f"Current buffer size: {current_size}")
                print(f"Next trigger at: {self.next_trigger_size}")
                print(f"{'='*80}\n")
                return True
        
        return False

    def sample_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
    def sample_data_mix(self, batch_size):
        batch_online = self.replay_buffer.sample_batch(batch_size)
        batch_offline = self.replay_buffer_offline.sample_batch(batch_size)

        batch = {
            'obs1': np.concatenate([batch_online['obs1'], batch_offline['obs1']], axis=0),
            'obs2': np.concatenate([batch_online['obs2'], batch_offline['obs2']], axis=0),
            'acts': np.concatenate([batch_online['acts'], batch_offline['acts']], axis=0),
            'rews': np.concatenate([batch_online['rews'], batch_offline['rews']], axis=0),
            'done': np.concatenate([batch_online['done'], batch_offline['done']], axis=0),
        }

        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def _extract_and_remove_reference_batches(self):
        """
        Sample reference batches and remove them from both replay buffers.
        """
        print(f"  Sampling {self.num_reference_batches} reference batches...")
        
        samples_per_batch_per_buffer = self.batch_size
        total_online_samples = self.num_reference_batches * samples_per_batch_per_buffer
        total_offline_samples = self.num_reference_batches * samples_per_batch_per_buffer
        
        if self.replay_buffer.size < total_online_samples:
            total_online_samples = min(total_online_samples, self.replay_buffer.size)
        if self.replay_buffer_offline.size < total_offline_samples:
            total_offline_samples = min(total_offline_samples, self.replay_buffer_offline.size)
        
        online_indices = np.random.choice(self.replay_buffer.size, size=total_online_samples, replace=False)
        offline_indices = np.random.choice(self.replay_buffer_offline.size, size=total_offline_samples, replace=False)
        
        online_data = self.replay_buffer.sample_all()
        offline_data = self.replay_buffer_offline.sample_all()
        
        extracted_online = {
            'obs1': online_data['obs1'][online_indices].copy(),
            'obs2': online_data['obs2'][online_indices].copy(),
            'acts': online_data['acts'][online_indices].copy(),
            'rews': online_data['rews'][online_indices].copy(),
            'done': online_data['done'][online_indices].copy(),
        }
        
        extracted_offline = {
            'obs1': offline_data['obs1'][offline_indices].copy(),
            'obs2': offline_data['obs2'][offline_indices].copy(),
            'acts': offline_data['acts'][offline_indices].copy(),
            'rews': offline_data['rews'][offline_indices].copy(),
            'done': offline_data['done'][offline_indices].copy(),
        }
        
        self.extracted_online_indices = online_indices
        self.extracted_offline_indices = offline_indices
        self.extracted_online_data = extracted_online
        self.extracted_offline_data = extracted_offline
        
        online_mask = np.ones(self.replay_buffer.size, dtype=bool)
        online_mask[online_indices] = False
        
        offline_mask = np.ones(self.replay_buffer_offline.size, dtype=bool)
        offline_mask[offline_indices] = False
        
        # Rebuild buffers
        remaining_online = {k: v[online_mask] for k, v in online_data.items()}
        remaining_offline = {k: v[offline_mask] for k, v in offline_data.items()}
        
        old_online_size = self.replay_buffer.size
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        for i in range(len(remaining_online['obs1'])):
            self.replay_buffer.store(
                remaining_online['obs1'][i], remaining_online['acts'][i],
                remaining_online['rews'][i], remaining_online['obs2'][i], remaining_online['done'][i]
            )
        
        old_offline_size = self.replay_buffer_offline.size
        self.replay_buffer_offline = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        for i in range(len(remaining_offline['obs1'])):
            self.replay_buffer_offline.store(
                remaining_offline['obs1'][i], remaining_offline['acts'][i],
                remaining_offline['rews'][i], remaining_offline['obs2'][i], remaining_offline['done'][i]
            )
        
        print(f"  Online buffer: {old_online_size} -> {self.replay_buffer.size}")
        print(f"  Offline buffer: {old_offline_size} -> {self.replay_buffer_offline.size}")
        
        static_batches = []
        for b in range(self.num_reference_batches):
            start_idx = b * samples_per_batch_per_buffer
            end_idx = (b + 1) * samples_per_batch_per_buffer
            
            batch_obs = np.concatenate([
                extracted_online['obs1'][start_idx:end_idx],
                extracted_offline['obs1'][start_idx:end_idx]
            ], axis=0)
            
            obs_tensor = torch.Tensor(batch_obs).to(self.device)
            static_batches.append(obs_tensor)
        
        self.static_reference_batches = static_batches
        print(f"  Created {len(static_batches)} static reference batches for monitoring")
        return static_batches

    def _restore_reference_batches(self):
        """Restore the extracted reference batches back to their respective buffers."""
        if self.extracted_online_data is None or self.extracted_offline_data is None:
            print("  Warning: No extracted data to restore")
            return
        
        print(f"  Restoring reference batches to replay buffers...")
        
        for i in range(len(self.extracted_online_data['obs1'])):
            self.replay_buffer.store(
                self.extracted_online_data['obs1'][i], self.extracted_online_data['acts'][i],
                self.extracted_online_data['rews'][i], self.extracted_online_data['obs2'][i],
                self.extracted_online_data['done'][i]
            )
        
        for i in range(len(self.extracted_offline_data['obs1'])):
            self.replay_buffer_offline.store(
                self.extracted_offline_data['obs1'][i], self.extracted_offline_data['acts'][i],
                self.extracted_offline_data['rews'][i], self.extracted_offline_data['obs2'][i],
                self.extracted_offline_data['done'][i]
            )
        
        self.static_reference_batches = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None

    def evaluate_lcb_on_static_batches(self, beta=1.0):
        """
        Evaluate Lower Confidence Bound (LCB) on static reference batches.
        LCB = Mean - Beta * Std
        
        Returns:
            Mean Q, Std Q, LCB
        """
        if self.static_reference_batches is None or len(self.static_reference_batches) == 0:
            return 0.0, 0.0, 0.0
        
        total_mean = 0.0
        total_std = 0.0
        total_lcb = 0.0
        
        with torch.no_grad():
            for obs_tensor in self.static_reference_batches:
                # Get policy actions (fixed policy)
                a_tilda, _, _, _, _, _ = self.policy_net.forward(obs_tensor)
                
                # Get Q-values from all ensemble members
                q_list = []
                for q_i in range(self.num_Q):
                    q_val = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_list.append(q_val)
                q_cat = torch.cat(q_list, 1) # [Batch, Num_Q]
                
                # Calculate Statistics per state
                q_mean = torch.mean(q_cat, dim=1)
                q_std = torch.std(q_cat, dim=1)
                q_lcb = q_mean - beta * q_std
                
                total_mean += q_mean.mean().item()
                total_std += q_std.mean().item()
                total_lcb += q_lcb.mean().item()
        
        N = len(self.static_reference_batches)
        return total_mean / N, total_std / N, total_lcb / N

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        with torch.no_grad():
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for q_i in range(self.num_Q):
                q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            
            sample_idxs = None
            
            if self.q_target_mode == 'min':
                q_target = torch.min(q_prediction_next_cat, 1, keepdim=True)[0]
            elif self.q_target_mode == 'ave':
                q_target = torch.mean(q_prediction_next_cat, 1, keepdim=True)
            elif self.q_target_mode == 'rem':
                num_mins_to_use = int(self.num_min)
                sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
                q_prediction_next_cat_min = q_prediction_next_cat[:, sample_idxs]
                q_target = torch.min(q_prediction_next_cat_min, 1, keepdim=True)[0]
            else:
                num_mins_to_use = get_probabilistic_num_min(self.num_min)
                sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
                q_prediction_next_cat_min = q_prediction_next_cat[:, sample_idxs]
                q_target = torch.min(q_prediction_next_cat_min, 1, keepdim=True)[0]
            y_q = rews_tensor + self.gamma * (1 - done_tensor) * (
                    q_target - self.alpha * log_prob_a_tilda_next
            )
        return y_q, sample_idxs

    def train(self, logger, current_env_step=None):
        """Standard online training update."""
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):

            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(
                    obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

                if current_env_step is not None:
                    wandb.log({
                        "policy_loss": policy_loss.cpu().item(), 
                        "mean_loss_q": q_loss_all.cpu().item() / self.num_Q
                    }, step=current_env_step)

        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    @staticmethod
    def expectile_loss(diff, expectile=0.5):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def finetune_offline_lcb(self, epochs, test_env=None, current_env_step=None):
        """
        FASPEQ-LCB: Offline stabilization monitoring explicitly the LCB (Mean - Std).
        We STOP when LCB starts decreasing (i.e., variance penalty outweighs mean gain).
        """
        print(f"\n{'='*80}")
        print(f"STARTING FASPEQ-LCB OFFLINE STABILIZATION (Optimizing LCB = Mean - Std)")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval}")
        print(f"Patience: {self.val_patience}")
        print(f"{'='*80}\n")
        
        # Step 1: Extract reference batches and remove from buffers
        self._extract_and_remove_reference_batches()
        
        # Log initial LCB metrics
        init_mean, init_std, init_lcb = self.evaluate_lcb_on_static_batches(beta=1.0)
        
        # We want to MAXIMIZE LCB. 
        best_lcb = init_lcb
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "Val_Q_Mean": init_mean,
                "Val_Q_Std": init_std,
                "Val_LCB": init_lcb,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"Initial: Mean={init_mean:.4f}, Std={init_std:.4f}, LCB={init_lcb:.4f}\n")
        
        epochs_performed = 0
        
        try:
            for e in range(epochs):
                epochs_performed = e
                
                # Check metrics at regular intervals
                if e > 0 and e % self.val_check_interval == 0:
                    cur_mean, cur_std, cur_lcb = self.evaluate_lcb_on_static_batches(beta=1.0)
                    
                    if current_env_step is not None:
                        wandb.log({
                            "Val_Q_Mean": cur_mean,
                            "Val_Q_Std": cur_std,
                            "Val_LCB": cur_lcb,
                            "offline_epoch": e,
                            "offline_training": 1
                        }, step=current_env_step)
                    
                    # Logic: Maximize LCB
                    # If current LCB is higher than best, we update best and reset patience
                    if cur_lcb > best_lcb:
                        improvement = (cur_lcb - best_lcb)
                        best_lcb = cur_lcb
                        steps_without_improvement = 0
                        print(f"Epoch {e:6d}: LCB improved to {cur_lcb:.4f} (Mean {cur_mean:.2f} | Std {cur_std:.2f}) ✓")
                    else:
                        steps_without_improvement += self.val_check_interval
                        print(f"Epoch {e:6d}: LCB {cur_lcb:.4f} (Mean {cur_mean:.2f} | Std {cur_std:.2f}) - No improv. for {steps_without_improvement}")
                    
                    # Stop if no improvement for val_patience steps
                    if steps_without_improvement >= self.val_patience:
                        print(f"\n{'='*80}")
                        print(f"EARLY STOPPING TRIGGERED: Variance Penalty Dominated Signal")
                        print(f"Epoch: {e}")
                        print(f"Best LCB: {best_lcb:.4f}")
                        print(f"Current LCB: {cur_lcb:.4f}")
                        print(f"{'='*80}\n")
                        if current_env_step is not None:
                            wandb.log({
                                "offline_early_stop_epoch": e,
                                "offline_training": 0
                            }, step=current_env_step)
                        break
                
                # Evaluate on test environment every 5000 epochs
                if test_env and (e + 1) % 5000 == 0:
                    test_rw = test_agent(self, test_env, 1000, None)
                    if current_env_step is not None:
                        wandb.log({
                            "OfflineEvalReward": np.mean(test_rw),
                            "offline_training": 1
                        }, step=current_env_step)
                
                # Train Q-networks only (Standard loop without reference batches)
                if self.o2o:
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(self.batch_size)
                else:
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
                
                """Q loss with expectile loss"""
                y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = []
                for q_i in range(self.num_Q):
                    q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                    q_prediction_list.append(q_prediction)
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                q_loss_all = self.expectile_loss(q_prediction_cat - y_q,).mean() * self.num_Q

                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].zero_grad()
                q_loss_all.backward()

                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].step()

                for q_i in range(self.num_Q):
                    soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)
        
        finally:
            # Step 3: Restore reference batches to buffers (always executed)
            print("\nRestoring reference batches to replay buffers...")
            self._restore_reference_batches()
        
        final_mean, final_std, final_lcb = self.evaluate_lcb_on_static_batches()
        print(f"\nFASPEQ-LCB OFFLINE STABILIZATION COMPLETED")
        print(f"Final: Mean={final_mean:.4f}, Std={final_std:.4f}, LCB={final_lcb:.4f}")
        
        if current_env_step is not None:
            wandb.log({
                "Val_LCB": final_lcb,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs_performed