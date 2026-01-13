### AGENT CODE FOR FASPEQ ALGORITHM - Fixed Reference Batches for Policy Loss Monitoring


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


class EffectiveRankMonitor:
    """
    Monitor effective rank and gradient diversity for plasticity tracking.
    Based on "Learning Continually by Spectral Regularization" (ICLR 2025)
    """
    
    @staticmethod
    def compute_effective_rank(singular_values: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Compute effective rank from singular values.
        erank(G) = -Σ σ̄ᵢ log σ̄ᵢ (entropy of singular value distribution)
        """
        sv_sum = singular_values.sum()
        if sv_sum < eps:
            return 0.0
        
        normalized_sv = singular_values / sv_sum
        normalized_sv = normalized_sv[normalized_sv > eps]
        
        if len(normalized_sv) == 0:
            return 0.0
        
        log_sv = torch.log(normalized_sv)
        effective_rank = -torch.sum(normalized_sv * log_sv).item()
        
        return effective_rank
    
    @staticmethod
    def compute_condition_number(singular_values: torch.Tensor, eps: float = 1e-10) -> float:
        """Compute condition number (ratio of largest to smallest singular value)."""
        if len(singular_values) == 0:
            return float('inf')
        
        max_sv = singular_values[0]
        min_sv = singular_values[-1]
        
        if min_sv < eps:
            return float('inf')
        
        return (max_sv / min_sv).item()
    
    @staticmethod
    def compute_gradient_metrics(gradients: torch.Tensor) -> dict:
        """Compute spectral metrics for a gradient matrix."""
        if gradients.dim() == 1:
            gradients = gradients.unsqueeze(1)
        
        try:
            U, S, Vh = torch.linalg.svd(gradients, full_matrices=False)
        except:
            return {
                'effective_rank': 0.0,
                'condition_number': float('inf'),
                'spectral_norm': 0.0,
                'stable_rank': 0.0
            }
        
        effective_rank = EffectiveRankMonitor.compute_effective_rank(S)
        condition_number = EffectiveRankMonitor.compute_condition_number(S)
        spectral_norm = S[0].item() if len(S) > 0 else 0.0
        
        frobenius_norm_sq = torch.sum(S ** 2).item()
        stable_rank = frobenius_norm_sq / (spectral_norm ** 2 + 1e-10)
        
        return {
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'spectral_norm': spectral_norm,
            'stable_rank': stable_rank
        }


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
                 num_reference_batches=10
                 ):
        """
        FASPEQ Agent with fixed reference batches for policy loss monitoring during offline stabilization.
        
        New parameters:
            num_reference_batches: Number of batches to use as static reference for policy loss (default: 10)
        """
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
        
        # Validation replay buffer (not used in FASPEQ but kept for compatibility)
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
            # FASPEQ: Fixed 10k intervals
            self.next_trigger_size = 10000
        self.auto_stab = auto_stab
        
        # FASPEQ-specific: Number of reference batches for policy loss monitoring
        self.num_reference_batches = num_reference_batches
        
        # Storage for static reference batches during offline stabilization
        self.static_reference_batches = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None
        
        # Effective rank monitor
        self.effective_rank_monitor = EffectiveRankMonitor()

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

    def get_exploration_action_o2o(self, obs, env, timestep):
        with torch.no_grad():
            if timestep > self.start_steps:
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

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor, deterministic=False,
                                                                                   return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def store_data(self, o, a, r, o2, d):
        """Store data in main replay buffer and probabilistically in validation buffer."""
        self.replay_buffer.store(o, a, r, o2, d)
        
        if np.random.rand() < self.val_buffer_prob:
            self.val_replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        """Store offline data in offline buffer."""
        self.replay_buffer_offline.store(o, a, r, o2, d)

    def populate_val_buffer_from_offline(self):
        """Populate validation buffer with a fraction of offline data."""
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
        """Check if offline stabilization should be triggered."""
        current_size = self.replay_buffer.size

        if self.auto_stab:
            if self.val_replay_buffer.size < 1000:
                return False
            if current_size >= self.next_trigger_size:
                self.last_trigger_buffer_size = current_size
                self.next_trigger_size = int(current_size * self.adaptive_trigger_expansion_rate)
                
                print(f"\n{'='*80}")
                print(f"OFFLINE STABILIZATION TRIGGERED (FASPEQ - Adaptive)")
                print(f"Current buffer size: {current_size}")
                print(f"Next trigger at: {self.next_trigger_size}")
                print(f"{'='*80}\n")
                return True
        else:
            # FASPEQ: Fixed 10k intervals
            if current_size >= self.next_trigger_size:
                self.last_trigger_buffer_size = current_size
                self.next_trigger_size = self.last_trigger_buffer_size + 10000
                
                print(f"\n{'='*80}")
                print(f"OFFLINE STABILIZATION TRIGGERED (FASPEQ - Fixed Interval)")
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
        FASPEQ: Sample reference batches and remove them from both replay buffers.
        
        This method:
        1. Samples num_reference_batches mixed batches (half online, half offline each)
        2. Stores the data and indices
        3. Removes the sampled transitions from both buffers by rebuilding them
        
        Returns:
            List of sampled batches as tensors for policy loss evaluation
        """
        print(f"  Sampling {self.num_reference_batches} reference batches...")
        
        # Calculate samples per batch from each buffer
        samples_per_batch_per_buffer = self.batch_size  # Each mixed batch has batch_size from each buffer
        total_online_samples = self.num_reference_batches * samples_per_batch_per_buffer
        total_offline_samples = self.num_reference_batches * samples_per_batch_per_buffer
        
        # Check buffer sizes
        if self.replay_buffer.size < total_online_samples:
            print(f"  Warning: Online buffer too small ({self.replay_buffer.size} < {total_online_samples})")
            total_online_samples = min(total_online_samples, self.replay_buffer.size)
        
        if self.replay_buffer_offline.size < total_offline_samples:
            print(f"  Warning: Offline buffer too small ({self.replay_buffer_offline.size} < {total_offline_samples})")
            total_offline_samples = min(total_offline_samples, self.replay_buffer_offline.size)
        
        # Sample unique indices from each buffer
        online_indices = np.random.choice(self.replay_buffer.size, size=total_online_samples, replace=False)
        offline_indices = np.random.choice(self.replay_buffer_offline.size, size=total_offline_samples, replace=False)
        
        # Get all data from buffers
        online_data = self.replay_buffer.sample_all()
        offline_data = self.replay_buffer_offline.sample_all()
        
        # Extract the reference data
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
        
        # Store for later restoration
        self.extracted_online_indices = online_indices
        self.extracted_offline_indices = offline_indices
        self.extracted_online_data = extracted_online
        self.extracted_offline_data = extracted_offline
        
        # Create mask for remaining data (indices NOT in extracted set)
        online_mask = np.ones(self.replay_buffer.size, dtype=bool)
        online_mask[online_indices] = False
        
        offline_mask = np.ones(self.replay_buffer_offline.size, dtype=bool)
        offline_mask[offline_indices] = False
        
        # Rebuild online buffer without extracted samples
        remaining_online = {
            'obs1': online_data['obs1'][online_mask],
            'obs2': online_data['obs2'][online_mask],
            'acts': online_data['acts'][online_mask],
            'rews': online_data['rews'][online_mask],
            'done': online_data['done'][online_mask],
        }
        
        # Rebuild offline buffer without extracted samples
        remaining_offline = {
            'obs1': offline_data['obs1'][offline_mask],
            'obs2': offline_data['obs2'][offline_mask],
            'acts': offline_data['acts'][offline_mask],
            'rews': offline_data['rews'][offline_mask],
            'done': offline_data['done'][offline_mask],
        }
        
        # Clear and repopulate online buffer
        old_online_size = self.replay_buffer.size
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        for i in range(len(remaining_online['obs1'])):
            self.replay_buffer.store(
                remaining_online['obs1'][i],
                remaining_online['acts'][i],
                remaining_online['rews'][i],
                remaining_online['obs2'][i],
                remaining_online['done'][i]
            )
        
        # Clear and repopulate offline buffer
        old_offline_size = self.replay_buffer_offline.size
        self.replay_buffer_offline = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        for i in range(len(remaining_offline['obs1'])):
            self.replay_buffer_offline.store(
                remaining_offline['obs1'][i],
                remaining_offline['acts'][i],
                remaining_offline['rews'][i],
                remaining_offline['obs2'][i],
                remaining_offline['done'][i]
            )
        
        print(f"  Online buffer: {old_online_size} -> {self.replay_buffer.size} (removed {total_online_samples})")
        print(f"  Offline buffer: {old_offline_size} -> {self.replay_buffer_offline.size} (removed {total_offline_samples})")
        
        # Create static reference batches as list of tensors
        static_batches = []
        samples_per_batch = samples_per_batch_per_buffer
        
        for b in range(self.num_reference_batches):
            start_idx = b * samples_per_batch
            end_idx = (b + 1) * samples_per_batch
            
            # Combine online and offline portions for this batch
            batch_obs = np.concatenate([
                extracted_online['obs1'][start_idx:end_idx],
                extracted_offline['obs1'][start_idx:end_idx]
            ], axis=0)
            
            obs_tensor = torch.Tensor(batch_obs).to(self.device)
            static_batches.append(obs_tensor)
        
        self.static_reference_batches = static_batches
        print(f"  Created {len(static_batches)} static reference batches for policy loss monitoring")
        
        return static_batches

    def _restore_reference_batches(self):
        """
        FASPEQ: Restore the extracted reference batches back to their respective buffers.
        """
        if self.extracted_online_data is None or self.extracted_offline_data is None:
            print("  Warning: No extracted data to restore")
            return
        
        print(f"  Restoring reference batches to replay buffers...")
        
        old_online_size = self.replay_buffer.size
        old_offline_size = self.replay_buffer_offline.size
        
        # Restore online data
        for i in range(len(self.extracted_online_data['obs1'])):
            self.replay_buffer.store(
                self.extracted_online_data['obs1'][i],
                self.extracted_online_data['acts'][i],
                self.extracted_online_data['rews'][i],
                self.extracted_online_data['obs2'][i],
                self.extracted_online_data['done'][i]
            )
        
        # Restore offline data
        for i in range(len(self.extracted_offline_data['obs1'])):
            self.replay_buffer_offline.store(
                self.extracted_offline_data['obs1'][i],
                self.extracted_offline_data['acts'][i],
                self.extracted_offline_data['rews'][i],
                self.extracted_offline_data['obs2'][i],
                self.extracted_offline_data['done'][i]
            )
        
        print(f"  Online buffer: {old_online_size} -> {self.replay_buffer.size}")
        print(f"  Offline buffer: {old_offline_size} -> {self.replay_buffer_offline.size}")
        
        # Clear stored data
        self.static_reference_batches = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None

    def evaluate_policy_loss_on_static_batches(self):
        """
        FASPEQ: Evaluate policy loss on the static reference batches.
        
        Policy loss = α*entropy - Q(s, π(s))
        
        Returns:
            Average policy loss across all static reference batches
        """
        if self.static_reference_batches is None or len(self.static_reference_batches) == 0:
            return 0.0
        
        total_policy_loss = 0.0
        
        with torch.no_grad():
            for obs_tensor in self.static_reference_batches:
                # Get policy actions and log probs (POLICY IS FIXED, no gradient)
                a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
                
                # Compute Q-values for policy actions
                q_a_tilda_list = []
                for q_i in range(self.num_Q):
                    q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                
                # Policy loss: α*log_prob - Q(s, π(s))
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                total_policy_loss += policy_loss.item()
        
        average_policy_loss = total_policy_loss / len(self.static_reference_batches)
        return average_policy_loss

    def get_td_error(self):
        with torch.no_grad():
            batch = self.replay_buffer.sample_batch(1)
            obs_tensor = Tensor(batch['obs1']).to(self.device)
            obs_next_tensor = Tensor(batch['obs2']).to(self.device)
            acts_tensor = Tensor(batch['acts']).to(self.device)
            rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
            done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)

            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction = self.q_net_list[0](torch.cat([obs_tensor, acts_tensor], 1))
            td_error = (y_q - q_prediction).abs().item()
        return td_error

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

    def compute_effective_rank_metrics(self, batch_size=256):
        """Compute effective rank metrics for Q-networks."""
        metrics = {}
        
        for q_idx, q_net in enumerate(self.q_net_list):
            batch = self.replay_buffer.sample_batch(batch_size)
            obs_tensor = torch.Tensor(batch['obs1']).to(self.device)
            acts_tensor = torch.Tensor(batch['acts']).to(self.device)
            
            q_net.zero_grad()
            inputs = torch.cat([obs_tensor, acts_tensor], 1)
            inputs.requires_grad = True
            
            outputs = q_net(inputs)
            
            gradients = []
            for i in range(batch_size):
                q_net.zero_grad()
                if inputs.grad is not None:
                    inputs.grad.zero_()
                outputs[i].backward(retain_graph=True)
                if inputs.grad is not None:
                    gradients.append(inputs.grad[i].flatten())
            
            if gradients:
                gradient_matrix = torch.stack(gradients).T
                rank_metrics = self.effective_rank_monitor.compute_gradient_metrics(gradient_matrix)
                
                for key, value in rank_metrics.items():
                    metrics[f'q{q_idx}_{key}'] = value
        
        return metrics

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

    def evaluate_validation_loss(self, batch_size=256):
        """Evaluate Q-loss on validation buffer."""
        if self.val_replay_buffer.size == 0:
            return 0.0
        
        val_data = self.val_replay_buffer.sample_all()
        obs_full = val_data['obs1']
        obs_next_full = val_data['obs2']
        acts_full = val_data['acts']
        rews_full = val_data['rews']
        done_full = val_data['done']

        num_samples = len(obs_full)
        if num_samples == 0:
            return 0.0

        num_batches = (num_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            obs_tensor = torch.Tensor(obs_full[start_idx:end_idx]).to(self.device)
            obs_next_tensor = torch.Tensor(obs_next_full[start_idx:end_idx]).to(self.device)
            acts_tensor = torch.Tensor(acts_full[start_idx:end_idx]).to(self.device)
            rews_tensor = torch.Tensor(rews_full[start_idx:end_idx]).unsqueeze(1).to(self.device)
            done_tensor = torch.Tensor(done_full[start_idx:end_idx]).unsqueeze(1).to(self.device)

            with torch.no_grad():
                y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = [
                    q_net(torch.cat([obs_tensor, acts_tensor], 1)) for q_net in self.q_net_list
                ]
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q

                q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q
                total_loss += q_loss_all.item()

        average_loss = total_loss / num_batches
        return average_loss

    def evaluate_policy_loss(self, num_batches=10, batch_size=256):
        """
        Evaluate policy loss on training buffer with FIXED policy.
        For non-FASPEQ methods or fallback.
        """
        if self.replay_buffer.size == 0:
            return 0.0
        
        total_policy_loss = 0.0
        
        with torch.no_grad():
            for _ in range(num_batches):
                if self.o2o:
                    obs_tensor, _, _, _, _ = self.sample_data_mix(batch_size)
                else:
                    obs_tensor, _, _, _, _ = self.sample_data(batch_size)
                
                a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
                
                q_a_tilda_list = []
                for q_i in range(self.num_Q):
                    q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                total_policy_loss += policy_loss.item()
        
        average_policy_loss = total_policy_loss / num_batches
        return average_policy_loss

    def evaluate_policy_loss_on_validation(self, batch_size=256):
        """Evaluate policy loss on ENTIRE validation buffer with FIXED policy."""
        if self.val_replay_buffer.size == 0:
            return 0.0
        
        val_data = self.val_replay_buffer.sample_all()
        obs_full = val_data['obs1']
        
        num_samples = len(obs_full)
        if num_samples == 0:
            return 0.0
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_policy_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                obs_tensor = torch.Tensor(obs_full[start_idx:end_idx]).to(self.device)
                
                a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
                
                q_a_tilda_list = []
                for q_i in range(self.num_Q):
                    q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                
                policy_loss_sum = (self.alpha * log_prob_a_tilda - ave_q).sum()
                total_policy_loss += policy_loss_sum.item()
                total_samples += (end_idx - start_idx)
        
        average_policy_loss = total_policy_loss / total_samples
        return average_policy_loss

    def finetune_offline_policy_faspeq(self, epochs, test_env=None, current_env_step=None):
        """ 
        FASPEQ: Finetune with policy loss monitoring using STATIC reference batches.
        
        Key differences from PASPEQ:
        1. Sample 10 mixed batches once at the start of stabilization
        2. Remove those samples from both replay buffers
        3. Use them as static reference throughout the stabilization phase
        4. Restore them to buffers after stabilization ends
        
        Args:
            epochs: Maximum number of offline training epochs
            test_env: Environment for evaluation (optional)
            current_env_step: Current environment step for wandb logging
            
        Returns:
            Number of epochs actually performed
        """
        print(f"\n{'='*80}")
        print(f"STARTING FASPEQ OFFLINE STABILIZATION")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval} epochs")
        print(f"Patience: {self.val_patience} epochs")
        print(f"Monitoring: Policy loss with FIXED policy on STATIC reference batches")
        print(f"Number of reference batches: {self.num_reference_batches}")
        print(f"{'='*80}\n")
        
        # FASPEQ Step 1: Extract reference batches and remove from buffers
        print("Step 1: Extracting and removing reference batches from replay buffers...")
        self._extract_and_remove_reference_batches()
        
        # Log initial policy loss on static reference batches
        initial_loss = self.evaluate_policy_loss_on_static_batches()
        best_policy_loss = initial_loss
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed_Static": initial_loss,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"\nInitial policy loss (fixed policy, static batches): {initial_loss:.6f}\n")
        
        epochs_performed = 0
        
        try:
            for e in range(epochs):
                epochs_performed = e
                
                # Print policy loss every 100 epochs for monitoring
                if e > 0 and e % 100 == 0:
                    current_policy_loss = self.evaluate_policy_loss_on_static_batches()
                    print(f"  Epoch {e:6d}: Policy loss = {current_policy_loss:.6f}")
                
                # Check policy loss at regular intervals for early stopping
                if e > 0 and e % self.val_check_interval == 0:
                    policy_loss = self.evaluate_policy_loss_on_static_batches()
                    
                    if current_env_step is not None:
                        wandb.log({
                            "PolicyLoss_Fixed_Static": policy_loss,
                            "offline_epoch": e,
                            "offline_training": 1
                        }, step=current_env_step)
                    
                    # Early stopping logic based on policy loss
                    improvement = (best_policy_loss - policy_loss) / best_policy_loss * 100 if best_policy_loss > 0 else 0
                    if policy_loss < best_policy_loss * 0.999:  # 0.1% improvement threshold
                        best_policy_loss = policy_loss
                        steps_without_improvement = 0
                        print(f"Epoch {e:6d}: Policy loss improved to {policy_loss:.6f} (↓ {improvement:.2f}%) ✓")
                    else:
                        steps_without_improvement += self.val_check_interval
                        trend = "↑" if policy_loss > best_policy_loss else "→"
                        print(f"Epoch {e:6d}: Policy loss {policy_loss:.6f} {trend} (no improvement for {steps_without_improvement} steps)")
                    
                    # Stop if no improvement for val_patience steps
                    if steps_without_improvement >= self.val_patience:
                        print(f"\n{'='*80}")
                        print(f"EARLY STOPPING TRIGGERED")
                        print(f"Epoch: {e}")
                        print(f"No improvement for {steps_without_improvement} steps")
                        print(f"Best policy loss: {best_policy_loss:.6f}")
                        print(f"Current policy loss: {policy_loss:.6f}")
                        print(f"{'='*80}\n")
                        if current_env_step is not None:
                            wandb.log({
                                "offline_early_stop_epoch": e,
                                "offline_training": 0
                            }, step=current_env_step)
                        epochs_performed = e
                        break
                
                # Evaluate on test environment every 5000 epochs
                if test_env and (e + 1) % 5000 == 0:
                    test_rw = test_agent(self, test_env, 1000, None)
                    if current_env_step is not None:
                        wandb.log({
                            "OfflineEvalReward": np.mean(test_rw),
                            "offline_training": 1
                        }, step=current_env_step)
                
                # Sample data and train Q-networks only (policy stays fixed)
                # Note: We sample from the buffers WITHOUT the reference batches
                if self.o2o:
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                        self.batch_size)
                else:
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                        self.batch_size)
                
                """Q loss with expectile loss"""
                y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = []
                for q_i in range(self.num_Q):
                    q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                    q_prediction_list.append(q_prediction)
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                q_loss_all = self.expectile_loss(q_prediction_cat - y_q,).mean() * self.num_Q

                # Update Q-networks only
                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].zero_grad()
                q_loss_all.backward()

                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].step()

                # Update target networks
                for q_i in range(self.num_Q):
                    soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                    self.polyak)
            else:
                # Loop completed without early stopping
                epochs_performed = epochs
        
        finally:
            # FASPEQ Step 3: Restore reference batches to buffers (always executed)
            print("\nStep 3: Restoring reference batches to replay buffers...")
            self._restore_reference_batches()
        
        # Log completion
        final_policy_loss = self.evaluate_policy_loss()  # Use regular evaluation after restore
        print(f"\n{'='*80}")
        print(f"FASPEQ OFFLINE STABILIZATION COMPLETED")
        print(f"{'='*80}")
        print(f"Epochs performed: {epochs_performed}")
        print(f"Initial policy loss: {initial_loss:.6f}")
        print(f"Final policy loss: {final_policy_loss:.6f}")
        print(f"Total improvement: {(initial_loss - final_policy_loss)/initial_loss*100:.2f}%")
        print(f"{'='*80}\n")
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed": final_policy_loss,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs_performed

    # Keep other finetuning methods for compatibility
    def finetune_offline_auto(self, epochs, test_env=None, current_env_step=None):
        """Finetune with validation-based early stopping (Q-loss based)."""
        print(f"\nStarting offline stabilization (max {epochs} epochs)...")
        
        initial_loss = self.evaluate_validation_loss()
        best_val_loss = initial_loss
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "ValLoss": initial_loss,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"Initial validation loss: {initial_loss:.4f}")
       
        for e in range(epochs):
            if e > 0 and e % self.val_check_interval == 0:
                val_loss = self.evaluate_validation_loss()
                
                if current_env_step is not None:
                    wandb.log({
                        "ValLoss": val_loss,
                        "offline_epoch": e,
                        "offline_training": 1
                    }, step=current_env_step)
                
                if val_loss < best_val_loss * 0.999:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    print(f"Epoch {e}: Val loss improved to {val_loss:.4f}")
                else:
                    steps_without_improvement += self.val_check_interval
                    print(f"Epoch {e}: Val loss {val_loss:.4f} (no improvement for {steps_without_improvement} steps)")
                
                if steps_without_improvement >= self.val_patience:
                    print(f"\nEarly stopping at epoch {e}: No improvement for {steps_without_improvement} steps")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    if current_env_step is not None:
                        wandb.log({
                            "offline_early_stop_epoch": e,
                            "offline_training": 0
                        }, step=current_env_step)
                    return e
            
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step is not None:
                    wandb.log({
                        "OfflineEvalReward": np.mean(test_rw),
                        "offline_training": 1
                    }, step=current_env_step)
            
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                    self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                self.batch_size)
            
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
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                self.polyak)
        
        final_val_loss = self.evaluate_validation_loss()
        print(f"\nOffline stabilization completed: {epochs} epochs")
        print(f"Final validation loss: {final_val_loss:.4f}")
        
        if current_env_step is not None:
            wandb.log({
                "ValLoss": final_val_loss,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs

    def finetune_offline(self, epochs, test_env=None, current_env_step=None):
        """Finetune the model on the replay buffer data (fixed epochs, no early stopping)."""
        initial_loss = self.evaluate_validation_loss()
        if current_env_step is not None:
            wandb.log({"ValLoss": initial_loss}, step=current_env_step)
       
        for e in range(epochs):
            if e % 1000 == 0 and e > 0:
                val_loss = self.evaluate_validation_loss()
                if current_env_step is not None:
                    wandb.log({"ValLoss": val_loss}, step=current_env_step)
            
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step is not None:
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)
            
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                    self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                self.batch_size)
            
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
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                self.polyak)

    def finetune_offline_policy(self, epochs, test_env=None, current_env_step=None):
        """PASPEQ: Finetune with policy loss monitoring (samples new batches each check)."""
        print(f"\n{'='*80}")
        print(f"STARTING PASPEQ OFFLINE STABILIZATION")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval} epochs")
        print(f"Patience: {self.val_patience} epochs")
        print(f"Monitoring: Policy loss with FIXED policy on training buffer")
        print(f"{'='*80}\n")
        
        initial_loss = self.evaluate_policy_loss()
        best_policy_loss = initial_loss
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed": initial_loss,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"Initial policy loss (fixed policy): {initial_loss:.6f}\n")
       
        for e in range(epochs):
            if e > 0 and e % 100 == 0:
                current_policy_loss = self.evaluate_policy_loss()
                print(f"  Epoch {e:6d}: Policy loss = {current_policy_loss:.6f}")
            
            if e > 0 and e % self.val_check_interval == 0:
                policy_loss = self.evaluate_policy_loss()
                
                if current_env_step is not None:
                    wandb.log({
                        "PolicyLoss_Fixed": policy_loss,
                        "offline_epoch": e,
                        "offline_training": 1
                    }, step=current_env_step)
                
                improvement = (best_policy_loss - policy_loss) / best_policy_loss * 100 if best_policy_loss > 0 else 0
                if policy_loss < best_policy_loss * 0.999:
                    best_policy_loss = policy_loss
                    steps_without_improvement = 0
                    print(f"Epoch {e:6d}: Policy loss improved to {policy_loss:.6f} (↓ {improvement:.2f}%) ✓")
                else:
                    steps_without_improvement += self.val_check_interval
                    trend = "↑" if policy_loss > best_policy_loss else "→"
                    print(f"Epoch {e:6d}: Policy loss {policy_loss:.6f} {trend} (no improvement for {steps_without_improvement} steps)")
                
                if steps_without_improvement >= self.val_patience:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"Epoch: {e}")
                    print(f"No improvement for {steps_without_improvement} steps")
                    print(f"Best policy loss: {best_policy_loss:.6f}")
                    print(f"Current policy loss: {policy_loss:.6f}")
                    print(f"{'='*80}\n")
                    if current_env_step is not None:
                        wandb.log({
                            "offline_early_stop_epoch": e,
                            "offline_training": 0
                        }, step=current_env_step)
                    return e
            
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step is not None:
                    wandb.log({
                        "OfflineEvalReward": np.mean(test_rw),
                        "offline_training": 1
                    }, step=current_env_step)
            
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                    self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                self.batch_size)
            
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
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                self.polyak)
        
        final_policy_loss = self.evaluate_policy_loss()
        print(f"\n{'='*80}")
        print(f"PASPEQ OFFLINE STABILIZATION COMPLETED")
        print(f"{'='*80}")
        print(f"Epochs performed: {epochs}")
        print(f"Initial policy loss: {initial_loss:.6f}")
        print(f"Final policy loss: {final_policy_loss:.6f}")
        print(f"Total improvement: {(initial_loss - final_policy_loss)/initial_loss*100:.2f}%")
        print(f"{'='*80}\n")
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed": final_policy_loss,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs

    def finetune_offline_policy_auto(self, epochs, test_env=None, current_env_step=None):
        """APASPEQ: Finetune with policy loss monitoring on validation set."""
        print(f"\n{'='*80}")
        print(f"STARTING APASPEQ OFFLINE STABILIZATION")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval} epochs")
        print(f"Patience: {self.val_patience} epochs")
        print(f"Monitoring: Policy loss with FIXED policy on validation buffer")
        print(f"Validation buffer size: {self.val_replay_buffer.size}")
        print(f"{'='*80}\n")
        
        initial_loss = self.evaluate_policy_loss_on_validation()
        best_policy_loss = initial_loss
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed_Val": initial_loss,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"Initial validation policy loss (fixed policy): {initial_loss:.6f}\n")
       
        for e in range(epochs):
            if e > 0 and e % 100 == 0:
                current_policy_loss = self.evaluate_policy_loss_on_validation()
                print(f"  Epoch {e:6d}: Validation policy loss = {current_policy_loss:.6f}")
            
            if e > 0 and e % self.val_check_interval == 0:
                policy_loss = self.evaluate_policy_loss_on_validation()
                
                if current_env_step is not None:
                    wandb.log({
                        "PolicyLoss_Fixed_Val": policy_loss,
                        "offline_epoch": e,
                        "offline_training": 1
                    }, step=current_env_step)
                
                improvement = (best_policy_loss - policy_loss) / best_policy_loss * 100 if best_policy_loss > 0 else 0
                if policy_loss < best_policy_loss * 0.999:
                    best_policy_loss = policy_loss
                    steps_without_improvement = 0
                    print(f"Epoch {e:6d}: Validation policy loss improved to {policy_loss:.6f} (↓ {improvement:.2f}%) ✓")
                else:
                    steps_without_improvement += self.val_check_interval
                    trend = "↑" if policy_loss > best_policy_loss else "→"
                    print(f"Epoch {e:6d}: Validation policy loss {policy_loss:.6f} {trend} (no improvement for {steps_without_improvement} steps)")
                
                if steps_without_improvement >= self.val_patience:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"Epoch: {e}")
                    print(f"No improvement for {steps_without_improvement} steps")
                    print(f"Best validation policy loss: {best_policy_loss:.6f}")
                    print(f"Current validation policy loss: {policy_loss:.6f}")
                    print(f"{'='*80}\n")
                    if current_env_step is not None:
                        wandb.log({
                            "offline_early_stop_epoch": e,
                            "offline_training": 0
                        }, step=current_env_step)
                    return e
            
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step is not None:
                    wandb.log({
                        "OfflineEvalReward": np.mean(test_rw),
                        "offline_training": 1
                    }, step=current_env_step)
            
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                    self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                self.batch_size)
            
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
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                self.polyak)
        
        final_policy_loss = self.evaluate_policy_loss_on_validation()
        print(f"\n{'='*80}")
        print(f"APASPEQ OFFLINE STABILIZATION COMPLETED")
        print(f"{'='*80}")
        print(f"Epochs performed: {epochs}")
        print(f"Initial validation policy loss: {initial_loss:.6f}")
        print(f"Final validation policy loss: {final_policy_loss:.6f}")
        print(f"Total improvement: {(initial_loss - final_policy_loss)/initial_loss*100:.2f}%")
        print(f"{'='*80}\n")
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed_Val": final_policy_loss,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs
