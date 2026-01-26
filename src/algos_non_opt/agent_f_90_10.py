### AGENT CODE FOR FASPEQ_90_10 ALGORITHM - 10% Online Buffer Validation Set


import copy

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.algos_non_opt.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict, soft_update_policy
import wandb

from src.algos_non_opt.core import test_agent


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
                 validation_fraction=0.1
                 ):
        """
        FASPEQ_90_10 Agent: Validation set is 10% of online replay buffer size.
        
        Key difference from FASPEQ:
            validation_fraction: Fraction of online buffer to use as validation (default: 0.1 = 10%)
            The same number of samples is taken from the offline buffer.
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
        
        # Validation replay buffer (not used in FASPEQ_90_10 but kept for compatibility)
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
            # FASPEQ_90_10: Fixed 10k intervals
            self.next_trigger_size = 10000
        self.auto_stab = auto_stab
        
        # FASPEQ_90_10-specific: Validation fraction of online buffer
        self.validation_fraction = validation_fraction
        
        # Storage for static reference data during offline stabilization
        self.static_reference_obs = None  # Combined online+offline observations
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
                print(f"OFFLINE STABILIZATION TRIGGERED (FASPEQ_90_10 - Adaptive)")
                print(f"Current buffer size: {current_size}")
                print(f"Next trigger at: {self.next_trigger_size}")
                print(f"{'='*80}\n")
                return True
        else:
            # FASPEQ_90_10: Fixed 10k intervals
            if current_size >= self.next_trigger_size:
                self.last_trigger_buffer_size = current_size
                self.next_trigger_size = self.last_trigger_buffer_size + 10000
                
                print(f"\n{'='*80}")
                print(f"OFFLINE STABILIZATION TRIGGERED (FASPEQ_90_10 - Fixed Interval)")
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

    def _extract_and_remove_validation_set(self):
        """
        FASPEQ_90_10: Sample 10% of online buffer and equal amount from offline buffer.
        
        This method:
        1. Calculates 10% of current online replay buffer size
        2. Samples that number of transitions from online buffer
        3. Samples the same number from offline buffer
        4. Combines them into a mixed validation set
        5. Removes sampled transitions from both buffers
        
        Returns:
            Combined observation tensor for policy loss evaluation
        """
        # Calculate validation set size: 10% of online buffer
        num_val_samples = int(self.replay_buffer.size * self.validation_fraction)
        
        print(f"  Validation set sizing:")
        print(f"    Online buffer size: {self.replay_buffer.size}")
        print(f"    Validation fraction: {self.validation_fraction * 100:.1f}%")
        print(f"    Samples from each buffer: {num_val_samples}")
        print(f"    Total validation samples: {num_val_samples * 2}")
        
        # Ensure we have enough samples
        if num_val_samples < 1:
            print(f"  Warning: Online buffer too small for 10% validation")
            num_val_samples = max(1, self.replay_buffer.size)
        
        if self.replay_buffer.size < num_val_samples:
            print(f"  Warning: Online buffer smaller than requested samples")
            num_val_samples = self.replay_buffer.size
        
        if self.replay_buffer_offline.size < num_val_samples:
            print(f"  Warning: Offline buffer smaller than online 10%")
            num_val_samples = min(num_val_samples, self.replay_buffer_offline.size)
        
        # Sample unique indices from each buffer
        online_indices = np.random.choice(self.replay_buffer.size, size=num_val_samples, replace=False)
        offline_indices = np.random.choice(self.replay_buffer_offline.size, size=num_val_samples, replace=False)
        
        # Get all data from buffers
        online_data = self.replay_buffer.sample_all()
        offline_data = self.replay_buffer_offline.sample_all()
        
        # Extract the validation data
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
        
        print(f"  Online buffer: {old_online_size} -> {self.replay_buffer.size} (removed {num_val_samples})")
        print(f"  Offline buffer: {old_offline_size} -> {self.replay_buffer_offline.size} (removed {num_val_samples})")
        
        # Create combined validation set (observations only for policy loss)
        combined_obs = np.concatenate([
            extracted_online['obs1'],
            extracted_offline['obs1']
        ], axis=0)
        
        self.static_reference_obs = torch.Tensor(combined_obs).to(self.device)
        print(f"  Created validation set with {len(combined_obs)} total observations")
        
        return self.static_reference_obs

    def _restore_validation_set(self):
        """
        FASPEQ_90_10: Restore the extracted validation data back to their respective buffers.
        """
        if self.extracted_online_data is None or self.extracted_offline_data is None:
            print("  Warning: No extracted data to restore")
            return
        
        print(f"  Restoring validation data to replay buffers...")
        
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
        self.static_reference_obs = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None

    def evaluate_policy_loss_on_validation_set(self):
        """
        FASPEQ_90_10: Evaluate policy loss on the static validation set.
        
        Policy loss = α*entropy - Q(s, π(s))
        
        Returns:
            Average policy loss across all validation samples
        """
        if self.static_reference_obs is None or len(self.static_reference_obs) == 0:
            return 0.0
        
        with torch.no_grad():
            # Get policy actions and log probs
            a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(self.static_reference_obs)
            
            # Compute Q-values for policy actions
            q_a_tilda_list = []
            for q_i in range(self.num_Q):
                q_a_tilda = self.q_net_list[q_i](torch.cat([self.static_reference_obs, a_tilda], 1))
                q_a_tilda_list.append(q_a_tilda)
            q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
            ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
            
            # Policy loss: α*log_prob - Q(s, π(s))
            policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
        
        return policy_loss.item()

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
                raise ValueError(f"Unknown q_target_mode: {self.q_target_mode}")
            
            y_q = rews_tensor + self.gamma * (1 - done_tensor) * (q_target - self.alpha * log_prob_a_tilda_next)
        return y_q, sample_idxs

    def expectile_loss(self, diff, expectile=0.8):
        """Asymmetric squared loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * diff ** 2

    def train(self, logger, current_env_step=None):
        if self.__get_current_num_data() < self.start_steps:
            return
        
        for _ in range(self.utd_ratio):
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(
                    self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(
                    self.batch_size)
            
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

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            """Policy loss"""
            a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
            q_a_tilda_list = []
            for q_i in range(self.num_Q):
                q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                q_a_tilda_list.append(q_a_tilda)
            q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
            ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
            policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            """Alpha loss"""
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_prob_a_tilda.detach() + self.target_entropy)).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0)

            """Update target networks"""
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                               self.polyak)

            """Logging"""
            logger.store(Q1Vals=q_prediction_list[0].detach().cpu().numpy(),
                        LossQ1=q_loss_all.item(),
                        LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                        LossPi=policy_loss.item(),
                        Alpha=self.alpha,
                        LossAlpha=alpha_loss.item(),
                        PreTanh=a_tilda.detach().cpu().numpy())

    def evaluate_policy_loss(self):
        """Evaluate policy loss on a sample from training buffers."""
        with torch.no_grad():
            if self.o2o:
                obs_tensor, _, _, _, _ = self.sample_data_mix(self.batch_size * 10)
            else:
                obs_tensor, _, _, _, _ = self.sample_data(self.batch_size * 10)
            
            a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
            
            q_a_tilda_list = []
            for q_i in range(self.num_Q):
                q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                q_a_tilda_list.append(q_a_tilda)
            q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
            ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
            
            policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
        
        return policy_loss.item()

    def evaluate_validation_loss(self):
        """Evaluate Q-network loss on validation buffer."""
        if self.val_replay_buffer.size < self.batch_size:
            return float('inf')
        
        with torch.no_grad():
            batch = self.val_replay_buffer.sample_batch(min(self.batch_size * 5, self.val_replay_buffer.size))
            obs_tensor = Tensor(batch['obs1']).to(self.device)
            obs_next_tensor = Tensor(batch['obs2']).to(self.device)
            acts_tensor = Tensor(batch['acts']).to(self.device)
            rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
            done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)

            y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction = self.q_net_list[0](torch.cat([obs_tensor, acts_tensor], 1))
            val_loss = self.mse_criterion(q_prediction, y_q)
        
        return val_loss.item()

    def compute_effective_rank_metrics(self, batch_size=256):
        """Compute effective rank metrics for policy and Q networks."""
        metrics = {}
        
        if self.o2o and self.replay_buffer_offline.size >= batch_size:
            obs_tensor, _, _, _, _ = self.sample_data_mix(batch_size)
        elif self.replay_buffer.size >= batch_size:
            obs_tensor, _, _, _, _ = self.sample_data(batch_size)
        else:
            return metrics
        
        # Policy network gradients
        self.policy_optimizer.zero_grad()
        a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
        
        q_a_tilda_list = []
        for q_i in range(self.num_Q):
            q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
        policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
        policy_loss.backward()
        
        # Collect policy gradients
        policy_grads = []
        for param in self.policy_net.parameters():
            if param.grad is not None:
                policy_grads.append(param.grad.view(-1))
        
        if policy_grads:
            policy_grad_tensor = torch.cat(policy_grads)
            policy_metrics = self.effective_rank_monitor.compute_gradient_metrics(policy_grad_tensor)
            metrics['policy_effective_rank'] = policy_metrics['effective_rank']
            metrics['policy_condition_number'] = policy_metrics['condition_number']
            metrics['policy_spectral_norm'] = policy_metrics['spectral_norm']
            metrics['policy_stable_rank'] = policy_metrics['stable_rank']
        
        # Q-network gradients
        for q_i in range(min(2, self.num_Q)):  # Only track first 2 Q-networks
            self.q_optimizer_list[q_i].zero_grad()
            
            if self.o2o and self.replay_buffer_offline.size >= batch_size:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(batch_size)
            
            y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_loss = self.mse_criterion(q_prediction, y_q)
            q_loss.backward()
            
            q_grads = []
            for param in self.q_net_list[q_i].parameters():
                if param.grad is not None:
                    q_grads.append(param.grad.view(-1))
            
            if q_grads:
                q_grad_tensor = torch.cat(q_grads)
                q_metrics = self.effective_rank_monitor.compute_gradient_metrics(q_grad_tensor)
                metrics[f'q{q_i+1}_effective_rank'] = q_metrics['effective_rank']
                metrics[f'q{q_i+1}_condition_number'] = q_metrics['condition_number']
                metrics[f'q{q_i+1}_spectral_norm'] = q_metrics['spectral_norm']
                metrics[f'q{q_i+1}_stable_rank'] = q_metrics['stable_rank']
        
        return metrics

    def finetune_offline_policy_faspeq_90_10(self, epochs, test_env=None, current_env_step=None):
        """ 
        FASPEQ_90_10: Finetune with policy loss monitoring using 10% of online buffer as validation.
        
        Key differences from FASPEQ:
        1. Validation set size = 10% of online replay buffer
        2. Sample that amount from both online and offline buffers
        3. Remove sampled data during stabilization, restore after
        
        Args:
            epochs: Maximum number of offline training epochs
            test_env: Environment for evaluation (optional)
            current_env_step: Current environment step for wandb logging
            
        Returns:
            Number of epochs actually performed
        """
        print(f"\n{'='*80}")
        print(f"STARTING FASPEQ_90_10 OFFLINE STABILIZATION")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval} epochs")
        print(f"Patience: {self.val_patience} epochs")
        print(f"Monitoring: Policy loss on 10% validation set")
        print(f"Validation fraction: {self.validation_fraction * 100:.1f}%")
        print(f"{'='*80}\n")
        
        # Step 1: Extract validation set and remove from buffers
        print("Step 1: Extracting validation set (10% of online buffer from each source)...")
        self._extract_and_remove_validation_set()
        
        # Log initial policy loss on validation set
        initial_loss = self.evaluate_policy_loss_on_validation_set()
        best_policy_loss = initial_loss
        steps_without_improvement = 0
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed_Val90_10": initial_loss,
                "offline_training": 1
            }, step=current_env_step)
        
        print(f"\nInitial policy loss (validation set): {initial_loss:.6f}\n")
        
        epochs_performed = 0
        
        try:
            for e in range(epochs):
                epochs_performed = e
                
                # Print policy loss every 100 epochs for monitoring
                if e > 0 and e % 100 == 0:
                    current_policy_loss = self.evaluate_policy_loss_on_validation_set()
                    print(f"  Epoch {e:6d}: Policy loss = {current_policy_loss:.6f}")
                
                # Check policy loss at regular intervals for early stopping
                if e > 0 and e % self.val_check_interval == 0:
                    policy_loss = self.evaluate_policy_loss_on_validation_set()
                    
                    if current_env_step is not None:
                        wandb.log({
                            "PolicyLoss_Fixed_Val90_10": policy_loss,
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
            # Step 2: Restore validation set to buffers (always executed)
            print("\nStep 2: Restoring validation set to replay buffers...")
            self._restore_validation_set()
        
        # Log completion
        final_policy_loss = self.evaluate_policy_loss()  # Use regular evaluation after restore
        print(f"\n{'='*80}")
        print(f"FASPEQ_90_10 OFFLINE STABILIZATION COMPLETED")
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

    # Keep compatibility methods
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
