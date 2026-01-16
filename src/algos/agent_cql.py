### AGENT CODE FOR CQL (Conservative Q-Learning) ALGORITHM
### Based on: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
### Reference implementation: https://github.com/young-geng/CQL
### Adapted to match FASPEQ/IQL codebase structure

import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict
import wandb

from src.algos.core import test_agent


class Agent(object):
    """
    CQL Agent - Conservative Q-Learning for Offline RL.
    
    CQL learns conservative Q-values by adding a regularizer that:
    1. Minimizes Q-values under the learned policy (OOD actions)
    2. Maximizes Q-values under the dataset distribution (in-distribution actions)
    
    CQL Loss = TD_loss + α * (E_π[Q(s,a)] - E_D[Q(s,a)])
    
    In practice, uses logsumexp for soft maximum approximation:
    CQL Loss = TD_loss + α * (logsumexp_a Q(s,a) - E_D[Q(s,a)])
    
    Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
                 policy_update_delay=1,
                 target_drop_rate=0.0, layer_norm=False, o2o=False,
                 # CQL specific parameters
                 cql_alpha=5.0, cql_n_actions=10, cql_temp=1.0,
                 cql_max_target_backup=False, cql_clip_diff_min=-np.inf,
                 cql_clip_diff_max=np.inf, cql_lagrange=False,
                 cql_target_action_gap=1.0,
                 # Mixing ratio for online fine-tuning
                 mixing_ratio=0.5,
                 # Validation buffer params (kept for compatibility)
                 val_buffer_prob=0.0, val_buffer_offline_frac=0.0,
                 val_check_interval=1000, val_patience=5000,
                 adaptive_trigger_expansion_rate=1.1, auto_stab=False,
                 num_reference_batches=10
                 ):
        """
        CQL Agent.
        
        CQL specific parameters:
            cql_alpha: Weight for conservative regularizer (default: 5.0)
            cql_n_actions: Number of actions to sample for CQL logsumexp (default: 10)
            cql_temp: Temperature for CQL logsumexp (default: 1.0)
            cql_lagrange: Use Lagrangian version of CQL (default: False)
            cql_target_action_gap: Target action gap for Lagrange version (default: 1.0)
            mixing_ratio: Ratio of offline data in mixed batches during fine-tuning (default: 0.5)
        """
        # Policy network (SAC-style)
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        
        # Q networks (ensemble)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                            layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                                   layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        
        # CQL specific hyperparameters
        self.cql_alpha = cql_alpha
        self.cql_n_actions = cql_n_actions
        self.cql_temp = cql_temp
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.mixing_ratio = mixing_ratio
        
        # Lagrange multiplier for CQL (if using Lagrange version)
        if cql_lagrange:
            self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.cql_alpha_optimizer = optim.Adam([self.log_cql_alpha], lr=lr)
        
        # Auto alpha for SAC entropy
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = -act_dim
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
        
        self.o2o = o2o
        
        # Replay buffers
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.replay_buffer_offline = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.val_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size * 0.2))
        
        # Loss criterion
        self.mse_criterion = nn.MSELoss()
        
        # Store hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
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
        
        # Validation and adaptive triggering (kept for compatibility)
        self.val_buffer_prob = val_buffer_prob
        self.val_buffer_offline_frac = val_buffer_offline_frac
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.adaptive_trigger_expansion_rate = adaptive_trigger_expansion_rate
        self.last_trigger_buffer_size = 0
        self.next_trigger_size = 10000
        self.auto_stab = auto_stab
        self.num_reference_batches = num_reference_batches
        
        # Static reference batches (kept for compatibility)
        self.static_reference_batches = None
        self.extracted_online_indices = None
        self.extracted_offline_indices = None
        self.extracted_online_data = None
        self.extracted_offline_data = None

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
        """Store data in main (online) replay buffer."""
        self.replay_buffer.store(o, a, r, o2, d)
        if np.random.rand() < self.val_buffer_prob:
            self.val_replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        """Store offline data in offline buffer."""
        self.replay_buffer_offline.store(o, a, r, o2, d)

    def check_should_trigger_offline_stabilization(self):
        """CQL doesn't use offline stabilization, always returns False."""
        return False

    def sample_data(self, batch_size):
        """Sample from online buffer only."""
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
    def sample_data_mix(self, batch_size):
        """
        Sample mixed batch from online and offline buffers based on mixing_ratio.
        mixing_ratio=0.5 means 50% offline, 50% online.
        """
        offline_batch_size = int(batch_size * self.mixing_ratio)
        online_batch_size = batch_size - offline_batch_size
        
        # Handle case where online buffer might be too small
        if self.replay_buffer.size < online_batch_size:
            online_batch_size = max(1, self.replay_buffer.size)
            offline_batch_size = batch_size - online_batch_size
        
        batch_offline = self.replay_buffer_offline.sample_batch(offline_batch_size)
        batch_online = self.replay_buffer.sample_batch(online_batch_size)

        batch = {
            'obs1': np.concatenate([batch_offline['obs1'], batch_online['obs1']], axis=0),
            'obs2': np.concatenate([batch_offline['obs2'], batch_online['obs2']], axis=0),
            'acts': np.concatenate([batch_offline['acts'], batch_online['acts']], axis=0),
            'rews': np.concatenate([batch_offline['rews'], batch_online['rews']], axis=0),
            'done': np.concatenate([batch_offline['done'], batch_online['done']], axis=0),
        }

        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
    def sample_data_offline_only(self, batch_size):
        """Sample batch from offline buffer only (for offline pretraining)."""
        batch = self.replay_buffer_offline.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def compute_cql_q_loss(self, obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor):
        """
        CQL Q Loss: Standard TD loss + Conservative regularizer.
        
        CQL regularizer = α * (logsumexp_a Q(s,a) - E_D[Q(s,a)])
        
        This pushes down Q-values for OOD actions (sampled from policy)
        while keeping Q-values high for in-distribution actions (from dataset).
        """
        batch_size = obs_tensor.shape[0]
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. STANDARD TD TARGET COMPUTATION
        # ═══════════════════════════════════════════════════════════════════
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, _, _, next_log_probs, _, _ = self.policy_net.forward(
                obs_next_tensor, deterministic=False, return_log_prob=True
            )
            
            # Compute target Q-values
            if self.cql_max_target_backup:
                # Max over sampled actions (more conservative)
                next_q_values = []
                for _ in range(self.cql_n_actions):
                    next_a, _, _, next_lp, _, _ = self.policy_net.forward(
                        obs_next_tensor, deterministic=False, return_log_prob=True
                    )
                    q_vals = torch.stack([self.q_target_net_list[q_i](
                        torch.cat([obs_next_tensor, next_a], 1)
                    ) for q_i in range(self.num_Q)], dim=0)
                    next_q_values.append(q_vals.min(dim=0)[0])
                target_q_values = torch.stack(next_q_values, dim=0).max(dim=0)[0]
            else:
                # Standard: min over Q-networks, entropy bonus
                target_q_list = []
                for q_i in range(self.num_Q):
                    target_q = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, next_actions], 1))
                    target_q_list.append(target_q)
                target_q_values = torch.min(torch.cat(target_q_list, dim=1), dim=1, keepdim=True)[0]
                target_q_values = target_q_values - self.alpha * next_log_probs
            
            # TD target
            td_target = rews_tensor + self.gamma * (1 - done_tensor) * target_q_values

        # ═══════════════════════════════════════════════════════════════════
        # 2. COMPUTE Q-VALUES FOR DATASET ACTIONS
        # ═══════════════════════════════════════════════════════════════════
        q_data_list = []
        for q_i in range(self.num_Q):
            q_data = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_data_list.append(q_data)

        # ═══════════════════════════════════════════════════════════════════
        # 3. CQL CONSERVATIVE REGULARIZER
        # ═══════════════════════════════════════════════════════════════════
        
        # Sample random actions uniformly
        random_actions = torch.FloatTensor(
            self.cql_n_actions, batch_size, self.act_dim
        ).uniform_(-1, 1).to(self.device)
        random_log_prob = np.log(0.5 ** self.act_dim)  # Uniform distribution log prob
        
        # Sample actions from current policy
        policy_actions_list = []
        policy_log_probs_list = []
        for _ in range(self.cql_n_actions):
            pi_actions, _, _, pi_log_probs, _, _ = self.policy_net.forward(
                obs_tensor, deterministic=False, return_log_prob=True
            )
            policy_actions_list.append(pi_actions.unsqueeze(0))
            policy_log_probs_list.append(pi_log_probs.unsqueeze(0))
        
        policy_actions = torch.cat(policy_actions_list, dim=0)  # [n_actions, batch, act_dim]
        policy_log_probs = torch.cat(policy_log_probs_list, dim=0)  # [n_actions, batch, 1]
        
        # Also sample actions from current policy at next states (for importance sampling)
        next_policy_actions_list = []
        next_policy_log_probs_list = []
        for _ in range(self.cql_n_actions):
            next_pi_actions, _, _, next_pi_log_probs, _, _ = self.policy_net.forward(
                obs_next_tensor, deterministic=False, return_log_prob=True
            )
            next_policy_actions_list.append(next_pi_actions.unsqueeze(0))
            next_policy_log_probs_list.append(next_pi_log_probs.unsqueeze(0))
        
        next_policy_actions = torch.cat(next_policy_actions_list, dim=0)
        next_policy_log_probs = torch.cat(next_policy_log_probs_list, dim=0)
        
        # Compute CQL loss for each Q-network
        cql_loss_total = 0
        td_loss_total = 0
        q_values_for_log = []
        cql_diff_for_log = []
        
        for q_i in range(self.num_Q):
            # Q-values for random actions: [n_actions, batch, 1]
            q_random = torch.stack([
                self.q_net_list[q_i](torch.cat([obs_tensor, random_actions[j]], 1))
                for j in range(self.cql_n_actions)
            ], dim=0)
            
            # Q-values for policy actions: [n_actions, batch, 1]
            q_policy = torch.stack([
                self.q_net_list[q_i](torch.cat([obs_tensor, policy_actions[j]], 1))
                for j in range(self.cql_n_actions)
            ], dim=0)
            
            # Q-values for next policy actions: [n_actions, batch, 1]
            q_next_policy = torch.stack([
                self.q_net_list[q_i](torch.cat([obs_tensor, next_policy_actions[j]], 1))
                for j in range(self.cql_n_actions)
            ], dim=0)
            
            # Importance sampling weights (subtract log probs for proper weighting)
            # For random actions: subtract uniform log prob
            q_random_is = q_random / self.cql_temp - random_log_prob
            # For policy actions: subtract policy log prob
            q_policy_is = q_policy / self.cql_temp - policy_log_probs / self.cql_temp
            # For next policy actions: subtract next policy log prob  
            q_next_policy_is = q_next_policy / self.cql_temp - next_policy_log_probs / self.cql_temp
            
            # Concatenate all Q-values: [3*n_actions, batch, 1]
            cat_q = torch.cat([q_random_is, q_policy_is, q_next_policy_is], dim=0)
            
            # LogSumExp over actions (conservative estimate)
            cql_logsumexp = torch.logsumexp(cat_q, dim=0) * self.cql_temp
            
            # CQL penalty: logsumexp - Q(s, a_data)
            cql_diff = cql_logsumexp - q_data_list[q_i]
            
            # Clip the difference (optional, for stability)
            cql_diff = torch.clamp(cql_diff, self.cql_clip_diff_min, self.cql_clip_diff_max)
            
            cql_loss = cql_diff.mean()
            
            # TD loss
            td_loss = self.mse_criterion(q_data_list[q_i], td_target)
            
            cql_loss_total += cql_loss
            td_loss_total += td_loss
            q_values_for_log.append(q_data_list[q_i].mean().item())
            cql_diff_for_log.append(cql_diff.mean().item())
        
        # Get current CQL alpha (potentially learned via Lagrange)
        if self.cql_lagrange:
            current_cql_alpha = torch.exp(self.log_cql_alpha).clamp(min=0.0, max=1e6)
        else:
            current_cql_alpha = self.cql_alpha
        
        # Total Q loss
        total_q_loss = td_loss_total + current_cql_alpha * cql_loss_total
        
        # Lagrange alpha update
        alpha_loss = 0.0
        if self.cql_lagrange:
            # Update alpha to maintain target action gap
            alpha_loss = -self.log_cql_alpha * (cql_loss_total.detach() / self.num_Q - self.cql_target_action_gap)
        
        return total_q_loss, td_loss_total.item() / self.num_Q, cql_loss_total.item() / self.num_Q, \
               np.mean(q_values_for_log), np.mean(cql_diff_for_log), alpha_loss, current_cql_alpha

    def compute_policy_loss(self, obs_tensor):
        """
        SAC-style policy loss: maximize Q-value minus entropy penalty.
        """
        actions, _, _, log_probs, _, _ = self.policy_net.forward(
            obs_tensor, deterministic=False, return_log_prob=True
        )
        
        # Get Q-values for policy actions
        q_values_list = []
        for q_i in range(self.num_Q):
            q_val = self.q_net_list[q_i](torch.cat([obs_tensor, actions], 1))
            q_values_list.append(q_val)
        
        # Use minimum Q-value (for stability)
        q_values = torch.min(torch.cat(q_values_list, dim=1), dim=1, keepdim=True)[0]
        
        # Policy loss: minimize (alpha * log_prob - Q)
        policy_loss = (self.alpha * log_probs - q_values).mean()
        
        return policy_loss, log_probs.mean().item()

    def train(self, logger, current_env_step=None):
        """
        CQL training update for online fine-tuning phase.
        
        Uses mixed batches from offline and online data.
        """
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        
        # Initialize logging variables
        td_loss_val, cql_loss_val, q_mean = 0.0, 0.0, 0.0
        policy_loss_val, log_pi_mean = 0.0, 0.0
        alpha_loss_val = 0.0
        cql_alpha_val = self.cql_alpha
        
        for i_update in range(num_update):
            # Sample mixed batch (offline + online)
            if self.o2o and self.replay_buffer.size > 0:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = \
                    self.sample_data_mix(self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = \
                    self.sample_data(self.batch_size)

            # ═══════════════════════════════════════════════════════════════════
            # 1. Q-FUNCTION UPDATE (CQL)
            # ═══════════════════════════════════════════════════════════════════
            total_q_loss, td_loss_val, cql_loss_val, q_mean, cql_diff, alpha_loss, cql_alpha_val = \
                self.compute_cql_q_loss(
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
                )
            
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            total_q_loss.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()
            
            # Update CQL alpha if using Lagrange
            if self.cql_lagrange:
                self.cql_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.cql_alpha_optimizer.step()

            # ═══════════════════════════════════════════════════════════════════
            # 2. POLICY UPDATE (SAC-style)
            # ═══════════════════════════════════════════════════════════════════
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                policy_loss, log_pi_mean = self.compute_policy_loss(obs_tensor)
                policy_loss_val = policy_loss.item()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # SAC entropy alpha update
                if self.auto_alpha:
                    _, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(
                        obs_tensor, deterministic=False, return_log_prob=True
                    )
                    alpha_loss_sac = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss_sac.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                    alpha_loss_val = alpha_loss_sac.item()

            # ═══════════════════════════════════════════════════════════════════
            # 3. TARGET NETWORK UPDATE
            # ═══════════════════════════════════════════════════════════════════
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

        # Log to EpochLogger (required)
        logger.store(Q1Vals=q_mean, LossQ1=td_loss_val,
                     LogPi=log_pi_mean, LossPi=policy_loss_val,
                     Alpha=self.alpha, LossAlpha=alpha_loss_val,
                     PreTanh=0.0, LossV=cql_loss_val, VMean=cql_diff if num_update > 0 else 0.0)
        
        # WandB logging
        if num_update > 0 and current_env_step is not None:
            wandb.log({
                "CQL/td_loss": td_loss_val,
                "CQL/cql_loss": cql_loss_val,
                "CQL/policy_loss": policy_loss_val,
                "CQL/q_mean": q_mean,
                "CQL/alpha": self.alpha,
                "CQL/cql_alpha": cql_alpha_val if isinstance(cql_alpha_val, float) else cql_alpha_val.item(),
            }, step=current_env_step)

    def train_offline(self, epochs, test_env=None, current_env_step=None, log_interval=1000, max_ep_len=1000):
        """
        CQL offline pretraining phase.
        
        Uses only offline data with full CQL conservative regularizer.
        """
        print(f"\n{'='*80}")
        print(f"STARTING CQL OFFLINE TRAINING")
        print(f"{'='*80}")
        print(f"Epochs: {epochs}")
        print(f"CQL Alpha: {self.cql_alpha}")
        print(f"CQL N Actions: {self.cql_n_actions}")
        print(f"CQL Lagrange: {self.cql_lagrange}")
        if self.cql_lagrange:
            print(f"CQL Target Action Gap: {self.cql_target_action_gap}")
        print(f"Offline buffer size: {self.replay_buffer_offline.size}")
        print(f"{'='*80}\n")
        
        for e in range(epochs):
            # Sample from offline buffer only
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = \
                self.sample_data_offline_only(self.batch_size)
            
            # Q-function update with CQL
            total_q_loss, td_loss, cql_loss, q_mean, cql_diff, alpha_loss, cql_alpha_val = \
                self.compute_cql_q_loss(
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
                )
            
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            total_q_loss.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()
            
            # Update CQL alpha if using Lagrange
            if self.cql_lagrange:
                self.cql_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.cql_alpha_optimizer.step()
            
            # Policy update
            policy_loss, log_pi_mean = self.compute_policy_loss(obs_tensor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # SAC entropy alpha update
            if self.auto_alpha:
                _, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(
                    obs_tensor, deterministic=False, return_log_prob=True
                )
                alpha_loss_sac = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss_sac.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.cpu().exp().item()
            
            # Target network update
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)
            
            # Logging
            if (e + 1) % log_interval == 0:
                cql_alpha_log = cql_alpha_val if isinstance(cql_alpha_val, float) else cql_alpha_val.item()
                print(f"Epoch {e+1}/{epochs}: TD={td_loss:.4f}, CQL={cql_loss:.4f}, "
                      f"Pi={policy_loss.item():.4f}, Q={q_mean:.2f}, CQL_α={cql_alpha_log:.4f}")
                
                wandb.log({
                    "CQL_offline/td_loss": td_loss,
                    "CQL_offline/cql_loss": cql_loss,
                    "CQL_offline/policy_loss": policy_loss.item(),
                    "CQL_offline/q_mean": q_mean,
                    "CQL_offline/cql_alpha": cql_alpha_log,
                    "CQL_offline/epoch": e + 1,
                    "offline_training": 1
                }, step=e + 1)
                
                # Evaluation
                if test_env is not None:
                    test_rw = test_agent(self, test_env, max_ep_len, None)
                    print(f"  Eval reward: {np.mean(test_rw):.2f}")
                    wandb.log({
                        "OfflineEvalRewards": np.mean(test_rw),
                    }, step=e + 1)
        
        print(f"\n{'='*80}")
        print(f"CQL OFFLINE TRAINING COMPLETED")
        print(f"{'='*80}\n")
        
        wandb.log({"offline_training": 0}, step=epochs)
        
        return epochs
