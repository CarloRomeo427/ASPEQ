### AGENT CODE FOR SPEQ ALGORITHM 


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
        
        Args:
            singular_values: Tensor of singular values
            eps: Small constant for numerical stability
            
        Returns:
            Effective rank value
        """
        # Normalize singular values
        sv_sum = singular_values.sum()
        if sv_sum < eps:
            return 0.0
        
        normalized_sv = singular_values / sv_sum
        
        # Filter out near-zero values
        normalized_sv = normalized_sv[normalized_sv > eps]
        
        if len(normalized_sv) == 0:
            return 0.0
        
        # Compute effective rank: -Σ σ̄ᵢ log σ̄ᵢ
        log_sv = torch.log(normalized_sv)
        effective_rank = -torch.sum(normalized_sv * log_sv).item()
        
        return effective_rank
    
    @staticmethod
    def compute_condition_number(singular_values: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Compute condition number (ratio of largest to smallest singular value).
        """
        if len(singular_values) == 0:
            return float('inf')
        
        max_sv = singular_values[0]  # Assuming sorted in descending order
        min_sv = singular_values[-1]
        
        if min_sv < eps:
            return float('inf')
        
        return (max_sv / min_sv).item()
    
    @staticmethod
    def compute_gradient_metrics(gradients: torch.Tensor) -> dict:
        """
        Compute spectral metrics for a gradient matrix.
        
        Args:
            gradients: Tensor of shape (num_params, batch_size) representing
                      per-example gradients
                      
        Returns:
            Dictionary containing effective_rank, condition_number, spectral_norm, stable_rank
        """
        if gradients.dim() == 1:
            gradients = gradients.unsqueeze(1)
        
        # Compute SVD
        try:
            U, S, Vh = torch.linalg.svd(gradients, full_matrices=False)
        except:
            # Fallback for numerical issues
            return {
                'effective_rank': 0.0,
                'condition_number': float('inf'),
                'spectral_norm': 0.0,
                'stable_rank': 0.0
            }
        
        # Compute metrics
        effective_rank = EffectiveRankMonitor.compute_effective_rank(S)
        condition_number = EffectiveRankMonitor.compute_condition_number(S)
        spectral_norm = S[0].item() if len(S) > 0 else 0.0
        
        # Stable rank: ||G||_F^2 / ||G||_2^2
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
                 target_drop_rate=0.0, layer_norm=False, o2o=False
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
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.replay_buffer_offline = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
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
        self.replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        self.replay_buffer_offline.store(o, a, r, o2, d)

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
        batch_offline = self.replay_buffer.sample_batch(batch_size)

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

    def get_td_error(self):
        with torch.no_grad():
            batch = self.replay_buffer.sample_all()
            result = np.zeros([1, len(batch['obs1'])])
            for i in range(0, len(batch['obs1']), 1000):
                obs_tensor = Tensor(batch['obs1'][i:i + 1000]).to(self.device)
                obs_next_tensor = Tensor(batch['obs2'][i:i + 1000]).to(self.device)
                acts_tensor = Tensor(batch['acts'][i:i + 1000]).to(self.device)
                rews_tensor = Tensor(batch['rews'][i:i + 1000]).unsqueeze(1).to(self.device)
                done_tensor = Tensor(batch['done'][i:i + 1000]).unsqueeze(1).to(self.device)

                y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = []
                for q_i in range(self.num_Q):
                    q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                    q_prediction_list.append(q_prediction)
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                q_loss_all = ((q_prediction_cat - y_q) ** 2).mean(1)
                result[0, i:i + 1000] = q_loss_all.cpu().numpy()

        return result

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            if self.q_target_mode == 'min':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.q_target_net_list[sample_idx](
                        torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            if self.q_target_mode == 'ave':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = torch.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_ave - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            if self.q_target_mode == 'rem':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                rem_weight = Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.device)
                normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.num_Q)
                rem_weight = rem_weight / normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_rem - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
        return y_q, sample_idxs

    def compute_effective_rank_metrics(self, batch_size=128):
        """
        Compute effective rank and gradient diversity metrics for Q-networks and policy.
        This is computationally expensive, so call it periodically (e.g., every 1000-5000 steps).
        
        Returns:
            Dictionary with metrics for Q1, Q2, and policy networks
        """
        metrics = {}
        
        # Sample a batch
        if self.o2o:
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(batch_size)
        else:
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(batch_size)
        
        # ========== Q-Network Metrics ==========
        for q_i in range(self.num_Q):
            q_net = self.q_net_list[q_i]
            
            # Compute per-example gradients for Q-network
            per_example_grads = []
            
            for i in range(batch_size):
                # Single example
                obs_i = obs_tensor[i:i+1]
                acts_i = acts_tensor[i:i+1]
                obs_next_i = obs_next_tensor[i:i+1]
                rews_i = rews_tensor[i:i+1]
                done_i = done_tensor[i:i+1]
                
                # Compute Q-loss for this example
                y_q, _ = self.get_redq_q_target_no_grad(obs_next_i, rews_i, done_i)
                q_pred = q_net(torch.cat([obs_i, acts_i], 1))
                loss = F.mse_loss(q_pred, y_q)
                
                # Compute gradients
                self.q_optimizer_list[q_i].zero_grad()
                loss.backward(retain_graph=False)
                
                # Collect gradients
                grad_vec = []
                for param in q_net.parameters():
                    if param.grad is not None:
                        grad_vec.append(param.grad.flatten().detach())
                
                if len(grad_vec) > 0:
                    per_example_grads.append(torch.cat(grad_vec))
            
            # Stack into gradient matrix (num_params x batch_size)
            if len(per_example_grads) > 0:
                gradient_matrix = torch.stack(per_example_grads, dim=1)
                
                # Compute metrics
                q_metrics = self.effective_rank_monitor.compute_gradient_metrics(gradient_matrix)
                metrics[f'Q{q_i+1}_effective_rank'] = q_metrics['effective_rank']
                metrics[f'Q{q_i+1}_condition_number'] = q_metrics['condition_number']
                metrics[f'Q{q_i+1}_spectral_norm'] = q_metrics['spectral_norm']
                metrics[f'Q{q_i+1}_stable_rank'] = q_metrics['stable_rank']
            
            # Zero out gradients
            self.q_optimizer_list[q_i].zero_grad()
        
        # ========== Policy Network Metrics ==========
        per_example_policy_grads = []
        
        for i in range(batch_size):
            obs_i = obs_tensor[i:i+1]
            
            # Compute policy loss for this example
            a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_i)
            
            # Get Q-values for policy gradient
            q_list = []
            for q_net in self.q_net_list:
                with torch.no_grad():
                    q_val = q_net(torch.cat([obs_i, a_tilda], 1))
                    q_list.append(q_val)
            
            ave_q = torch.mean(torch.cat(q_list, 1), dim=1, keepdim=True)
            policy_loss = (self.alpha * log_prob_a_tilda - ave_q)
            
            # Compute gradients
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            
            # Collect gradients
            grad_vec = []
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.flatten().detach())
            
            if len(grad_vec) > 0:
                per_example_policy_grads.append(torch.cat(grad_vec))
        
        # Stack into gradient matrix
        if len(per_example_policy_grads) > 0:
            policy_gradient_matrix = torch.stack(per_example_policy_grads, dim=1)
            
            # Compute metrics
            policy_metrics = self.effective_rank_monitor.compute_gradient_metrics(policy_gradient_matrix)
            metrics['policy_effective_rank'] = policy_metrics['effective_rank']
            metrics['policy_condition_number'] = policy_metrics['condition_number']
            metrics['policy_spectral_norm'] = policy_metrics['spectral_norm']
            metrics['policy_stable_rank'] = policy_metrics['stable_rank']
        
        # Zero out gradients
        self.policy_optimizer.zero_grad()
        
        return metrics

    def train(self, logger, current_env_step=None):
        """
        Train the agent for one step.
        
        Args:
            logger: Logger object for storing training metrics
            current_env_step: Current environment step for wandb logging (optional)
        """
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

            # Only log at the last update and only if current_env_step is provided
            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

                # Log to wandb with explicit step if provided
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
        """
        Evaluate Q-loss on the entire validation replay buffer by iterating over it in batches.
        Returns the average Q-loss across all validation data.
        """
        val_data = self.replay_buffer.sample_all()
        obs_full = val_data['obs1']
        obs_next_full = val_data['obs2']
        acts_full = val_data['acts']
        rews_full = val_data['rews']
        done_full = val_data['done']

        num_samples = len(obs_full)
        if num_samples == 0:
            return 0.0  # No validation data available

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

    def finetune_offline(self, epochs, test_env=None, current_env_step=None):
        """ 
        Finetune the model on the replay buffer data.
        
        Args:
            epochs: Number of offline training epochs
            test_env: Environment for evaluation (optional)
            current_env_step: Current environment step for wandb logging
        """

        # Log initial validation loss with explicit step
        initial_loss = self.evaluate_validation_loss()
        if current_env_step is not None:
            wandb.log({"ValLoss": initial_loss}, step=current_env_step)
       
        for e in range(epochs):
            # Log validation loss every 1000 epochs
            if e % 1000 == 0 and e > 0:
                val_loss = self.evaluate_validation_loss()
                if current_env_step is not None:
                    # Use same env step for all offline updates within this offline training phase
                    wandb.log({"ValLoss": val_loss}, step=current_env_step)
            
            # Evaluate on test environment every 5000 epochs
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step is not None:
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)
            
            # Sample data and train
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

            # Update Q-networks
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            # Update target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i],
                                                self.polyak)