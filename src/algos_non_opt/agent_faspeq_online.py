### AGENT CODE FOR FASPEQ ALGORITHM - Pure Online Version
### Static Reference Batches for Policy Loss Monitoring (No Offline Dataset)


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
        if len(singular_values) == 0:
            return float('inf')
        
        max_sv = singular_values[0]
        min_sv = singular_values[-1]
        
        if min_sv < eps:
            return float('inf')
        
        return (max_sv / min_sv).item()
    
    @staticmethod
    def compute_gradient_metrics(gradients: torch.Tensor) -> dict:
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
                 target_drop_rate=0.0, layer_norm=False,
                 val_check_interval=1000, val_patience=5000,
                 num_reference_batches=10
                 ):
        """
        FASPEQ Agent - Pure Online Version
        Uses static reference batches from online buffer for policy loss monitoring.
        
        Args:
            num_reference_batches: Number of batches to extract as static reference (default: 10)
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
        
        # Main replay buffer (online only)
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
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
        
        # Offline stabilization parameters
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.num_reference_batches = num_reference_batches
        
        # Fixed 10k interval triggering
        self.last_trigger_buffer_size = 0
        self.next_trigger_size = 10000
        
        # Storage for static reference batches during offline stabilization
        self.static_reference_batches = None
        self.extracted_indices = None
        self.extracted_data = None
        
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
        """Store data in main replay buffer."""
        self.replay_buffer.store(o, a, r, o2, d)

    def check_should_trigger_offline_stabilization(self):
        """Check if offline stabilization should be triggered (every 10k steps)."""
        current_size = self.replay_buffer.size

        if current_size >= self.next_trigger_size:
            self.last_trigger_buffer_size = current_size
            self.next_trigger_size = self.last_trigger_buffer_size + 10000
            
            print(f"\n{'='*80}")
            print(f"OFFLINE STABILIZATION TRIGGERED (FASPEQ Online - Fixed Interval)")
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

    def _extract_and_remove_reference_batches(self):
        """
        Sample reference batches from online buffer and remove them.
        
        Returns:
            List of sampled batches as tensors for policy loss evaluation
        """
        print(f"  Sampling {self.num_reference_batches} reference batches from online buffer...")
        
        total_samples = self.num_reference_batches * self.batch_size
        
        if self.replay_buffer.size < total_samples:
            print(f"  Warning: Buffer too small ({self.replay_buffer.size} < {total_samples})")
            total_samples = min(total_samples, self.replay_buffer.size)
        
        # Sample unique indices
        indices = np.random.choice(self.replay_buffer.size, size=total_samples, replace=False)
        
        # Get all data from buffer
        all_data = self.replay_buffer.sample_all()
        
        # Extract the reference data
        extracted = {
            'obs1': all_data['obs1'][indices].copy(),
            'obs2': all_data['obs2'][indices].copy(),
            'acts': all_data['acts'][indices].copy(),
            'rews': all_data['rews'][indices].copy(),
            'done': all_data['done'][indices].copy(),
        }
        
        # Store for later restoration
        self.extracted_indices = indices
        self.extracted_data = extracted
        
        # Create mask for remaining data
        mask = np.ones(self.replay_buffer.size, dtype=bool)
        mask[indices] = False
        
        # Rebuild buffer without extracted samples
        remaining = {
            'obs1': all_data['obs1'][mask],
            'obs2': all_data['obs2'][mask],
            'acts': all_data['acts'][mask],
            'rews': all_data['rews'][mask],
            'done': all_data['done'][mask],
        }
        
        old_size = self.replay_buffer.size
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        for i in range(len(remaining['obs1'])):
            self.replay_buffer.store(
                remaining['obs1'][i],
                remaining['acts'][i],
                remaining['rews'][i],
                remaining['obs2'][i],
                remaining['done'][i]
            )
        
        print(f"  Buffer: {old_size} -> {self.replay_buffer.size} (removed {total_samples})")
        
        # Create static reference batches as list of tensors
        static_batches = []
        for b in range(self.num_reference_batches):
            start_idx = b * self.batch_size
            end_idx = min((b + 1) * self.batch_size, total_samples)
            
            obs_tensor = torch.Tensor(extracted['obs1'][start_idx:end_idx]).to(self.device)
            static_batches.append(obs_tensor)
        
        self.static_reference_batches = static_batches
        print(f"  Created {len(static_batches)} static reference batches")
        
        return static_batches

    def _restore_reference_batches(self):
        """Restore the extracted reference batches back to the replay buffer."""
        if self.extracted_data is None:
            print("  Warning: No extracted data to restore")
            return
        
        print(f"  Restoring reference batches to replay buffer...")
        
        old_size = self.replay_buffer.size
        
        for i in range(len(self.extracted_data['obs1'])):
            self.replay_buffer.store(
                self.extracted_data['obs1'][i],
                self.extracted_data['acts'][i],
                self.extracted_data['rews'][i],
                self.extracted_data['obs2'][i],
                self.extracted_data['done'][i]
            )
        
        print(f"  Buffer: {old_size} -> {self.replay_buffer.size}")
        
        # Clear stored data
        self.static_reference_batches = None
        self.extracted_indices = None
        self.extracted_data = None

    def evaluate_policy_loss_on_static_batches(self):
        """
        Evaluate policy loss on the static reference batches.
        Policy loss = α*entropy - Q(s, π(s))
        """
        if self.static_reference_batches is None or len(self.static_reference_batches) == 0:
            return 0.0
        
        total_policy_loss = 0.0
        
        with torch.no_grad():
            for obs_tensor in self.static_reference_batches:
                a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
                
                q_a_tilda_list = []
                for q_i in range(self.num_Q):
                    q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                total_policy_loss += policy_loss.item()
        
        average_policy_loss = total_policy_loss / len(self.static_reference_batches)
        return average_policy_loss

    def evaluate_policy_loss(self, batch_size=256):
        """Evaluate policy loss on samples from the replay buffer."""
        if self.replay_buffer.size < batch_size:
            return 0.0
        
        total_policy_loss = 0.0
        num_batches = 10
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = self.replay_buffer.sample_batch(batch_size)
                obs_tensor = torch.Tensor(batch['obs1']).to(self.device)
                
                a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor)
                
                q_a_tilda_list = []
                for q_i in range(self.num_Q):
                    q_a_tilda = self.q_net_list[q_i](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                total_policy_loss += policy_loss.item()
        
        return total_policy_loss / num_batches

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

    def expectile_loss(self, diff, expectile=0.7):
        """Expectile loss for IQL-style Q-learning."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

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

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            """Policy loss"""
            a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)

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
                alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.cpu().exp().item()
            else:
                alpha_loss = Tensor([0])

            """Soft update target networks"""
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                         LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction_list[0].detach().cpu().numpy(),
                         Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                         PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))
        
        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    def finetune_offline_faspeq(self, epochs, test_env=None, current_env_step=None):
        """ 
        FASPEQ: Finetune with policy loss monitoring using STATIC reference batches.
        
        Pure online version:
        1. Sample 10 batches from online buffer at start
        2. Remove those samples from buffer
        3. Use them as static reference throughout stabilization
        4. Restore them after stabilization ends
        """
        print(f"\n{'='*80}")
        print(f"STARTING FASPEQ OFFLINE STABILIZATION (Online Only)")
        print(f"{'='*80}")
        print(f"Max epochs: {epochs}")
        print(f"Check interval: {self.val_check_interval} epochs")
        print(f"Patience: {self.val_patience} epochs")
        print(f"Number of reference batches: {self.num_reference_batches}")
        print(f"{'='*80}\n")
        
        # Step 1: Extract reference batches and remove from buffer
        print("Step 1: Extracting and removing reference batches...")
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
        
        print(f"\nInitial policy loss (static batches): {initial_loss:.6f}\n")
        
        epochs_performed = 0
        
        try:
            for e in range(epochs):
                epochs_performed = e
                
                if e > 0 and e % 100 == 0:
                    current_policy_loss = self.evaluate_policy_loss_on_static_batches()
                    print(f"  Epoch {e:6d}: Policy loss = {current_policy_loss:.6f}")
                
                if e > 0 and e % self.val_check_interval == 0:
                    policy_loss = self.evaluate_policy_loss_on_static_batches()
                    
                    if current_env_step is not None:
                        wandb.log({
                            "PolicyLoss_Fixed_Static": policy_loss,
                            "offline_epoch": e,
                            "offline_training": 1
                        }, step=current_env_step)
                    
                    improvement = (best_policy_loss - policy_loss) / abs(best_policy_loss) * 100 if best_policy_loss != 0 else 0
                    if policy_loss < best_policy_loss * 0.999:
                        best_policy_loss = policy_loss
                        steps_without_improvement = 0
                        print(f"Epoch {e:6d}: Policy loss improved to {policy_loss:.6f} (↓ {abs(improvement):.2f}%) ✓")
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
                        epochs_performed = e
                        break
                
                if test_env and (e + 1) % 5000 == 0:
                    test_rw = test_agent(self, test_env, 1000, None)
                    if current_env_step is not None:
                        wandb.log({
                            "OfflineEvalReward": np.mean(test_rw),
                            "offline_training": 1
                        }, step=current_env_step)
                
                # Train Q-networks only (policy stays fixed)
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
                
                y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = []
                for q_i in range(self.num_Q):
                    q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                    q_prediction_list.append(q_prediction)
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                q_loss_all = self.expectile_loss(q_prediction_cat - y_q).mean() * self.num_Q

                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].zero_grad()
                q_loss_all.backward()

                for q_i in range(self.num_Q):
                    self.q_optimizer_list[q_i].step()

                for q_i in range(self.num_Q):
                    soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)
            else:
                epochs_performed = epochs
        
        finally:
            # Step 3: Restore reference batches (always executed)
            print("\nStep 3: Restoring reference batches...")
            self._restore_reference_batches()
        
        final_policy_loss = self.evaluate_policy_loss()
        print(f"\n{'='*80}")
        print(f"FASPEQ OFFLINE STABILIZATION COMPLETED")
        print(f"{'='*80}")
        print(f"Epochs performed: {epochs_performed}")
        print(f"Initial policy loss: {initial_loss:.6f}")
        print(f"Final policy loss: {final_policy_loss:.6f}")
        if initial_loss != 0:
            print(f"Total improvement: {(initial_loss - final_policy_loss)/abs(initial_loss)*100:.2f}%")
        print(f"{'='*80}\n")
        
        if current_env_step is not None:
            wandb.log({
                "PolicyLoss_Fixed": final_policy_loss,
                "offline_training": 0
            }, step=current_env_step)
        
        return epochs_performed