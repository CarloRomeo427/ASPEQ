"""
IQL Agent - Implicit Q-Learning for Offline-to-Online RL
========================================================
Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022

Key differences from SAC:
- Adds Value network V(s) trained via expectile regression
- Q-network bootstraps from V(s') instead of min Q with policy actions  
- Policy trained via advantage-weighted regression (AWR)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.algos.agent_base import BaseAgent
from src.algos.core import test_agent


EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile loss: L_tau(u) = |tau - 1(u < 0)| * u^2"""
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class ValueNetwork(nn.Module):
    """State value function V(s) for IQL."""
    
    def __init__(self, obs_dim: int, hidden_sizes: tuple = (256, 256), layer_norm: bool = True):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class IQLAgent(BaseAgent):
    """
    IQL Agent for Offline-to-Online RL.
    
    Args:
        iql_tau: Expectile for value function training (0.5-0.99, default: 0.7)
                 Higher values focus more on high-value actions
        iql_beta: Temperature for advantage-weighted policy extraction (default: 3.0)
                  Lower values make policy more greedy
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
        policy_update_delay: int = 1,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = False,
        iql_tau: float = 0.7,
        iql_beta: float = 3.0,
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm, o2o
        )
        
        # IQL-specific hyperparameters
        self.iql_tau = iql_tau
        self.iql_beta = iql_beta
        
        # Value network V(s)
        self.value_net = ValueNetwork(obs_dim, hidden_sizes, layer_norm).to(device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
    
    def compute_value_loss(self, obs: torch.Tensor, acts: torch.Tensor):
        """
        IQL Value Loss: Expectile regression on Q - V.
        L_V = E_{(s,a)~D}[L_tau(Q(s,a) - V(s))]
        """
        with torch.no_grad():
            q_values = self._get_min_q_target(obs, acts)
        
        v_values = self.value_net(obs)
        value_loss = asymmetric_l2_loss(q_values - v_values, self.iql_tau)
        
        return value_loss, v_values.mean().item()
    
    def compute_q_loss(self, obs, obs_next, acts, rews, done):
        """
        IQL Q Loss: TD loss bootstrapping from V(s') instead of Q(s', π(s')).
        L_Q = E[(r + γ(1-d)V(s') - Q(s,a))^2]
        """
        with torch.no_grad():
            next_v = self.value_net(obs_next)
            target_q = rews + self.gamma * (1 - done) * next_v
        
        q_loss_total = 0
        q_values = []
        for q_net in self.q_net_list:
            q_pred = q_net(torch.cat([obs, acts], 1))
            q_loss_total += self.mse_criterion(q_pred, target_q)
            q_values.append(q_pred.mean().item())
        
        return q_loss_total, np.mean(q_values)
    
    def compute_policy_loss(self, obs: torch.Tensor, acts: torch.Tensor):
        """
        IQL Policy Loss: Advantage-Weighted Regression (AWR).
        L_π = E[exp((Q(s,a) - V(s)) / β) * log π(a|s)]
        """
        with torch.no_grad():
            q_values = self._get_min_q_target(obs, acts)
            v_values = self.value_net(obs)
            adv = q_values - v_values
            exp_adv = torch.exp(adv / self.iql_beta).clamp(max=EXP_ADV_MAX)
        
        # Get policy distribution and compute log prob of dataset actions
        _, mean, log_std, _, _, _ = self.policy_net.forward(obs, deterministic=False, return_log_prob=True)
        
        acts_normalized = acts / self.act_limit
        eps = 1e-6
        acts_clamped = acts_normalized.clamp(-1 + eps, 1 - eps)
        pretanh_actions = torch.atanh(acts_clamped)
        
        std = log_std.exp()
        var = std ** 2
        log_prob_gaussian = -0.5 * (((pretanh_actions - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        log_prob_gaussian = log_prob_gaussian.sum(dim=-1, keepdim=True)
        log_jacobian = torch.log(1 - acts_clamped ** 2 + eps).sum(dim=-1, keepdim=True)
        log_prob = log_prob_gaussian - log_jacobian
        
        policy_loss = -(exp_adv * log_prob).mean()
        return policy_loss, log_prob.mean().item()
    
    def _get_min_q_target(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        """Get min Q-value from target networks."""
        q_values = [q_target(torch.cat([obs, acts], 1)) for q_target in self.q_target_net_list]
        return torch.min(torch.cat(q_values, dim=1), dim=1, keepdim=True)[0]
    
    def train(self, current_env_step: int = None):
        """IQL training update."""
        num_update = 0 if self.buffer_size <= self.delay_update_steps else self.utd_ratio
        
        if num_update == 0:
            return 0.0, 0.0
        
        q_loss_val, policy_loss_val = 0.0, 0.0
        
        for i_update in range(num_update):
            if self.o2o:
                obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, obs_next, acts, rews, done = self.sample_data(self.batch_size)
            
            # 1. Value function update
            value_loss, _ = self.compute_value_loss(obs, acts)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # 2. Q-function update
            q_loss, _ = self.compute_q_loss(obs, obs_next, acts, rews, done)
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            q_loss_val = q_loss.item()
            
            # 3. Policy update
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                policy_loss, _ = self.compute_policy_loss(obs, acts)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                policy_loss_val = policy_loss.item()
                
                # Alpha update
                if self.auto_alpha:
                    _, _, _, log_prob, _, _ = self.policy_net.forward(obs, deterministic=False, return_log_prob=True)
                    self.update_alpha(log_prob)
            
            # 4. Target network update
            self.update_target_networks()
        
        return policy_loss_val, q_loss_val / self.num_Q
    
    def train_offline(self, epochs: int, test_env=None, current_env_step: int = None, log_interval: int = 10000):
        """Pure offline IQL training with periodic logging every 10k updates."""
        print(f"IQL offline training: {epochs} steps")
        
        for e in range(epochs):
            obs, obs_next, acts, rews, done = self.sample_data_offline_only(self.batch_size)
            
            value_loss, _ = self.compute_value_loss(obs, acts)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            q_loss, _ = self.compute_q_loss(obs, obs_next, acts, rews, done)
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            policy_loss, _ = self.compute_policy_loss(obs, acts)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.update_target_networks()
            
            if (e + 1) % log_interval == 0:
                print(f"  Offline pretraining epoch {e+1}/{epochs}")
                if test_env is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    print(f"    EvalReward: {np.mean(test_rw):.2f}")
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=e + 1)
        
        print(f"IQL offline pretraining complete: {epochs} epochs")
        return epochs
