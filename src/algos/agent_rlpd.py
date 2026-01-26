"""
RLPD Agent - RL with Prior Data
===============================
Reference: Ball et al., "Efficient Online Reinforcement Learning with Offline Data", ICML 2023

RLPD is essentially SAC with:
- Large ensemble (num_Q=10) 
- High UTD ratio (utd_ratio=20)
- 50/50 online/offline data mixing during training

Simple but effective approach for offline-to-online RL.
"""

import numpy as np
import torch
import wandb

from src.algos.agent_base import BaseAgent


class RLPDAgent(BaseAgent):
    """
    RLPD Agent: SAC with large ensemble and high UTD ratio.
    
    Default settings from paper:
    - num_Q=10 (large Q ensemble)
    - utd_ratio=20 (high update-to-data ratio)
    - 50/50 online/offline data mixing
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
        utd_ratio: int = 20,  # High UTD ratio
        num_Q: int = 10,       # Large ensemble
        policy_update_delay: int = 1,
        target_drop_rate: float = 0.0,
        layer_norm: bool = True,
        o2o: bool = True,      # Always O2O for RLPD
    ):
        super().__init__(
            env_name, obs_dim, act_dim, act_limit, device,
            hidden_sizes, replay_size, batch_size, lr, gamma, polyak,
            alpha, auto_alpha, target_entropy, start_steps, utd_ratio,
            num_Q, policy_update_delay, target_drop_rate, layer_norm, o2o
        )
    
    def train(self, current_env_step: int = None):
        """RLPD training: SAC with mixed online/offline batches."""
        num_update = 0 if self.buffer_size <= self.delay_update_steps else self.utd_ratio
        
        if num_update == 0:
            return 0.0, 0.0
        
        q_loss_val, policy_loss_val = 0.0, 0.0
        
        for i_update in range(num_update):
            if self.o2o and self.replay_buffer_offline.size > 0:
                obs, obs_next, acts, rews, done = self.sample_data_mix(self.batch_size)
            else:
                obs, obs_next, acts, rews, done = self.sample_data(self.batch_size)
            
            q_target = self.get_sac_q_target(obs_next, rews, done)
            
            q_loss_total = 0
            for q_net in self.q_net_list:
                q_pred = q_net(torch.cat([obs, acts], 1))
                q_loss_total += self.mse_criterion(q_pred, q_target)
            
            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss_total.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()
            
            q_loss_val = q_loss_total.item() / self.num_Q
            
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                action, _, _, log_prob, _, _ = self.policy_net.forward(obs)
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(False)
                
                q_values = [q_net(torch.cat([obs, action], 1)) for q_net in self.q_net_list]
                q_mean_val = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
                
                policy_loss = (self.alpha * log_prob - q_mean_val).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(True)
                
                self.policy_optimizer.step()
                policy_loss_val = policy_loss.item()
                
                self.update_alpha(log_prob)
            
            self.update_target_networks()
        
        return policy_loss_val, q_loss_val
