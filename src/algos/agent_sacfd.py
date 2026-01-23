import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict

class Agent(object):
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, num_Q=2, target_drop_rate=0.0, layer_norm=False
                 ):
        """
        SACfD Agent: Standard SAC Agent that supports offline data initialization.
        """
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        
        self.q_net_list, self.q_target_net_list = [], []
        for _ in range(num_Q):
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
        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.num_Q = num_Q
        self.start_steps = start_steps

        # Automatic entropy tuning
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = - act_dim
            elif target_entropy == 'mbpo':
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2
                try:
                    self.target_entropy = mbpo_target_entropy_dict[env_name]
                except:
                    self.target_entropy = -2
            else:
                self.target_entropy = target_entropy
                
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        
        # Standard Replay Buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.mse_criterion = nn.MSELoss()

    def __get_current_num_data(self):
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        with torch.no_grad():
            # If buffer has data (offline or online) > start_steps, use policy
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
        """Store online interaction data."""
        self.replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        """
        Store offline demonstration data. 
        In SACfD, this simply populates the main replay buffer.
        """
        self.replay_buffer.store(o, a, r, o2, d)

    def sample_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        return (torch.Tensor(batch['obs1']).to(self.device),
                torch.Tensor(batch['obs2']).to(self.device),
                torch.Tensor(batch['acts']).to(self.device),
                torch.Tensor(batch['rews']).unsqueeze(1).to(self.device),
                torch.Tensor(batch['done']).unsqueeze(1).to(self.device))

    def train(self, logger):
        # Sample from the buffer (which may contain mixed offline/online data)
        obs, obs2, acts, rews, done = self.sample_data(self.batch_size)

        # ------------------------------------
        # 1. Update Q-functions
        # ------------------------------------
        with torch.no_grad():
            next_action, _, _, next_log_prob, _, _ = self.policy_net(obs2)
            
            # Compute target Q-value
            target_q_list = [q_target(torch.cat([obs2, next_action], 1)) for q_target in self.q_target_net_list]
            target_q_min = torch.min(torch.cat(target_q_list, 1), dim=1, keepdim=True)[0]
            
            # Soft Bellman backup
            y_q = rews + self.gamma * (1 - done) * (target_q_min - self.alpha * next_log_prob)

        # Compute current Q-values
        q_list = [q(torch.cat([obs, acts], 1)) for q in self.q_net_list]
        q_loss_all = sum([self.mse_criterion(q, y_q) for q in q_list])

        # Optimize Q-networks
        for q_opt in self.q_optimizer_list:
            q_opt.zero_grad()
        q_loss_all.backward()
        for q_opt in self.q_optimizer_list:
            q_opt.step()

        # ------------------------------------
        # 2. Update Policy
        # ------------------------------------
        # Re-parameterized sampling
        a_tilda, _, _, log_prob, _, _ = self.policy_net(obs)
        
        # Compute Q-values for new actions
        q_a_list = [q(torch.cat([obs, a_tilda], 1)) for q in self.q_net_list]
        q_a_min = torch.min(torch.cat(q_a_list, 1), dim=1, keepdim=True)[0]
        
        # Policy loss: Alpha * LogPi - Q
        policy_loss = (self.alpha * log_prob - q_a_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ------------------------------------
        # 3. Update Alpha (Entropy Temperature)
        # ------------------------------------
        alpha_loss = torch.tensor(0.0).to(self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.cpu().exp().item()

        # ------------------------------------
        # 4. Update Target Networks
        # ------------------------------------
        for i in range(self.num_Q):
            soft_update_model1_with_model2(self.q_target_net_list[i], self.q_net_list[i], self.polyak)

        # ------------------------------------
        # 5. Logging
        # ------------------------------------
        logger.store(LossPi=policy_loss.item(), 
                     LossQ1=q_loss_all.item() / self.num_Q, 
                     LossAlpha=alpha_loss.item(), 
                     Q1Vals=q_list[0].detach().cpu().numpy(),
                     Alpha=self.alpha, 
                     LogPi=log_prob.detach().cpu().numpy())