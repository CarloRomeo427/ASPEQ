### AGENT CODE FOR A3SPEQ ALGORITHM
### A3RL Prioritization + FASPEQ Structure
### Key: Density + priorities updated at EVERY gradient step (like A3RL)
### FASPEQ elements: UTD=1, 2 Q-critics, dropout, offline stabilization phases

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.algos_non_opt.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict
import wandb
from src.algos_non_opt.core import test_agent


class DensityNetwork(nn.Module):
    """
    Estimates density ratio w(s,a) = d^on(s,a) / d^off(s,a).
    Architecture: MLP with ReLU hidden, Softplus output (ensures w >= 0).
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, device):
        super().__init__()
        layers = []
        input_dim = obs_dim + act_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers).to(device)
        
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return F.softplus(self.net(x))


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
                 val_check_interval=1000, val_patience=5000,
                 num_reference_batches=10,
                 # A3RL parameters
                 prio_zeta=0.3,
                 adv_temp_xi=0.03,
                 num_density_nets=10,
                 beta_start=0.4,
                 beta_end=1.0,
                 beta_annealing_steps=100000
                 ):
        """
        A3SPEQ Agent: A3RL prioritization + FASPEQ structure.
        
        A3RL elements (updated EVERY gradient step):
        - Density network ensemble
        - Priority calculation with density LCB + advantage LCB
        - IS-weighted critic loss
        
        FASPEQ elements:
        - UTD=1
        - 2 Q-critics (not 10)
        - Dropout on critics
        - Offline stabilization phases with early stopping
        """
        # Policy network
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        
        # Q-network ensemble (FASPEQ: typically 2 critics with dropout)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, 
                           target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes,
                                   target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        
        # A3RL: Density network ensemble
        self.num_density_nets = num_density_nets
        self.density_net_list = []
        self.density_optimizer_list = []
        for _ in range(num_density_nets):
            d_net = DensityNetwork(obs_dim, act_dim, hidden_sizes, device)
            self.density_net_list.append(d_net)
            self.density_optimizer_list.append(optim.Adam(d_net.parameters(), lr=lr))
        
        # A3RL hyperparameters
        self.zeta = prio_zeta
        self.xi = adv_temp_xi
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.total_gradient_steps = 0
        
        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = -act_dim
            elif target_entropy == 'mbpo':
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2
                self.target_entropy = mbpo_target_entropy_dict.get(env_name, -2)
            else:
                self.target_entropy = target_entropy
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
        
        # Store dimensions and hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.polyak = polyak
        self.start_steps = start_steps
        self.delay_update_steps = start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.utd_ratio = utd_ratio
        self.num_Q = num_Q
        self.num_min = num_min
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm
        
        # FASPEQ: Offline stabilization parameters
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience
        self.num_reference_batches = num_reference_batches
        self.next_trigger_size = 10000
        
        # Storage for reference batches during offline stabilization
        self.static_reference_batches = None
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

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(
                obs_tensor, deterministic=False, return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        return torch.mean(q_prediction_cat, dim=1)

    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        self.replay_buffer_offline.store(o, a, r, o2, d)

    def check_should_trigger_offline_stabilization(self):
        """Check if offline stabilization should be triggered (every 10k steps)."""
        current_size = self.replay_buffer.size
        if current_size >= self.next_trigger_size:
            self.next_trigger_size += 10000
            print(f"\n{'='*60}")
            print(f"OFFLINE STABILIZATION TRIGGERED at buffer size {current_size}")
            print(f"Next trigger at: {self.next_trigger_size}")
            print(f"{'='*60}\n")
            return True
        return False

    # =========================================================================
    # A3RL Core: Density update + Priority calculation + Sampling
    # Called at EVERY gradient step (both online and offline)
    # =========================================================================
    def update_density_ensemble(self, obs_on, acts_on, obs_off, acts_off):
        """Update density networks with JS divergence loss."""
        eps = 1e-6
        total_loss = 0.0
        
        for i, d_net in enumerate(self.density_net_list):
            w_on = d_net(obs_on, acts_on)
            w_off = d_net(obs_off, acts_off)
            
            # JS Divergence variational form
            term_on = torch.log(2 * w_on / (w_on + 1 + eps) + eps)
            term_off = torch.log((w_off + 1) / 2 + eps)
            loss = -(torch.mean(term_on) - torch.mean(term_off))
            
            self.density_optimizer_list[i].zero_grad()
            loss.backward()
            self.density_optimizer_list[i].step()
            total_loss += loss.item()
        
        return total_loss / self.num_density_nets

    def calculate_priorities(self, obs, acts, is_online):
        """
        Calculate priorities: σ = (I^off·w_lcb + I^on) · exp(ξ·A_lcb)
        """
        with torch.no_grad():
            # Density LCB
            ws = torch.stack([d_net(obs, acts) for d_net in self.density_net_list], dim=0)
            w_mean = ws.mean(dim=0)
            w_std = ws.std(dim=0)
            w_lcb = torch.clamp(w_mean - w_std, min=0.0)
            
            # Advantage LCB
            qs = torch.stack([q_net(torch.cat([obs, acts], dim=1)) 
                             for q_net in self.q_net_list], dim=0)
            q_mean = qs.mean(dim=0)
            q_std = qs.std(dim=0)
            q_lcb = q_mean - q_std
            
            a_pi, _, _, log_pi, _, _ = self.policy_net(obs)
            qs_pi = torch.stack([q_net(torch.cat([obs, a_pi], dim=1)) 
                                for q_net in self.q_net_list], dim=0)
            v_mean = qs_pi.mean(dim=0) - self.alpha * log_pi
            v_std = qs_pi.std(dim=0)
            v_ucb = v_mean + v_std
            adv_lcb = q_lcb - v_ucb
            
            # Priority
            is_on = is_online.squeeze()
            density_term = torch.where(
                is_on.unsqueeze(1),
                torch.ones_like(w_lcb),
                w_lcb
            )
            sigma = density_term * torch.exp(self.xi * adv_lcb)
            sigma_zeta = torch.pow(sigma.squeeze() + 1e-8, self.zeta)
            
            return sigma_zeta.cpu().numpy()

    def sample_and_update_a3rl(self, batch_size):
        """
        A3RL sampling: sample pool → update density → compute priorities → resample → return with IS weights.
        This is called at EVERY gradient step.
        """
        half_N = batch_size // 2
        
        # Sample pool from both buffers
        batch_on = self.replay_buffer.sample_batch(min(half_N, self.replay_buffer.size))
        batch_off = self.replay_buffer_offline.sample_batch(min(half_N, self.replay_buffer_offline.size))
        
        actual_on = len(batch_on['obs1'])
        actual_off = len(batch_off['obs1'])
        pool_size = actual_on + actual_off
        
        obs_on = torch.Tensor(batch_on['obs1']).to(self.device)
        acts_on = torch.Tensor(batch_on['acts']).to(self.device)
        obs_off = torch.Tensor(batch_off['obs1']).to(self.device)
        acts_off = torch.Tensor(batch_off['acts']).to(self.device)
        
        # Update density networks (A3RL does this every step)
        density_loss = self.update_density_ensemble(obs_on, acts_on, obs_off, acts_off)
        
        # Build full pool tensors
        obs = torch.cat([obs_on, obs_off], dim=0)
        obs_next = torch.Tensor(np.concatenate([batch_on['obs2'], batch_off['obs2']])).to(self.device)
        acts = torch.cat([acts_on, acts_off], dim=0)
        rews = torch.Tensor(np.concatenate([batch_on['rews'], batch_off['rews']])).unsqueeze(1).to(self.device)
        dones = torch.Tensor(np.concatenate([batch_on['done'], batch_off['done']])).unsqueeze(1).to(self.device)
        
        is_online = torch.cat([
            torch.ones(actual_on, dtype=torch.bool, device=self.device),
            torch.zeros(actual_off, dtype=torch.bool, device=self.device)
        ]).unsqueeze(1)
        
        # Calculate priorities
        sigma_zeta = self.calculate_priorities(obs, acts, is_online)
        probs = sigma_zeta / (sigma_zeta.sum() + 1e-8)
        
        # Resample from pool
        idx = np.random.choice(pool_size, size=batch_size, p=probs, replace=True)
        
        # Compute IS weights
        beta = self.beta_start + (self.beta_end - self.beta_start) * \
               min(1.0, self.total_gradient_steps / self.beta_annealing_steps)
        sampled_probs = probs[idx]
        is_weights = (1.0 / (pool_size * sampled_probs + 1e-8)) ** beta
        is_weights = is_weights / is_weights.max()
        is_weights_tensor = torch.Tensor(is_weights).unsqueeze(1).to(self.device)
        
        return (obs[idx], obs_next[idx], acts[idx], rews[idx], dones[idx], 
                is_weights_tensor, density_loss, probs.std())

    # =========================================================================
    # Q-target computation
    # =========================================================================
    def get_q_target(self, obs_next, rews, dones):
        with torch.no_grad():
            a_next, _, _, log_prob_next, _, _ = self.policy_net.forward(obs_next)
            q_next_list = [q_target(torch.cat([obs_next, a_next], 1)) 
                          for q_target in self.q_target_net_list]
            q_next_cat = torch.cat(q_next_list, 1)
            
            if self.q_target_mode == 'min':
                q_target = torch.min(q_next_cat, 1, keepdim=True)[0]
            elif self.q_target_mode == 'ave':
                q_target = torch.mean(q_next_cat, 1, keepdim=True)
            else:
                num_mins = get_probabilistic_num_min(self.num_min)
                sample_idxs = np.random.choice(self.num_Q, num_mins, replace=False)
                q_target = torch.min(q_next_cat[:, sample_idxs], 1, keepdim=True)[0]
            
            y_q = rews + self.gamma * (1 - dones) * (q_target - self.alpha * log_prob_next)
        return y_q

    # =========================================================================
    # Main training step (online): A3RL prioritization + FASPEQ UTD=1
    # =========================================================================
    def train(self, logger, current_env_step=None):
        """
        Online training with A3RL prioritization at every step.
        FASPEQ structure: UTD=1, 2 critics, dropout.
        """
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        
        for i_update in range(num_update):
            self.total_gradient_steps += 1
            
            if self.o2o:
                # A3RL: sample + update density + compute priorities + IS weights
                obs, obs_next, acts, rews, dones, is_weights, density_loss, prio_std = \
                    self.sample_and_update_a3rl(self.batch_size)
            else:
                # Standard sampling (no offline data)
                batch = self.replay_buffer.sample_batch(self.batch_size)
                obs = Tensor(batch['obs1']).to(self.device)
                obs_next = Tensor(batch['obs2']).to(self.device)
                acts = Tensor(batch['acts']).to(self.device)
                rews = Tensor(batch['rews']).unsqueeze(1).to(self.device)
                dones = Tensor(batch['done']).unsqueeze(1).to(self.device)
                is_weights = torch.ones(self.batch_size, 1).to(self.device)
                density_loss, prio_std = 0.0, 0.0

            # Q loss with IS weighting
            y_q = self.get_q_target(obs_next, rews, dones)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q))
            
            td_errors = (q_cat - y_q_expanded) ** 2
            q_loss = (is_weights * td_errors).mean() * self.num_Q

            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()

            # Policy and alpha loss
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                a_tilda, _, _, log_prob, _, pretanh = self.policy_net.forward(obs)
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(False)
                q_vals = [q_net(torch.cat([obs, a_tilda], 1)) for q_net in self.q_net_list]
                ave_q = torch.mean(torch.cat(q_vals, 1), dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob - ave_q).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                for q_net in self.q_net_list:
                    q_net.requires_grad_(True)

                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            # Soft update targets
            for q_net, q_target in zip(self.q_net_list, self.q_target_net_list):
                soft_update_model1_with_model2(q_target, q_net, self.polyak)

            # Logging
            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), 
                            LossQ1=q_loss.cpu().item() / self.num_Q,
                            LossAlpha=alpha_loss.cpu().item(), 
                            Q1Vals=q_preds[0].detach().cpu().numpy(),
                            Alpha=self.alpha, 
                            LogPi=log_prob.detach().cpu().numpy(),
                            PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

                if current_env_step is not None and self.o2o:
                    beta = self.beta_start + (self.beta_end - self.beta_start) * \
                           min(1.0, self.total_gradient_steps / self.beta_annealing_steps)
                    wandb.log({
                        "policy_loss": policy_loss.cpu().item(),
                        "q_loss": q_loss.cpu().item() / self.num_Q,
                        "density_loss": density_loss,
                        "priority_std": prio_std,
                        "beta": beta
                    }, step=current_env_step)

        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    # =========================================================================
    # FASPEQ: Offline Stabilization with A3RL prioritization
    # =========================================================================
    def _extract_reference_batches(self):
        """Extract static reference batches for policy loss monitoring."""
        samples_per_batch = self.batch_size
        total_samples = self.num_reference_batches * samples_per_batch
        
        total_online = min(total_samples, self.replay_buffer.size)
        total_offline = min(total_samples, self.replay_buffer_offline.size)
        
        online_indices = np.random.choice(self.replay_buffer.size, size=total_online, replace=False)
        offline_indices = np.random.choice(self.replay_buffer_offline.size, size=total_offline, replace=False)
        
        online_data = self.replay_buffer.sample_all()
        offline_data = self.replay_buffer_offline.sample_all()
        
        self.extracted_online_data = {k: online_data[k][online_indices].copy() for k in online_data}
        self.extracted_offline_data = {k: offline_data[k][offline_indices].copy() for k in offline_data}
        
        # Create static batches for monitoring
        static_batches = []
        samples_per_batch = min(samples_per_batch, total_online, total_offline)
        
        for b in range(self.num_reference_batches):
            start = b * samples_per_batch
            end = min((b + 1) * samples_per_batch, total_online, total_offline)
            if start >= end:
                break
            batch_obs = np.concatenate([
                self.extracted_online_data['obs1'][start:end],
                self.extracted_offline_data['obs1'][start:end]
            ], axis=0)
            static_batches.append(torch.Tensor(batch_obs).to(self.device))
        
        self.static_reference_batches = static_batches
        print(f"  Created {len(static_batches)} reference batches for policy loss monitoring")

    def evaluate_policy_loss_on_static_batches(self):
        """Evaluate policy loss on static reference batches."""
        if not self.static_reference_batches:
            return 0.0
        
        total_loss = 0.0
        with torch.no_grad():
            for obs in self.static_reference_batches:
                a, _, _, log_prob, _, _ = self.policy_net.forward(obs)
                q_vals = [q_net(torch.cat([obs, a], 1)) for q_net in self.q_net_list]
                ave_q = torch.mean(torch.cat(q_vals, 1), dim=1, keepdim=True)
                loss = (self.alpha * log_prob - ave_q).mean()
                total_loss += loss.item()
        
        return total_loss / len(self.static_reference_batches)

    def finetune_offline(self, epochs, test_env=None, current_env_step=None):
        """
        FASPEQ offline stabilization with A3RL prioritization.
        
        A3RL elements: density + priorities updated every step.
        FASPEQ elements: Q-only training, early stopping on policy loss.
        """
        print(f"\n{'='*60}")
        print(f"A3SPEQ OFFLINE STABILIZATION")
        print(f"Max epochs: {epochs}, Patience: {self.val_patience}")
        print(f"A3RL: density + priorities updated every step")
        print(f"{'='*60}\n")
        
        self._extract_reference_batches()
        
        initial_loss = self.evaluate_policy_loss_on_static_batches()
        best_loss = initial_loss
        steps_no_improve = 0
        
        print(f"Initial policy loss: {initial_loss:.6f}\n")
        
        if current_env_step:
            wandb.log({"PolicyLoss_Static": initial_loss, "offline_training": 1}, step=current_env_step)
        
        epochs_done = 0
        
        for e in range(epochs):
            epochs_done = e
            self.total_gradient_steps += 1
            
            # A3RL: sample + update density + compute priorities
            obs, obs_next, acts, rews, dones, is_weights, density_loss, prio_std = \
                self.sample_and_update_a3rl(self.batch_size)
            
            # Q-network update only (no policy update during stabilization)
            y_q = self.get_q_target(obs_next, rews, dones)
            q_preds = [q_net(torch.cat([obs, acts], 1)) for q_net in self.q_net_list]
            q_cat = torch.cat(q_preds, dim=1)
            y_q_expanded = y_q.expand((-1, self.num_Q))
            
            td_errors = (q_cat - y_q_expanded) ** 2
            q_loss = (is_weights * td_errors).mean() * self.num_Q

            for q_opt in self.q_optimizer_list:
                q_opt.zero_grad()
            q_loss.backward()
            for q_opt in self.q_optimizer_list:
                q_opt.step()

            for q_net, q_target in zip(self.q_net_list, self.q_target_net_list):
                soft_update_model1_with_model2(q_target, q_net, self.polyak)
            
            # Logging
            if e > 0 and e % 500 == 0:
                curr_loss = self.evaluate_policy_loss_on_static_batches()
                print(f"  Epoch {e:5d}: Policy loss = {curr_loss:.6f}, "
                      f"Density loss = {density_loss:.6f}, Priority std = {prio_std:.6f}")
            
            # Early stopping check
            if e > 0 and e % self.val_check_interval == 0:
                curr_loss = self.evaluate_policy_loss_on_static_batches()
                
                if current_env_step:
                    wandb.log({
                        "PolicyLoss_Static": curr_loss,
                        "offline_density_loss": density_loss,
                        "offline_priority_std": prio_std,
                        "offline_epoch": e,
                        "offline_training": 1
                    }, step=current_env_step)
                
                if curr_loss < best_loss * 0.999:
                    best_loss = curr_loss
                    steps_no_improve = 0
                    print(f"Epoch {e:5d}: Policy loss improved to {curr_loss:.6f} ✓")
                else:
                    steps_no_improve += self.val_check_interval
                    print(f"Epoch {e:5d}: No improvement ({steps_no_improve}/{self.val_patience})")
                
                if steps_no_improve >= self.val_patience:
                    print(f"\nEarly stopping at epoch {e}")
                    break
            
            # Periodic evaluation
            if test_env and (e + 1) % 5000 == 0:
                test_rw = test_agent(self, test_env, 1000, None)
                if current_env_step:
                    wandb.log({"OfflineEvalReward": np.mean(test_rw)}, step=current_env_step)
        
        final_loss = self.evaluate_policy_loss_on_static_batches()
        print(f"\n{'='*60}")
        print(f"Stabilization complete: {epochs_done} epochs")
        print(f"Policy loss: {initial_loss:.6f} → {final_loss:.6f}")
        print(f"{'='*60}\n")
        
        self.static_reference_batches = None
        self.extracted_online_data = None
        self.extracted_offline_data = None
        
        if current_env_step:
            wandb.log({"PolicyLoss_Final": final_loss, "offline_training": 0}, step=current_env_step)
        
        return epochs_done