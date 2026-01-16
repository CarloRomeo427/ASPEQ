### A3RL: Active Advantage-Aligned Online Reinforcement Learning with Offline Data
### Paper: https://arxiv.org/abs/2502.07937v3
###
### Implementation follows Algorithm 1 exactly with equations:
###   Priority (Eq. 4): p_i = σ_i^ζ / Σ_k σ_k^ζ
###   Sigma: σ_i = I^off · w(s,a) · exp(ξ·A(s,a)) + I^on · exp(ξ·A(s,a))
###   Density LCB (Eq. 5): w(s,a) = ŵ(s,a) - Ĉ_w(s,a)
###   Advantage LCB (Eq. 6): A(s,a) = Â(s,a) - Ĉ_A(s,a)
###   Importance weights (Eq. 3): u_i ∝ (1/(|R|·p_i))^β

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from src.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict, test_agent


class DensityNetwork(nn.Module):
    """
    Estimates density ratio w(s,a) = d^on(s,a) / d^off(s,a).
    
    Paper page 4: "approximates w(s,a) by training a neural network w_ψ(s,a)...
    with parameters ensuring that the outputs remain non-negative through 
    the use of activation function."
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
        return F.softplus(self.net(x))  # Non-negative output


class Agent(object):
    """
    A3RL Agent implementing Algorithm 1 from the paper.
    
    Key components:
    - Confidence-aware active advantage-aligned sampling (Eq. 4-6)
    - Density ratio estimation via JS divergence
    - REDQ-style ensemble Q-learning with subset sampling
    - Importance sampling correction (Eq. 3)
    
    Hyperparameters from Table 2:
    - E = 10 (ensemble size)
    - G = 20 (gradient steps)
    - ζ = 0.3 (priority exponent)
    - ξ = 0.03 (advantage temperature)
    - Z = 2 (subset size for REDQ target)
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=1, num_Q=10, num_min=2, q_target_mode='min',
                 policy_update_delay=1,
                 target_drop_rate=0.0, layer_norm=True, o2o=False,
                 # A3RL parameters (Table 2)
                 gradient_steps_g=20,
                 prio_zeta=0.3,
                 adv_temp_xi=0.03,
                 # Compatibility args
                 val_buffer_prob=0.0, val_buffer_offline_frac=0.0,
                 val_check_interval=1000, val_patience=5000,
                 adaptive_trigger_expansion_rate=1.1, auto_stab=False,
                 num_reference_batches=10):
        """
        Initialize A3RL Agent (Algorithm 1, Lines 1-4).
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak  # ρ (critic EMA weight)
        self.replay_size = replay_size
        self.batch_size = batch_size  # N
        self.start_steps = start_steps
        self.num_Q = num_Q  # E
        self.num_min = num_min  # Z
        self.utd_ratio = utd_ratio
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm
        self.o2o = o2o
        
        # A3RL parameters
        self.gradient_steps = gradient_steps_g  # G
        self.zeta = prio_zeta  # ζ
        self.xi = adv_temp_xi  # ξ
        
        self.delay_update_steps = start_steps if delay_update_steps == 'auto' else delay_update_steps

        # Line 2: Initialize Actor ϕ
        self.policy_net = TanhGaussianPolicy(
            obs_dim, act_dim, hidden_sizes, action_limit=act_limit
        ).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Line 2: Initialize Critic ensemble θ_i and targets θ'_i = θ_i
        self.q_net_list = []
        self.q_target_net_list = []
        self.q_optimizer_list = []
        
        for _ in range(num_Q):
            q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, 
                       target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            q_target = Mlp(obs_dim + act_dim, 1, hidden_sizes,
                          target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(device)
            q_target.load_state_dict(q_net.state_dict())
            
            self.q_net_list.append(q_net)
            self.q_target_net_list.append(q_target)
            self.q_optimizer_list.append(optim.Adam(q_net.parameters(), lr=lr))

        # Density network ensemble for w(s,a) estimation
        self.density_net_list = []
        self.density_optimizer_list = []
        for _ in range(num_Q):
            d_net = DensityNetwork(obs_dim, act_dim, hidden_sizes, device)
            self.density_net_list.append(d_net)
            self.density_optimizer_list.append(optim.Adam(d_net.parameters(), lr=lr))

        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = -act_dim
            elif target_entropy == 'mbpo':
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2
                self.target_entropy = mbpo_target_entropy_dict.get(env_name, -act_dim)
            else:
                self.target_entropy = target_entropy
                
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None

        # Line 4: Initialize buffers - D (offline), R (online)
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.replay_buffer_offline = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.val_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size * 0.2))
        
        # Beta annealing for importance sampling (Eq. 3)
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_steps = 100000
        self.current_step = 0
        
        # Compatibility
        self.val_buffer_prob = val_buffer_prob
        self.val_buffer_offline_frac = val_buffer_offline_frac
        self.val_check_interval = val_check_interval
        self.val_patience = val_patience

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
            action_tensor, _, _, log_prob, _, _ = self.policy_net.forward(
                obs_tensor, deterministic=False, return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        q_list = [self.q_net_list[i](torch.cat([obs_tensor, acts_tensor], 1)) for i in range(self.num_Q)]
        return torch.mean(torch.cat(q_list, dim=1), dim=1)

    def store_data(self, o, a, r, o2, d):
        """Line 8: R ← R ∪ {(s_t, a_t, r_t, s_{t+1})}"""
        self.replay_buffer.store(o, a, r, o2, d)

    def store_data_offline(self, o, a, r, o2, d):
        self.replay_buffer_offline.store(o, a, r, o2, d)

    def populate_val_buffer_from_offline(self):
        pass

    def check_should_trigger_offline_stabilization(self):
        return False

    def sample_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        return (Tensor(batch['obs1']).to(self.device),
                Tensor(batch['obs2']).to(self.device),
                Tensor(batch['acts']).to(self.device),
                Tensor(batch['rews']).unsqueeze(1).to(self.device),
                Tensor(batch['done']).unsqueeze(1).to(self.device))

    def sample_data_mix(self, batch_size):
        b_on = self.replay_buffer.sample_batch(batch_size)
        b_off = self.replay_buffer_offline.sample_batch(batch_size)
        batch = {k: np.concatenate([b_on[k], b_off[k]], axis=0) 
                 for k in ['obs1', 'obs2', 'acts', 'rews', 'done']}
        return (Tensor(batch['obs1']).to(self.device),
                Tensor(batch['obs2']).to(self.device),
                Tensor(batch['acts']).to(self.device),
                Tensor(batch['rews']).unsqueeze(1).to(self.device),
                Tensor(batch['done']).unsqueeze(1).to(self.device))

    def update_density_ensemble(self, obs_on, acts_on, obs_off, acts_off):
        """
        Line 10: Update density ensemble using R_N
        
        Paper page 4: JS divergence variational form
        L^DR(ψ) = E_{x~P}[f'(w_ψ(x))] - E_{x~Q}[f*(f'(w_ψ(x)))]
        
        For JS: f(y) = y·log(2y/(y+1)) + log((y+1)/2)
        f'(y) = log(2y/(y+1))
        f*(f'(w)) = log((w+1)/2)
        
        Sample from B^on for P, from B^off for Q
        """
        if len(obs_on) == 0 or len(obs_off) == 0:
            return 0.0

        loss_total = 0.0
        eps = 1e-6
        
        for i, d_net in enumerate(self.density_net_list):
            w_on = d_net(obs_on, acts_on)
            w_off = d_net(obs_off, acts_off)
            
            # E_{x~P}[f'(w)] = E[log(2w/(w+1))]
            term_on = torch.log(2 * w_on / (w_on + 1 + eps) + eps)
            
            # E_{x~Q}[f*(f'(w))] = E[log((w+1)/2)]
            term_off = torch.log((w_off + 1) / 2 + eps)
            
            # Maximize lower bound → minimize negative
            loss = -(torch.mean(term_on) - torch.mean(term_off))
            
            self.density_optimizer_list[i].zero_grad()
            loss.backward()
            self.density_optimizer_list[i].step()
            loss_total += loss.item()
            
        return loss_total / self.num_Q

    def calculate_priorities(self, obs, acts, is_online):
        """
        Line 11: Calculate priority P_R of R_N via (4)
        
        Eq. 4: p_i = σ_i^ζ / Σ_k σ_k^ζ
        
        σ_i = I^off · w(s_i,a_i) · exp(ξ·A(s_i,a_i)) + I^on · exp(ξ·A(s_i,a_i))
            = (I^off · w(s_i,a_i) + I^on) · exp(ξ·A(s_i,a_i))
        
        Eq. 5: w(s,a) = ŵ(s,a) - Ĉ_w(s,a)  [density LCB]
        Eq. 6: A(s,a) = Â(s,a) - Ĉ_A(s,a)  [advantage LCB]
        
        Paper: "For the advantage term, to enhance robustness, we use the 
        pessimistic CDQ Q estimation, while incorporating uncertainty estimation
        for the value function under the current policy."
        """
        with torch.no_grad():
            N = obs.shape[0]
            
            # === Density LCB (Eq. 5): w = ŵ - Ĉ_w ===
            ws = torch.stack([d_net(obs, acts) for d_net in self.density_net_list], dim=0)  # [E,N,1]
            w_mean = ws.mean(dim=0)  # ŵ
            w_std = ws.std(dim=0)    # Ĉ_w
            w_lcb = torch.clamp(w_mean - w_std, min=0.0)
            
            # === Advantage LCB (Eq. 6): A = Â - Ĉ_A ===
            # Q(s,a) ensemble estimates
            qs = torch.stack([q_net(torch.cat([obs, acts], dim=1)) 
                             for q_net in self.q_net_list], dim=0)  # [E,N,1]
            q_mean = qs.mean(dim=0)
            q_std = qs.std(dim=0)
            
            # Pessimistic Q (CDQ-style LCB)
            q_lcb = q_mean - q_std
            
            # V(s) = E_{a'~π}[Q(s,a') - α·log π(a'|s)]
            # Sample action from current policy for V estimation
            a_pi, _, _, log_pi, _, _ = self.policy_net(obs)
            qs_pi = torch.stack([q_net(torch.cat([obs, a_pi], dim=1)) 
                                for q_net in self.q_net_list], dim=0)
            v_mean = qs_pi.mean(dim=0) - self.alpha * log_pi
            v_std = qs_pi.std(dim=0)
            
            # For A_lcb = Q_lcb - V_ucb (conservative advantage)
            v_ucb = v_mean + v_std
            adv_lcb = q_lcb - v_ucb
            
            # === Priority σ (Eq. 4) ===
            # σ = (I^off · w + I^on) · exp(ξ · A)
            is_on = is_online.squeeze()  # [N]
            
            # "we set the density ratio to 1 for transitions from the online dataset"
            density_term = torch.where(
                is_on.unsqueeze(1),
                torch.ones_like(w_lcb),
                w_lcb
            )
            
            adv_term = torch.exp(self.xi * adv_lcb)
            sigma = density_term * adv_term  # [N,1]
            
            # p_i = σ_i^ζ / Σ_k σ_k^ζ
            sigma_zeta = torch.pow(sigma.squeeze() + 1e-8, self.zeta)
            
            return sigma_zeta.cpu().numpy()

    def train(self, logger, current_env_step=None):
        """
        Algorithm 1, Lines 9-21: Main training loop per environment step.
        
        Line 9:  Sample N/2 from R and N/2 from D → R_N
        Line 10: Update density ensemble
        Line 11: Calculate priorities
        Lines 12-21: G gradient steps with prioritized sampling
        """
        # Buffer size checks
        half_N = self.batch_size // 2
        if self.replay_buffer.size < half_N:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=self.alpha, LogPi=0, PreTanh=0)
            return
        if self.o2o and self.replay_buffer_offline.size < half_N:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=self.alpha, LogPi=0, PreTanh=0)
            return
        if self.__get_current_num_data() <= self.delay_update_steps:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=self.alpha, LogPi=0, PreTanh=0)
            return

        self.current_step += 1
        
        # === Line 9: Form R_N of size N ===
        # "Randomly sample a subset of size N/2 from online buffer R 
        #  and size N/2 from offline buffer D"
        batch_on = self.replay_buffer.sample_batch(half_N)
        batch_off = self.replay_buffer_offline.sample_batch(half_N) if self.o2o else self.replay_buffer.sample_batch(half_N)
        
        obs = torch.Tensor(np.concatenate([batch_on['obs1'], batch_off['obs1']])).to(self.device)
        obs_next = torch.Tensor(np.concatenate([batch_on['obs2'], batch_off['obs2']])).to(self.device)
        acts = torch.Tensor(np.concatenate([batch_on['acts'], batch_off['acts']])).to(self.device)
        rews = torch.Tensor(np.concatenate([batch_on['rews'], batch_off['rews']])).unsqueeze(1).to(self.device)
        dones = torch.Tensor(np.concatenate([batch_on['done'], batch_off['done']])).unsqueeze(1).to(self.device)
        
        # Online indicator: first half online, second half offline
        is_online = torch.cat([
            torch.ones(half_N, dtype=torch.bool, device=self.device),
            torch.zeros(half_N, dtype=torch.bool, device=self.device)
        ]).unsqueeze(1)
        
        # === Line 10: Update density ensemble ===
        if self.o2o:
            self.update_density_ensemble(obs[:half_N], acts[:half_N], obs[half_N:], acts[half_N:])
        
        # === Line 11: Calculate priorities ===
        sigma_zeta = self.calculate_priorities(obs, acts, is_online)
        probs = sigma_zeta / (sigma_zeta.sum() + 1e-8)
        
        # Beta annealing (Eq. 3): "β anneals from β_0 ∈ (0,1) to 1"
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.current_step / self.beta_steps)
        
        pool_size = self.batch_size  # |R_N| = N
        pool_idx = np.arange(pool_size)
        
        loss_q_sum, loss_pi_sum, alpha_loss_sum = 0.0, 0.0, 0.0
        num_pi_updates = 0
        q_vals_last, log_pi_last, pretanh_last = None, None, None

        # === Line 12: For g = 1,...,G ===
        for g in range(self.gradient_steps):
            
            # === Line 13: Sample b_N of size N according to P_R from R_N ===
            idx = np.random.choice(pool_idx, size=self.batch_size, p=probs, replace=True)
            
            b_obs = obs[idx]
            b_next = obs_next[idx]
            b_acts = acts[idx]
            b_rews = rews[idx]
            b_dones = dones[idx]
            b_probs = probs[idx]
            
            # === Line 17: Importance weight (Eq. 3) ===
            # u_i ∝ (1/(|R|·p_i))^β, normalized to max=1
            weights = (1.0 / (pool_size * b_probs + 1e-8)) ** beta
            weights = weights / weights.max()
            weights_t = torch.Tensor(weights).unsqueeze(1).to(self.device)
            
            # === Lines 14-15: Calculate target y ===
            with torch.no_grad():
                a_next, _, _, log_prob_next, _, _ = self.policy_net(b_next)
                
                # Line 14: Sample Z indices
                z_idx = np.random.choice(self.num_Q, size=self.num_min, replace=False)
                
                # min_{i∈Z} Q_{θ'_i}(s',a')
                target_qs = torch.stack([
                    self.q_target_net_list[i](torch.cat([b_next, a_next], dim=1))
                    for i in z_idx
                ], dim=0)
                min_q_next = torch.min(target_qs, dim=0)[0]
                
                # y = r + γ(min Q - α log π)
                y = b_rews + self.gamma * (1 - b_dones) * (min_q_next - self.alpha * log_prob_next)
            
            # === Lines 16-18: Update critics ===
            q_preds = [self.q_net_list[i](torch.cat([b_obs, b_acts], dim=1)) for i in range(self.num_Q)]
            q_cat = torch.cat(q_preds, dim=1)
            y_exp = y.expand(-1, self.num_Q)
            
            # Weighted MSE: ℓ = Σ_i u_i·(y - Q_{θ_i})^2
            q_loss = (weights_t * (q_cat - y_exp) ** 2).mean() * self.num_Q
            
            for opt in self.q_optimizer_list:
                opt.zero_grad()
            q_loss.backward()
            for opt in self.q_optimizer_list:
                opt.step()
            
            loss_q_sum += q_loss.item() / self.num_Q
            
            # === Line 19: Update targets ===
            for i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[i], self.q_net_list[i], self.polyak)
            
            # === Lines 20-21: Update actor ===
            if ((g + 1) % self.policy_update_delay == 0) or (g == self.gradient_steps - 1):
                a_new, _, _, log_prob_new, _, pretanh = self.policy_net(b_obs)
                
                # Freeze Q gradients
                for q in self.q_net_list:
                    q.requires_grad_(False)
                
                # (1/E) Σ Q_{θ_i}(s,a)
                q_vals = torch.cat([q(torch.cat([b_obs, a_new], dim=1)) for q in self.q_net_list], dim=1)
                avg_q = q_vals.mean(dim=1, keepdim=True)
                
                # Maximize Q - α log π → minimize α log π - Q
                pi_loss = (self.alpha * log_prob_new - avg_q).mean()
                
                self.policy_optimizer.zero_grad()
                pi_loss.backward()
                self.policy_optimizer.step()
                
                for q in self.q_net_list:
                    q.requires_grad_(True)
                
                loss_pi_sum += pi_loss.item()
                num_pi_updates += 1
                
                q_vals_last = q_preds[0].detach().cpu().numpy()
                log_pi_last = log_prob_new.detach().cpu().numpy()
                pretanh_last = pretanh.abs().detach().cpu().numpy().reshape(-1)
                
                # Update α
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                    alpha_loss_sum += alpha_loss.item()

        # === Logging ===
        logger.store(
            LossPi=loss_pi_sum / max(num_pi_updates, 1),
            LossQ1=loss_q_sum / self.gradient_steps,
            LossAlpha=alpha_loss_sum / max(num_pi_updates, 1) if self.auto_alpha else 0,
            Q1Vals=q_vals_last if q_vals_last is not None else 0,
            Alpha=self.alpha,
            LogPi=log_pi_last if log_pi_last is not None else 0,
            PreTanh=pretanh_last if pretanh_last is not None else 0
        )

        if current_env_step is not None:
            wandb.log({
                "policy_loss": loss_pi_sum / max(num_pi_updates, 1),
                "mean_loss_q": loss_q_sum / self.gradient_steps,
                "alpha": self.alpha
            }, step=current_env_step)