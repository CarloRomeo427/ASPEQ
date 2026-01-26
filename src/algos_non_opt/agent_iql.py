### AGENT CODE FOR IQL (Implicit Q-Learning) ALGORITHM
### Adapted from gwthomas/IQL-PyTorch to match FASPEQ codebase structure

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.algos_non_opt.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict
import wandb

from src.algos_non_opt.core import test_agent


EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u, tau):
    """
    Expectile loss: L_tau(u) = |tau - 1(u < 0)| * u^2
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class ValueNetwork(nn.Module):
    """State value function V(s) for IQL."""
    def __init__(self, obs_dim, hidden_sizes=(256, 256), layer_norm=False):
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
    
    def forward(self, obs):
        return self.net(obs)


class Agent(object):
    """
    IQL Agent - Implicit Q-Learning for Offline-to-Online RL.
    
    Key differences from SAC:
    - Adds Value network V(s) trained via expectile regression
    - Q-network bootstraps from V(s') instead of min Q with policy actions
    - Policy trained via advantage-weighted regression (AWR)
    
    Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=1, num_Q=2, num_min=2, q_target_mode='min',
                 policy_update_delay=1,
                 target_drop_rate=0.0, layer_norm=False, o2o=False,
                 # IQL-specific parameters
                 iql_tau=0.7, iql_beta=3.0,
                 # Validation buffer params (kept for compatibility)
                 val_buffer_prob=0.0, val_buffer_offline_frac=0.0,
                 val_check_interval=1000, val_patience=5000,
                 adaptive_trigger_expansion_rate=1.1, auto_stab=False,
                 num_reference_batches=10
                 ):
        """
        IQL Agent.
        
        IQL-specific parameters:
            iql_tau: Expectile for value function training (default: 0.7)
                     Higher values (0.7-0.9) focus more on high-value actions
            iql_beta: Temperature for advantage-weighted policy extraction (default: 3.0)
                      Lower values make policy more greedy
        """
        # Policy network - NOW WITH LAYER_NORM SUPPORT
        self.policy_net = TanhGaussianPolicy(
            obs_dim, act_dim, hidden_sizes, 
            action_limit=act_limit,
            layer_norm=layer_norm  # Pass layer_norm to policy network
        ).to(device)
        
        # Q networks (same as SAC)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                            layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                                   layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        
        # Value network V(s) - IQL specific
        self.value_net = ValueNetwork(obs_dim, hidden_sizes, layer_norm=layer_norm).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # IQL-specific hyperparameters
        self.iql_tau = iql_tau      # Expectile for value function
        self.iql_beta = iql_beta    # Temperature for AWR policy extraction
        
        # Auto alpha (kept for compatibility, but IQL doesn't use entropy bonus in standard form)
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
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_exploration_action_o2o(self, obs, env, timestep):
        with torch.no_grad():
            if timestep > self.start_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
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
        if num_samples == 0:
            return
            
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
        """IQL doesn't use offline stabilization, always returns False."""
        return False

    def sample_data(self, batch_size):
        """Sample batch from online replay buffer."""
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next_tensor = torch.FloatTensor(batch['obs2']).to(self.device)
        acts_tensor = torch.FloatTensor(batch['acts']).to(self.device)
        rews_tensor = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
    def sample_data_mix(self, batch_size):
        """Sample mixed batch from online and offline buffers (50/50 split)."""
        batch_online = self.replay_buffer.sample_batch(batch_size)
        batch_offline = self.replay_buffer_offline.sample_batch(batch_size)

        batch = {
            'obs1': np.concatenate([batch_online['obs1'], batch_offline['obs1']], axis=0),
            'obs2': np.concatenate([batch_online['obs2'], batch_offline['obs2']], axis=0),
            'acts': np.concatenate([batch_online['acts'], batch_offline['acts']], axis=0),
            'rews': np.concatenate([batch_online['rews'], batch_offline['rews']], axis=0),
            'done': np.concatenate([batch_online['done'], batch_offline['done']], axis=0),
        }

        obs_tensor = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next_tensor = torch.FloatTensor(batch['obs2']).to(self.device)
        acts_tensor = torch.FloatTensor(batch['acts']).to(self.device)
        rews_tensor = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    
    def sample_data_offline_only(self, batch_size):
        """Sample batch from offline buffer only."""
        batch = self.replay_buffer_offline.sample_batch(batch_size)
        obs_tensor = torch.FloatTensor(batch['obs1']).to(self.device)
        obs_next_tensor = torch.FloatTensor(batch['obs2']).to(self.device)
        acts_tensor = torch.FloatTensor(batch['acts']).to(self.device)
        rews_tensor = torch.FloatTensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = torch.FloatTensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def compute_iql_value_loss(self, obs_tensor, acts_tensor):
        """
        IQL Value Loss: Expectile regression on Q - V.
        
        L_V = E_{(s,a)~D}[L_tau(Q(s,a) - V(s))]
        
        where L_tau(u) = |tau - 1(u < 0)| * u^2
        """
        with torch.no_grad():
            # Get Q-values for dataset actions (use min of Q ensemble for conservatism)
            q_values_list = []
            for q_i in range(self.num_Q):
                q_val = self.q_target_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_values_list.append(q_val)
            q_values = torch.min(torch.cat(q_values_list, dim=1), dim=1, keepdim=True)[0]
        
        # Compute V(s)
        v_values = self.value_net(obs_tensor)
        
        # Expectile loss: L_tau(Q - V)
        u = q_values - v_values
        value_loss = asymmetric_l2_loss(u, self.iql_tau)
        
        return value_loss, v_values.mean().item()

    def compute_iql_q_loss(self, obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor):
        """
        IQL Q Loss: Standard TD loss but bootstrapping from V(s') instead of Q(s', π(s')).
        
        L_Q = E_{(s,a,r,s')~D}[(r + γ * (1-d) * V(s') - Q(s,a))^2]
        """
        with torch.no_grad():
            # Target: r + γ * V(s')
            next_v = self.value_net(obs_next_tensor)
            target_q = rews_tensor + self.gamma * (1 - done_tensor) * next_v
        
        # Current Q predictions
        q_loss_total = 0
        q_values_for_log = []
        for q_i in range(self.num_Q):
            q_pred = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_loss_total += self.mse_criterion(q_pred, target_q)
            q_values_for_log.append(q_pred.mean().item())
        
        return q_loss_total, np.mean(q_values_for_log)

    def compute_iql_policy_loss(self, obs_tensor, acts_tensor):
        """
        IQL Policy Loss: Advantage-Weighted Regression (AWR).
        
        L_π = E_{(s,a)~D}[exp((Q(s,a) - V(s)) / β) * log π(a|s)]
        
        The advantage is clamped to avoid numerical instability.
        """
        with torch.no_grad():
            # Compute advantage A(s,a) = Q(s,a) - V(s)
            q_values_list = []
            for q_i in range(self.num_Q):
                q_val = self.q_target_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_values_list.append(q_val)
            q_values = torch.min(torch.cat(q_values_list, dim=1), dim=1, keepdim=True)[0]
            v_values = self.value_net(obs_tensor)
            
            # Advantage weights with clipping for numerical stability
            adv = q_values - v_values
            exp_adv = torch.exp(adv / self.iql_beta).clamp(max=EXP_ADV_MAX)
        
        # Get policy distribution parameters
        # TanhGaussianPolicy.forward returns: (action, mean, log_std, log_prob, pretanh_action, tanh_action)
        _, mean, log_std, _, _, _ = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=True)
        
        # Compute log probability of dataset actions under current policy
        # For tanh-squashed Gaussian: log π(a|s) = log N(atanh(a); μ, σ) - log(1 - tanh²(atanh(a)))
        # Since acts_tensor are already squashed, we need to invert
        
        # Normalize actions to [-1, 1] range for log-prob computation
        acts_normalized = acts_tensor / self.act_limit
        
        eps = 1e-6
        acts_clamped = acts_normalized.clamp(-1 + eps, 1 - eps)
        pretanh_actions = torch.atanh(acts_clamped)
        
        std = log_std.exp()
        
        # Gaussian log prob
        var = std ** 2
        log_prob_gaussian = -0.5 * (((pretanh_actions - mean) ** 2) / var + 2 * log_std + np.log(2 * np.pi))
        log_prob_gaussian = log_prob_gaussian.sum(dim=-1, keepdim=True)
        
        # Jacobian correction for tanh squashing
        log_jacobian = torch.log(1 - acts_clamped ** 2 + eps).sum(dim=-1, keepdim=True)
        
        log_prob_data_actions = log_prob_gaussian - log_jacobian
        
        policy_loss = -(exp_adv * log_prob_data_actions).mean()
        
        return policy_loss, log_prob_data_actions.mean().item()

    def train(self, logger, current_env_step=None):
        """
        IQL training update.
        
        Update order:
        1. Value function (expectile regression)
        2. Q-functions (TD with V bootstrap)
        3. Policy (advantage-weighted regression)
        """
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        
        # Initialize logging variables with defaults
        q_loss_val, q_mean = 0.0, 0.0
        value_loss_val, v_mean = 0.0, 0.0
        policy_loss_val, log_pi_mean = 0.0, 0.0
        alpha_loss_val = 0.0
        
        for i_update in range(num_update):
            # Sample batch
            if self.o2o:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_mix(self.batch_size)
            else:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            # ═══════════════════════════════════════════════════════════════════
            # 1. VALUE FUNCTION UPDATE (Expectile Regression)
            # ═══════════════════════════════════════════════════════════════════
            value_loss, v_mean = self.compute_iql_value_loss(obs_tensor, acts_tensor)
            value_loss_val = value_loss.item()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # ═══════════════════════════════════════════════════════════════════
            # 2. Q-FUNCTION UPDATE (TD with V bootstrap)
            # ═══════════════════════════════════════════════════════════════════
            q_loss, q_mean = self.compute_iql_q_loss(obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor)
            q_loss_val = q_loss.item()
            
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            # ═══════════════════════════════════════════════════════════════════
            # 3. POLICY UPDATE (Advantage-Weighted Regression)
            # ═══════════════════════════════════════════════════════════════════
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                policy_loss, log_pi_mean = self.compute_iql_policy_loss(obs_tensor, acts_tensor)
                policy_loss_val = policy_loss.item()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # Alpha update (optional, for compatibility)
                if self.auto_alpha:
                    _, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=True)
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                    alpha_loss_val = alpha_loss.item()
                else:
                    alpha_loss_val = 0.0

            # ═══════════════════════════════════════════════════════════════════
            # 4. TARGET NETWORK UPDATE (Soft update)
            # ═══════════════════════════════════════════════════════════════════
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

        # Always log to logger (required by EpochLogger)
        logger.store(Q1Vals=q_mean, LossQ1=q_loss_val,
                     LogPi=log_pi_mean, LossPi=policy_loss_val,
                     Alpha=self.alpha, LossAlpha=alpha_loss_val,
                     PreTanh=0.0, LossV=value_loss_val, VMean=v_mean)
        
        # WandB logging only when updates occurred
        if num_update > 0 and current_env_step is not None:
            wandb.log({
                "IQL/q_loss": q_loss_val,
                "IQL/value_loss": value_loss_val,
                "IQL/policy_loss": policy_loss_val,
                "IQL/q_mean": q_mean,
                "IQL/v_mean": v_mean,
                "IQL/alpha": self.alpha,
            }, step=current_env_step)

    def train_offline(self, epochs, test_env=None, current_env_step=None, log_interval=1000):
        """
        Pure offline IQL training (before online finetuning).
        
        Args:
            epochs: Number of gradient steps
            test_env: Environment for evaluation
            current_env_step: Current environment step for logging
            log_interval: How often to log and evaluate
        """
        print(f"\n{'='*80}")
        print(f"STARTING IQL OFFLINE TRAINING")
        print(f"{'='*80}")
        print(f"Epochs: {epochs}")
        print(f"Tau (expectile): {self.iql_tau}")
        print(f"Beta (AWR temperature): {self.iql_beta}")
        print(f"Offline buffer size: {self.replay_buffer_offline.size}")
        print(f"{'='*80}\n")
        
        for e in range(epochs):
            # Sample from offline buffer only
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data_offline_only(self.batch_size)
            
            # Value update
            value_loss, v_mean = self.compute_iql_value_loss(obs_tensor, acts_tensor)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            # Q update
            q_loss, q_mean = self.compute_iql_q_loss(obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor)
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()
            
            # Policy update
            policy_loss, log_pi_mean = self.compute_iql_policy_loss(obs_tensor, acts_tensor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Target update
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)
            
            # Logging
            if (e + 1) % log_interval == 0:
                print(f"Epoch {e+1}/{epochs}: Q_loss={q_loss.item():.4f}, V_loss={value_loss.item():.4f}, Pi_loss={policy_loss.item():.4f}")
                
                wandb.log({
                    "IQL_offline/q_loss": q_loss.item(),
                    "IQL_offline/value_loss": value_loss.item(),
                    "IQL_offline/policy_loss": policy_loss.item(),
                    "IQL_offline/epoch": e + 1,
                    "offline_training": 1
                }, step=e + 1)
                
                # Evaluation
                if test_env is not None:
                    test_rw = test_agent(self, test_env, 1000, None)
                    print(f"  Eval reward: {np.mean(test_rw):.2f}")
                    wandb.log({
                        "OfflineEvalRewards": np.mean(test_rw),
                    }, step=e + 1)
        
        print(f"\n{'='*80}")
        print(f"IQL OFFLINE TRAINING COMPLETED")
        print(f"{'='*80}\n")
        
        wandb.log({"offline_training": 0}, step=epochs)
        
        return epochs