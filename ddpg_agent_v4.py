"""
ddpg_agent_v4.py

DDPG Agent v4 — Pulsed Hot-Air Reheating Process Control
CBIC × TUAT | Julian Evan Chrisnanto | 2026

═══════════════════════════════════════════════════════════════════════
CHANGES FROM v3 (based on CLIENT_Data_0427.xlsx)
═══════════════════════════════════════════════════════════════════════

NEW PROCESS UNDERSTANDING:
  The real production line uses short hot-air pulses (≈20 sec, ~100°C)
  applied early in the process while the bulk is still in the mushy
  zone (62–72°C). This is NOT a post-solidification reheat — it is a
  mushy-zone surface treatment to prevent/heal sink marks.

  Total target: ≤ 30 min (production line: ~24.5 min observed)

NEW ACTION SPACE (8 dimensions):
  a[0]: dT_reheat         hot-air temperature adjustment     [±10°C]
  a[1]: d_h_hot           hot-air convection intensity       [±15 W/m²K]
  a[2]: dt_pulse_start    first pulse timing adjustment      [±30 sec]
  a[3]: dt_pulse_dur      pulse duration adjustment          [±10 sec]
  a[4]: dt_pulse_interval inter-pulse interval adjustment    [±30 sec]
  a[5]: dn_pulses         number of pulses adjustment        [±2]
  a[6]: dT_cool           cooling environment adjustment     [±5°C]
  a[7]: dt_cold_start     cold-air phase start adjustment    [±120 sec]

STATE SPACE (64 dimensions):
  Spatial field observations at 16 key (r,z) probe locations:
    - T_surface: T at 4 surface points (r=R, z=0.25H, 0.5H, 0.75H, H)
    - T_core:    T at 4 core points   (r=0, z=0.25H, 0.5H, 0.75H, H)
    - phi:       φ at 4 surface points + 4 core points
  Process state (8 dims):
    - t_norm, DI_current, phi_max, phi_top, fl_mean
    - Bi_current, T_env_current, heal_effectiveness
  Process parameters (8 dims):
    - T_fill_norm, T_reheat_norm, h_cool_norm, h_hot_norm
    - t_pulse_start_norm, t_pulse_dur_norm, t_pulse_interval_norm, n_pulses_norm
  History (16 dims):
    - Last 4 DI values, last 4 phi_top values,
      last 4 T_surface[top] values, last 4 healing_event flags

REWARD FUNCTION:
  r = -w1×DI - w2×phi_max - w3×phi_top - w4×sink_indicator
      + w5×heal_bonus - w6×time_penalty - w7×energy_cost

  heal_bonus: positive reward when φ decreases during hot-air pulse
  time_penalty: penalizes total process time > 25 min
  energy_cost: penalizes high h_hot (energy consumption of hot air)

NETWORKS:
  Actor:  state(64) → action(8), 3 hidden layers [512, 256, 128]
  Critic: state(64) + action(8) → Q-value, 3 hidden layers [512, 256, 128]
  Both use LayerNorm for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

STATE_DIM  = 64
ACTION_DIM = 8

# Action scaling (maps tanh output [-1,1] to physical range)
ACTION_SCALES = {
    'dT_reheat':          10.0,   # °C
    'd_h_hot':            15.0,   # W/m²K
    'dt_pulse_start':     30.0,   # sec
    'dt_pulse_dur':       10.0,   # sec
    'dt_pulse_interval':  30.0,   # sec
    'dn_pulses':           2.0,   # count (round to int in env)
    'dT_cool':             5.0,   # °C
    'dt_cold_start':     120.0,   # sec
}

ACTION_NAMES = list(ACTION_SCALES.keys())
ACTION_SCALE_ARRAY = np.array(list(ACTION_SCALES.values()), dtype=np.float32)


# ============================================================================
# NETWORKS
# ============================================================================

class ActorNetworkV4(nn.Module):
    """
    Policy network: state(64) → action(8).
    Maps current process state to parameter adjustments.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),   nn.LayerNorm(hidden),   nn.ReLU(),
            nn.Linear(hidden,        256),  nn.LayerNorm(256),      nn.ReLU(),
            nn.Linear(256,           128),  nn.LayerNorm(128),      nn.ReLU(),
            nn.Linear(128,    action_dim),
        )
        # Small init for output layer → conservative initial policy
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)

    def forward(self, state):
        return torch.tanh(self.net(state))


class CriticNetworkV4(nn.Module):
    """
    Q-value network: (state(64), action(8)) → scalar Q.
    State and action processed separately before merging.
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=512):
        super().__init__()
        self.state_enc  = nn.Sequential(
            nn.Linear(state_dim,  384), nn.LayerNorm(384), nn.ReLU(),
        )
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim,  64), nn.ReLU(),
        )
        self.merged = nn.Sequential(
            nn.Linear(384 + 64, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden,      256), nn.LayerNorm(256),    nn.ReLU(),
            nn.Linear(256,           1),
        )
        nn.init.uniform_(self.merged[-1].weight, -3e-3, 3e-3)

    def forward(self, state, action):
        s = self.state_enc(state)
        a = self.action_enc(action)
        return self.merged(torch.cat([s, a], dim=1))


# ============================================================================
# PRIORITIZED REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Experience replay with priority-based sampling.
    High-DI (bad) experiences are replayed more often so the agent
    learns to avoid critical crack states.
    """
    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=50_000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_end    = beta_end
        self.beta_inc    = (beta_end - beta_start) / beta_steps
        self.buffer      = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos         = 0

    def push(self, state, action, reward, next_state, done, di_val=0.0):
        # Priority = |reward| + DI bonus (bad experiences → higher priority)
        priority = abs(reward) + 1.0 + di_val * 2.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(self.beta_end, self.beta + self.beta_inc)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
            indices,
            torch.FloatTensor(weights).unsqueeze(1),
        )

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DDPG AGENT V4
# ============================================================================

class DDPGAgentV4:
    """
    DDPG v4 agent for pulsed hot-air reheating process control.

    Key differences from v3:
    - 8D action space targeting pulsed process parameters
    - 64D state space with healing effectiveness tracking
    - Physics-constrained action clipping (prevents physically impossible params)
    - Prioritized replay with DI-weighted sampling
    """

    # Process parameter bounds (for action clipping)
    PARAM_BOUNDS = {
        'T_reheat':         (353.15, 393.15),   # 80–120°C in K
        'h_hot':            (20.0,   70.0),      # W/m²K
        't_pulse_start':    (15.0,   300.0),     # 15–300 sec
        't_pulse_dur':      (10.0,   120.0),     # 10–120 sec
        't_pulse_interval': (30.0,   600.0),     # 30–600 sec
        'n_pulses':         (1.0,    10.0),      # 1–10
        'T_cool':           (289.15, 303.15),    # 16–30°C in K
        't_cold_start':     (600.0, 1800.0),     # 10–30 min in sec
    }

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 gamma=0.99, tau=0.005, actor_lr=1e-4, critic_lr=3e-4,
                 batch_size=128, buffer_size=100_000):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Actor + target actor
        self.actor        = ActorNetworkV4(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetworkV4(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic + target critic
        self.critic        = CriticNetworkV4(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetworkV4(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = PrioritizedReplayBuffer(capacity=buffer_size)

        # Exploration noise — starts high, decays
        self.noise_scale = 0.3
        self.noise_min   = 0.02
        self.noise_decay = 0.9995

        self.steps_done = 0

    def select_action(self, state, noise_scale=None, deterministic=False):
        """
        Select action given state.
        noise_scale=0.0 for pure inference, >0 for exploration.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()

        if not deterministic:
            ns = noise_scale if noise_scale is not None else self.noise_scale
            noise = np.random.normal(0, ns, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def action_to_params(self, action_raw, current_params):
        """
        Convert normalized action [-1,1] to physical parameter deltas.
        Applies physics-based bounds to prevent invalid configurations.

        Returns dict of new process parameters.
        """
        delta = action_raw * ACTION_SCALE_ARRAY
        new_p = dict(current_params)

        new_p['T_reheat']         = float(np.clip(
            current_params['T_reheat'] + delta[0],
            *self.PARAM_BOUNDS['T_reheat']))
        new_p['h_hot']            = float(np.clip(
            current_params['h_hot'] + delta[1],
            *self.PARAM_BOUNDS['h_hot']))
        new_p['t_pulse_start']    = float(np.clip(
            current_params['t_pulse_start'] + delta[2],
            *self.PARAM_BOUNDS['t_pulse_start']))
        new_p['t_pulse_dur']      = float(np.clip(
            current_params['t_pulse_dur'] + delta[3],
            *self.PARAM_BOUNDS['t_pulse_dur']))
        new_p['t_pulse_interval'] = float(np.clip(
            current_params['t_pulse_interval'] + delta[4],
            *self.PARAM_BOUNDS['t_pulse_interval']))
        new_p['n_pulses']         = float(np.clip(
            round(current_params['n_pulses'] + delta[5]),
            *self.PARAM_BOUNDS['n_pulses']))
        new_p['T_cool']           = float(np.clip(
            current_params['T_cool'] + delta[6],
            *self.PARAM_BOUNDS['T_cool']))
        new_p['t_cold_start']     = float(np.clip(
            current_params['t_cold_start'] + delta[7],
            *self.PARAM_BOUNDS['t_cold_start']))

        # Physics constraint: cold start must be after last pulse
        last_pulse_end = (new_p['t_pulse_start']
                          + (new_p['n_pulses'] - 1) * new_p['t_pulse_interval']
                          + new_p['t_pulse_dur'])
        new_p['t_cold_start'] = max(new_p['t_cold_start'], last_pulse_end + 60.0)

        # Physics constraint: total time must be ≤ 1800 sec (30 min)
        new_p['t_cold_start'] = min(new_p['t_cold_start'], 1500.0)

        return new_p

    def update(self):
        """Single DDPG update step with prioritized replay."""
        if len(self.replay) < self.batch_size:
            return None

        (states, actions, rewards, next_states,
         dones, indices, weights) = self.replay.sample(self.batch_size)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)
        weights     = weights.to(self.device)

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next       = self.critic_target(next_states, next_actions)
            q_target     = rewards + self.gamma * (1 - dones) * q_next

        q_pred   = self.critic(states, actions)
        td_error = (q_pred - q_target).detach().cpu().numpy().squeeze()

        critic_loss = (weights * F.mse_loss(q_pred, q_target, reduction='none')).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Update priorities
        self.replay.update_priorities(indices, td_error)

        # ── Actor update ──────────────────────────────────────────────────
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft target update ────────────────────────────────────────────
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        # Decay exploration noise
        self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        self.steps_done += 1

        return {
            'critic_loss': float(critic_loss.item()),
            'actor_loss':  float(actor_loss.item()),
            'noise':       self.noise_scale,
        }

    def save(self, path):
        torch.save({
            'actor':              self.actor.state_dict(),
            'actor_target':       self.actor_target.state_dict(),
            'critic':             self.critic.state_dict(),
            'critic_target':      self.critic_target.state_dict(),
            'actor_opt':          self.actor_opt.state_dict(),
            'critic_opt':         self.critic_opt.state_dict(),
            'steps_done':         self.steps_done,
            'noise_scale':        self.noise_scale,
        }, path)

    def load(self, path, inference_only=False):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck['actor'])
        self.actor_target.load_state_dict(ck['actor_target'])
        if not inference_only:
            self.critic.load_state_dict(ck['critic'])
            self.critic_target.load_state_dict(ck['critic_target'])
            if 'actor_opt' in ck:
                self.actor_opt.load_state_dict(ck['actor_opt'])
            if 'critic_opt' in ck:
                self.critic_opt.load_state_dict(ck['critic_opt'])
            self.steps_done  = ck.get('steps_done', 0)
            self.noise_scale = ck.get('noise_scale', 0.3)
        self.actor.eval()
        print(f"[DDPGv4] Loaded from {path} "
              f"(steps={self.steps_done}, noise={self.noise_scale:.3f})")

    def get_recommended_parameters(self, state, current_params):
        """
        Return human-readable parameter recommendations.
        For dashboard display and operator guidance.
        """
        action = self.select_action(state, deterministic=True)
        new_p  = self.action_to_params(action, current_params)
        total_t = (new_p['t_cold_start']
                   + new_p['n_pulses'] * new_p['t_pulse_dur'] + 180) / 60.0

        return {
            'T_reheat_C':          float(new_p['T_reheat'] - 273.15),
            'h_hot_W_m2K':         float(new_p['h_hot']),
            't_pulse_start_sec':   float(new_p['t_pulse_start']),
            't_pulse_dur_sec':     float(new_p['t_pulse_dur']),
            't_pulse_interval_sec':float(new_p['t_pulse_interval']),
            'n_pulses':            int(round(new_p['n_pulses'])),
            'T_cool_C':            float(new_p['T_cool'] - 273.15),
            't_cold_start_min':    float(new_p['t_cold_start'] / 60.0),
            'estimated_total_min': float(total_t),
            'within_30min_target': total_t <= 30.0,
        }


def load_agent(checkpoint_path, inference_only=True):
    """Convenience function — load inference-ready agent."""
    agent = DDPGAgentV4()
    agent.load(checkpoint_path, inference_only=inference_only)
    return agent
