"""
train_drl_agent_v4.py

DRL Training for Pulsed Hot-Air Reheating Process Control
CBIC × TUAT | Julian Evan Chrisnanto | 2026

═══════════════════════════════════════════════════════════════════════
PROCESS BEING OPTIMIZED (from CLIENT_Data_0427.xlsx)
═══════════════════════════════════════════════════════════════════════

Actual CBIC Production Line Process:
  1. Fill at T_fill ≈ 75°C (back-fill) or 82°C (front-fill)
  2. Apply N short hot-air pulses (≈20 sec each, ~100°C surface)
     while bulk is in mushy zone (62–72°C) — t ≈ 0.5–6 min
  3. Continue cooling under ambient conditions (21–25°C)
  4. Apply cold air (16°C) at t ≈ 21 min for final solidification
  Total: ~24.5 min (well within 30-min target)

Key DRL task:
  Find the optimal combination of:
  - T_reheat (hot-air temperature)
  - h_hot (hot-air convection intensity = heat transfer rate)
  - t_pulse_start (when to start the first pulse)
  - t_pulse_dur (how long each pulse lasts)
  - t_pulse_interval (time between pulses)
  - n_pulses (total number of pulses)
  - T_cool (ambient cooling temperature)
  - t_cold_start (when to switch to cold air)

  Such that: DI → 0, φ_top → 0, total time ≤ 30 min

ENVIRONMENT:
  PhysicsEnvV4 wraps the trained MS-PINN v4 and computes:
  - State vector (64D) from PINN field predictions
  - Reward based on DI, φ, healing effectiveness, time penalty
  - Episode: 150 steps × Δt = 12 sec = 30-min process window

TRAINING PROTOCOL:
  5,000 episodes, batch_size=128, prioritized replay
  Exploration: σ noise 0.3 → 0.02 (exponential decay)
  Reference baselines tracked: production (observed), client best ①
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ddpg_agent_v4 import DDPGAgentV4, ACTION_SCALE_ARRAY
from train_pinn_physics_enforced_v4 import (
    MultiPhysicsPINNV4, MATERIAL, NORM, GEO, PROCESS,
    T_CYCLE, T_ROOM, sample_process_params
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PHYSICS ENVIRONMENT V4
# ============================================================================

class PhysicsEnvV4:
    """
    Differentiable physics environment backed by MS-PINN v4.

    State (64D):
      [0:8]   T_surface at 4 probe z-locations (r=R)
      [8:16]  T_core at 4 probe z-locations (r=0)
      [16:24] phi at 4 surface + 4 core probe locations
      [24:32] process state: t_norm, DI, phi_max, phi_top, fl_mean,
                             Bi, T_env, heal_effectiveness
      [32:40] process params (normalized): T_fill, T_reheat, h_cool,
                             h_hot, t_ps, t_pd, t_pi, n_pulses
      [40:64] history buffer: DI[t-4:t], phi_top[t-4:t],
                              T_surf_top[t-4:t], heal_events[t-4:t]

    Reward:
      r = -4.0×DI - 3.0×phi_max - 2.0×phi_top
          - 2.5×sink_indicator
          + 5.0×heal_bonus
          - 0.5×time_penalty
          - 0.3×energy_cost
          + 10.0×success_bonus (if DI < 0.10 at end)
    """

    # Probe locations
    Z_PROBES   = np.array([0.25, 0.50, 0.75, 1.00]) * GEO.H  # 4 z-heights
    R_SURFACE  = GEO.R
    R_CORE     = 1e-4  # near-zero (avoid singularity)

    # Episode parameters
    N_STEPS    = 150
    DT_STEP    = 12.0   # sec per step = 30 min / 150 steps

    # Reward weights
    W_DI       = 4.0
    W_PHI_MAX  = 3.0
    W_PHI_TOP  = 2.0
    W_SINK     = 2.5
    W_HEAL     = 5.0
    W_TIME     = 0.5
    W_ENERGY   = 0.3
    W_SUCCESS  = 10.0

    # Success threshold
    DI_SAFE    = 0.15
    DI_TARGET  = 0.10

    def __init__(self, pinn_model):
        self.pinn   = pinn_model
        self.pinn.eval()
        self.reset()

    def reset(self, params=None):
        """Reset environment. Optional params dict for curriculum."""
        self.step_count  = 0
        self.t_current   = 0.0  # seconds
        self.history_DI       = [0.0] * 4
        self.history_phi_top  = [0.0] * 4
        self.history_T_surf   = [MATERIAL.T_liquidus - 273.15] * 4  # °C
        self.history_heal     = [0.0] * 4
        self.prev_phi_top     = 0.0
        self.prev_DI          = 0.0
        self.cumulative_energy= 0.0
        self.healing_count    = 0

        # Default process parameters (production line baseline)
        if params is None:
            T_fill_base = np.random.uniform(74, 83) + 273.15
            self.current_params = {
                'T_fill':          T_fill_base,
                'T_cool':          (np.random.uniform(20, 25)) + 273.15,
                'T_reheat':        (np.random.uniform(85, 105)) + 273.15,
                'T_cold':          (16.0) + 273.15,
                'h_cool':          np.random.uniform(5, 12),
                'h_hot':           np.random.uniform(30, 55),
                'h_cold':          20.0,
                't_pulse_start':   np.random.uniform(20, 60),   # sec
                't_pulse_dur':     np.random.uniform(15, 30),   # sec
                't_pulse_interval':np.random.uniform(60, 180),  # sec
                'n_pulses':        float(np.random.randint(3, 8)),
                't_cold_start':    np.random.uniform(1000, 1400),  # sec
                'T_container':     (23.0) + 273.15,
                'viscosity':       0.3,
                'humidity':        0.4,
                'Gc':              80.0,
            }
        else:
            self.current_params = dict(params)

        return self._get_state()

    def _query_pinn(self, t_s, params):
        """
        Query PINN at current time and all probe locations.
        Returns dict with field values at probe points.
        """
        r_pts, z_pts = [], []
        # Surface probes
        for zp in self.Z_PROBES:
            r_pts.append(self.R_SURFACE); z_pts.append(zp)
        # Core probes
        for zp in self.Z_PROBES:
            r_pts.append(self.R_CORE); z_pts.append(zp)

        n = len(r_pts)
        r = torch.tensor(r_pts, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        z = torch.tensor(z_pts, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        t = torch.full((n, 1), t_s, dtype=torch.float32, device=DEVICE)

        bc = lambda v: torch.full((n, 1), float(v), dtype=torch.float32, device=DEVICE)
        p  = params

        with torch.no_grad():
            out = self.pinn.predict_fields(
                r, z, t,
                bc(p['T_fill']),     bc(p['T_cool']),   bc(p['T_reheat']),
                bc(p['T_cold']),     bc(p['h_cool']),   bc(p['h_hot']),
                bc(p['t_pulse_start']), bc(p['t_pulse_dur']),
                bc(p['t_pulse_interval']), bc(p['n_pulses']),
                bc(p['t_cold_start']),    bc(p['T_container']),
                bc(p['viscosity']),  bc(p['humidity']),  bc(p['Gc']),
            )

        T   = out['T'].squeeze().cpu().numpy()         # [8] Kelvin
        phi = out['phi'].squeeze().cpu().numpy()        # [8]
        fl  = out['f_l'].squeeze().cpu().numpy()        # [8]
        T_e = float(out['T_env'].mean().cpu())
        h_d = float(out['heal_driver'].mean().cpu()) if 'heal_driver' in out else 0.0

        return {
            'T_surf_K':    T[:4],                    # surface temps [K]
            'T_core_K':    T[4:],                    # core temps [K]
            'phi_surf':    phi[:4],
            'phi_core':    phi[4:],
            'fl_surf':     fl[:4],
            'fl_core':     fl[4:],
            'T_env_K':     T_e,
            'heal_driver': h_d,
        }

    def _compute_DI(self, fields):
        """
        Robust Damage Indicator computation.

        Fix for DI=0.500 lock:
          The untrained PINN outputs Sigmoid(net) ≈ 0.5 everywhere,
          giving phi_max ≈ 0.5 uniformly. The old code returned phi_max
          directly → DI stuck at 0.500 for ALL parameter combinations
          → agent received zero gradient signal.

        New approach (physics-proxy primary):
          1. Always compute physics-based DI from Biot + temperature
             → This varies meaningfully with process parameters
          2. Check if PINN phi shows spatial variation (std > 0.08)
             → Only then trust it as a modifier
          3. Final DI = physics_DI * (1 - trust) + pinn_DI * trust
        """
        phi_all = np.concatenate([fields['phi_surf'], fields['phi_core']])
        phi_std = float(np.std(phi_all))
        phi_max = float(np.max(phi_all))
        phi_mean= float(np.mean(phi_all))

        # ── Physics proxy (always computed, varies with params) ───────────
        h_now   = self.current_params['h_hot'] if self._in_pulse() \
                  else self.current_params['h_cool']
        Bi      = h_now * GEO.R / MATERIAL.k_thermal
        Bi_risk = float(np.clip((Bi - 0.20) / 0.70, 0.0, 1.0))

        Ts  = float(fields['T_surf_K'][-1]) - 273.15
        Tc  = float(fields['T_core_K'][-1]) - 273.15
        Te  = float(fields['T_env_K']) - 273.15
        Tf  = self.current_params['T_fill'] - 273.15
        fl  = float(np.mean(fields['fl_surf']))

        dT_drive  = max(0.0, Tf - Te)
        cool_sev  = float(np.clip(dT_drive / 60.0, 0.0, 1.0))

        # Phase weight: peak risk in mushy zone
        if fl > 0.90:
            phase_w = 0.55          # mostly liquid
        elif fl > 0.10:
            phase_w = float(np.clip(1.0 - 1.6 * abs(fl - 0.50), 0.30, 1.0))
        else:
            phase_w = 0.50          # fully solid, residual risk

        T_avg     = (Ts + Tc) / 2.0
        mushy_prx = float(np.exp(-((T_avg - 67.0)**2) / (5.5**2)))

        # Healing reduction: during hot-air pulse, surface re-softens
        heal_reduce = 0.0
        if self._in_pulse():
            T_surf_above_sol = max(0.0, Ts - 62.0)  # how far above solidus
            heal_reduce = float(np.clip(T_surf_above_sol / 10.0 * 0.3, 0.0, 0.3))

        physics_DI = float(np.clip((
            0.45 * Bi_risk * cool_sev * phase_w +
            0.25 * Bi_risk * phase_w +
            0.20 * mushy_prx * Bi_risk +
            0.10 * Bi_risk
        ) * 1.20 - heal_reduce, 0.0, 1.0))

        # ── PINN modifier (only when phi shows real spatial structure) ────
        # Threshold: std > 0.08 means PINN has learned non-trivial distribution
        pinn_trust = float(np.clip((phi_std - 0.08) / 0.15, 0.0, 1.0))

        if pinn_trust > 0.05:
            # Structured PINN DI: weighted combination of field values
            phi_surf_mean = float(np.mean(fields['phi_surf']))
            phi_top       = float(fields['phi_surf'][-1])   # top surface
            pinn_DI = float(np.clip(
                0.35 * phi_max +
                0.30 * float(np.percentile(phi_all, 85)) +
                0.20 * phi_surf_mean +
                0.15 * phi_top,
                0.0, 1.0
            ))
            # Blend physics proxy with PINN
            DI = (1.0 - pinn_trust) * physics_DI + pinn_trust * pinn_DI
        else:
            DI = physics_DI

        return float(np.clip(DI, 0.0, 1.0))

    def _in_pulse(self):
        """Returns True if current time is during a hot-air pulse."""
        p = self.current_params
        t = self.t_current
        for i in range(int(p['n_pulses'])):
            on  = p['t_pulse_start'] + i * p['t_pulse_interval']
            off = on + p['t_pulse_dur']
            if on <= t < off:
                return True
        return False

    def _get_state(self):
        """Build 64D state vector from current PINN evaluation."""
        if self.step_count == 0:
            # Initial state — analytically from fill conditions
            T_f  = self.current_params['T_fill']
            T_50 = T_f * 0.95   # approximation: slight cooling
            state = np.zeros(64, dtype=np.float32)
            # Surface temps (normalized to [0,1])
            state[0:4]  = (T_f - 273.15) / 80.0  # all at fill temp
            state[4:8]  = (T_f - 273.15) / 80.0  # core same at t=0
            state[8:16] = 0.0   # phi = 0 at start
            state[16]   = 0.0   # t_norm
            state[17]   = 0.0   # DI
            state[24:32] = self._encode_params()
            return state

        fields = self._query_pinn(self.t_current, self.current_params)

        state = np.zeros(64, dtype=np.float32)

        # [0:4] Surface temperature (normalized [0,1]: 0=room, 1=fill)
        state[0:4]  = (fields['T_surf_K'] - 273.15) / 80.0

        # [4:8] Core temperature
        state[4:8]  = (fields['T_core_K'] - 273.15) / 80.0

        # [8:12] Phi surface
        state[8:12] = np.clip(fields['phi_surf'], 0, 1)

        # [12:16] Phi core
        state[12:16] = np.clip(fields['phi_core'], 0, 1)

        # [16:24] Process state scalars
        phi_max = float(np.max(np.concatenate([fields['phi_surf'],
                                                fields['phi_core']])))
        phi_top = float(fields['phi_surf'][-1])   # top surface
        fl_mean = float(np.mean(np.concatenate([fields['fl_surf'],
                                                 fields['fl_core']])))
        h_now   = (self.current_params['h_hot']
                   if self._in_pulse() else self.current_params['h_cool'])
        Bi_now  = h_now * GEO.R / MATERIAL.k_thermal
        DI_now  = self._compute_DI(fields)

        # Healing effectiveness: how much phi decreased this step
        heal_eff = float(np.clip(self.prev_phi_top - phi_top, 0, 1))

        state[16] = self.t_current / T_CYCLE             # t_norm
        state[17] = DI_now                                # DI
        state[18] = phi_max                               # phi_max
        state[19] = phi_top                               # phi_top
        state[20] = fl_mean                               # fl_mean
        state[21] = float(np.clip(Bi_now / 2.0, 0, 1))  # Bi_norm
        state[22] = float((fields['T_env_K'] - 273.15) / 80.0)  # T_env_norm
        state[23] = float(np.clip(heal_eff * 10.0, 0, 1))        # heal_eff

        # [24:32] Process parameters (normalized)
        state[24:32] = self._encode_params()

        # [32:48] DI history (last 4 values)
        state[32:36] = self.history_DI

        # [36:40] phi_top history
        state[36:40] = self.history_phi_top

        # [40:44] T_surface[top] history (normalized)
        state[40:44] = [v / 80.0 for v in self.history_T_surf]

        # [44:48] Healing event history (1 if healing occurred)
        state[44:48] = self.history_heal

        # [48:64] spare — zero-padded (for future extensions)
        state[48:64] = 0.0

        return state

    def _encode_params(self):
        """Encode current process parameters to [0,1] normalized vector (8D)."""
        p = self.current_params
        return np.array([
            (p['T_fill']    - 348.15) / 8.0,                    # 75-83°C
            (p['T_reheat']  - 353.15) / 40.0,                   # 80-120°C
            p['h_cool']     / 25.0,                              # 0-25
            p['h_hot']      / 70.0,                              # 0-70
            p['t_pulse_start']    / 300.0,                       # 0-300s
            p['t_pulse_dur']      / 120.0,                       # 0-120s
            p['t_pulse_interval'] / 600.0,                       # 0-600s
            p['n_pulses']   / 10.0,                              # 0-10
        ], dtype=np.float32)

    def step(self, action):
        """
        Execute one environment step.
        action: normalized action from DDPG agent (8D, [-1,1])

        Fixes vs original:
          - DI varies with parameters (physics-proxy-first)
          - Reward normalized to [-10, +10] range (not -850)
          - Sink/heal indicators use temperature + Biot, not phi
          - done condition extended: max 150 steps or 30 min
        """
        self.current_params = self._agent_action(action)
        self.t_current += self.DT_STEP
        self.step_count += 1

        fields  = self._query_pinn(self.t_current, self.current_params)
        DI_now  = self._compute_DI(fields)
        in_pulse= self._in_pulse()

        Ts_top  = float(fields['T_surf_K'][-1]) - 273.15   # °C, top surface
        Tc_top  = float(fields['T_core_K'][-1]) - 273.15
        fl_top  = float(fields['fl_surf'][-1])
        fl_mean = float(np.mean(np.concatenate([fields['fl_surf'],
                                                 fields['fl_core']])))
        t_min   = self.t_current / 60.0
        p       = self.current_params

        # ── Physics-based sink-mark indicator ─────────────────────────────
        # Sink marks form when: (a) top surface solidifies fast, AND
        # (b) bulk below is still liquid → can't support surface shrinkage
        # Proxy: large Biot × fraction already solid at top
        h_now   = p['h_hot'] if in_pulse else p['h_cool']
        Bi      = h_now * GEO.R / MATERIAL.k_thermal
        solid_top = max(0.0, 1.0 - fl_top)
        sink_indicator = float(np.clip(
            Bi * solid_top * (1.0 - fl_mean) * 0.5, 0.0, 1.0
        ))

        # ── Healing bonus ──────────────────────────────────────────────────
        # During pulse: measure if surface temperature rose toward solidus
        # This indicates effective surface re-softening
        heal_bonus = 0.0
        if in_pulse:
            T_rise_toward_sol = max(0.0, Ts_top - 55.0)   # above 55°C = helpful
            T_normalized_rise = float(np.clip(T_rise_toward_sol / 10.0, 0.0, 1.0))
            heal_bonus = T_normalized_rise * 0.8
            # Extra bonus if surface is near mushy zone during pulse (62-72°C)
            if 60.0 <= Ts_top <= 74.0:
                heal_bonus += 0.5
            self.healing_count += 1

        # ── DI improvement reward ──────────────────────────────────────────
        di_improvement = float(max(0.0, self.prev_DI - DI_now))
        di_improvement_bonus = di_improvement * 3.0

        # ── Time penalty ───────────────────────────────────────────────────
        time_penalty = float(max(0.0, (t_min - 25.0) / 5.0))

        # ── Energy cost ────────────────────────────────────────────────────
        energy_cost = float(p['h_hot'] / 70.0) * 0.1 if in_pulse else 0.0

        # ── Mushy zone traversal speed reward ─────────────────────────────
        # Reward slow, controlled transition through mushy zone (62-72°C)
        # Fast traversal (large Bi) → cracks. Slow = gentle.
        mushy_control = 0.0
        T_avg = (Ts_top + Tc_top) / 2.0
        if 62.0 <= T_avg <= 72.0:
            # In mushy zone: reward low Biot (controlled cooling)
            bi_risk = float(np.clip((Bi - 0.2) / 0.8, 0.0, 1.0))
            mushy_control = float(np.clip((1.0 - bi_risk) * 0.5, 0.0, 0.5))

        # ── Normalized reward  ──────────────────────────────────────────────
        # Scale: DI ∈ [0,1] → reward ∈ [-4, 0] for DI term alone
        # Total range approximately [-6, +3] per step
        reward = (
            - 4.0 * DI_now              # primary: minimize damage
            - 1.0 * sink_indicator      # minimize sink marks
            + 2.0 * heal_bonus          # reward effective heating
            + 1.5 * di_improvement_bonus # reward DI reduction
            + 1.0 * mushy_control        # reward gentle mushy traversal
            - 0.5 * time_penalty         # mild time pressure
            - 0.1 * energy_cost          # mild energy cost
        )

        # Done condition
        done = (self.step_count >= self.N_STEPS or self.t_current >= T_CYCLE)

        # Terminal bonuses/penalties
        if done:
            total_time_min = p['t_cold_start'] / 60.0 + 3.0
            if DI_now < self.DI_TARGET:                  # DI < 0.10
                reward += self.W_SUCCESS                  # +10.0
            elif DI_now < self.DI_SAFE:                  # DI < 0.25
                reward += self.W_SUCCESS * 0.5            # +5.0
            if total_time_min > 30.0:                    # time exceeded
                reward -= 3.0

        # Update history
        self.history_DI      = self.history_DI[1:]      + [DI_now]
        self.history_phi_top = self.history_phi_top[1:]  + [
            float(fields['phi_surf'][-1])]
        self.history_T_surf  = self.history_T_surf[1:]   + [Ts_top]
        self.history_heal    = self.history_heal[1:]     + [
            1.0 if (in_pulse and heal_bonus > 0.3) else 0.0]
        self.prev_phi_top = float(fields['phi_surf'][-1])
        self.prev_DI      = DI_now

        next_state = self._get_state()

        info = {
            'DI': DI_now, 'phi_max': float(np.max(np.concatenate(
                [fields['phi_surf'], fields['phi_core']]))),
            'phi_top': float(fields['phi_surf'][-1]),
            'T_surf_top_C': Ts_top, 'fl_top': fl_top,
            'sink_indicator': sink_indicator,
            'heal_bonus': heal_bonus,
            'in_pulse': in_pulse,
            'heal_events': self.healing_count,
            't_min': t_min, 'reward': reward,
            'Bi': Bi,
        }
        return next_state, reward, done, info

    def _agent_action(self, action_raw):
        """Apply normalized action to current params with physics constraints."""
        delta = np.array(action_raw, dtype=np.float32) * ACTION_SCALE_ARRAY
        p     = dict(self.current_params)

        BOUNDS = DDPGAgentV4.PARAM_BOUNDS
        p['T_reheat']         = float(np.clip(p['T_reheat'] + delta[0],          *BOUNDS['T_reheat']))
        p['h_hot']            = float(np.clip(p['h_hot'] + delta[1],             *BOUNDS['h_hot']))
        p['t_pulse_start']    = float(np.clip(p['t_pulse_start'] + delta[2],     *BOUNDS['t_pulse_start']))
        p['t_pulse_dur']      = float(np.clip(p['t_pulse_dur'] + delta[3],       *BOUNDS['t_pulse_dur']))
        p['t_pulse_interval'] = float(np.clip(p['t_pulse_interval'] + delta[4],  *BOUNDS['t_pulse_interval']))
        p['n_pulses']         = float(np.clip(round(p['n_pulses'] + delta[5]),   *BOUNDS['n_pulses']))
        p['T_cool']           = float(np.clip(p['T_cool'] + delta[6],            *BOUNDS['T_cool']))
        p['t_cold_start']     = float(np.clip(p['t_cold_start'] + delta[7],      *BOUNDS['t_cold_start']))

        # Physics constraint: cold start after last pulse
        last_end = (p['t_pulse_start']
                    + (p['n_pulses'] - 1) * p['t_pulse_interval']
                    + p['t_pulse_dur'])
        p['t_cold_start'] = max(p['t_cold_start'], last_end + 60.0)
        p['t_cold_start'] = min(p['t_cold_start'], 1500.0)
        return p


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_risk_level(DI):
    if DI < 0.10:  return "SAFE",     "#2ecc71"
    if DI < 0.25:  return "CAUTION",  "#f1c40f"
    if DI < 0.50:  return "WARNING",  "#e67e22"
    return               "CRITICAL", "#e74c3c"


class TrainingMetrics:
    """Tracks and plots DRL training progress."""
    def __init__(self):
        self.episode_rewards   = []
        self.episode_DI_final  = []
        self.episode_DI_min    = []
        self.episode_DI_max    = []
        self.success_rates     = []   # rolling 100-ep window
        self.critic_losses     = []
        self.actor_losses      = []
        self.healing_counts    = []

    def record_episode(self, total_reward, DI_final, DI_min, DI_max,
                       healing_count, critic_loss=None, actor_loss=None):
        self.episode_rewards.append(total_reward)
        self.episode_DI_final.append(DI_final)
        self.episode_DI_min.append(DI_min)
        self.episode_DI_max.append(DI_max)
        self.healing_counts.append(healing_count)

        # Rolling success rate (DI_final < 0.25)
        window = min(100, len(self.episode_DI_final))
        recent = self.episode_DI_final[-window:]
        self.success_rates.append(
            sum(1 for d in recent if d < 0.25) / window * 100.0
        )
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)

    def plot(self, save_dir, episode):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(f"DRL v4 Training — Episode {episode} "
                     f"| Pulsed Hot-Air Reheating Process",
                     fontsize=12, fontweight='bold')

        ep = np.arange(len(self.episode_rewards))
        smooth = lambda x, w=50: (
            np.convolve(x, np.ones(w)/w, mode='valid') if len(x) >= w else x
        )

        ax = axes[0, 0]
        ax.plot(ep, self.episode_rewards, alpha=0.3, color='royalblue')
        rw_s = smooth(self.episode_rewards)
        ax.plot(np.arange(len(rw_s)), rw_s, color='royalblue', lw=2,
                label='100-ep smooth')
        ax.axhline(0, color='red', ls='--', lw=1, label='Zero baseline')
        ax.set_title('Episode Reward'); ax.set_xlabel('Episode')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.fill_between(ep, self.episode_DI_min, self.episode_DI_max,
                        alpha=0.25, color='salmon', label='DI range')
        ax.plot(ep, self.episode_DI_final, alpha=0.4, color='red', lw=1)
        di_s = smooth(self.episode_DI_final)
        ax.plot(np.arange(len(di_s)), di_s, color='darkred', lw=2,
                label='Final DI (smooth)')
        ax.axhline(0.10, color='green',  ls='--', lw=1.5, label='Target (0.10)')
        ax.axhline(0.25, color='orange', ls='--', lw=1,   label='CAUTION')
        ax.axhline(0.50, color='red',    ls='--', lw=1,   label='WARNING')
        ax.set_ylim(0, 1); ax.set_title('Damage Indicator (DI)')
        ax.set_xlabel('Episode'); ax.legend(fontsize=7)

        ax = axes[0, 2]
        ax.plot(ep, self.success_rates, color='green', lw=2)
        ax.fill_between(ep, 0, self.success_rates, alpha=0.2, color='green')
        ax.axhline(95, color='green', ls='--', lw=1, label='Target 95%')
        ax.set_ylim(0, 105); ax.set_title('Success Rate (DI<0.25, 100-ep window)')
        ax.set_xlabel('Episode'); ax.legend(fontsize=8)

        ax = axes[1, 0]
        if self.critic_losses:
            ax.semilogy(self.critic_losses, alpha=0.7, color='steelblue')
        ax.set_title('Critic Loss'); ax.set_xlabel('Update Step')

        ax = axes[1, 1]
        if self.actor_losses:
            ax.plot(self.actor_losses, alpha=0.7, color='coral')
        ax.set_title('Actor Loss'); ax.set_xlabel('Update Step')

        ax = axes[1, 2]
        ax.plot(ep, self.healing_counts, alpha=0.6, color='purple', lw=1)
        hc_s = smooth(self.healing_counts)
        ax.plot(np.arange(len(hc_s)), hc_s, color='purple', lw=2,
                label='Healing events (smooth)')
        ax.set_title('Hot-Air Healing Events / Episode')
        ax.set_xlabel('Episode'); ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/drl_training_v4_ep{episode}.png", dpi=110)
        plt.close()


# ============================================================================
# CURRICULUM LEARNING SCHEDULER
# ============================================================================

class CurriculumSchedulerV4:
    """
    Progressive difficulty increase for DRL training.

    Stage 1 (ep 0-500):    Easy — moderate Bi, good T_reheat range
    Stage 2 (ep 500-1500): Medium — wider param range
    Stage 3 (ep 1500-3500): Hard — full param range including bad inits
    Stage 4 (ep 3500+):    Random — fully stochastic initial conditions
    """
    def __init__(self):
        self.stage = 1

    def get_initial_params(self, episode):
        if episode < 500:
            # Stage 1: Start close to production baseline
            return {
                'T_fill':          (np.random.uniform(74, 79)) + 273.15,
                'T_cool':          (22.0) + 273.15,
                'T_reheat':        (np.random.uniform(90, 105)) + 273.15,
                'T_cold':          (16.0) + 273.15,
                'h_cool':          np.random.uniform(6, 10),
                'h_hot':           np.random.uniform(35, 50),
                'h_cold':          20.0,
                't_pulse_start':   np.random.uniform(25, 50),
                't_pulse_dur':     np.random.uniform(18, 25),
                't_pulse_interval':np.random.uniform(70, 120),
                'n_pulses':        float(np.random.randint(4, 7)),
                't_cold_start':    np.random.uniform(1150, 1350),
                'T_container':     (23.0) + 273.15,
                'viscosity':       0.3, 'humidity': 0.4, 'Gc': 80.0,
            }
        elif episode < 1500:
            # Stage 2: Wider range
            return {
                'T_fill':          (np.random.uniform(73, 83)) + 273.15,
                'T_cool':          (np.random.uniform(20, 25)) + 273.15,
                'T_reheat':        (np.random.uniform(80, 110)) + 273.15,
                'T_cold':          (np.random.uniform(14, 19)) + 273.15,
                'h_cool':          np.random.uniform(5, 14),
                'h_hot':           np.random.uniform(25, 60),
                'h_cold':          np.random.uniform(15, 25),
                't_pulse_start':   np.random.uniform(15, 80),
                't_pulse_dur':     np.random.uniform(15, 45),
                't_pulse_interval':np.random.uniform(60, 200),
                'n_pulses':        float(np.random.randint(3, 8)),
                't_cold_start':    np.random.uniform(1000, 1450),
                'T_container':     (np.random.uniform(21, 25)) + 273.15,
                'viscosity':       np.random.uniform(0.2, 0.5),
                'humidity':        np.random.uniform(0.3, 0.6),
                'Gc':              np.random.uniform(60, 120),
            }
        else:
            # Stage 3 & 4: Full random including bad conditions
            return {
                'T_fill':          (np.random.uniform(72, 84)) + 273.15,
                'T_cool':          (np.random.uniform(18, 27)) + 273.15,
                'T_reheat':        (np.random.uniform(75, 120)) + 273.15,
                'T_cold':          (np.random.uniform(12, 22)) + 273.15,
                'h_cool':          np.random.uniform(3, 18),
                'h_hot':           np.random.uniform(20, 70),
                'h_cold':          np.random.uniform(10, 30),
                't_pulse_start':   np.random.uniform(10, 120),
                't_pulse_dur':     np.random.uniform(10, 90),
                't_pulse_interval':np.random.uniform(30, 400),
                'n_pulses':        float(np.random.randint(1, 10)),
                't_cold_start':    np.random.uniform(800, 1500),
                'T_container':     (np.random.uniform(20, 26)) + 273.15,
                'viscosity':       np.random.uniform(0.1, 0.8),
                'humidity':        np.random.uniform(0.2, 0.7),
                'Gc':              np.random.uniform(50, 140),
            }

        self.stage = (1 if episode < 500 else 2 if episode < 1500 else 3)


# ============================================================================
# REFERENCE BASELINES
# ============================================================================

BASELINES = {
    'production_observed': {
        # From CLIENT_Data_0427.xlsx production example
        'T_fill':          75.0 + 273.15,
        'T_reheat':       100.0 + 273.15,
        'h_cool':           8.0,
        'h_hot':           45.0,
        't_pulse_start':   34.0,   # sec (~0.57 min)
        't_pulse_dur':     20.0,   # sec
        't_pulse_interval':88.0,   # sec average
        'n_pulses':         5.0,
        't_cold_start':  1278.0,   # sec (21.3 min)
        'T_cool':         21.0 + 273.15,
        'T_cold':         16.0 + 273.15,
        'h_cold':         20.0,
        'T_container':    23.0 + 273.15,
        'viscosity':       0.3, 'humidity': 0.4, 'Gc': 80.0,
    },
    'client_best_trial1': {
        # Trial ①: 60°C insulation cap, 40 min total
        'T_fill':          82.0 + 273.15,
        'T_reheat':        60.0 + 273.15,   # ambient temp = 60°C (no pulse)
        'h_cool':           4.0,
        'h_hot':            4.0,
        't_pulse_start':  900.0,
        't_pulse_dur':      1.0,
        't_pulse_interval':9999.0,
        'n_pulses':         0.0,   # no pulses
        't_cold_start':  2400.0,   # 40 min
        'T_cool':         60.0 + 273.15,
        'T_cold':         23.0 + 273.15,
        'h_cold':          4.0,
        'T_container':    23.0 + 273.15,
        'viscosity':       0.3, 'humidity': 0.4, 'Gc': 80.0,
    },
    'bad_blast_cool': {
        # RT blast (worst case — Bi≈0.9)
        'T_fill':          82.0 + 273.15,
        'T_reheat':        23.0 + 273.15,
        'h_cool':          18.0,
        'h_hot':           18.0,
        't_pulse_start':    0.0,
        't_pulse_dur':      1.0,
        't_pulse_interval':9999.0,
        'n_pulses':         0.0,
        't_cold_start':  1200.0,
        'T_cool':         23.0 + 273.15,
        'T_cold':         16.0 + 273.15,
        'h_cold':         20.0,
        'T_container':    23.0 + 273.15,
        'viscosity':       0.3, 'humidity': 0.4, 'Gc': 80.0,
    },
}


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_drl_v4(pinn_checkpoint='pinn_outputs_v4/pinn_v4_best.pth',
                 n_episodes=5_000,
                 save_dir='drl_outputs_v4',
                 warmup_steps=2_000,
                 update_every=2,
                 n_updates=2):
    """
    Train DDPG agent v4 on the pulsed hot-air reheating environment.

    Args:
        pinn_checkpoint: path to trained PINN v4 model
        n_episodes: total training episodes
        save_dir: output directory
        warmup_steps: random exploration before training starts
        update_every: update networks every N environment steps
        n_updates: number of gradient updates per call

    Outputs:
        drl_outputs_v4/agent_best.pth     — best agent checkpoint
        drl_outputs_v4/agent_final.pth    — final agent checkpoint
        drl_outputs_v4/training_curves.png
        drl_outputs_v4/optimal_params.json
    """
    Path(save_dir).mkdir(exist_ok=True)

    # ── Load PINN ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DRL v4 Training — Pulsed Hot-Air Reheating")
    print("  CBIC × TUAT | Julian Evan Chrisnanto | 2026")
    print(f"{'='*60}")

    pinn = MultiPhysicsPINNV4().to(DEVICE)
    if Path(pinn_checkpoint).exists():
        ck = torch.load(pinn_checkpoint, map_location=DEVICE, weights_only=False)
        state = ck.get('model_state_dict', ck)
        pinn.load_state_dict(state)
        print(f"  PINN loaded from {pinn_checkpoint}")
    else:
        print(f"  WARNING: PINN checkpoint not found at {pinn_checkpoint}")
        print(f"  Using randomly initialized PINN — results will be less accurate.")
        print(f"  Run train_pinn_physics_enforced_v4.py first for best results.")
    pinn.eval()

    # ── Initialize ────────────────────────────────────────────────────────
    env       = PhysicsEnvV4(pinn)
    agent     = DDPGAgentV4()
    metrics   = TrainingMetrics()
    curriculum= CurriculumSchedulerV4()

    best_DI   = float('inf')
    best_ep   = 0
    total_steps = 0

    print(f"\n  Episodes: {n_episodes}")
    print(f"  DDPG state_dim={agent.state_dim}, action_dim={agent.action_dim}")
    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"  Warmup: {warmup_steps} random steps")
    print(f"{'='*60}\n")

    # ── Compute baselines ─────────────────────────────────────────────────
    print("Computing reference baselines...")
    baseline_results = {}
    for name, bp in BASELINES.items():
        env_bl = PhysicsEnvV4(pinn)
        env_bl.reset(params=bp)
        di_vals = []
        for _ in range(env.N_STEPS):
            action = np.zeros(agent.action_dim)  # no adjustment
            _, _, done, info = env_bl.step(action)
            di_vals.append(info['DI'])
            if done:
                break
        baseline_results[name] = {
            'DI_final': di_vals[-1],
            'DI_min':   min(di_vals),
            'DI_max':   max(di_vals),
            'risk':     compute_risk_level(di_vals[-1])[0],
        }
        print(f"  {name:30s}: DI_final={di_vals[-1]:.3f} [{baseline_results[name]['risk']}]")

    print()
    # Save baseline results
    with open(f"{save_dir}/baselines.json", 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────────
    for ep in range(n_episodes):
        init_params = curriculum.get_initial_params(ep)
        state = env.reset(params=init_params)

        ep_reward   = 0.0
        ep_DI_vals  = []
        ep_steps    = 0
        critic_losses = []
        actor_losses  = []

        while True:
            # Random warmup or agent action
            if total_steps < warmup_steps:
                action = np.random.uniform(-1, 1, agent.action_dim)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_DI_vals.append(info['DI'])
            total_steps += 1

            # Store experience
            agent.replay.push(
                state, action, reward, next_state, float(done),
                di_val=info['DI']
            )

            state = next_state

            # Update networks
            if (total_steps >= warmup_steps
                    and total_steps % update_every == 0):
                for _ in range(n_updates):
                    upd = agent.update()
                    if upd is not None:
                        critic_losses.append(upd['critic_loss'])
                        actor_losses.append(upd['actor_loss'])

            ep_steps += 1
            if done:
                break

        # ── Episode post-processing ────────────────────────────────────────
        DI_final = ep_DI_vals[-1] if ep_DI_vals else 1.0
        DI_min   = min(ep_DI_vals) if ep_DI_vals else 1.0
        DI_max   = max(ep_DI_vals) if ep_DI_vals else 1.0
        c_loss   = np.mean(critic_losses) if critic_losses else 0.0
        a_loss   = np.mean(actor_losses)  if actor_losses  else 0.0

        metrics.record_episode(
            ep_reward, DI_final, DI_min, DI_max,
            env.healing_count, c_loss, a_loss
        )

        # Save best model (from ep > 10 instead of ep > 100)
        if DI_final < best_DI and ep > 10:
            best_DI = DI_final
            best_ep = ep
            agent.save(f"{save_dir}/agent_best.pth")

            # Save optimal params
            risk, _ = compute_risk_level(DI_final)
            total_t_min = env.current_params.get('t_cold_start', 1278) / 60.0 + 3.0
            opt_params  = {k: float(v) for k, v in env.current_params.items()
                           if isinstance(v, (int, float))}
            opt_params['_DI_final']    = float(DI_final)
            opt_params['_DI_min']      = float(DI_min)
            opt_params['_risk']        = risk
            opt_params['_episode']     = ep
            opt_params['_total_min']   = float(total_t_min)
            opt_params['_heal_events'] = env.healing_count
            with open(f"{save_dir}/optimal_params.json", 'w') as f:
                json.dump(opt_params, f, indent=2)

        # Logging
        if ep % 50 == 0 or ep < 20:
            risk, _  = compute_risk_level(DI_final)
            succ_r   = metrics.success_rates[-1] if metrics.success_rates else 0.0
            bi_now   = env.current_params['h_cool'] * GEO.R / MATERIAL.k_thermal
            print(f"Ep {ep:5d}/{n_episodes}  "
                  f"R={ep_reward:7.1f}  "
                  f"DI={DI_final:.3f}[{DI_min:.2f}-{DI_max:.2f}] [{risk:8s}]  "
                  f"succ={succ_r:5.1f}%  "
                  f"Bi={bi_now:.2f}  "
                  f"noise={agent.noise_scale:.3f}  "
                  f"heal={env.healing_count}  "
                  f"best={best_DI:.3f}@{best_ep}")

        # Periodic checkpoints and plots
        if ep % 500 == 0 and ep > 0:
            agent.save(f"{save_dir}/agent_ep{ep}.pth")
            metrics.plot(save_dir, ep)

    # ── Final saves ───────────────────────────────────────────────────────
    agent.save(f"{save_dir}/agent_final.pth")
    metrics.plot(save_dir, n_episodes)

    # ── Final summary ─────────────────────────────────────────────────────
    risk, _ = compute_risk_level(best_DI)
    print(f"\n{'='*60}")
    print(f"  DRL v4 Training Complete")
    print(f"  Best DI: {best_DI:.4f} [{risk}] at episode {best_ep}")
    print(f"  Final success rate: {metrics.success_rates[-1]:.1f}%")
    print(f"  Optimal params saved: {save_dir}/optimal_params.json")
    print(f"{'='*60}")

    # Load and display optimal params
    if Path(f"{save_dir}/optimal_params.json").exists():
        with open(f"{save_dir}/optimal_params.json") as f:
            opt = json.load(f)
        print("\n  DRL-Optimal Process Parameters:")
        print(f"    T_reheat:          {opt.get('T_reheat',0)-273.15:.1f} °C")
        print(f"    h_hot:             {opt.get('h_hot',0):.1f} W/m²K")
        print(f"    t_pulse_start:     {opt.get('t_pulse_start',0)/60:.2f} min "
              f"({opt.get('t_pulse_start',0):.0f} sec)")
        print(f"    t_pulse_dur:       {opt.get('t_pulse_dur',0):.0f} sec")
        print(f"    t_pulse_interval:  {opt.get('t_pulse_interval',0):.0f} sec")
        print(f"    n_pulses:          {opt.get('n_pulses',0):.0f}")
        print(f"    t_cold_start:      {opt.get('t_cold_start',0)/60:.1f} min")
        print(f"    Peak DI:           {opt.get('_DI_final',1):.4f} [{opt.get('_risk','?')}]")
        print(f"    Total time:        {opt.get('_total_min',0):.1f} min")
        print(f"    Healing events:    {opt.get('_heal_events',0)}")
        print(f"    Within 30-min:     {'✓' if opt.get('_total_min',99)<30 else '✗'}")
    return agent, metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    agent, metrics = train_drl_v4(
        pinn_checkpoint='pinn_outputs_v4/pinn_v4_best.pth',
        n_episodes=5_000,
        save_dir='drl_outputs_v4',
    )
    print("\nDRL v4 training complete.")
