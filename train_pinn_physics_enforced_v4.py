"""
train_pinn_physics_enforced_v4.py

MS-PINN v4 — Pulsed Hot-Air Reheating for Sink-Mark Elimination
CBIC / TUAT — Julian Evan Chrisnanto

═══════════════════════════════════════════════════════════════════════
KEY CHANGES FROM v3 (based on CLIENT_Data_0427.xlsx, 2026-04-27)
═══════════════════════════════════════════════════════════════════════

NEW PROCESS UNDERSTANDING (from xlsx data):
  The actual production line does NOT cool to 37°C first then reheat.
  Instead, it applies short hot-air pulses (≈20 sec, 100°C surface)
  EARLY in the process (t=0.57, 1.13, 2.90, 4.03, 5.83 min) while
  the bulk is STILL IN THE MUSHY ZONE (T_bulk=73→59°C). This is a
  mushy-zone surface treatment, not post-solidification healing.
  Cold air (16°C) is applied from t≈21.3 min for final cooling.
  Total process: ~24.5 min ← within the 30-min production target.

  Two verified strategies observed:
    ① Verification Trial 1: Insulation cap near top (59-63°C ambient),
      slow room-temp cool. Best quality but 40 min total — too slow.
    ② Verification Trial 2: Room-temp cool + periodic 90°C hot-air
      1-min pulses at t=10,14.5,19 min. Post-solidification reheat.
      Surface temp rises 44→49°C (BELOW solidus 62°C) — surface
      softening rather than true re-melting.
    ③ Production line: Back-fill, 5× 20-sec 100°C hot-air pulses
      early (t<6 min), final cold air 16°C at t≈21 min. ≈24.5 min.

NEW PHYSICS IN v4:
  A. Pulsed boundary condition:
     h_conv(t) and T_env(t) switch between cooling and heating values
     at defined pulse intervals. Robin BC changes sign of heat flux.
  
  B. Phase-field healing term (reversible φ):
     Standard AT2 model is irreversible (φ monotone increasing).
     v4 adds: ∂φ/∂t += -M_heal × ReLU(T − T_solidus) × φ × g_heal(φ)
     where g_heal(φ) = 4φ(1−φ) ensures healing only where 0<φ<1.
     Physical meaning: when T > T_solidus (semi-liquid), surface
     tension + liquid flow closes crack openings → φ decreases.
  
  C. Data-informed training (client measurements):
     L_data uses 10 supervision points from experimental temperature
     profiles (both trials), ensuring PINN matches observed dynamics.
  
  D. Sink-mark indicator (volume deficit at top):
     L_sink penalizes large shrinkage at z=H during mushy→solid
     transition. Based on dilatometry: ~8% volume contraction.

NEW NETWORKS:
  ThermalNetworkV4:    12 inputs (adds T_reheat, h_conv_reheat,
                        T_cold, t_pulse_start, t_pulse_dur,
                        t_pulse_interval, n_pulses)
  PhaseFieldNetworkV4:  9 inputs (adds heal_driver = ReLU(T−T_sol))
  MechanicalNetworkV4:  8 inputs (unchanged from v3)

TRAINING SCHEDULE:
  Phase 1 (0–20k):   Thermal only + data matching
  Phase 2 (20k–45k): Coupled thermal + phase field + healing
  Phase 3 (45k–60k): Full multi-physics + sink-mark loss

Material:  LCWT401 (cyclopentasiloxane + stearyl alcohol + PE wax)
Geometry:  R=1.25 cm, H=4.0 cm (unchanged)
Device:    CUDA if available, else CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[v4] Device: {DEVICE}")
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# GEOMETRY
# ============================================================================

class Geometry:
    R = 0.0125   # m  — 1.25 cm radius
    H = 0.04     # m  — 4.0 cm height

GEO = Geometry()
T_CYCLE = 1800.0          # s  — 30 min max
T_ROOM  = 273.15 + 23.0   # K

# ============================================================================
# NORMALIZATION SCALES (recalibrated from v3)
# ============================================================================

class NormalizationScalesV4:
    L_ref      = GEO.R
    T_ref      = 273.15 + 23.0   # K  baseline room temp
    T_scale    = 80.0             # K  spans room→fill
    t_ref      = T_CYCLE          # 1800 s
    sigma_ref  = 5.0e5            # Pa
    Gc_ref     = 200.0            # J/m²
    h_ref      = 25.0             # W/m²K  (cooling baseline)
    h_hot_ref  = 60.0             # W/m²K  (hot-air baseline — higher)
    l0         = 0.0005           # m  phase-field length scale
    k_stab     = 1e-6             # phase-field stability
    alpha_vol  = 0.08             # volume contraction fraction in mushy→solid

NORM = NormalizationScalesV4()

# ============================================================================
# MATERIAL PROPERTIES — LCWT401 (verified against CBIC xlsx 2026-03-26)
# ============================================================================

class MaterialPropertiesV4:
    # Thermal transition (from melting range 62–72°C)
    T_solidus    = 62.0 + 273.15   # 335.15 K
    T_liquidus   = 72.0 + 273.15   # 345.15 K
    T_midpoint   = (T_solidus + T_liquidus) / 2.0

    # Thermal properties
    rho          = 990.0           # kg/m³  (from 0.99 g/cm³)
    cp           = 2000.0          # J/kg/K
    k_thermal    = 0.25            # W/m/K
    L_latent     = 150_000.0       # J/kg

    # Mechanical
    alpha_th     = 8e-5            # 1/K  thermal expansion
    E_base       = 1.37e6          # Pa   (from compression test)
    nu           = 0.40            # —    (viscoelastic wax)
    sigma_y_sol  = 3.5e5           # Pa   (from xlsx)

    # Fracture
    Gc_base      = 80.0            # J/m² (soft wax)
    l0           = NORM.l0
    k_stab       = NORM.k_stab

    # Healing
    M_heal       = 0.3             # healing rate constant (1/K·s normalized)

    T_ROOM = T_ROOM

    def liquid_fraction(self, T):
        liq  = T >= self.T_liquidus
        sol  = T <= self.T_solidus
        mush = ~liq & ~sol
        fl   = torch.zeros_like(T)
        fl[liq] = 1.0
        T_m = T[mush]
        fl[mush] = 0.5 * (1.0 + torch.sin(
            torch.pi * (T_m - self.T_midpoint) /
            (self.T_liquidus - self.T_solidus)
        ))
        return fl

    def yield_strength(self, T, fl):
        pf  = torch.clamp((1.0 - fl)**2.5, min=0.01)
        tf  = torch.clamp(1.0 - 0.008*(T - self.T_solidus), min=0.01, max=1.0)
        return self.sigma_y_sol * pf * tf

    def effective_Gc(self, Gc_input, humidity_norm):
        return Gc_input * (1.0 - 0.3 * torch.clamp(humidity_norm, 0.0, 1.0))

    def phase_degraded_modulus(self, phi):
        return (1.0 - self.k_stab) * (1.0 - phi)**2 + self.k_stab

    def healing_driver(self, T):
        """Driving force for crack healing — activated above solidus."""
        return F.relu(T - self.T_solidus) / 10.0  # normalized [K/10]

MATERIAL = MaterialPropertiesV4()

# ============================================================================
# PULSED PROCESS BOUNDARY CONDITION (the new physics core of v4)
# ============================================================================

class PulsedProcessV4:
    """
    Computes T_env(t) and h_conv(t) for the production pulsed process.

    Timeline (from CLIENT_Data_0427.xlsx production example):
      Phase 0: Cooling, T_env=T_cool (21°C), h_conv=h_cool
      Phase 1: Hot-air pulses (n_pulses times):
               T_env=T_reheat (≈100°C surface), h_conv=h_hot (≈50 W/m²K)
               pulse duration ≈ 20 sec, interval controlled by DRL
      Phase 2: Continued cooling at T_cool
      Phase 3: Final cold-air cooling, T_env=T_cold (16°C), h_conv=h_cold
    """

    @staticmethod
    def _to_field(val, ref):
        """
        Ensure val broadcasts correctly against ref (1-D tensor of size N).
        Handles: Python scalar, 0-D tensor, 1-D tensor of size N.
        Returns a 1-D tensor of size N on the same device as ref.
        """
        if not torch.is_tensor(val):
            return torch.full_like(ref, float(val))
        if val.dim() == 0 or val.numel() == 1:
            return torch.full_like(ref, val.item())
        # Already shape [N] — detach from any gradient tape so
        # comparison ops stay in-place without graph issues
        return val.to(ref.device)

    def compute_T_env(self, t_s, T_fill, T_cool, T_reheat, T_cold,
                      t_pulse_start_s, t_pulse_dur_s, t_pulse_interval_s,
                      n_pulses, t_cold_start_s):
        """
        t_s: 1-D time tensor [N] in seconds.
        All other args may be Python scalars OR 1-D tensors of size N.
        Returns T_env(t) as 1-D tensor [N].
        """
        f = self._to_field      # alias

        T_cool_f    = f(T_cool,    t_s)
        T_reheat_f  = f(T_reheat,  t_s)
        T_cold_f    = f(T_cold,    t_s)
        tps         = f(t_pulse_start_s,    t_s)
        tpd         = f(t_pulse_dur_s,      t_s)
        tpi         = f(t_pulse_interval_s, t_s)
        tcs         = f(t_cold_start_s,     t_s)

        # Max pulses across batch (scalar) — safe for loop bound
        n = int(n_pulses.max().item()) if torch.is_tensor(n_pulses) \
            else int(n_pulses)
        n = max(n, 0)

        T_e = T_cool_f.clone()

        for i in range(n):
            pulse_on  = tps + i * tpi
            pulse_off = pulse_on + tpd
            mask = (t_s >= pulse_on) & (t_s < pulse_off)
            T_e = torch.where(mask, T_reheat_f, T_e)

        cold_mask = t_s >= tcs
        T_e = torch.where(cold_mask, T_cold_f, T_e)
        return T_e

    def compute_h_conv(self, t_s, h_cool, h_hot, h_cold,
                       t_pulse_start_s, t_pulse_dur_s, t_pulse_interval_s,
                       n_pulses, t_cold_start_s):
        """Returns h_conv(t) as 1-D tensor [N]."""
        f = self._to_field

        h_cool_f = f(h_cool, t_s)
        h_hot_f  = f(h_hot,  t_s)
        h_cold_f = f(h_cold, t_s)
        tps      = f(t_pulse_start_s,    t_s)
        tpd      = f(t_pulse_dur_s,      t_s)
        tpi      = f(t_pulse_interval_s, t_s)
        tcs      = f(t_cold_start_s,     t_s)

        n = int(n_pulses.max().item()) if torch.is_tensor(n_pulses) \
            else int(n_pulses)
        n = max(n, 0)

        h = h_cool_f.clone()

        for i in range(n):
            pulse_on  = tps + i * tpi
            pulse_off = pulse_on + tpd
            mask = (t_s >= pulse_on) & (t_s < pulse_off)
            h = torch.where(mask, h_hot_f, h)

        cold_mask = t_s >= tcs
        h = torch.where(cold_mask, h_cold_f, h)
        return h

PROCESS = PulsedProcessV4()

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class ThermalNetworkV4(nn.Module):
    """
    Predicts normalized temperature field T(r,z,t; process_params).

    Inputs (12, all normalized to [0,1] or [-1,1]):
      r_n, z_n, t_n:           spatial-temporal coordinates
      T_fill_n:                 fill temperature
      T_cool_n:                 cooling environment temp
      T_reheat_n:               hot-air temperature
      T_cold_n:                 cold-air final phase temp
      h_cool_n:                 cooling convection coefficient
      h_hot_n:                  hot-air convection coefficient
      t_pulse_start_n:          first pulse start time
      t_pulse_dur_n:            pulse duration
      t_pulse_interval_n:       between-pulse interval
    """
    def __init__(self, hidden=256, n_layers=7):
        super().__init__()
        layers = [nn.Linear(12, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, r_n, z_n, t_n, T_fill_n, T_cool_n, T_reheat_n,
                T_cold_n, h_cool_n, h_hot_n, t_ps_n, t_pd_n, t_pi_n):
        x = torch.cat([r_n, z_n, t_n, T_fill_n, T_cool_n, T_reheat_n,
                        T_cold_n, h_cool_n, h_hot_n, t_ps_n, t_pd_n, t_pi_n], dim=1)
        return self.net(x)


class PhaseFieldNetworkV4(nn.Module):
    """
    Predicts phase-field damage φ(r,z,t; params).
    Includes healing driver input so network learns healing physics.

    Inputs (9):
      r_n, z_n, t_n:     coordinates
      T_n:               normalized temperature
      f_l:               liquid fraction
      H_norm:            crack driving history
      Gc_norm:           fracture toughness
      viscosity_n:       viscosity
      heal_driver:       ReLU(T - T_solidus)/10 — healing activation
    """
    def __init__(self, hidden=256, n_layers=7):
        super().__init__()
        layers = [nn.Linear(9, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, r_n, z_n, t_n, T_n, f_l, H_norm, Gc_n, visc_n, heal_drv):
        x = torch.cat([r_n, z_n, t_n, T_n, f_l, H_norm, Gc_n, visc_n, heal_drv], dim=1)
        return self.net(x)


class MechanicalNetworkV4(nn.Module):
    """
    Predicts stress tensor components.
    Same architecture as v3 (6 outputs: σ_rr, σ_zz, σ_θθ, σ_rz, σ_rθ, σ_zθ).

    Inputs (8):
      r_n, z_n, t_n, T_n, f_l, phi, viscosity_n, humidity_n
    """
    def __init__(self, hidden=192, n_layers=6):
        super().__init__()
        layers = [nn.Linear(8, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*layers)

    def forward(self, r_n, z_n, t_n, T_n, f_l, phi, visc_n, humid_n):
        x = torch.cat([r_n, z_n, t_n, T_n, f_l, phi, visc_n, humid_n], dim=1)
        out = self.net(x)
        return {
            'sigma_rr':     0.5 * torch.tanh(out[:, 0:1]),
            'sigma_zz':     0.5 * torch.tanh(out[:, 1:2]),
            'sigma_theta':  0.5 * torch.tanh(out[:, 2:3]),
            'sigma_rz':     0.3 * torch.tanh(out[:, 3:4]),
            'sigma_rtheta': 0.2 * torch.tanh(out[:, 4:5]),
            'sigma_ztheta': 0.2 * torch.tanh(out[:, 5:6]),
        }


class MultiPhysicsPINNV4(nn.Module):
    """
    Complete multi-physics PINN v4 with:
    - Pulsed thermal BC
    - Phase-field crack formation AND healing
    - Mechanical stress
    - Data-informed training from CLIENT_Data_0427.xlsx
    """
    def __init__(self):
        super().__init__()
        self.thermal_net    = ThermalNetworkV4()
        self.phasefield_net = PhaseFieldNetworkV4()
        self.mechanical_net = MechanicalNetworkV4()
        self.mat  = MaterialPropertiesV4()
        self.norm = NormalizationScalesV4()

    def _broadcast(self, val, ref):
        n = ref.shape[0]
        if not torch.is_tensor(val):
            return torch.full((n, 1), float(val), dtype=ref.dtype, device=ref.device)
        if val.numel() == 1:
            return torch.full((n, 1), val.item(), dtype=ref.dtype, device=ref.device)
        if val.dim() == 1:
            return val.unsqueeze(1)
        return val

    def forward_thermal(self, r, z, t,
                        T_fill, T_cool, T_reheat, T_cold,
                        h_cool, h_hot,
                        t_pulse_start, t_pulse_dur, t_pulse_interval,
                        n_pulses, t_cold_start, T_container):
        """
        Correction-based temperature field.

        Instead of T = T_container + T_range * Sigmoid(net)  [broken: net=0.5 → midpoint]
        We use: T = T_analytical(r,z,t) + delta_T * tanh(net)

        where T_analytical is a physics-based analytical baseline that:
          - Satisfies T(r,z,0) = T_fill  exactly at t=0
          - Decays toward T_env(t) with spatial variation
          - Captures pulse heating events approximately

        The network only needs to learn CORRECTIONS (±delta_T°C) to this
        baseline → fast convergence, proper IC satisfaction, data learning.
        """
        mat  = self.mat
        norm = self.norm

        r_n   = r        / GEO.R
        z_n   = z        / GEO.H
        t_n   = t        / norm.t_ref
        Tf_n  = (T_fill   - norm.T_ref) / norm.T_scale
        Tc_n  = (T_cool   - norm.T_ref) / norm.T_scale
        Tr_n  = (T_reheat - norm.T_ref) / norm.T_scale
        Tcd_n = (T_cold   - norm.T_ref) / norm.T_scale
        hc_n  = h_cool  / norm.h_ref
        hh_n  = h_hot   / norm.h_hot_ref
        tps_n = t_pulse_start    / norm.t_ref
        tpd_n = t_pulse_dur      / norm.t_ref
        tpi_n = t_pulse_interval / norm.t_ref

        # Network output: raw correction signal ∈ (-∞, +∞)
        # Use tanh on output layer → bounded [-1, 1]
        delta_raw = self.thermal_net(
            r_n, z_n, t_n, Tf_n, Tc_n, Tr_n, Tcd_n,
            hc_n, hh_n, tps_n, tpd_n, tpi_n
        )  # ∈ [0,1] due to Sigmoid — we remap: [-1,1] via 2*x-1
        delta_norm = 2.0 * delta_raw - 1.0   # ∈ [-1, 1]

        # ── Analytical baseline ────────────────────────────────────────────
        # Compute current environmental temperature
        t_s_1d = t.squeeze(1)
        np_1d  = n_pulses.squeeze(1)   if (torch.is_tensor(n_pulses)   and n_pulses.dim()   > 1) else n_pulses
        tcs_1d = t_cold_start.squeeze(1) if (torch.is_tensor(t_cold_start) and t_cold_start.dim() > 1) else t_cold_start

        T_env_now = PROCESS.compute_T_env(
            t_s_1d,
            T_fill.squeeze(1), T_cool.squeeze(1),
            T_reheat.squeeze(1), T_cold.squeeze(1),
            t_pulse_start.squeeze(1), t_pulse_dur.squeeze(1),
            t_pulse_interval.squeeze(1),
            np_1d, tcs_1d
        ).unsqueeze(1)

        # Time constant for cooling: based on geometry and convection
        Bi  = h_cool * GEO.R / mat.k_thermal   # Biot number
        tau = torch.clamp(
            GEO.R**2 * mat.rho * mat.cp / (mat.k_thermal * (Bi + 0.1) * 10.0),
            min=120.0, max=1200.0
        )

        # Spatial profile: top surface cools faster than bottom (z-dependent)
        # z_factor: 1 at z=H (top, most exposed), ~0.6 at z=0 (bottom, container)
        z_factor = 0.6 + 0.4 * (z / GEO.H)

        # Cooling factor: exponential toward T_env
        cool_factor = torch.exp(-t * z_factor / tau)

        # Analytical: T_env + (T_fill - T_env) * exp(-t/tau) * z_factor
        # At t=0: T_analytical = T_env + (T_fill - T_env) = T_fill ✓
        T_analytical = T_env_now + (T_fill - T_env_now) * cool_factor

        # ── Add network correction (±15°C max) ────────────────────────────
        delta_T_scale = 15.0   # Maximum correction in Kelvin
        T = T_analytical + delta_T_scale * delta_norm

        # Physical bounds: never below ambient-10, never above fill+5
        T_min = torch.minimum(T_env_now - 10.0, T_container - 5.0)
        T_max = T_fill + 5.0
        T = torch.clamp(T, T_min, T_max)

        fl = mat.liquid_fraction(T)

        # Return T_raw as the normalized temperature for downstream use
        # T_norm ∈ [0,1] where 0=T_container, 1=T_fill
        T_norm = torch.clamp((T - T_container) / (T_fill - T_container + 1e-6), 0.0, 1.0)

        return T, fl, T_norm, T_env_now

    def forward_phase_field(self, r, z, t, T_raw, fl, T, Gc, viscosity):
        mat  = self.mat
        norm = self.norm

        r_n     = r / GEO.R
        z_n     = z / GEO.H
        t_n     = t / norm.t_ref
        Gc_n    = Gc / norm.Gc_ref

        # Crack driving: peaks in mushy zone
        mushy_wt = torch.clamp(fl * (1.0 - fl) * 4.0, 0.0, 1.0)
        T_prox   = torch.clamp(
            1.0 - torch.abs(T - mat.T_midpoint) /
            (mat.T_liquidus - mat.T_solidus), 0.0, 1.0
        )
        H_norm = mushy_wt * T_prox * torch.clamp(t_n * 2.0, 0.0, 1.0)

        # Healing driver — activated above solidus (material re-softens)
        heal_drv = mat.healing_driver(T)

        phi = self.phasefield_net(
            r_n, z_n, t_n, T_raw, fl, H_norm, Gc_n, viscosity, heal_drv
        )
        return phi, H_norm, heal_drv

    def forward_mechanical(self, r, z, t, T_raw, fl, phi, viscosity, humidity):
        return self.mechanical_net(
            r / GEO.R, z / GEO.H, t / self.norm.t_ref,
            T_raw, fl, phi, viscosity, humidity
        )

    def predict_fields(self, r, z, t,
                       T_fill, T_cool, T_reheat, T_cold,
                       h_cool, h_hot,
                       t_pulse_start, t_pulse_dur, t_pulse_interval,
                       n_pulses, t_cold_start,
                       T_container, viscosity, humidity, Gc):
        """Main inference interface compatible with v4 Streamlit dashboard."""
        bc = self._broadcast
        (T_fill_t, T_cool_t, T_reheat_t, T_cold_t,
         h_cool_t, h_hot_t,
         tps_t, tpd_t, tpi_t,
         np_t, tcs_t,
         Tc_t, vi_t, hu_t, Gc_t) = [
            bc(v, r) for v in (
                T_fill, T_cool, T_reheat, T_cold,
                h_cool, h_hot,
                t_pulse_start, t_pulse_dur, t_pulse_interval,
                n_pulses, t_cold_start,
                T_container, viscosity, humidity, Gc
            )
        ]

        T, fl, T_raw, T_env = self.forward_thermal(
            r, z, t,
            T_fill_t, T_cool_t, T_reheat_t, T_cold_t,
            h_cool_t, h_hot_t, tps_t, tpd_t, tpi_t,
            np_t, tcs_t, Tc_t
        )
        phi, H_norm, heal_drv = self.forward_phase_field(
            r, z, t, T_raw, fl, T, Gc_t, vi_t
        )
        mech = self.forward_mechanical(r, z, t, T_raw, fl, phi, vi_t, hu_t)

        mat  = self.mat
        norm = self.norm

        Gc_eff  = mat.effective_Gc(Gc_t, hu_t)
        s_rr    = mech['sigma_rr']
        s_zz    = mech['sigma_zz']
        s_th    = mech['sigma_theta']
        s_rz    = mech['sigma_rz']

        sigma_eq = torch.sqrt(0.5*(
            (s_rr-s_zz)**2 + (s_zz-s_th)**2 +
            (s_th-s_rr)**2 + 6*s_rz**2
        ) + 1e-8)
        sigma_crit = mat.yield_strength(T, fl) / norm.sigma_ref
        g_phi      = mat.phase_degraded_modulus(phi)

        return {
            'T': T, 'f_l': fl, 'phi': phi,
            'H_norm': H_norm, 'T_env': T_env,
            'heal_driver': heal_drv,
            'sigma_rr': s_rr, 'sigma_zz': s_zz,
            'sigma_theta': s_th, 'sigma_rz': s_rz,
            'sigma_eq': sigma_eq, 'sigma_crit': sigma_crit,
            'Gc_eff': Gc_eff, 'g_phi': g_phi,
        }


# ============================================================================
# COLLOCATION POINT SAMPLER
# ============================================================================

def sample_collocation(n_domain=3000, n_surface=600, n_top=400,
                        n_bottom=300, n_initial=500, device=DEVICE):
    """
    Sample collocation points across the cylindrical geometry.
    Includes extra surface and top-surface points where hot-air acts.
    """
    # Interior domain points
    r_d = torch.rand(n_domain, 1, device=device) * GEO.R
    z_d = torch.rand(n_domain, 1, device=device) * GEO.H
    t_d = torch.rand(n_domain, 1, device=device) * T_CYCLE

    # Lateral surface (r=R) — where Robin BC / hot-air acts
    r_s = torch.full((n_surface, 1), GEO.R, device=device)
    z_s = torch.rand(n_surface, 1, device=device) * GEO.H
    t_s = torch.rand(n_surface, 1, device=device) * T_CYCLE

    # Top surface (z=H) — sink-mark formation zone
    r_top = torch.rand(n_top, 1, device=device) * GEO.R
    z_top = torch.full((n_top, 1), GEO.H, device=device)
    t_top = torch.rand(n_top, 1, device=device) * T_CYCLE

    # Bottom surface (z=0) — container contact
    r_bot = torch.rand(n_bottom, 1, device=device) * GEO.R
    z_bot = torch.zeros(n_bottom, 1, device=device)
    t_bot = torch.rand(n_bottom, 1, device=device) * T_CYCLE

    # Initial condition (t=0, T=T_fill everywhere)
    r_ic = torch.rand(n_initial, 1, device=device) * GEO.R
    z_ic = torch.rand(n_initial, 1, device=device) * GEO.H
    t_ic = torch.zeros(n_initial, 1, device=device)

    return {
        'domain':  (r_d,  z_d,  t_d),
        'surface': (r_s,  z_s,  t_s),
        'top':     (r_top, z_top, t_top),
        'bottom':  (r_bot, z_bot, t_bot),
        'initial': (r_ic, z_ic, t_ic),
    }


def sample_process_params(n, device=DEVICE):
    """
    Sample realistic process parameter ranges for training.
    Based on CLIENT_Data_0427.xlsx and physical constraints.
    """
    T_ROOM_K = MATERIAL.T_ROOM
    T_SOL_K  = MATERIAL.T_solidus

    return {
        # Fill temperature: 75-82°C (back-fill 75, front-fill 82)
        'T_fill': (torch.rand(n,1,device=device)*7 + 75 + 273.15),
        # Cooling environment: 21-25°C (lab/production)
        'T_cool': (torch.rand(n,1,device=device)*4 + 21 + 273.15),
        # Hot-air reheat temp: 80-100°C (control variable)
        'T_reheat': (torch.rand(n,1,device=device)*20 + 80 + 273.15),
        # Cold-air final phase: 16-21°C (production uses 16°C)
        'T_cold': (torch.rand(n,1,device=device)*5 + 16 + 273.15),
        # Container/mold wall: ~23°C
        'T_container': (torch.rand(n,1,device=device)*3 + 22 + 273.15),
        # Cooling convection: 5-15 W/m²K (natural+forced cool)
        'h_cool': (torch.rand(n,1,device=device)*10 + 5),
        # Hot-air convection: 30-60 W/m²K (hot air has high h)
        'h_hot': (torch.rand(n,1,device=device)*30 + 30),
        # Cold air convection: 15-25 W/m²K (forced cold air)
        'h_cold': (torch.rand(n,1,device=device)*10 + 15),
        # First pulse start: 0.5-3 min = 30-180 s
        't_pulse_start': (torch.rand(n,1,device=device)*150 + 30),
        # Pulse duration: 15-90 sec (production: 20 sec, lab: 60 sec)
        't_pulse_dur': (torch.rand(n,1,device=device)*75 + 15),
        # Pulse interval: 60-360 sec (between pulses)
        't_pulse_interval': (torch.rand(n,1,device=device)*300 + 60),
        # Number of pulses: 2-7 (production uses 5)
        'n_pulses': (torch.randint(2, 8, (n,1), device=device).float()),
        # Cold start time: 18-25 min = 1080-1500 s
        't_cold_start': (torch.rand(n,1,device=device)*420 + 1080),
        # Material params
        'viscosity': (torch.rand(n,1,device=device)*0.8 + 0.1),
        'humidity':  (torch.rand(n,1,device=device)*0.6 + 0.2),
        'Gc':        (torch.rand(n,1,device=device)*120 + 60),
    }


# ============================================================================
# CLIENT EXPERIMENTAL DATA (from CLIENT_Data_0427.xlsx)
# These are supervision points for L_data
# ============================================================================

def get_client_data_points(device=DEVICE):
    """
    Experimental temperature data for data-informed PINN training.
    Returns (r, z, t, T_measured) tensors at observed surface points.

    Verification Trial 1 (insulation cap, front-fill, 82°C):
      r=R (surface), z=H (top surface measurement)
      Times: 0, 10, 20, 30, 40 min
      T_surface: 82, 70.5, 66.2, 48.7, 45.2 °C

    Verification Trial 2 (room-temp + periodic reheat, front-fill):
      r=R, z=H/2 (representative surface measurement)
      Times and temps including pre/post reheat measurements

    Production Example (back-fill, ~75°C fill):
      Subset of 736 data points — 15 representative ones
    """
    # Trial 1 measurements (at surface r≈R, z≈H)
    t1_times  = [0, 10, 20, 30, 40]  # min
    t1_T_surf = [82, 70.5, 66.2, 48.7, 45.2]  # °C

    # Trial 2 measurements (surface r≈R, z≈H)
    # Before/after reheat events at t=10/11, 14.5/15.5, 19/20 min
    t2_times  = [0, 4.5, 10, 11, 14.5, 15.5, 19, 20, 25, 30]  # min
    t2_T_surf = [82, 52, 44, 49, 44, 49.3, 44, 47.7, 41, 38]  # °C

    # Production data — 15 key representative points from 736
    # (from bulk temperature data, r≈0 = core measurement)
    prod_times = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16, 19, 21, 22, 24.5]  # min
    prod_T     = [74.9, 72.5, 68.5, 65.1, 63.0, 60.0, 57.0, 55.1,
                  50.1, 46.5, 45.2, 44.4, 43.9, 43.8, 43.7]  # °C

    # Assemble as tensors — format: (r, z, t_s, T_K)
    data_pts = []

    # Trial 1: r=R (surface), z=H (top)
    for tm, Ts in zip(t1_times, t1_T_surf):
        data_pts.append((GEO.R, GEO.H, tm*60, Ts + 273.15))

    # Trial 2: r=R (surface), z=H (top)
    for tm, Ts in zip(t2_times, t2_T_surf):
        data_pts.append((GEO.R, GEO.H * 0.9, tm*60, Ts + 273.15))

    # Production: r=0 (core/bulk measurement), z=H/2 (mid-height)
    for tm, Tb in zip(prod_times, prod_T):
        data_pts.append((0.0 + 1e-4, GEO.H * 0.5, tm*60, Tb + 273.15))

    r_data = torch.tensor([p[0] for p in data_pts], dtype=torch.float32, device=device).unsqueeze(1)
    z_data = torch.tensor([p[1] for p in data_pts], dtype=torch.float32, device=device).unsqueeze(1)
    t_data = torch.tensor([p[2] for p in data_pts], dtype=torch.float32, device=device).unsqueeze(1)
    T_data = torch.tensor([p[3] for p in data_pts], dtype=torch.float32, device=device).unsqueeze(1)

    return r_data, z_data, t_data, T_data


# ============================================================================
# PHYSICS LOSS FUNCTIONS
# ============================================================================

def compute_thermal_residual(model, r, z, t, params, create_graph=True):
    """
    Heat equation residual (cylindrical coordinates):
      ρ(c_p + L∂f_l/∂T) ∂T/∂t = k[∂²T/∂r² + (1/r)∂T/∂r + ∂²T/∂z²]

    Returns L_pde (scalar).
    """
    mat  = model.mat
    norm = model.norm

    r_req = r.detach().requires_grad_(True)
    z_req = z.detach().requires_grad_(True)
    t_req = t.detach().requires_grad_(True)

    n = r_req.shape[0]
    bc = model._broadcast

    T_fill     = bc(params['T_fill'],     r_req)
    T_cool     = bc(params['T_cool'],     r_req)
    T_reheat   = bc(params['T_reheat'],   r_req)
    T_cold     = bc(params['T_cold'],     r_req)
    h_cool     = bc(params['h_cool'],     r_req)
    h_hot      = bc(params['h_hot'],      r_req)
    tps        = bc(params['t_pulse_start'],    r_req)
    tpd        = bc(params['t_pulse_dur'],      r_req)
    tpi        = bc(params['t_pulse_interval'], r_req)
    n_p        = bc(params['n_pulses'],   r_req)
    tcs        = bc(params['t_cold_start'], r_req)
    T_cont     = bc(params['T_container'], r_req)
    visc       = bc(params['viscosity'],  r_req)

    T, fl, T_raw, _ = model.forward_thermal(
        r_req, z_req, t_req,
        T_fill, T_cool, T_reheat, T_cold,
        h_cool, h_hot, tps, tpd, tpi, n_p, tcs, T_cont
    )

    # Derivatives
    dT_dt  = torch.autograd.grad(T, t_req, torch.ones_like(T),
                                  create_graph=create_graph, retain_graph=True)[0]
    dT_dr  = torch.autograd.grad(T, r_req, torch.ones_like(T),
                                  create_graph=create_graph, retain_graph=True)[0]
    dT_dz  = torch.autograd.grad(T, z_req, torch.ones_like(T),
                                  create_graph=create_graph, retain_graph=True)[0]
    d2T_dr2 = torch.autograd.grad(dT_dr, r_req, torch.ones_like(dT_dr),
                                   create_graph=create_graph, retain_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z_req, torch.ones_like(dT_dz),
                                   create_graph=create_graph, retain_graph=True)[0]

    # Latent heat term: L * ∂f_l/∂t
    dfl_dt = torch.autograd.grad(fl, t_req, torch.ones_like(fl),
                                  create_graph=create_graph, retain_graph=True)[0]

    lap_T = d2T_dr2 + dT_dr / (r_req + 1e-8) + d2T_dz2
    lhs   = mat.rho * (mat.cp + mat.L_latent * dfl_dt / (dT_dt + 1e-8)) * dT_dt
    rhs   = mat.k_thermal * lap_T

    res = (lhs - rhs) / (mat.rho * mat.cp * 80.0 / T_CYCLE)  # normalized
    return (res**2).mean()


def compute_robin_bc_residual(model, r_s, z_s, t_s, params):
    """
    Lateral surface Robin BC:
      -k ∂T/∂r|_{r=R} = h_conv(t) × (T(R,z,t) − T_env(t))

    During hot-air pulses: h=h_hot, T_env=T_reheat (large flux inward)
    During cooling:        h=h_cool, T_env=T_cool (flux outward)
    During cold air:       h=h_cold, T_env=T_cold
    """
    mat = model.mat
    bc  = model._broadcast

    r_req = r_s.detach().requires_grad_(True)

    T_fill   = bc(params['T_fill'],   r_req)
    T_cool   = bc(params['T_cool'],   r_req)
    T_reheat = bc(params['T_reheat'], r_req)
    T_cold   = bc(params['T_cold'],   r_req)
    h_cool   = bc(params['h_cool'],   r_req)
    h_hot    = bc(params['h_hot'],    r_req)
    tps      = bc(params['t_pulse_start'],    r_req)
    tpd      = bc(params['t_pulse_dur'],      r_req)
    tpi      = bc(params['t_pulse_interval'], r_req)
    n_p      = bc(params['n_pulses'],  r_req)
    tcs      = bc(params['t_cold_start'], r_req)
    T_cont   = bc(params['T_container'], r_req)

    T, _, _, T_env = model.forward_thermal(
        r_req, z_s, t_s,
        T_fill, T_cool, T_reheat, T_cold,
        h_cool, h_hot, tps, tpd, tpi, n_p, tcs, T_cont
    )

    dT_dr = torch.autograd.grad(T, r_req, torch.ones_like(T), create_graph=True)[0]

    h_now = PROCESS.compute_h_conv(
        t_s.squeeze(1),
        h_cool.squeeze(1), h_hot.squeeze(1),
        bc(params.get('h_cold', h_cool), r_req).squeeze(1),
        tps.squeeze(1), tpd.squeeze(1), tpi.squeeze(1),
        n_p.squeeze(1), tcs.squeeze(1)
    ).unsqueeze(1)

    # Robin: -k dT/dr = h*(T - T_env)
    residual = mat.k_thermal * dT_dr + h_now * (T - T_env)
    normalized = residual / (mat.k_thermal * 80.0 / GEO.R)
    return (normalized**2).mean()


def compute_phase_field_residual(model, r, z, t, params):
    """
    Modified phase-field PDE with healing:
      ∂φ/∂t = M_φ × [-φ/l₀² + 2(1-k)(1-φ)H_crack - (κ/l₀)∇²φ]
               - M_heal × heal_driver × φ × 4φ(1-φ)

    The healing term: when T > T_solidus, φ is driven toward 0
    (liquid/semi-solid fills crack → damage repairs).
    """
    mat  = model.mat
    norm = model.norm
    bc   = model._broadcast

    r_req = r.detach().requires_grad_(True)
    z_req = z.detach().requires_grad_(True)
    t_req = t.detach().requires_grad_(True)

    # Get temperature field first
    T_fill   = bc(params['T_fill'],   r_req)
    T_cool   = bc(params['T_cool'],   r_req)
    T_reheat = bc(params['T_reheat'], r_req)
    T_cold   = bc(params['T_cold'],   r_req)
    h_cool   = bc(params['h_cool'],   r_req)
    h_hot    = bc(params['h_hot'],    r_req)
    tps      = bc(params['t_pulse_start'],    r_req)
    tpd      = bc(params['t_pulse_dur'],      r_req)
    tpi      = bc(params['t_pulse_interval'], r_req)
    n_p      = bc(params['n_pulses'],  r_req)
    tcs      = bc(params['t_cold_start'], r_req)
    T_cont   = bc(params['T_container'], r_req)
    Gc_inp   = bc(params['Gc'],        r_req)
    visc     = bc(params['viscosity'], r_req)

    T, fl, T_raw, _ = model.forward_thermal(
        r_req, z_req, t_req,
        T_fill, T_cool, T_reheat, T_cold,
        h_cool, h_hot, tps, tpd, tpi, n_p, tcs, T_cont
    )

    phi, H_norm, heal_drv = model.forward_phase_field(
        r_req, z_req, t_req, T_raw.detach(), fl.detach(),
        T.detach(), Gc_inp, visc
    )

    dphi_dt = torch.autograd.grad(phi, t_req, torch.ones_like(phi),
                                   create_graph=True, retain_graph=True)[0]
    dphi_dr = torch.autograd.grad(phi, r_req, torch.ones_like(phi),
                                   create_graph=True, retain_graph=True)[0]
    dphi_dz = torch.autograd.grad(phi, z_req, torch.ones_like(phi),
                                   create_graph=True, retain_graph=True)[0]
    d2phi_dr2 = torch.autograd.grad(dphi_dr, r_req, torch.ones_like(dphi_dr),
                                     create_graph=True, retain_graph=True)[0]
    d2phi_dz2 = torch.autograd.grad(dphi_dz, z_req, torch.ones_like(dphi_dz),
                                     create_graph=True, retain_graph=True)[0]

    l0   = norm.l0
    k_st = norm.k_stab
    lap_phi = d2phi_dr2 + dphi_dr/(r_req+1e-8) + d2phi_dz2

    # Standard crack evolution (AT2 model)
    Gc_eff  = mat.effective_Gc(Gc_inp, bc(params.get('humidity', 0.3), r_req))
    crack_rhs = (
        - phi / (l0**2) + 2*(1-k_st)*(1-phi)*H_norm
        + (Gc_eff / (mat.Gc_base * l0)) * lap_phi
    )

    # Healing term: activated when T > T_solidus (material re-softens)
    # 4φ(1-φ) peaks at φ=0.5 — healing most effective at partial damage
    healing_rhs = mat.M_heal * heal_drv * phi * 4 * phi * (1 - phi)

    M_phi = 1.0  # phase-field mobility (normalized)
    rhs_total = M_phi * crack_rhs - healing_rhs

    res = dphi_dt - rhs_total
    # Normalize by phase-field scale
    scale = 1.0 / (T_CYCLE * l0)
    return (res * scale)**2 * (1/(scale**2 + 1e-10))
    # Simpler: just MSE of residual
    return (res**2).mean()


def compute_data_loss(model, r_d, z_d, t_d, T_measured, params_for_data):
    """
    Data supervision loss from client experimental measurements.
    Uses MAE + MSE combined for better gradient signal at all error scales.
    Normalization: divide by T_scale=80K so loss is O(1) when error~80K.
    """
    bc = model._broadcast
    p  = params_for_data

    T_pred, _, _, _ = model.forward_thermal(
        r_d, z_d, t_d,
        bc(p['T_fill'],          r_d),
        bc(p['T_cool'],          r_d),
        bc(p['T_reheat'],        r_d),
        bc(p['T_cold'],          r_d),
        bc(p['h_cool'],          r_d),
        bc(p['h_hot'],           r_d),
        bc(p['t_pulse_start'],   r_d),
        bc(p['t_pulse_dur'],     r_d),
        bc(p['t_pulse_interval'],r_d),
        bc(p['n_pulses'],        r_d),
        bc(p['t_cold_start'],    r_d),
        bc(p['T_container'],     r_d),
    )

    T_scale = 80.0   # K — normalization
    err     = (T_pred - T_measured) / T_scale

    # Huber-like loss: L1 for large errors (robust), L2 for small errors (smooth)
    mae  = err.abs().mean()
    mse  = (err ** 2).mean()
    loss = 0.5 * mae + 0.5 * mse

    return loss


def compute_sink_mark_loss(model, r_top, z_top, t_top, params):
    """
    Penalize volume contraction at top surface during mushy→solid.
    Sink marks form when bulk shrinks (≈8%) and top surface loses material.
    High φ at z=H + low fl → severe sink. We penalize φ_top × (1 - fl_top).
    """
    bc = model._broadcast
    T_fill   = bc(params['T_fill'],   r_top)
    T_cool   = bc(params['T_cool'],   r_top)
    T_reheat = bc(params['T_reheat'], r_top)
    T_cold   = bc(params['T_cold'],   r_top)
    h_cool   = bc(params['h_cool'],   r_top)
    h_hot    = bc(params['h_hot'],    r_top)
    tps      = bc(params['t_pulse_start'],    r_top)
    tpd      = bc(params['t_pulse_dur'],      r_top)
    tpi      = bc(params['t_pulse_interval'], r_top)
    n_p      = bc(params['n_pulses'],  r_top)
    tcs      = bc(params['t_cold_start'], r_top)
    T_cont   = bc(params['T_container'], r_top)
    Gc_inp   = bc(params['Gc'],        r_top)
    visc     = bc(params['viscosity'], r_top)

    T, fl, T_raw, _ = model.forward_thermal(
        r_top, z_top, t_top,
        T_fill, T_cool, T_reheat, T_cold,
        h_cool, h_hot, tps, tpd, tpi, n_p, tcs, T_cont
    )
    phi, _, _ = model.forward_phase_field(
        r_top, z_top, t_top, T_raw.detach(), fl.detach(),
        T.detach(), Gc_inp, visc
    )

    # Sink indicator: high φ AND low fl (solid material with cracks = sink mark)
    sink = phi * (1.0 - fl)
    return sink.mean()


def compute_healing_effectiveness_loss(model, r, z, t, params):
    """
    Encourages φ to decrease during hot-air pulse windows.
    Computes φ just before and during a pulse, penalizes if φ doesn't decrease.
    """
    bc = model._broadcast
    n  = r.shape[0]

    tps = bc(params['t_pulse_start'],    r)
    tpd = bc(params['t_pulse_dur'],      r)

    # Sample time points: just before pulse start and just after
    t_before = tps - 5.0   # 5 sec before pulse
    t_during = tps + tpd/2  # mid-pulse

    t_before = torch.clamp(t_before, 0.1, T_CYCLE)
    t_during = torch.clamp(t_during, 0.1, T_CYCLE)

    def get_phi(t_eval):
        T_fill   = bc(params['T_fill'],   r)
        T_cool   = bc(params['T_cool'],   r)
        T_reheat = bc(params['T_reheat'], r)
        T_cold   = bc(params['T_cold'],   r)
        h_cool   = bc(params['h_cool'],   r)
        h_hot    = bc(params['h_hot'],    r)
        tpi      = bc(params['t_pulse_interval'], r)
        n_p      = bc(params['n_pulses'],  r)
        tcs      = bc(params['t_cold_start'], r)
        T_cont   = bc(params['T_container'], r)
        Gc_inp   = bc(params['Gc'],        r)
        visc     = bc(params['viscosity'], r)

        T, fl, T_raw, _ = model.forward_thermal(
            r, z, t_eval,
            T_fill, T_cool, T_reheat, T_cold,
            h_cool, h_hot, bc(params['t_pulse_start'], r), tpd, tpi, n_p, tcs, T_cont
        )
        phi, _, _ = model.forward_phase_field(
            r, z, t_eval, T_raw.detach(), fl.detach(), T.detach(), Gc_inp, visc
        )
        return phi

    phi_before = get_phi(t_before)
    phi_during = get_phi(t_during)

    # Penalize if φ does NOT decrease during healing (φ_during should ≤ φ_before)
    healing_deficit = F.relu(phi_during - phi_before + 0.02)
    return healing_deficit.mean()


# ============================================================================
# ADAPTIVE LOSS WEIGHT CONTROLLER
# ============================================================================

class AdaptiveLossWeightsV4:
    """
    Tracks and adapts loss component weights using gradient statistics.
    v4 adds weights for healing and sink-mark losses.
    """
    def __init__(self):
        self.weights = {
            'pde':    1.0,
            'robin':  2.0,  # higher because BC is crucial for hot-air effect
            'ic':     5.0,
            'phase':  0.8,
            'data':   10.0, # highest — experimental data is ground truth
            'heal':   3.0,  # healing enforcement
            'sink':   2.0,  # sink-mark prevention
            'bottom': 1.5,
        }
        self.running_mean = {k: 1.0 for k in self.weights}
        self.alpha = 0.98  # EMA factor

    def update(self, loss_dict):
        for k, v in loss_dict.items():
            if k in self.running_mean:
                self.running_mean[k] = (self.alpha * self.running_mean[k]
                                        + (1 - self.alpha) * float(v))
        # Re-balance weights toward equal loss contributions
        mean_val = np.mean(list(self.running_mean.values()))
        for k in self.weights:
            if self.running_mean[k] > 1e-10:
                self.weights[k] = mean_val / (self.running_mean[k] + 1e-8)
        # Clamp weights
        for k in self.weights:
            self.weights[k] = float(np.clip(self.weights[k], 0.1, 50.0))
        # Always keep data loss weight high (experimental ground truth)
        self.weights['data'] = max(self.weights['data'], 8.0)

    def total_loss(self, loss_dict):
        total = sum(self.weights[k] * loss_dict[k]
                    for k in loss_dict if k in self.weights)
        return total, {k: self.weights[k] * v for k, v in loss_dict.items()
                       if k in self.weights}


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_pinn_v4(n_iterations=60_000, save_dir='pinn_outputs_v4'):
    """
    Three-phase training schedule — FIXED VERSION:
      Phase 1 (0–15k):   Data-first: IC + data + lightweight Robin BC
      Phase 2 (15k–40k): + PDE thermal residual + phase field
      Phase 3 (40k–60k): + Full multi-physics + sink mark + healing

    Key fixes vs original:
      1. No try/except — errors surface immediately for debugging
      2. Params broadcast to SAME size as collocation slice used
      3. IC loss normalized by T_scale (80K), not 6400
      4. Data loss uses Huber (MAE+MSE) not MSE/6400
      5. PDE gradients computed on requires_grad tensors properly
      6. L-BFGS closure samples consistently sized batches
    """
    Path(save_dir).mkdir(exist_ok=True)

    model = MultiPhysicsPINNV4().to(DEVICE)
    opt_adam  = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam, T_max=n_iterations, eta_min=1e-5
    )

    loss_weights = AdaptiveLossWeightsV4()

    # Client data supervision points (fixed, float32)
    r_data, z_data, t_data, T_measured = get_client_data_points(DEVICE)

    # Best-known process parameters (production line)
    DATA_PARAMS = {
        'T_fill':          float(75 + 273.15),
        'T_cool':          float(21 + 273.15),
        'T_reheat':        float(100 + 273.15),
        'T_cold':          float(16 + 273.15),
        'h_cool':          8.0,
        'h_hot':           45.0,
        'h_cold':          20.0,
        't_pulse_start':   35.0,
        't_pulse_dur':     20.0,
        't_pulse_interval':88.0,
        'n_pulses':        5.0,
        't_cold_start':    1278.0,
        'T_container':     float(23 + 273.15),
        'viscosity':       0.3,
        'humidity':        0.4,
        'Gc':              80.0,
    }

    # Trial 1 params (insulation cap, 82°C fill)
    TRIAL1_PARAMS = {
        'T_fill':          float(82 + 273.15),
        'T_cool':          float(61 + 273.15),   # avg zone temp
        'T_reheat':        float(63 + 273.15),
        'T_cold':          float(23 + 273.15),
        'h_cool':          4.0,
        'h_hot':           4.0,
        'h_cold':          4.0,
        't_pulse_start':   9999.0,
        't_pulse_dur':     1.0,
        't_pulse_interval':9999.0,
        'n_pulses':        0.0,
        't_cold_start':    2400.0,
        'T_container':     float(23 + 273.15),
        'viscosity':       0.3,
        'humidity':        0.4,
        'Gc':              80.0,
    }

    losses_history = {k: [] for k in ['total','pde','robin','ic','phase',
                                        'data','heal','sink','bottom']}

    print(f"\n{'='*60}")
    print(f"Training MS-PINN v4 (FIXED) for {n_iterations} iterations")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training data points: {r_data.shape[0]}")
    print(f"  [Fix] Correction-based T reconstruction (analytical baseline + δT)")
    print(f"  [Fix] Huber data loss (not MSE/6400)")
    print(f"  [Fix] No try/except — proper gradient flow")
    print(f"  [Fix] Params broadcast to match slice size exactly")
    print(f"{'='*60}\n")

    def _make_params(scalar_dict, n_pts):
        """Broadcast scalar param dict to n_pts collocation points."""
        bc = model._broadcast
        ref = torch.zeros(n_pts, 1, device=DEVICE)
        return {k: bc(v, ref) for k, v in scalar_dict.items()}

    def _make_rand_params(n_pts):
        """Sample one random param set broadcast to n_pts points."""
        n_b = 1
        p1  = sample_process_params(n_b, DEVICE)
        bc  = model._broadcast
        ref = torch.zeros(n_pts, 1, device=DEVICE)
        return {k: bc(v[0], ref) for k, v in p1.items()}

    best_data_loss = float('inf')

    for step in range(n_iterations):
        model.train()
        opt_adam.zero_grad()

        pts = sample_collocation(
            n_domain=800, n_surface=300, n_top=200,
            n_bottom=150, n_initial=200, device=DEVICE
        )
        r_d, z_d, t_d     = pts['domain']
        r_s, z_s, t_s_pts = pts['surface']
        r_top,z_top,t_top = pts['top']
        r_bot,z_bot,t_bot = pts['bottom']
        r_ic, z_ic, t_ic  = pts['initial']

        loss_dict = {}

        # ══ Phase 1 (0–15k): Data-first ══════════════════════════════════
        if step < 15_000:

            # DATA LOSS — primary objective, both experimental conditions
            # Production params
            pd_n   = _make_params(DATA_PARAMS, r_data.shape[0])
            T_pred, _, _, _ = model.forward_thermal(
                r_data, z_data, t_data,
                pd_n['T_fill'], pd_n['T_cool'], pd_n['T_reheat'], pd_n['T_cold'],
                pd_n['h_cool'], pd_n['h_hot'], pd_n['t_pulse_start'],
                pd_n['t_pulse_dur'], pd_n['t_pulse_interval'],
                pd_n['n_pulses'], pd_n['t_cold_start'], pd_n['T_container'],
            )
            err   = (T_pred - T_measured) / 80.0
            loss_dict['data'] = 0.5 * err.abs().mean() + 0.5 * (err**2).mean()

            # IC LOSS — T(r,z,0) = T_fill (using random params for generalization)
            pd_ic  = _make_rand_params(r_ic.shape[0])
            T_ic, _, _, _ = model.forward_thermal(
                r_ic, z_ic, t_ic,
                pd_ic['T_fill'], pd_ic['T_cool'], pd_ic['T_reheat'], pd_ic['T_cold'],
                pd_ic['h_cool'], pd_ic['h_hot'], pd_ic['t_pulse_start'],
                pd_ic['t_pulse_dur'], pd_ic['t_pulse_interval'],
                pd_ic['n_pulses'], pd_ic['t_cold_start'], pd_ic['T_container'],
            )
            loss_dict['ic'] = ((T_ic - pd_ic['T_fill']) / 80.0).pow(2).mean()

            # ROBIN BC (lightweight — ensures surface BC is learned)
            N_s = 150
            pd_s = _make_rand_params(N_s)
            T_s, _, _, T_env_s = model.forward_thermal(
                r_s[:N_s].detach().requires_grad_(True),
                z_s[:N_s], t_s_pts[:N_s],
                pd_s['T_fill'], pd_s['T_cool'], pd_s['T_reheat'], pd_s['T_cold'],
                pd_s['h_cool'], pd_s['h_hot'], pd_s['t_pulse_start'],
                pd_s['t_pulse_dur'], pd_s['t_pulse_interval'],
                pd_s['n_pulses'], pd_s['t_cold_start'], pd_s['T_container'],
            )
            r_s_rg = r_s[:N_s].detach().requires_grad_(True)
            T_s2, _, _, T_env_s2 = model.forward_thermal(
                r_s_rg, z_s[:N_s], t_s_pts[:N_s],
                pd_s['T_fill'], pd_s['T_cool'], pd_s['T_reheat'], pd_s['T_cold'],
                pd_s['h_cool'], pd_s['h_hot'], pd_s['t_pulse_start'],
                pd_s['t_pulse_dur'], pd_s['t_pulse_interval'],
                pd_s['n_pulses'], pd_s['t_cold_start'], pd_s['T_container'],
            )
            dT_dr = torch.autograd.grad(
                T_s2.sum(), r_s_rg, create_graph=True, retain_graph=True
            )[0]
            h_bc  = pd_s['h_cool'][:N_s]
            robin = MATERIAL.k_thermal * dT_dr + h_bc * (T_s2 - T_env_s2)
            loss_dict['robin'] = (robin / (MATERIAL.k_thermal * 80.0 / GEO.R)).pow(2).mean()

            # Placeholders
            for k in ['pde','phase','heal','sink','bottom']:
                loss_dict[k] = torch.tensor(0.0, device=DEVICE)

        # ══ Phase 2 (15k–40k): + PDE + Phase field ═══════════════════════
        elif step < 40_000:

            # Data loss (always)
            pd_n = _make_params(DATA_PARAMS, r_data.shape[0])
            T_pred, _, _, _ = model.forward_thermal(
                r_data, z_data, t_data,
                pd_n['T_fill'], pd_n['T_cool'], pd_n['T_reheat'], pd_n['T_cold'],
                pd_n['h_cool'], pd_n['h_hot'], pd_n['t_pulse_start'],
                pd_n['t_pulse_dur'], pd_n['t_pulse_interval'],
                pd_n['n_pulses'], pd_n['t_cold_start'], pd_n['T_container'],
            )
            err = (T_pred - T_measured) / 80.0
            loss_dict['data'] = 0.5 * err.abs().mean() + 0.5 * (err**2).mean()

            # IC loss
            pd_ic = _make_rand_params(r_ic.shape[0])
            T_ic, _, _, _ = model.forward_thermal(
                r_ic, z_ic, t_ic,
                pd_ic['T_fill'], pd_ic['T_cool'], pd_ic['T_reheat'], pd_ic['T_cold'],
                pd_ic['h_cool'], pd_ic['h_hot'], pd_ic['t_pulse_start'],
                pd_ic['t_pulse_dur'], pd_ic['t_pulse_interval'],
                pd_ic['n_pulses'], pd_ic['t_cold_start'], pd_ic['T_container'],
            )
            loss_dict['ic'] = ((T_ic - pd_ic['T_fill']) / 80.0).pow(2).mean()

            # PDE residual (heat equation in cylindrical coords)
            N_pde = 300
            pd_pde = _make_rand_params(N_pde)
            r_rg   = r_d[:N_pde].detach().requires_grad_(True)
            z_rg   = z_d[:N_pde].detach().requires_grad_(True)
            t_rg   = t_d[:N_pde].detach().requires_grad_(True)

            T_pde, fl_pde, _, _ = model.forward_thermal(
                r_rg, z_rg, t_rg,
                pd_pde['T_fill'], pd_pde['T_cool'], pd_pde['T_reheat'], pd_pde['T_cold'],
                pd_pde['h_cool'], pd_pde['h_hot'], pd_pde['t_pulse_start'],
                pd_pde['t_pulse_dur'], pd_pde['t_pulse_interval'],
                pd_pde['n_pulses'], pd_pde['t_cold_start'], pd_pde['T_container'],
            )

            ones = torch.ones_like(T_pde)
            dT_dt   = torch.autograd.grad(T_pde, t_rg, ones, create_graph=True, retain_graph=True)[0]
            dT_dr_  = torch.autograd.grad(T_pde, r_rg, ones, create_graph=True, retain_graph=True)[0]
            dT_dz_  = torch.autograd.grad(T_pde, z_rg, ones, create_graph=True, retain_graph=True)[0]
            d2T_dr2 = torch.autograd.grad(dT_dr_, r_rg, ones, create_graph=True, retain_graph=True)[0]
            d2T_dz2 = torch.autograd.grad(dT_dz_, z_rg, ones, create_graph=True, retain_graph=True)[0]

            lap_T   = d2T_dr2 + dT_dr_ / (r_rg + 1e-5) + d2T_dz2
            dfl_dT  = torch.autograd.grad(fl_pde, T_pde, ones,
                                           create_graph=True, retain_graph=True)[0] \
                      if T_pde.requires_grad else torch.zeros_like(T_pde)
            eff_cp  = MATERIAL.cp + MATERIAL.L_latent * dfl_dT.detach().clamp(-10, 10)
            res_pde = (MATERIAL.rho * eff_cp * dT_dt - MATERIAL.k_thermal * lap_T)
            scale   = MATERIAL.rho * MATERIAL.cp * 80.0 / T_CYCLE
            loss_dict['pde'] = (res_pde / (scale + 1e-8)).pow(2).mean()

            # Phase field (lightweight)
            N_pf = 200
            pd_pf = _make_rand_params(N_pf)
            T_pf, fl_pf, T_norm_pf, _ = model.forward_thermal(
                r_d[:N_pf], z_d[:N_pf], t_d[:N_pf],
                pd_pf['T_fill'], pd_pf['T_cool'], pd_pf['T_reheat'], pd_pf['T_cold'],
                pd_pf['h_cool'], pd_pf['h_hot'], pd_pf['t_pulse_start'],
                pd_pf['t_pulse_dur'], pd_pf['t_pulse_interval'],
                pd_pf['n_pulses'], pd_pf['t_cold_start'], pd_pf['T_container'],
            )
            phi_pf, _, _ = model.forward_phase_field(
                r_d[:N_pf], z_d[:N_pf], t_d[:N_pf],
                T_norm_pf.detach(), fl_pf.detach(), T_pf.detach(),
                pd_pf['Gc'], pd_pf['viscosity'],
            )
            # Phase field should be in [0,1] and increase in mushy zone
            mushy = (fl_pf.detach() > 0.1) & (fl_pf.detach() < 0.9)
            if mushy.sum() > 0:
                # φ should be > 0 in mushy zone (some damage)
                phi_mushy = phi_pf[mushy]
                loss_dict['phase'] = F.relu(0.05 - phi_mushy).mean()
            else:
                loss_dict['phase'] = torch.tensor(0.0, device=DEVICE)

            # Robin BC
            N_s   = 150
            pd_s  = _make_rand_params(N_s)
            r_s_rg = r_s[:N_s].detach().requires_grad_(True)
            T_s_rg, _, _, T_env_rg = model.forward_thermal(
                r_s_rg, z_s[:N_s], t_s_pts[:N_s],
                pd_s['T_fill'], pd_s['T_cool'], pd_s['T_reheat'], pd_s['T_cold'],
                pd_s['h_cool'], pd_s['h_hot'], pd_s['t_pulse_start'],
                pd_s['t_pulse_dur'], pd_s['t_pulse_interval'],
                pd_s['n_pulses'], pd_s['t_cold_start'], pd_s['T_container'],
            )
            dT_dr_s = torch.autograd.grad(T_s_rg.sum(), r_s_rg, create_graph=True)[0]
            robin   = MATERIAL.k_thermal * dT_dr_s + pd_s['h_cool'][:N_s] * (T_s_rg - T_env_rg)
            loss_dict['robin'] = (robin / (MATERIAL.k_thermal * 80.0 / GEO.R)).pow(2).mean()

            # Heal + sink placeholder
            for k in ['heal','sink','bottom']:
                loss_dict[k] = torch.tensor(0.0, device=DEVICE)

        # ══ Phase 3 (40k–60k): Full multi-physics ═══════════════════════
        else:

            # Data (always)
            pd_n = _make_params(DATA_PARAMS, r_data.shape[0])
            T_pred, _, _, _ = model.forward_thermal(
                r_data, z_data, t_data,
                pd_n['T_fill'], pd_n['T_cool'], pd_n['T_reheat'], pd_n['T_cold'],
                pd_n['h_cool'], pd_n['h_hot'], pd_n['t_pulse_start'],
                pd_n['t_pulse_dur'], pd_n['t_pulse_interval'],
                pd_n['n_pulses'], pd_n['t_cold_start'], pd_n['T_container'],
            )
            err = (T_pred - T_measured) / 80.0
            loss_dict['data'] = 0.5*err.abs().mean() + 0.5*(err**2).mean()

            # IC
            pd_ic = _make_rand_params(r_ic.shape[0])
            T_ic, _, _, _ = model.forward_thermal(
                r_ic, z_ic, t_ic,
                pd_ic['T_fill'], pd_ic['T_cool'], pd_ic['T_reheat'], pd_ic['T_cold'],
                pd_ic['h_cool'], pd_ic['h_hot'], pd_ic['t_pulse_start'],
                pd_ic['t_pulse_dur'], pd_ic['t_pulse_interval'],
                pd_ic['n_pulses'], pd_ic['t_cold_start'], pd_ic['T_container'],
            )
            loss_dict['ic'] = ((T_ic - pd_ic['T_fill']) / 80.0).pow(2).mean()

            # Lighter PDE
            N_pde = 200
            pd_pde = _make_rand_params(N_pde)
            r_rg = r_d[:N_pde].detach().requires_grad_(True)
            z_rg = z_d[:N_pde].detach().requires_grad_(True)
            t_rg = t_d[:N_pde].detach().requires_grad_(True)
            T_pde, fl_pde, _, _ = model.forward_thermal(
                r_rg, z_rg, t_rg,
                pd_pde['T_fill'], pd_pde['T_cool'], pd_pde['T_reheat'], pd_pde['T_cold'],
                pd_pde['h_cool'], pd_pde['h_hot'], pd_pde['t_pulse_start'],
                pd_pde['t_pulse_dur'], pd_pde['t_pulse_interval'],
                pd_pde['n_pulses'], pd_pde['t_cold_start'], pd_pde['T_container'],
            )
            ones = torch.ones_like(T_pde)
            dT_dt  = torch.autograd.grad(T_pde, t_rg, ones, create_graph=True, retain_graph=True)[0]
            dT_dr_ = torch.autograd.grad(T_pde, r_rg, ones, create_graph=True, retain_graph=True)[0]
            dT_dz_ = torch.autograd.grad(T_pde, z_rg, ones, create_graph=True, retain_graph=True)[0]
            d2T_dr2= torch.autograd.grad(dT_dr_, r_rg, ones, create_graph=True, retain_graph=True)[0]
            d2T_dz2= torch.autograd.grad(dT_dz_, z_rg, ones, create_graph=True, retain_graph=True)[0]
            lap_T  = d2T_dr2 + dT_dr_/(r_rg+1e-5) + d2T_dz2
            res_pde= MATERIAL.rho*MATERIAL.cp*dT_dt - MATERIAL.k_thermal*lap_T
            scale  = MATERIAL.rho*MATERIAL.cp*80.0/T_CYCLE
            loss_dict['pde'] = (res_pde/(scale+1e-8)).pow(2).mean()

            # Phase field + healing
            N_pf = 150
            pd_pf = _make_rand_params(N_pf)
            T_pf, fl_pf, T_norm_pf, _ = model.forward_thermal(
                r_d[:N_pf], z_d[:N_pf], t_d[:N_pf],
                pd_pf['T_fill'], pd_pf['T_cool'], pd_pf['T_reheat'], pd_pf['T_cold'],
                pd_pf['h_cool'], pd_pf['h_hot'], pd_pf['t_pulse_start'],
                pd_pf['t_pulse_dur'], pd_pf['t_pulse_interval'],
                pd_pf['n_pulses'], pd_pf['t_cold_start'], pd_pf['T_container'],
            )
            phi_pf, H_pf, hd_pf = model.forward_phase_field(
                r_d[:N_pf], z_d[:N_pf], t_d[:N_pf],
                T_norm_pf.detach(), fl_pf.detach(), T_pf.detach(),
                pd_pf['Gc'], pd_pf['viscosity'],
            )
            mushy = (fl_pf.detach() > 0.1) & (fl_pf.detach() < 0.9)
            loss_dict['phase'] = F.relu(0.05 - phi_pf[mushy]).mean() if mushy.sum()>0 \
                else torch.tensor(0.0, device=DEVICE)

            # Healing loss (φ should decrease when heal_driver active)
            healing_active = hd_pf > 0.05
            if healing_active.sum() > 0:
                phi_heal = phi_pf[healing_active]
                hd_heal  = hd_pf[healing_active]
                # Penalize high φ in healing zone — should approach 0
                loss_dict['heal'] = (phi_heal * hd_heal).mean()
            else:
                loss_dict['heal'] = torch.tensor(0.0, device=DEVICE)

            # Sink mark: φ_top × (1-fl) at z=H
            N_top = 100
            pd_top = _make_rand_params(N_top)
            T_top, fl_top, T_norm_top, _ = model.forward_thermal(
                r_top[:N_top], z_top[:N_top], t_top[:N_top],
                pd_top['T_fill'], pd_top['T_cool'], pd_top['T_reheat'], pd_top['T_cold'],
                pd_top['h_cool'], pd_top['h_hot'], pd_top['t_pulse_start'],
                pd_top['t_pulse_dur'], pd_top['t_pulse_interval'],
                pd_top['n_pulses'], pd_top['t_cold_start'], pd_top['T_container'],
            )
            phi_top, _, _ = model.forward_phase_field(
                r_top[:N_top], z_top[:N_top], t_top[:N_top],
                T_norm_top.detach(), fl_top.detach(), T_top.detach(),
                pd_top['Gc'], pd_top['viscosity'],
            )
            loss_dict['sink'] = (phi_top * (1.0 - fl_top.detach())).mean()

            # Bottom BC
            pd_bot = _make_rand_params(r_bot.shape[0])
            T_bot, _, _, _ = model.forward_thermal(
                r_bot, z_bot, t_bot,
                pd_bot['T_fill'], pd_bot['T_cool'], pd_bot['T_reheat'], pd_bot['T_cold'],
                pd_bot['h_cool'], pd_bot['h_hot'], pd_bot['t_pulse_start'],
                pd_bot['t_pulse_dur'], pd_bot['t_pulse_interval'],
                pd_bot['n_pulses'], pd_bot['t_cold_start'], pd_bot['T_container'],
            )
            loss_dict['bottom'] = ((T_bot - pd_bot['T_container']) / 80.0).pow(2).mean()

            # Robin
            N_s  = 120
            pd_s = _make_rand_params(N_s)
            r_s_rg = r_s[:N_s].detach().requires_grad_(True)
            T_s_rg, _, _, T_env_rg = model.forward_thermal(
                r_s_rg, z_s[:N_s], t_s_pts[:N_s],
                pd_s['T_fill'], pd_s['T_cool'], pd_s['T_reheat'], pd_s['T_cold'],
                pd_s['h_cool'], pd_s['h_hot'], pd_s['t_pulse_start'],
                pd_s['t_pulse_dur'], pd_s['t_pulse_interval'],
                pd_s['n_pulses'], pd_s['t_cold_start'], pd_s['T_container'],
            )
            dT_dr_s = torch.autograd.grad(T_s_rg.sum(), r_s_rg, create_graph=True)[0]
            robin   = MATERIAL.k_thermal*dT_dr_s + pd_s['h_cool'][:N_s]*(T_s_rg - T_env_rg)
            loss_dict['robin'] = (robin/(MATERIAL.k_thermal*80.0/GEO.R)).pow(2).mean()

        # ─── Adaptive weighting + backward ───────────────────────────────
        loss_weights.update({k: v.item() for k,v in loss_dict.items()})
        total, _ = loss_weights.total_loss(loss_dict)

        if torch.isfinite(total):
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_adam.step()
            scheduler.step()

            # Save best model by data loss
            dl = loss_dict['data'].item()
            if dl < best_data_loss:
                best_data_loss = dl
                torch.save({'model_state_dict': model.state_dict(),
                            'step': step,
                            'data_loss': dl}, f"{save_dir}/pinn_v4_best.pth")

        for k, v in loss_dict.items():
            losses_history[k].append(float(v.item()) if torch.is_tensor(v) else float(v))
        losses_history['total'].append(float(total.item()) if torch.isfinite(total) else 0.0)

        if step % 500 == 0:
            phase = 1 if step < 15000 else (2 if step < 40000 else 3)
            ic_v  = loss_dict['ic'].item()
            dt_v  = loss_dict['data'].item()
            pde_v = loss_dict['pde'].item()
            ph_v  = loss_dict['phase'].item()
            lr    = opt_adam.param_groups[0]['lr']
            print(f"[Ph{phase}] step {step:5d}  "
                  f"L_tot={float(total.item()):.4f}  "
                  f"L_data={dt_v:.4f}  "
                  f"L_ic={ic_v:.4f}  "
                  f"L_pde={pde_v:.4f}  "
                  f"L_pf={ph_v:.4f}  "
                  f"lr={lr:.2e}  "
                  f"best_data={best_data_loss:.4f}")

        if step % 10000 == 0 and step > 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'step': step, 'losses': losses_history},
                       f"{save_dir}/checkpoint_step{step}.pth")

    # ── L-BFGS fine-tuning (fixed: consistent batch size) ────────────────
    print("\nL-BFGS fine-tuning (300 steps)...")
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.05, max_iter=15,
        history_size=30, tolerance_grad=1e-6,
        line_search_fn='strong_wolfe'
    )

    lbfgs_losses = []

    def closure():
        opt_lbfgs.zero_grad()
        # Always use exact data points size (30 points)
        pd_n = _make_params(DATA_PARAMS, r_data.shape[0])
        T_pred, _, _, _ = model.forward_thermal(
            r_data, z_data, t_data,
            pd_n['T_fill'], pd_n['T_cool'], pd_n['T_reheat'], pd_n['T_cold'],
            pd_n['h_cool'], pd_n['h_hot'], pd_n['t_pulse_start'],
            pd_n['t_pulse_dur'], pd_n['t_pulse_interval'],
            pd_n['n_pulses'], pd_n['t_cold_start'], pd_n['T_container'],
        )
        err = (T_pred - T_measured) / 80.0
        L_data = 0.5*err.abs().mean() + 0.5*(err**2).mean()

        # Add IC on small fixed set
        r_ic_f  = torch.rand(50,1,device=DEVICE)*GEO.R
        z_ic_f  = torch.rand(50,1,device=DEVICE)*GEO.H
        t_ic_f  = torch.zeros(50,1,device=DEVICE)
        pd_ic_f = _make_params(DATA_PARAMS, 50)
        T_ic_f, _, _, _ = model.forward_thermal(
            r_ic_f, z_ic_f, t_ic_f,
            pd_ic_f['T_fill'], pd_ic_f['T_cool'], pd_ic_f['T_reheat'], pd_ic_f['T_cold'],
            pd_ic_f['h_cool'], pd_ic_f['h_hot'], pd_ic_f['t_pulse_start'],
            pd_ic_f['t_pulse_dur'], pd_ic_f['t_pulse_interval'],
            pd_ic_f['n_pulses'], pd_ic_f['t_cold_start'], pd_ic_f['T_container'],
        )
        L_ic = ((T_ic_f - pd_ic_f['T_fill']) / 80.0).pow(2).mean()

        L = 10.0 * L_data + 2.0 * L_ic
        if torch.isfinite(L):
            L.backward()
            lbfgs_losses.append(float(L.item()))
        return L

    for lb_step in range(300):
        try:
            opt_lbfgs.step(closure)
            if lb_step % 30 == 0 and lbfgs_losses:
                print(f"  L-BFGS {lb_step:3d}: loss={lbfgs_losses[-1]:.5f}")
        except Exception as e:
            print(f"  L-BFGS {lb_step}: {e}")
            break

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses':           losses_history,
        'best_data_loss':   best_data_loss,
        'norm_scales':      vars(NORM),
        'process_params_reference': DATA_PARAMS,
    }, f"{save_dir}/pinn_v4_final.pth")

    # Also update best if final is better
    pd_n = _make_params(DATA_PARAMS, r_data.shape[0])
    with torch.no_grad():
        T_final, _, _, _ = model.forward_thermal(
            r_data, z_data, t_data,
            pd_n['T_fill'], pd_n['T_cool'], pd_n['T_reheat'], pd_n['T_cold'],
            pd_n['h_cool'], pd_n['h_hot'], pd_n['t_pulse_start'],
            pd_n['t_pulse_dur'], pd_n['t_pulse_interval'],
            pd_n['n_pulses'], pd_n['t_cold_start'], pd_n['T_container'],
        )
    final_dl = float(((T_final - T_measured)/80.0).abs().mean().item())
    if final_dl < best_data_loss:
        torch.save({'model_state_dict': model.state_dict(), 'data_loss': final_dl},
                   f"{save_dir}/pinn_v4_best.pth")
        print(f"  Updated best model: data_loss={final_dl:.5f}")

    _plot_training_curves(losses_history, save_dir)

    print(f"\n✓ Training complete.")
    print(f"  Best data loss: {best_data_loss:.5f} (target: < 0.05)")
    print(f"  Model saved to {save_dir}/")
    return model


def _plot_training_curves(losses, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("MS-PINN v4 Training — Pulsed Reheat Process", fontsize=13)

    ax = axes[0, 0]
    ax.semilogy(losses['total'], label='Total', color='black', linewidth=1.5)
    ax.set_title("Total Loss"); ax.set_xlabel("Iteration"); ax.legend()

    ax = axes[0, 1]
    for k, col in [('pde','blue'), ('robin','red'), ('ic','green')]:
        if losses[k]:
            ax.semilogy(losses[k], label=k, color=col, alpha=0.8, linewidth=1)
    ax.set_title("Thermal Losses"); ax.legend()

    ax = axes[1, 0]
    for k, col in [('data','orange'), ('heal','purple'), ('sink','brown')]:
        if losses[k]:
            ax.semilogy([v+1e-10 for v in losses[k]], label=k, color=col, alpha=0.8)
    ax.set_title("Physics-Informed & Data Losses"); ax.legend()

    ax = axes[1, 1]
    if losses['phase']:
        ax.semilogy([v+1e-10 for v in losses['phase']], label='phase_field',
                    color='magenta', linewidth=1.2)
    ax.set_title("Phase-Field Loss"); ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves_v4.png", dpi=120)
    plt.close()
    print(f"  Training curves saved.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*65)
    print("  MS-PINN v4 Training  |  Pulsed Hot-Air Reheating Process")
    print("  CBIC × TUAT  |  Julian Evan Chrisnanto  |  2026")
    print("="*65)
    model = train_pinn_v4(n_iterations=60_000, save_dir='pinn_outputs_v4')
    print("\nPINN v4 training complete.")
