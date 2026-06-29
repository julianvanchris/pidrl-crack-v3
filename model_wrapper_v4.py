"""
model_wrapper_v4.py

Inference wrapper for trained MultiPhysicsPINNV4.
Drop-in replacement for model_wrapper_v3.py.

New predict_fields interface (v4):
  r, z, t,
  T_fill, T_cool, T_reheat, T_cold,
  h_cool, h_hot,
  t_pulse_start, t_pulse_dur, t_pulse_interval,
  n_pulses, t_cold_start,
  T_container, viscosity, humidity, Gc

Process model (from CLIENT_Data_0427.xlsx):
  LCWT401 back-fill: T_fill=75°C, T_cool=21°C
  Hot-air pulses: T_reheat≈100°C, h_hot≈45 W/m²K
  Pulse schedule: 5 pulses of 20 sec at 0.57, 1.13, 2.90, 4.03, 5.83 min
  Cold air: T_cold=16°C, starts at t≈21.3 min
  Total process: ~24.5 min (within 30-min target)
"""

import torch
import numpy as np
from pathlib import Path


def load_model(checkpoint_path):
    """
    Load trained PINN v4 model from checkpoint.
    Tries to import from training file first; falls back to standalone.
    """
    try:
        from train_pinn_physics_enforced_v4 import MultiPhysicsPINNV4
        model = MultiPhysicsPINNV4()
    except ImportError:
        model = MultiPhysicsPINNV4Standalone()

    if Path(checkpoint_path).exists():
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ck.get('model_state_dict', ck)
        model.load_state_dict(state)
        print(f"[v4] Model loaded from {checkpoint_path}")
    else:
        print(f"[v4] WARNING: Checkpoint not found at {checkpoint_path}. "
              f"Using untrained model with physics proxy DI.")
    model.eval()
    return model


def query_fields_v4(model, t_s, params, n_r=10, n_z=15):
    """
    Query PINN v4 over a (r,z) grid at time t_s.
    Returns dict with 2D arrays for each field.

    Args:
        model:   MultiPhysicsPINNV4 instance
        t_s:     time in seconds
        params:  dict with all process parameters
        n_r:     radial grid points
        n_z:     axial grid points

    Returns dict:
        T_C:       temperature in °C  (n_r, n_z)
        f_l:       liquid fraction    (n_r, n_z)
        phi:       phase field        (n_r, n_z)
        sigma_eq:  von Mises stress   (n_r, n_z)
        T_env_C:   current env temp °C (scalar)
        in_pulse:  bool — is hot-air pulse active?
        Bi:        current Biot number (scalar)
    """
    from train_pinn_physics_enforced_v4 import GEO, PROCESS

    r_lin = np.linspace(0, GEO.R, n_r)
    z_lin = np.linspace(0, GEO.H, n_z)
    RG, ZG = np.meshgrid(r_lin, z_lin, indexing='ij')  # (n_r, n_z)

    R_flat = RG.ravel()
    Z_flat = ZG.ravel()
    N      = len(R_flat)

    def bc(v):
        return torch.full((N, 1), float(v), dtype=torch.float32)

    r_t = torch.tensor(R_flat, dtype=torch.float32).unsqueeze(1)
    z_t = torch.tensor(Z_flat, dtype=torch.float32).unsqueeze(1)
    t_t = torch.full((N, 1), float(t_s))

    with torch.no_grad():
        out = model.predict_fields(
            r_t, z_t, t_t,
            bc(params['T_fill']),         bc(params['T_cool']),
            bc(params['T_reheat']),        bc(params['T_cold']),
            bc(params['h_cool']),          bc(params['h_hot']),
            bc(params['t_pulse_start']),   bc(params['t_pulse_dur']),
            bc(params['t_pulse_interval']),bc(params['n_pulses']),
            bc(params['t_cold_start']),    bc(params['T_container']),
            bc(params.get('viscosity', 0.3)),
            bc(params.get('humidity',  0.4)),
            bc(params.get('Gc',       80.0)),
        )

    T_K   = out['T'].squeeze().numpy().reshape(n_r, n_z)
    fl    = out['f_l'].squeeze().numpy().reshape(n_r, n_z)
    phi   = out['phi'].squeeze().numpy().reshape(n_r, n_z)
    sig   = out['sigma_eq'].squeeze().numpy().reshape(n_r, n_z)

    # Current environmental conditions
    t_tens = torch.tensor([t_s])
    T_env_K = PROCESS.compute_T_env(
        t_tens,
        torch.tensor([params['T_fill']]),
        torch.tensor([params['T_cool']]),
        torch.tensor([params['T_reheat']]),
        torch.tensor([params['T_cold']]),
        torch.tensor([params['t_pulse_start']]),
        torch.tensor([params['t_pulse_dur']]),
        torch.tensor([params['t_pulse_interval']]),
        params['n_pulses'],
        torch.tensor([params['t_cold_start']]),
    ).item()

    # Check if in pulse
    in_pulse = False
    for i in range(int(params['n_pulses'])):
        on  = params['t_pulse_start'] + i * params['t_pulse_interval']
        off = on + params['t_pulse_dur']
        if on <= t_s < off:
            in_pulse = True
            break

    h_now = params['h_hot'] if in_pulse else params['h_cool']
    from train_pinn_physics_enforced_v4 import MATERIAL
    Bi    = h_now * GEO.R / MATERIAL.k_thermal

    return {
        'T_C':      T_K - 273.15,
        'f_l':      fl,
        'phi':      phi,
        'sigma_eq': sig,
        'T_env_C':  T_env_K - 273.15,
        'in_pulse': in_pulse,
        'Bi':       Bi,
        'R_grid':   RG,
        'Z_grid':   ZG,
    }


def compute_DI_v4(fields, params):
    """
    Compute Damage Indicator from v4 field output.
    Uses PINN phi if available, else Biot-proxy.
    """
    phi_max = float(np.max(fields['phi']))
    if phi_max > 0.02:
        phi_mushy = fields['phi'][fields['f_l'] > 0.05]
        if len(phi_mushy) > 0:
            DI = float(np.clip(
                0.35*np.max(phi_mushy) + 0.25*np.percentile(fields['phi'], 90)
                + 0.25*np.mean(phi_mushy) + 0.15*phi_max,
                0, 1
            ))
        else:
            DI = phi_max
        return DI

    # Biot proxy (fallback when PINN not trained)
    Bi      = fields['Bi']
    T_surf  = float(np.mean(fields['T_C'][-1, :]))  # outer edge
    T_core  = float(np.mean(fields['T_C'][0,  :]))  # center
    T_env   = fields['T_env_C']
    T_fill  = params['T_fill'] - 273.15
    fl_mean = float(np.mean(fields['f_l']))

    Bi_risk  = float(np.clip((Bi - 0.20) / 0.70, 0.0, 1.0))
    dT_drive = max(0.0, T_fill - T_env)
    cool_sev = float(np.clip(dT_drive / 60.0, 0.0, 1.0))

    if fl_mean > 0.90:
        phase_w = 0.55
    elif fl_mean > 0.10:
        phase_w = float(np.clip(1.0 - 1.6*abs(fl_mean - 0.5), 0.3, 1.0))
    else:
        phase_w = 0.50

    T_avg    = (T_surf + T_core) / 2.0
    mushy_px = float(np.exp(-((T_avg - 67.0)**2) / (5.5**2)))

    DI = (
        0.45 * Bi_risk * cool_sev * phase_w +
        0.25 * Bi_risk * phase_w +
        0.20 * mushy_px * Bi_risk +
        0.10 * Bi_risk
    )
    return float(np.clip(DI * 1.20, 0.0, 1.0))


def compute_timeseries_v4(model, params, n_pts=50):
    """
    Compute full time-series of field observables over the process.
    Compatible with fig_combined_charts in Streamlit dashboard.

    Returns dict ready for fig_combined_charts(ts, label).
    """
    from train_pinn_physics_enforced_v4 import T_CYCLE, MATERIAL

    total_min = min((params['t_cold_start'] + 300) / 60.0, 35.0)
    times_min = np.linspace(0, total_min, n_pts)

    T_surf, T_core, T_env_l, dT, dTdt, DI_list = [], [], [], [], [], []
    prev_Ts  = params['T_fill'] - 273.15
    dt       = float(times_min[1] - times_min[0]) if n_pts > 1 else 1.0

    for tm in times_min:
        t_s    = tm * 60.0
        fields = query_fields_v4(model, t_s, params, n_r=8, n_z=10)
        Ts     = float(np.mean(fields['T_C'][-1, :]))  # surface
        Tc     = float(np.mean(fields['T_C'][0,  :]))  # core
        Te     = fields['T_env_C']
        DI_val = compute_DI_v4(fields, params)

        T_surf.append(Ts)
        T_core.append(Tc)
        T_env_l.append(Te)
        dT.append(Tc - Ts)
        dTdt.append((Ts - prev_Ts) / max(dt, 0.01))
        DI_list.append(DI_val)
        prev_Ts = Ts

    t_zone1_min = params['t_pulse_start'] / 60.0
    t_zone2_min = (params['t_cold_start'] - params['t_pulse_start']) / 60.0

    return {
        'times':  times_min.tolist(),
        'T_surf': T_surf,
        'T_core': T_core,
        'T_env':  T_env_l,
        'dT':     dT,
        'dTdt':   dTdt,
        'DI':     DI_list,
        'z1m':    t_zone1_min,
        'z2m':    t_zone2_min,
        'T_fill': params['T_fill'] - 273.15,
        # v4 extras
        'n_pulses':       int(params['n_pulses']),
        't_pulse_dur_s':  params['t_pulse_dur'],
        't_pulse_int_s':  params['t_pulse_interval'],
        't_cold_start_m': params['t_cold_start'] / 60.0,
        'T_reheat_C':     params['T_reheat'] - 273.15,
    }


# ============================================================================
# STANDALONE PINN V4 (for import when training file unavailable)
# ============================================================================

class MultiPhysicsPINNV4Standalone(torch.nn.Module):
    """
    Standalone v4 PINN for inference.
    Mirrors architecture from train_pinn_physics_enforced_v4.py exactly.
    """

    class _ThermalNet(torch.nn.Module):
        def __init__(self, hidden=256, n_layers=7):
            super().__init__()
            L = [torch.nn.Linear(12, hidden), torch.nn.Tanh()]
            for _ in range(n_layers - 1):
                L += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
            L += [torch.nn.Linear(hidden, 1), torch.nn.Sigmoid()]
            self.net = torch.nn.Sequential(*L)
        def forward(self, *args):
            return self.net(torch.cat(list(args), dim=1))

    class _PhaseNet(torch.nn.Module):
        def __init__(self, hidden=256, n_layers=7):
            super().__init__()
            L = [torch.nn.Linear(9, hidden), torch.nn.Tanh()]
            for _ in range(n_layers - 1):
                L += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
            L += [torch.nn.Linear(hidden, 1), torch.nn.Sigmoid()]
            self.net = torch.nn.Sequential(*L)
        def forward(self, *args):
            return self.net(torch.cat(list(args), dim=1))

    class _MechNet(torch.nn.Module):
        def __init__(self, hidden=192, n_layers=6):
            super().__init__()
            L = [torch.nn.Linear(8, hidden), torch.nn.Tanh()]
            for _ in range(n_layers - 1):
                L += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
            L += [torch.nn.Linear(hidden, 6)]
            self.net = torch.nn.Sequential(*L)
        def forward(self, *args):
            out = self.net(torch.cat(list(args), dim=1))
            return {
                'sigma_rr':     0.5 * torch.tanh(out[:, 0:1]),
                'sigma_zz':     0.5 * torch.tanh(out[:, 1:2]),
                'sigma_theta':  0.5 * torch.tanh(out[:, 2:3]),
                'sigma_rz':     0.3 * torch.tanh(out[:, 3:4]),
                'sigma_rtheta': 0.2 * torch.tanh(out[:, 4:5]),
                'sigma_ztheta': 0.2 * torch.tanh(out[:, 5:6]),
            }

    GEO_R = 0.0125
    GEO_H = 0.04

    def __init__(self):
        super().__init__()
        self.thermal_net    = self._ThermalNet()
        self.phasefield_net = self._PhaseNet()
        self.mechanical_net = self._MechNet()

    def _bc(self, v, ref):
        n = ref.shape[0]
        if not torch.is_tensor(v):
            return torch.full((n,1), float(v), dtype=ref.dtype, device=ref.device)
        if v.numel() == 1:
            return torch.full((n,1), v.item(), dtype=ref.dtype, device=ref.device)
        return v if v.dim() == 2 else v.unsqueeze(1)

    def predict_fields(self, r, z, t,
                       T_fill, T_cool, T_reheat, T_cold,
                       h_cool, h_hot,
                       t_pulse_start, t_pulse_dur, t_pulse_interval,
                       n_pulses, t_cold_start,
                       T_container, viscosity, humidity, Gc):
        bc = self._bc

        # Normalize inputs
        r_n  = r / self.GEO_R
        z_n  = z / self.GEO_H
        t_n  = t / 1800.0
        Tf_n = (bc(T_fill, r)   - 296.15) / 80.0
        Tc_n = (bc(T_cool, r)   - 296.15) / 80.0
        Tr_n = (bc(T_reheat, r) - 296.15) / 80.0
        Td_n = (bc(T_cold, r)   - 296.15) / 80.0
        hc_n = bc(h_cool, r)  / 25.0
        hh_n = bc(h_hot, r)   / 60.0
        tps_n= bc(t_pulse_start, r)    / 1800.0
        tpd_n= bc(t_pulse_dur, r)      / 1800.0
        tpi_n= bc(t_pulse_interval, r) / 1800.0

        T_raw = self.thermal_net(r_n, z_n, t_n, Tf_n, Tc_n, Tr_n, Td_n,
                                  hc_n, hh_n, tps_n, tpd_n, tpi_n)

        # Physical temperature reconstruction
        Tc_t   = bc(T_container, r)
        Tf_t   = bc(T_fill, r)
        tcs_t  = bc(t_cold_start, r)
        tau    = torch.clamp(tcs_t * 0.5 + 150.0, 120.0, 1200.0)
        cf     = torch.exp(-t / tau)
        T_range= (Tf_t - Tc_t) * (z / self.GEO_H) * cf
        T      = Tc_t + T_range * T_raw
        T      = torch.maximum(T, Tc_t - 5.0)
        T      = torch.minimum(T, Tf_t + 5.0)

        # Liquid fraction
        T_sol = 335.15; T_liq = 345.15; T_mid = 340.15
        liq = T >= T_liq; sol = T <= T_sol; mush = ~liq & ~sol
        fl  = torch.zeros_like(T)
        fl[liq] = 1.0
        T_m = T[mush]
        fl[mush] = 0.5*(1 + torch.sin(
            torch.pi*(T_m - T_mid)/(T_liq - T_sol)))

        # Mushy zone driving force
        mw    = torch.clamp(fl*(1-fl)*4.0, 0, 1)
        T_prx = torch.clamp(1 - torch.abs(T - T_mid)/(T_liq - T_sol), 0, 1)
        H_n   = mw * T_prx * torch.clamp(t_n*2.0, 0, 1)

        # Healing driver
        heal_drv = torch.nn.functional.relu(T - T_sol) / 10.0

        Gc_n = bc(Gc, r) / 200.0
        vi_t = bc(viscosity, r)

        phi = self.phasefield_net(r_n, z_n, t_n, T_raw, fl, H_n,
                                   Gc_n, vi_t, heal_drv)

        hu_t = bc(humidity, r)
        mech = self.mechanical_net(r_n, z_n, t_n, T_raw, fl, phi, vi_t, hu_t)

        s_rr = mech['sigma_rr']; s_zz = mech['sigma_zz']
        s_th = mech['sigma_theta']; s_rz = mech['sigma_rz']
        sigma_eq = torch.sqrt(0.5*(
            (s_rr-s_zz)**2+(s_zz-s_th)**2+(s_th-s_rr)**2+6*s_rz**2)+1e-8)

        # Simple T_env approximation
        T_env_now = bc(T_cool, r)

        return {
            'T': T, 'f_l': fl, 'phi': phi,
            'H_norm': H_n, 'T_env': T_env_now,
            'heal_driver': heal_drv,
            'sigma_rr': s_rr, 'sigma_zz': s_zz,
            'sigma_theta': s_th, 'sigma_rz': s_rz,
            'sigma_eq': sigma_eq,
            'sigma_crit': torch.ones_like(phi)*0.35,
            'Gc_eff': bc(Gc, r)*0.85, 'g_phi': (1-phi)**2,
        }
