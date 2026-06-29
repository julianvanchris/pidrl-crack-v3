"""
crack_predictor_v4.py

Damage assessment for the pulsed hot-air reheating process (v4).
Extended from v2 to handle:
  - Healing: φ can DECREASE during hot-air pulses → DI should reflect recovery
  - Sink-mark indicator: volume deficit at z=H during mushy→solid
  - Pulse timing: DI evolves non-monotonically (spikes → drops → settles)
  - Production timeline from CLIENT_Data_0427.xlsx

Damage Indicator (DI) scale:
  0.00 – 0.10: SAFE      — no visible defects
  0.10 – 0.25: CAUTION   — minor surface imperfections possible
  0.25 – 0.50: WARNING   — sink marks / craters likely
  0.50 – 0.80: CRITICAL  — severe cracking, production reject
  0.80 – 1.00: FAILURE   — immediate reject, process halt

Sink-mark Indicator (SI):
  Separate from DI. Measures shrinkage-driven surface deficit.
  SI > 0.3 → visible ヒケ (sink marks) expected at top surface.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================================
# MAIN DI COMPUTATION
# ============================================================================

def compute_damage_v4(phi, f_l, R_mesh, Z_mesh,
                      r_injection, z_injection,
                      heal_driver=None, in_pulse=False):
    """
    Compute Damage Indicator (DI) from phase-field + liquid fraction.

    Args:
        phi:          Phase field array (0=intact, 1=cracked)
        f_l:          Liquid fraction array (0=solid, 1=liquid)
        R_mesh:       Radial coordinate mesh (m)
        Z_mesh:       Vertical coordinate mesh (m)
        r_injection:  Injection radial position (m) — center for back-fill
        z_injection:  Injection height (m) — z=H for top-fill, z=0 for back-fill
        heal_driver:  Healing activation array (ReLU(T-T_sol)/10), optional
        in_pulse:     True if hot-air pulse is currently active

    Returns:
        DI:   float in [0, 1]
        SI:   float in [0, 1] — sink-mark indicator
        info: dict with component breakdown
    """

    # ── COMPONENT 1: Mushy zone damage ────────────────────────────────────
    mushy_mask = (f_l > 0.05) & (f_l < 0.95)
    if mushy_mask.sum() > 0:
        phi_mushy     = phi[mushy_mask]
        DI_mushy_mean = float(np.mean(phi_mushy))
        DI_mushy_max  = float(np.max(phi_mushy))
    else:
        DI_mushy_mean = float(np.mean(phi))
        DI_mushy_max  = float(np.max(phi))

    # ── COMPONENT 2: Top-surface damage (sink-mark zone) ─────────────────
    # Top 20% of height (z > 0.8H) — where sink marks form
    H_max      = Z_mesh.max()
    top_mask   = Z_mesh >= 0.80 * H_max
    top_solid  = top_mask & (f_l < 0.5)   # solidifying top region

    if top_mask.sum() > 0:
        DI_top = float(np.mean(phi[top_mask]))
    else:
        DI_top = 0.0

    # ── COMPONENT 3: Spatial weighting (proximity to fill point) ─────────
    dist = np.sqrt((R_mesh - r_injection)**2 + (Z_mesh - z_injection)**2)
    char = np.sqrt(R_mesh.max()**2 + H_max**2)
    prox = np.exp(-3.0 * (dist / (char + 1e-8))**2)
    DI_spatial = float(np.mean(phi * prox))

    # ── COMPONENT 4: Crack extent (volume fraction) ────────────────────────
    crack_fraction = float((phi > 0.3).sum() / phi.size)

    # ── COMPONENT 5: Top 10% worst regions ───────────────────────────────
    threshold_90 = float(np.percentile(phi.flatten(), 90))
    high_mask    = phi.flatten() >= threshold_90
    DI_high      = float(np.mean(phi.flatten()[high_mask])) if high_mask.sum() > 0 else DI_mushy_max

    # ── HEALING CORRECTION ─────────────────────────────────────────────────
    # During hot-air pulse: healing driver active → reduce DI contribution
    heal_reduction = 0.0
    if in_pulse and heal_driver is not None:
        # Mean healing driver over mushy/cracked zone
        cracked_mask = phi > 0.15
        if cracked_mask.sum() > 0 and heal_driver.shape == phi.shape:
            hd_cracked  = heal_driver[cracked_mask]
            heal_reduction = float(np.clip(np.mean(hd_cracked) * 0.4, 0, 0.3))
        elif heal_driver is not None:
            hd_val = float(np.mean(heal_driver)) if hasattr(heal_driver, '__len__') else float(heal_driver)
            heal_reduction = float(np.clip(hd_val * 0.3, 0, 0.25))

    # ── COMBINED DI ────────────────────────────────────────────────────────
    DI_raw = (
        0.30 * DI_mushy_max   +   # peak mushy zone damage (critical)
        0.25 * DI_high        +   # top 10% worst regions
        0.20 * DI_top         +   # top surface (sink-mark zone)
        0.15 * DI_mushy_mean  +   # mean mushy zone
        0.05 * DI_spatial     +   # proximity weighted
        0.05 * crack_fraction * 2.0   # crack extent
    )

    DI = float(np.clip(DI_raw - heal_reduction, 0.0, 1.0))

    # ── SINK-MARK INDICATOR ────────────────────────────────────────────────
    # High phi at top + material solidifying (shrinking) → sink mark
    if top_solid.sum() > 0:
        phi_top_solid = phi[top_solid]
        fl_top_solid  = f_l[top_solid]
        # Shrinkage drive: peaks when fl ≈ 0.3 (most dangerous solidification stage)
        shrink_weight = (1.0 - fl_top_solid) * np.exp(-2.0 * fl_top_solid)
        SI = float(np.clip(np.mean(phi_top_solid * shrink_weight) * 3.0, 0, 1))
    elif top_mask.sum() > 0:
        SI = float(np.clip(DI_top * 1.5, 0, 1))
    else:
        SI = float(np.clip(DI_raw * 0.5, 0, 1))

    info = {
        'DI_mushy_max':  DI_mushy_max,
        'DI_mushy_mean': DI_mushy_mean,
        'DI_top':        DI_top,
        'DI_spatial':    DI_spatial,
        'DI_high':       DI_high,
        'crack_fraction':crack_fraction,
        'heal_reduction':heal_reduction,
        'in_pulse':      in_pulse,
        'SI':            SI,
    }

    return DI, SI, info


# ============================================================================
# RISK LEVEL ASSESSMENT
# ============================================================================

def assess_risk_v4(DI, phi_max, SI=0.0):
    """
    Categorize crack + sink-mark risk.

    Returns:
        risk_level:     str ("SAFE" / "CAUTION" / "WARNING" / "CRITICAL" / "FAILURE")
        recommendation: actionable advice including hot-air pulse optimization
    """
    # Override with sink-mark check
    effective_DI = max(DI, SI * 0.6)

    if DI >= 0.80 or phi_max >= 0.75:
        level = "FAILURE"
        rec = (
            "Process halt recommended. Severe cracking detected. "
            "Actions: (1) Increase T_reheat to 95–105°C. "
            "(2) Add 2–3 more hot-air pulses early in process (t < 2 min). "
            "(3) Extend t_pulse_dur to 30–45 sec. "
            "(4) Reduce h_cool to ≤ 8 W/m²K. "
            "(5) Increase t_zone1 to ≥ 8 min before cold air phase. "
            "→ 処理停止推奨。深刻なクラック検出。"
            "T_reheat 95〜105°C、早期パルス追加、h_cool削減が必要。"
        )

    elif DI >= 0.50 or phi_max >= 0.45:
        level = "CRITICAL"
        rec = (
            "High crack + sink-mark risk. Immediate parameter adjustment needed. "
            "Actions: (1) Start hot-air pulses earlier (reduce t_pulse_start by 30–60 sec). "
            "(2) Increase T_reheat by 5–10°C (target: 95–100°C). "
            "(3) Increase n_pulses by 2. "
            "(4) Verify h_cool ≤ 10 W/m²K during cooling phase. "
            "→ 高リスク。パルス開始タイミングを早め、T_reheatを上げ、パルス数を増やしてください。"
        )

    elif DI >= 0.25 or effective_DI >= 0.20:
        level = "WARNING"
        rec = (
            "Moderate crack/sink-mark risk detected. Process adjustment recommended. "
            "Actions: (1) Fine-tune t_pulse_interval (reduce by 15–30 sec). "
            "(2) Consider extending t_pulse_dur by 5–10 sec. "
            "(3) Check that T_env1 ≥ 55°C during initial cooling. "
            f"Sink-mark indicator: SI={SI:.2f} {'(visible ヒケ likely)' if SI > 0.3 else '(minor)'}. "
            "→ 中程度のリスク。パルス間隔の最適化とゾーン温度の確認が必要。"
        )

    elif DI >= 0.10:
        level = "CAUTION"
        rec = (
            "Low-moderate risk. Process is acceptable but optimization possible. "
            "Actions: Consider reducing n_pulses by 1 to save energy, "
            "or reducing t_pulse_dur by 5 sec. "
            "Monitor for surface quality at end of batch. "
            "→ 軽微なリスク。プロセスは許容範囲内。エネルギー節約のためパルス最適化を検討。"
        )

    else:
        level = "SAFE"
        rec = (
            "Excellent process control. Crack and sink-mark risk minimal. "
            "Current hot-air pulse strategy is effective. "
            f"Healing events successfully reduced φ. SI={SI:.3f} ≈ 0 (no ヒケ expected). "
            "Consider cycle time optimization while maintaining quality. "
            "→ 優良。クラック・ヒケリスク最小。現在の温風パルス戦略は効果的です。"
        )

    return level, rec


# ============================================================================
# CRACK PATTERN ANALYSIS
# ============================================================================

def analyze_crack_pattern_v4(phi, f_l, R_mesh, Z_mesh,
                              heal_driver=None, pulse_times=None):
    """
    Spatial analysis of crack pattern including healing zones.

    Returns dict with:
        crack_centroid:    (r, z) damage center of mass
        crack_extent_r:    radial extent (m)
        crack_extent_z:    axial extent (m)
        mushy_overlap:     fraction of cracks in mushy zone
        top_concentration: fraction of cracks in top 20% (sink-mark risk)
        healing_zone:      fraction of cracked area where healing is active
        max_phi_location:  (r, z) of peak damage
    """
    crack_mask = phi > 0.20

    if crack_mask.sum() == 0:
        return {
            'crack_centroid':     (0.0, 0.0),
            'crack_extent_r':     0.0,
            'crack_extent_z':     0.0,
            'mushy_overlap':      0.0,
            'top_concentration':  0.0,
            'healing_zone':       0.0,
            'max_phi_location':   (0.0, 0.0),
            'phi_smooth_max':     float(phi.max()),
        }

    r_c = R_mesh[crack_mask]
    z_c = Z_mesh[crack_mask]
    w_c = phi[crack_mask]

    r_centroid  = float(np.average(r_c, weights=w_c))
    z_centroid  = float(np.average(z_c, weights=w_c))
    r_extent    = float(r_c.max() - r_c.min())
    z_extent    = float(z_c.max() - z_c.min())

    # Mushy zone overlap
    mushy_mask   = (f_l > 0.05) & (f_l < 0.95)
    mushy_overlap = float((crack_mask & mushy_mask).sum() / crack_mask.sum())

    # Top concentration (z > 0.8 H)
    H_max = Z_mesh.max()
    top_mask  = Z_mesh >= 0.80 * H_max
    top_conc  = float((crack_mask & top_mask).sum() / crack_mask.sum())

    # Healing zone (where heal_driver > 0.1 AND phi > 0.2)
    healing_zone = 0.0
    if heal_driver is not None and heal_driver.shape == phi.shape:
        active_heal  = heal_driver > 0.1
        heal_frac    = float((crack_mask & active_heal).sum() / crack_mask.sum())
        healing_zone = heal_frac

    # Peak damage location
    idx_max = np.unravel_index(np.argmax(phi), phi.shape)
    max_loc = (float(R_mesh[idx_max]), float(Z_mesh[idx_max]))

    # Smoothed phi for visualization
    phi_smooth = gaussian_filter(phi, sigma=1.2)

    return {
        'crack_centroid':     (r_centroid, z_centroid),
        'crack_extent_r':     r_extent,
        'crack_extent_z':     z_extent,
        'mushy_overlap':      mushy_overlap,
        'top_concentration':  top_conc,
        'healing_zone':       healing_zone,
        'max_phi_location':   max_loc,
        'phi_smooth_max':     float(phi_smooth.max()),
    }


# ============================================================================
# PULSE TIMELINE ANALYSIS
# ============================================================================

def analyze_pulse_effectiveness(DI_timeseries, times_min, params):
    """
    Analyze how effective each hot-air pulse was at reducing DI/φ.

    Args:
        DI_timeseries: list of DI values over time
        times_min:     list of times in minutes
        params:        process parameters dict

    Returns dict with per-pulse effectiveness metrics.
    """
    n       = int(params.get('n_pulses', 5))
    tps     = params.get('t_pulse_start', 35) / 60.0       # min
    tpd     = params.get('t_pulse_dur', 20) / 60.0         # min
    tpi     = params.get('t_pulse_interval', 88) / 60.0    # min
    times   = np.array(times_min)
    DI_arr  = np.array(DI_timeseries)

    def get_DI_at(t_target):
        idx = int(np.argmin(np.abs(times - t_target)))
        return float(DI_arr[idx])

    pulse_results = []
    for i in range(n):
        t_on   = tps + i * tpi
        t_off  = t_on + tpd
        t_after= t_off + tpd   # measure effect 1 pulse-duration later

        DI_before = get_DI_at(max(t_on - tpd, 0))
        DI_during = get_DI_at((t_on + t_off) / 2)
        DI_after  = get_DI_at(min(t_after, times[-1]))

        effectiveness = float(np.clip(DI_before - DI_after, -0.5, 0.5))

        pulse_results.append({
            'pulse_number':    i + 1,
            't_start_min':     float(t_on),
            't_end_min':       float(t_off),
            'DI_before':       DI_before,
            'DI_during':       DI_during,
            'DI_after':        DI_after,
            'effectiveness':   effectiveness,
            'healed':          effectiveness > 0.02,
        })

    total_reduction = float(DI_arr[0] - DI_arr[-1]) if len(DI_arr) > 1 else 0
    pulses_effective = sum(1 for p in pulse_results if p['healed'])

    return {
        'pulses':              pulse_results,
        'total_DI_reduction':  total_reduction,
        'pulses_effective':    pulses_effective,
        'pulses_total':        n,
        'effectiveness_pct':   float(pulses_effective / max(n, 1) * 100),
    }


# ============================================================================
# PROCESS SUMMARY
# ============================================================================

def summarize_process_v4(ts_dict, params, DI_final):
    """
    Generate a human-readable process summary for display.
    Compatible with Streamlit dashboard Results tab.

    Args:
        ts_dict:   timeseries dict from compute_timeseries_v4
        params:    process parameters dict
        DI_final:  final DI value

    Returns dict with formatted metrics for display.
    """
    times   = np.array(ts_dict['times'])
    DI_arr  = np.array(ts_dict['DI'])
    T_surf  = np.array(ts_dict['T_surf'])
    T_core  = np.array(ts_dict['T_core'])

    # Time in mushy zone (62–72°C)
    in_mushy = ((T_surf >= 62) & (T_surf <= 72))
    dt       = float(times[1] - times[0]) if len(times) > 1 else 0
    t_mushy_min = float(in_mushy.sum() * dt)

    # Cooling rate (initial)
    if len(T_surf) > 2:
        dT_dt = float((T_surf[1] - T_surf[0]) / max(dt, 0.01))
    else:
        dT_dt = 0.0

    # Max radial gradient
    dT_max = float(np.max(np.abs(np.array(ts_dict['dT']))))

    # Biot
    h_now = params['h_hot'] if params.get('n_pulses', 0) > 0 else params['h_cool']
    Bi    = h_now * 0.0125 / 0.25

    risk_level, _ = assess_risk_v4(DI_final, DI_final)

    total_t_min = params.get('t_cold_start', 1278) / 60.0 + 3.0  # rough total

    return {
        'risk_level':        risk_level,
        'DI_final':          float(DI_final),
        'DI_peak':           float(DI_arr.max()),
        'DI_min':            float(DI_arr.min()),
        'T_fill_C':          float(params['T_fill'] - 273.15),
        'T_reheat_C':        float(params['T_reheat'] - 273.15),
        'T_cool_C':          float(params['T_cool'] - 273.15),
        'T_cold_C':          float(params['T_cold'] - 273.15),
        'h_cool':            float(params['h_cool']),
        'h_hot':             float(params['h_hot']),
        'n_pulses':          int(params['n_pulses']),
        't_pulse_start_min': float(params['t_pulse_start'] / 60),
        't_pulse_dur_sec':   float(params['t_pulse_dur']),
        't_pulse_int_sec':   float(params['t_pulse_interval']),
        't_cold_start_min':  float(params['t_cold_start'] / 60),
        'total_time_min':    float(total_t_min),
        'within_30min':      total_t_min <= 30.0,
        'Biot_number':       float(Bi),
        'Biot_safe':         Bi <= 0.5,
        't_in_mushy_min':    float(t_mushy_min),
        'cooling_rate':      float(dT_dt),
        'dT_max_radial':     float(dT_max),
    }
