"""
crack_synthesis_v4.py

Synthetic crack + healing visualization for the pulsed hot-air process (v4).

Key differences from v2:
  - Cracks can HEAL (φ decreases) during hot-air pulses → crack markers fade
  - Sink-mark indicator drives surface dimple visualization at z=H
  - Pulse-timing-aware: crack density peaks in mushy zone traversal,
    then reduces during and after each hot-air pulse
  - Two distinct defect types:
      1. Volumetric cracks: branching ✕ marks in the bulk (from thermal stress)
      2. Surface sink marks: concentrated at top surface (from shrinkage)

Usage:
    from crack_synthesis_v4 import synthesize_cracks_v4, synthesize_sink_marks

    crack_field, sink_field = synthesize_cracks_v4(
        phi_field, stress_field, f_l,
        R_mesh, Z_mesh,
        injection_point=(0.0, H),   # back-fill: (0,0), front-fill: (0,H)
        severity=0.6,
        in_pulse=False,
        heal_driver=None,
    )
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================================
# MAIN CRACK + SINK SYNTHESIS
# ============================================================================

def synthesize_cracks_v4(phi_field, stress_field, f_l,
                          R_mesh, Z_mesh,
                          injection_point=(0.0, 0.04),
                          severity=0.5,
                          in_pulse=False,
                          heal_driver=None,
                          n_branches=5,
                          branch_probability=0.25,
                          seed=42):
    """
    Generate realistic crack patterns calibrated to the pulsed reheat process.

    During hot-air pulse (in_pulse=True):
      - New crack generation is suppressed (healing dominates)
      - Existing cracks fade toward center (surface healing)
      - net crack field = max(phi_field - heal_reduction, 0)

    Args:
        phi_field:         PINN phase field (0-1)
        stress_field:      Von Mises stress field
        f_l:               Liquid fraction
        R_mesh, Z_mesh:    Spatial coordinates
        injection_point:   (r, z) of material injection
        severity:          Overall crack severity (0-1, from DI)
        in_pulse:          True if hot-air pulse currently active
        heal_driver:       Array of healing activation (same shape as phi)
        n_branches:        Number of primary crack branches
        branch_probability:Probability of secondary branching
        seed:              Random seed for reproducibility

    Returns:
        synthetic_phi:     Enhanced crack field (0-1)
        sink_field:        Sink-mark field at top surface (0-1, only at z≈H)
    """
    np.random.seed(seed)
    shape   = phi_field.shape
    r_inj, z_inj = injection_point

    # ── Vulnerability map ──────────────────────────────────────────────────
    mushy  = (f_l > 0.15) & (f_l < 0.90)
    stress_thr = np.percentile(stress_field, 65)
    high_s = stress_field > stress_thr

    vuln = np.zeros_like(phi_field)
    vuln[mushy]  += 0.55
    vuln[high_s] += 0.45
    vuln += phi_field * 1.8
    vuln  = np.clip(vuln, 0, 1)
    vuln  = gaussian_filter(vuln, sigma=1.2)

    # Injection proximity weight (back-fill: center-bottom, front-fill: center-top)
    dist_inj = np.sqrt((R_mesh - r_inj)**2 + (Z_mesh - z_inj)**2)
    char_len  = np.sqrt(R_mesh.max()**2 + Z_mesh.max()**2)
    prox_wt   = np.exp(-2.5 * (dist_inj / (char_len + 1e-8))**2)

    # ── Healing suppression ────────────────────────────────────────────────
    heal_suppress = np.zeros_like(phi_field)
    if in_pulse:
        # During pulse: heal_driver drives crack closure
        if heal_driver is not None and heal_driver.shape == phi_field.shape:
            heal_suppress = np.clip(heal_driver * 0.6, 0, 0.5)
        else:
            # Approximate: surface heals, core lags
            # Surface = outer ring in R dimension
            surface_mask = R_mesh >= (R_mesh.max() * 0.75)
            heal_suppress[surface_mask] = 0.35
            heal_suppress = gaussian_filter(heal_suppress, sigma=1.5)

    # ── Generate crack field ───────────────────────────────────────────────
    if severity < 0.05:
        crack_field = phi_field * 0.5
    else:
        # Find injection point in grid
        dist_grid = np.sqrt((R_mesh - r_inj)**2 + (Z_mesh - z_inj)**2)
        inj_idx   = np.unravel_index(dist_grid.argmin(), shape)

        crack_field = np.zeros_like(phi_field)

        # Suppress crack generation during healing pulse
        effective_severity = severity * (0.3 if in_pulse else 1.0)

        if effective_severity > 0.05:
            n_cracks = max(2, int(n_branches * (0.4 + effective_severity)))
            angles   = np.linspace(0, 2*np.pi, n_cracks, endpoint=False)

            for angle in angles:
                dir_r = np.cos(angle) + np.random.normal(0, 0.15)
                dir_z = np.sin(angle) + np.random.normal(0, 0.15)
                mag   = np.sqrt(dir_r**2 + dir_z**2) + 1e-8
                dir_r /= mag; dir_z /= mag

                path = _propagate_v4(
                    start_idx=inj_idx,
                    direction=(dir_r, dir_z),
                    vulnerability=vuln,
                    R_mesh=R_mesh, Z_mesh=Z_mesh,
                    max_length=int(min(shape) * 0.75),
                    severity=effective_severity,
                    branch_prob=branch_probability * (0.5 if in_pulse else 1.0),
                )
                for (i, j), intensity in path:
                    if 0 <= i < shape[0] and 0 <= j < shape[1]:
                        crack_field[i, j] = max(crack_field[i, j], intensity)

        # Secondary cracks in high-stress regions
        if effective_severity > 0.35:
            sec = _secondary_cracks_v4(
                vuln, stress_field, int(4 * effective_severity),
                R_mesh, Z_mesh, seed=seed+1
            )
            crack_field = np.maximum(crack_field, sec)

    # Smooth for natural appearance
    crack_field = gaussian_filter(crack_field, sigma=0.7)
    crack_field = crack_field * (0.25 + 0.75 * vuln) * prox_wt
    crack_field += phi_field * 0.4
    crack_field  = np.clip(crack_field, 0, 1)

    # Apply healing reduction
    crack_field = np.clip(crack_field - heal_suppress, 0, 1)

    # ── Sink-mark field ────────────────────────────────────────────────────
    sink_field = _synthesize_sink_marks(
        phi_field, f_l, R_mesh, Z_mesh, severity, seed=seed+2
    )

    return crack_field, sink_field


def _propagate_v4(start_idx, direction, vulnerability, R_mesh, Z_mesh,
                  max_length=80, severity=0.5, branch_prob=0.25):
    """
    Propagate a crack path with physics-guided direction update.
    Same as v2 but with severity-scaled intensity for healing periods.
    """
    path = []
    current_idx = start_idx
    current_dir = np.array(direction, dtype=float)
    intensity   = 0.75 + 0.25 * severity

    for step in range(max_length):
        i, j = current_idx
        path.append(((i, j), intensity))
        intensity *= 0.994

        if intensity < 0.08:
            break

        # Gradient of vulnerability → cracks follow high-vuln regions
        di = dj = 0.0
        if 0 < i < vulnerability.shape[0] - 1:
            di = (vulnerability[i+1, j] - vulnerability[i-1, j]) / 2.0
        if 0 < j < vulnerability.shape[1] - 1:
            dj = (vulnerability[i, j+1] - vulnerability[i, j-1]) / 2.0

        grad_mag   = np.sqrt(di**2 + dj**2) + 1e-8
        current_dir = 0.72 * current_dir + 0.28 * np.array([dj, di]) / grad_mag
        current_dir += np.random.normal(0, 0.12, size=2)
        current_dir /= (np.linalg.norm(current_dir) + 1e-8)

        next_i = int(i + current_dir[1] * 1.4)
        next_j = int(j + current_dir[0] * 1.4)

        if (next_i < 0 or next_i >= vulnerability.shape[0]
                or next_j < 0 or next_j >= vulnerability.shape[1]):
            break

        current_idx = (next_i, next_j)

        # Branching
        if np.random.random() < branch_prob * severity:
            ba = np.random.uniform(-np.pi/3, np.pi/3)
            ca, sa = np.cos(ba), np.sin(ba)
            bd = np.array([
                current_dir[0]*ca - current_dir[1]*sa,
                current_dir[0]*sa + current_dir[1]*ca
            ])
            bp = _propagate_v4(
                current_idx, bd, vulnerability, R_mesh, Z_mesh,
                max_length=max_length//3,
                severity=severity * 0.65,
                branch_prob=branch_prob * 0.45,
            )
            path.extend(bp)

    return path


def _secondary_cracks_v4(vulnerability, stress_field, n_cracks,
                           R_mesh, Z_mesh, seed=100):
    """Small secondary cracks in high-stress regions."""
    secondary = np.zeros_like(vulnerability)
    shape = vulnerability.shape
    np.random.seed(seed)

    combined  = vulnerability * (stress_field / (stress_field.max() + 1e-8))
    threshold = np.percentile(combined.flatten(), 82)
    cands     = np.where(combined > threshold)

    if len(cands[0]) < n_cracks:
        return secondary

    indices = np.random.choice(len(cands[0]), size=n_cracks, replace=False)
    for idx in indices:
        ci, cj  = cands[0][idx], cands[1][idx]
        angle   = np.random.uniform(0, 2*np.pi)
        direction = (np.cos(angle), np.sin(angle))
        cp = _propagate_v4(
            (ci, cj), direction, vulnerability,
            R_mesh, Z_mesh,
            max_length=18, severity=0.28, branch_prob=0.08,
        )
        for (ii, jj), intensity in cp:
            if 0 <= ii < shape[0] and 0 <= jj < shape[1]:
                secondary[ii, jj] = max(secondary[ii, jj], intensity * 0.45)
    return secondary


def _synthesize_sink_marks(phi_field, f_l, R_mesh, Z_mesh,
                            severity, seed=200):
    """
    Generate sink-mark field concentrated at the top surface (z=H).

    Sink marks in LCWT401 form when:
      1. The top surface solidifies first (shrinks)
      2. The bulk below is still liquid/mushy (can't support surface)
      3. → Surface pulls inward → crater / dimple

    Output: 2D array same shape as phi_field, nonzero only near z=H.
    """
    np.random.seed(seed)
    sink = np.zeros_like(phi_field)
    H_max = Z_mesh.max()

    # Sink marks only at the top 15% of height
    top_mask = Z_mesh >= 0.85 * H_max

    if not top_mask.any() or severity < 0.05:
        return sink

    # Sink-mark severity: proportional to overall severity but concentrated
    sm_severity = float(np.clip(severity * 1.3, 0, 1))

    # The sink mark is radially centered (worst at r=0 for back-fill)
    r_norm = R_mesh / (R_mesh.max() + 1e-8)

    # Gaussian bell centered at r=0 (axisymmetric sink)
    r_profile = np.exp(-3.5 * r_norm**2)

    # z profile: strongest exactly at z=H, decays inward
    z_rel    = (Z_mesh - 0.85 * H_max) / (0.15 * H_max + 1e-8)
    z_profile = np.clip(z_rel, 0, 1) ** 0.5   # slow rise to surface

    # Base sink-mark field
    sink_base = sm_severity * r_profile * z_profile * top_mask

    # Add stochastic surface roughness (cratering pattern)
    n_craters = max(1, int(sm_severity * 6))
    for _ in range(n_craters):
        # Random position on top surface
        r_crater = np.random.uniform(0, R_mesh.max() * 0.6)
        # Find nearest grid point at top surface
        dist_crater = np.sqrt((R_mesh - r_crater)**2) * (Z_mesh >= 0.9*H_max)
        idx_c = np.unravel_index(
            (dist_crater + 1e6*(~top_mask)).argmin(), phi_field.shape
        )
        ic, jc = idx_c
        # Small Gaussian crater
        sigma_c = max(1, int(sm_severity * 2.5))
        for di in range(-sigma_c*3, sigma_c*3+1):
            for dj in range(-sigma_c*3, sigma_c*3+1):
                ni, nj = ic+di, jc+dj
                if (0 <= ni < phi_field.shape[0]
                        and 0 <= nj < phi_field.shape[1]
                        and top_mask[ni, nj]):
                    r_sq = di**2 + dj**2
                    sink[ni, nj] += (sm_severity * 0.4
                                     * np.exp(-r_sq / (2*sigma_c**2)))

    sink += sink_base
    sink  = np.clip(sink, 0, 1)
    sink  = gaussian_filter(sink, sigma=0.6) * top_mask
    return sink


# ============================================================================
# BLEND PINN + SYNTHETIC
# ============================================================================

def blend_pinn_and_synthetic_v4(pinn_phi, synthetic_phi, sink_field,
                                 blend_factor=0.65, in_pulse=False):
    """
    Blend PINN phase field with synthetic cracks.
    During healing pulse: reduce blend factor (trust PINN healing more).

    Returns (blended_crack, sink_field).
    """
    if in_pulse:
        blend_factor = blend_factor * 0.5   # less synthetic, more PINN healing

    blended = np.maximum(pinn_phi, synthetic_phi * blend_factor)
    blended += (1 - blend_factor) * pinn_phi + blend_factor * synthetic_phi
    blended  = np.clip(blended, 0, 1)

    return blended, sink_field


# ============================================================================
# PULSE EVENT MARKER
# ============================================================================

def get_pulse_windows(params):
    """
    Return list of (t_start_min, t_end_min) tuples for all hot-air pulses.
    Used for shading pulse windows on timeline charts.
    """
    n   = int(params.get('n_pulses', 5))
    tps = params.get('t_pulse_start', 35) / 60.0
    tpd = params.get('t_pulse_dur', 20) / 60.0
    tpi = params.get('t_pulse_interval', 88) / 60.0

    windows = []
    for i in range(n):
        t_on  = tps + i * tpi
        t_off = t_on + tpd
        windows.append((float(t_on), float(t_off)))
    return windows
