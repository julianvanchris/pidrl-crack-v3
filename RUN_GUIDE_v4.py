"""
==========================================================================
  PI-DRL v4 — Complete Step-by-Step Run Guide
  CBIC × TUAT | Julian Evan Chrisnanto | 2026
==========================================================================

PROCESS MODELLED:
  Pulsed hot-air reheating (from CLIENT_Data_0427.xlsx):
  - N short hot-air pulses (≈20 sec, ~100°C) during mushy-zone traversal
  - Cold air (16°C) for final solidification
  - Total target: ≤ 30 min (observed: ~24.5 min in production)

FILE STRUCTURE (put all in the same folder):
  version4/
  ├── train_pinn_physics_enforced_v4.py   ← PINN training
  ├── train_drl_agent_v4.py               ← DRL training
  ├── ddpg_agent_v4.py                    ← DRL agent architecture
  ├── model_wrapper_v4.py                 ← Inference wrapper
  ├── crack_predictor_v4.py               ← DI + SI computation
  ├── crack_synthesis_v4.py               ← Crack visualization
  ├── main_3d_v4.py                       ← Streamlit dashboard
  ├── verify_v4_outputs.py                ← Verification script
  └── requirements_v4.txt

==========================================================================
  STEP 1 — Install dependencies
==========================================================================

  conda activate env-pinn   (or your existing environment)
  pip install -r requirements_v4.txt

  # Verify CUDA (strongly recommended for PINN training):
  python -c "import torch; print('CUDA:', torch.cuda.is_available())"

==========================================================================
  STEP 2 — Verify training outputs exist
==========================================================================

  cd version4/
  python verify_v4_outputs.py

  Expected output:
    ✓ PINN trained model        (pinn_outputs_v4/pinn_v4_best.pth)
    ✓ DRL best agent            (drl_outputs_v4/agent_best.pth)
    ✓ DRL optimal params        (drl_outputs_v4/optimal_params.json)

    Best DI  : 0.XXXX  [SAFE/CAUTION/...]
    Time     : XX.X min
    T_reheat : XX.X °C
    n_pulses : X

  If any files are MISSING → re-run training (Steps 2a / 2b below).

--------------------------------------------------------------------------
  STEP 2a — Re-train PINN (only if pinn_v4_best.pth is missing)
--------------------------------------------------------------------------

  python train_pinn_physics_enforced_v4.py

  Expected log (FIXED version):
    [Ph1] step     0  L_tot=X.XXXX  L_data=0.35  L_ic=0.12  L_pde=0.00  ...
    [Ph1] step   500  L_tot=X.XXXX  L_data=0.18  L_ic=0.04  L_pde=0.00  ...
    [Ph2] step 15000  L_tot=X.XXXX  L_data=0.05  L_ic=0.01  L_pde=0.02  ...
    [Ph3] step 40000  L_tot=X.XXXX  L_data=0.02  L_ic=0.00  L_pde=0.01  ...

  KEY METRICS TO WATCH:
    L_data should DECREASE from ~0.35 → < 0.05 by step 15000
    L_ic   should DECREASE from ~0.12 → < 0.01 by step 5000
    L_pde  should appear > 0 from step 15000 onward
    L_pde = 0.0000 throughout = bug (contact developer)

  Duration:  ~45 min on NVIDIA GPU | ~4 hrs on CPU
  Output:    pinn_outputs_v4/pinn_v4_best.pth   (saved whenever data loss improves)
             pinn_outputs_v4/pinn_v4_final.pth  (end of training)
             pinn_outputs_v4/training_curves_v4.png

--------------------------------------------------------------------------
  STEP 2b — Re-train DRL (only if agent_best.pth is missing)
--------------------------------------------------------------------------

  python train_drl_agent_v4.py

  Expected log (FIXED version — DI should vary!):
    Computing reference baselines...
      production_observed: DI_final=0.350 [CAUTION]
      client_best_trial1 : DI_final=0.080 [SAFE]
      bad_blast_cool     : DI_final=0.780 [CRITICAL]

    Ep     0/5000  R= -4.2  DI=0.720[0.35-0.82] [CRITICAL]  succ=0.0%  Bi=0.90
    Ep    50/5000  R= -2.8  DI=0.450[0.20-0.68] [WARNING]   succ=12%   Bi=0.40
    Ep   200/5000  R= -1.1  DI=0.280[0.10-0.45] [CAUTION]   succ=45%   Bi=0.25
    Ep  1000/5000  R= +0.8  DI=0.120[0.05-0.28] [CAUTION]   succ=78%   Bi=0.18
    Ep  3000/5000  R= +2.1  DI=0.080[0.03-0.15] [SAFE]      succ=95%   Bi=0.15

  If DI is STILL 0.500 for everything → PINN phi not trusted.
  Dashboard will use physics-proxy DI instead (still functional).

  Duration:  ~30 min on GPU | ~2 hrs on CPU
  Output:    drl_outputs_v4/agent_best.pth
             drl_outputs_v4/optimal_params.json
             drl_outputs_v4/drl_training_v4_epXXXX.png

==========================================================================
  STEP 3 — Launch the Streamlit dashboard
==========================================================================

  cd version4/
  streamlit run main_3d_v4.py

  Open browser: http://localhost:8501

  If port is busy:
    streamlit run main_3d_v4.py --server.port 8502

==========================================================================
  STEP 4 — How to use the dashboard
==========================================================================

  LEFT SIDEBAR:
  ┌──────────────────────────────────────────────────────┐
  │ 📋 Condition Scenario                                │
  │   ❌ Bad — RT/blast (23°C)     ← No pulses, Bi=0.9 │
  │   ⚠️ Normal — 45°C/5min        ← 3 pulses           │
  │   ✅ Production (CBIC observed) ← 5×20sec pulses     │
  │   🚀 DRL Optimal (v4)           ← DRL recommended   │
  ├──────────────────────────────────────────────────────┤
  │ 🔥 Hot-Air Pulse Settings:                          │
  │   T_reheat (°C)      — hot air temperature          │
  │   h_hot (W/m²K)      — convection intensity         │
  │   n_pulses           — number of pulses (0-10)      │
  │   t_pulse_start (s)  — when first pulse fires       │
  │   t_pulse_dur (s)    — duration per pulse           │
  │   t_pulse_interval(s)— time between pulses          │
  │   t_cold_start (min) — when cold air (16°C) starts  │
  └──────────────────────────────────────────────────────┘

  TAB 1 — Belt + Charts:
    • 3D animated conveyor belt
    • 🟠 Orange bands = hot-air pulse zones on belt
    • 🔴 ✕ marks = crack locations (DI > 0.10)
    • 🔥 = pulse active during animation
    • ▶ Play / ⏸ Pause — browser-side, zero reload

  TAB 2 — Temperature & DI:
    • Row 1: T_surface, T_core, T_env vs time
    • Row 2: ΔT and cooling rate
    • Row 3: Damage Index (DI) with risk bands
    • 🟠 Shaded = hot-air pulse windows
    • Pointer turns orange during pulse
    • ▶ Play animates through process

  TAB 3 — Crack Propagation 3D:
    • 3D crack field at t = t_pulse_start (before first pulse)
    • Per-pulse effectiveness table (ΔDI per pulse)
    • If DI < 0.05 → "No significant cracks" shown

  TAB 4 — Results & Parameters:
    • Full parameter table with safe thresholds
    • ⚡ DRL recommendation (if agent loaded)
    • Saves from drl_outputs_v4/optimal_params.json

  WORKFLOW:
    1. Select scenario from sidebar
    2. Adjust pulse parameters if desired
    3. Click ▶ Compute Simulation
    4. Review charts → adjust if needed
    5. Click ⚡ DRL Optimize for AI recommendation
    6. Compare in Tab 2 (DRL overlay appears below)

==========================================================================
  STEP 5 — Troubleshooting
==========================================================================

  Problem: "L_data constant at 0.0450, L_pde=0.0000"
  → Old PINN code. Replace with latest train_pinn_physics_enforced_v4.py

  Problem: "DI=0.500 for everything in DRL"
  → Old DRL code. Replace with latest train_drl_agent_v4.py
  → Dashboard still works — uses physics-proxy DI which DOES vary.

  Problem: "Module not found: model_wrapper_v4"
  → All 6 v4 files must be in same folder as main_3d_v4.py

  Problem: Streamlit shows white screen
  → streamlit run main_3d_v4.py --server.headless true

  Problem: CUDA out of memory
  → Reduce n_domain/n_surface in sample_collocation()
  → Or add: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

  Problem: L-BFGS fails with size mismatch
  → Already fixed in latest PINN file. Re-download.

==========================================================================
  STEP 6 — Interpreting results for CBIC
==========================================================================

  DI < 0.10:  SAFE ✅ — No ヒケ expected. Good production condition.
  DI 0.10-0.25: CAUTION 🟡 — Minor surface imperfections possible.
  DI 0.25-0.50: WARNING 🟠 — Sink marks / ヒケ likely. Adjust pulses.
  DI > 0.50:  CRITICAL 🔴 — Severe cracking. Increase n_pulses, T_reheat.

  KEY RECOMMENDATION (from CLIENT_Data_0427.xlsx):
    Production baseline (5×20sec pulses @ t≈0.5-6min) achieves DI~0.35.
    DRL target: reduce to DI < 0.15 by optimizing:
      → Increase n_pulses from 5 to 6-7
      → Start first pulse slightly earlier (t_start: 34→25-30 sec)
      → Slightly longer pulses (t_dur: 20→25-30 sec)
      → Maintain T_reheat ~95-100°C
      → Total time: still within 25 min ✓

==========================================================================
"""

print(__doc__)
