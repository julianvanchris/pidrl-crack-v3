"""verify_v4_outputs.py — Step 1: verify training outputs."""
import json
from pathlib import Path

checks = {
    "pinn_outputs_v4/pinn_v4_best.pth":   "PINN trained model",
    "drl_outputs_v4/agent_best.pth":      "DRL best agent",
    "drl_outputs_v4/optimal_params.json": "DRL optimal params",
}

all_ok = True
for path, desc in checks.items():
    exists = Path(path).exists()
    size   = f"{Path(path).stat().st_size/1024:.1f} KB" if exists else "MISSING"
    status = "✓" if exists else "✗"
    print(f"  {status} {desc:35s} ({size})")
    if not exists:
        all_ok = False

if all_ok:
    with open("drl_outputs_v4/optimal_params.json") as f:
        opt = json.load(f)
    print(f"\n  Best DI  : {opt.get('_DI_final',1):.4f}  [{opt.get('_risk','?')}]")
    print(f"  Time     : {opt.get('_total_min',0):.1f} min")
    print(f"  T_reheat : {opt.get('T_reheat',0)-273.15:.1f} °C")
    print(f"  n_pulses : {int(opt.get('n_pulses',0))}")
    print(f"  t_start  : {opt.get('t_pulse_start',0)/60:.2f} min")
    print(f"  t_dur    : {opt.get('t_pulse_dur',0):.0f} sec")
    print("\n  ✓ Ready — proceed to Step 2.")
else:
    print("\n  ✗ Missing files — re-run training first.")
