"""The site computes in the browser (components/site/engine.js); the Python in main_3d_v4.py stays the reference.
This runs both on every preset (the timeline) and on the optimiser, and fails if they disagree.

    python tools/check_engine.py          (needs Node)
"""
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main_3d_v4 as M  # noqa: E402  (bare mode: Streamlit only warns)


def recipe(sc):
    return dict(T_fill=sc["T_fill"], zones=sc["zones"], reheat=sc["reheat"], h_cool=sc["h_cool"],
                h_reheat=sc.get("h_reheat", 12.0), melt=sc.get("melt", 67.0), late=sc.get("late_cool_T", 23.0))


def main():
    sys.stdout.reconfigure(encoding="utf-8")              # preset names carry emoji; Windows consoles default to cp932
    cases = {"timelines": [], "optimiser": []}
    py_t, py_o = {}, {}
    for name, sc in M.SCENARIOS.items():
        a = recipe(sc)
        ts = M.build_timeline(a["T_fill"], a["zones"], a["reheat"], a["h_cool"], a["h_reheat"], n_pts=120,
                              melt=a["melt"], late_cool_T=a["late"])
        cases["timelines"].append({"name": name, "args": a})
        py_t[name] = ts
    for name in (list(M.SCENARIOS)[2], list(M.SCENARIOS)[5], list(M.SCENARIOS)[0]):
        a = recipe(M.SCENARIOS[name])
        for scope in ("cooling", "all"):
            cases["optimiser"].append({"name": name, "scope": scope, "args": a})
            py_o[(name, scope)] = M.optimise(a["T_fill"], a["reheat"], a["h_cool"], a["h_reheat"], a["melt"], a["late"], scope)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cases, f)
    out = json.loads(subprocess.run(["node", str(ROOT / "tools" / "check_engine.js"), f.name], capture_output=True,
                                    text=True, encoding="utf-8", check=True).stdout)
    bad = []
    for r in out["timelines"]:
        p, j = py_t[r["name"]], r["js"]
        if len(p["times"]) != len(j["times"]):
            bad.append(f"{r['name']}: {len(p['times'])} vs {len(j['times'])} points"); continue
        worst = max(max(abs(x - y) for x, y in zip(p[k], j[k])) for k in ("times", "T_surf", "T_core", "DI"))
        for k in ("t_total", "t_reheat", "reheat_start", "reheat_end", "n_pulse", "pulse_sec"):
            if abs(float(p[k]) - float(j[k])) > 1e-9:
                bad.append(f"{r['name']}: {k} {p[k]} vs {j[k]}")
        if p["reheat_mode"] != j["reheat_mode"]:
            bad.append(f"{r['name']}: mode {p['reheat_mode']} vs {j['reheat_mode']}")
        if worst > 1e-9:
            bad.append(f"{r['name']}: arrays differ by {worst:.2e}")
        print(f"timeline  {r['name'][:46]:46s}  peak DI {max(p['DI']):.4f}  {p['t_total']:.1f} min  max diff {worst:.1e}")
    for r in out["optimiser"]:
        p, j = py_o[(r["name"], r["scope"])], r["js"]
        if set(p) != set(j):
            bad.append(f"optimiser {r['name']} {r['scope']}: kinds {sorted(p)} vs {sorted(j)}"); continue
        for k in p:
            if abs(p[k]["di"] - j[k]["di"]) > 1e-9 or abs(p[k]["t"] - j[k]["t"]) > 1e-9:
                bad.append(f"optimiser {r['name']} {r['scope']} {k}: {p[k]['di']:.6f}/{p[k]['t']} vs {j[k]['di']:.6f}/{j[k]['t']}")
        print(f"optimiser {r['name'][:38]:38s} {r['scope']:8s}  " + ", ".join(f"{k} {p[k]['di']:.3f}" for k in sorted(p)))
    print("OK" if not bad else "\n".join(["FAIL"] + bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
