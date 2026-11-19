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


def FIXES(n):
    """Lock sets for the site's Keep fixed: none, each one alone, a mixed set, and everything."""
    off = dict(n=False, T=[False] * n, D=[False] * n, hc=False, mode=False, rT=False, rD=False, hr=False, rN=False, rH=False)
    one = lambda **k: {**off, **k}
    return [("none", off), ("count", one(n=True)), ("zone 1 T", one(T=[True] + [False] * (n - 1))),
            ("all T", one(T=[True] * n)), ("last D", one(D=[False] * (n - 1) + [True])), ("all D", one(D=[True] * n)),
            ("air", one(hc=True)), ("timing", one(mode=True)), ("hot-air T", one(rT=True)), ("hot-air time", one(rD=True, mode=True)),
            ("reheat air", one(hr=True)), ("mixed", one(T=[True] + [False] * (n - 1), hc=True, rT=True)),
            ("burst count", one(rN=True)), ("first half", one(rH=True)),
            ("line limits", one(D=[True] * n, rN=True, rH=True, mode=True)),
            ("everything", one(n=True, T=[True] * n, D=[True] * n, hc=True, mode=True, rT=True, rD=True, hr=True))]


def check_fixed(r, a, fix):
    """Each result keeps every fixed setting exactly, and the recipe's own timing never scores worse than the recipe."""
    bad, tag = [], f"fixed {r['name'][:30]} [{r['label']}]"
    z0, reh0 = a["zones"], a["reheat"]
    for k, b in r["js"].items():
        z = b["zones"]
        if (fix["n"] or any(fix["T"]) or any(fix["D"])) and len(z) != len(z0):
            bad.append(f"{tag} {k}: {len(z)} zones, kept {len(z0)}")
            continue
        for i in range(len(z0)):
            if fix["T"][i] and z[i]["T"] != z0[i]["T"]:
                bad.append(f"{tag} {k}: zone {i + 1} T {z[i]['T']} vs {z0[i]['T']}")
            if fix["D"][i] and z[i]["duration"] != z0[i]["duration"]:
                bad.append(f"{tag} {k}: zone {i + 1} time {z[i]['duration']} vs {z0[i]['duration']}")
        if fix["hc"] and b["h_cool"] != a["h_cool"]:
            bad.append(f"{tag} {k}: h_cool {b['h_cool']}")
        if fix["hr"] and b["h_reh"] != a["h_reheat"]:
            bad.append(f"{tag} {k}: h_reheat {b['h_reh']}")
        if (fix["mode"] or fix["rD"]) and k != r["kind"]:
            bad.append(f"{tag}: timing {k}, kept {r['kind']}")
        if fix["rT"] and b["reh"]["T"] != reh0["T"]:
            bad.append(f"{tag} {k}: hot air {b['reh']['T']} vs {reh0['T']}")
        if fix["rN"] and not fix["rD"] and r["kind"] == "pulsed":
            lens = lambda rh: ([int(x[1]) for x in sorted(rh["bursts"], key=lambda x: x[0])] if rh.get("bursts")
                               else [int(rh.get("pulse_sec", 0))] * int(rh.get("pulses", 0)))
            if lens(b["reh"]) != lens(reh0) or len(lens(reh0)) == 0:
                bad.append(f"{tag} {k}: bursts {b['reh'].get('bursts')} vs {reh0.get('bursts')}")
        if fix["rH"]:                                     # the Python model is the judge of where the hot air ends
            tt = M.build_timeline(a["T_fill"], z, b["reh"], b["h_cool"], b["h_reh"], n_pts=80, melt=a["melt"], late_cool_T=a["late"])
            if tt["t_reheat"] > 0 and tt["reheat_end"] > tt["t_total"] / 2 + 1e-9:
                bad.append(f"{tag} {k}: hot air ends at {tt['reheat_end']:.2f} of {tt['t_total']:.2f} min")
        if fix["rD"]:
            for f in ("duration", "pulses", "pulse_sec", "pulse_window"):
                if b["reh"].get(f) != reh0.get(f):
                    bad.append(f"{tag} {k}: reheat {f} {b['reh'].get(f)} vs {reh0.get(f)}")
    if r["cur"] is not None:
        own = r["js"].get(r["kind"])
        if own is None or own["score"] > r["cur"] + 1e-12:
            bad.append(f"{tag}: worse than the recipe ({own and own['score']} vs {r['cur']})")
    if all([fix["n"], all(fix["T"]), all(fix["D"]), fix["hc"], fix["mode"], fix["rT"], fix["rD"], fix["hr"]]):
        if r["cur"] is not None and set(r["js"]) != {r["kind"]}:
            bad.append(f"{tag}: everything fixed, yet {sorted(r['js'])}")
    return bad


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
    for name in (list(M.SCENARIOS)[2], list(M.SCENARIOS)[5]):
        a = dict(recipe(M.SCENARIOS[name]), T_fill=60.0, melt=50.0)
        a["zones"] = [dict(z, T=min(z["T"], 59.0)) for z in a["zones"]]
        low = "low fill 60 C, melt 50 C: " + name
        cases["timelines"].append({"name": low, "args": a})
        py_t[low] = M.build_timeline(a["T_fill"], a["zones"], a["reheat"], a["h_cool"], a["h_reheat"], n_pts=120,
                                     melt=a["melt"], late_cool_T=a["late"])
        cases["optimiser"].append({"name": low, "scope": "all", "args": a})
        py_o[(low, "all")] = M.optimise(a["T_fill"], a["reheat"], a["h_cool"], a["h_reheat"], a["melt"], a["late"], "all")
    for name in (list(M.SCENARIOS)[2], list(M.SCENARIOS)[5], list(M.SCENARIOS)[0]):
        a = recipe(M.SCENARIOS[name])
        for scope in ("cooling", "all"):
            cases["optimiser"].append({"name": name, "scope": scope, "args": a})
            py_o[(name, scope)] = M.optimise(a["T_fill"], a["reheat"], a["h_cool"], a["h_reheat"], a["melt"], a["late"], scope)
    # fixed settings (JS only, the site's "Keep fixed"): every lock set on three presets
    cases["fixed"] = []
    for name in (list(M.SCENARIOS)[2], list(M.SCENARIOS)[5], list(M.SCENARIOS)[0]):
        a = recipe(M.SCENARIOS[name]); n = len(a["zones"])
        for label, fix in FIXES(n):
            cases["fixed"].append({"name": name, "label": label, "fix": fix, "args": a})
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
    for r in out["fixed"]:
        bad += check_fixed(r, recipe(M.SCENARIOS[r["name"]]), dict(FIXES(len(M.SCENARIOS[r["name"]]["zones"])))[r["label"]])
    print(f"fixed     {len(out['fixed'])} searches with settings kept: every kept setting unchanged, never worse than the recipe")
    print("OK" if not bad else "\n".join(["FAIL"] + bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
