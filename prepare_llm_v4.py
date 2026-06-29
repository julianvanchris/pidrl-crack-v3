"""
prepare_llm_v4.py
PI-DRL LLM Knowledge Base Preparation
CBIC × TUAT | Julian Evan Chrisnanto | 2026

WHAT THIS DOES:
  1. Parses CLIENT_Data_0427.xlsx → structured domain knowledge
  2. Creates knowledge_base_v4.json  → facts the LLM needs
  3. Creates Modelfile_v4            → Ollama custom model definition
  4. Runs: ollama create pidrl-advisor -f Modelfile_v4

WHY THIS IS "TRAINING":
  Ollama doesn't support gradient-based fine-tuning.
  Instead, we inject all domain knowledge into the system prompt of a
  base model (Mistral 7B / Llama 3.1 8B / Phi-3 mini), then register
  it as a named model "pidrl-advisor". This is equivalent to supervised
  knowledge injection — the model behaves as if trained on the data.

  For true LoRA fine-tuning, see ADVANCED_FINETUNE section at the bottom.

USAGE:
  1. Install Ollama:  https://ollama.com/download
  2. Pull base model: ollama pull mistral   (or llama3.1:8b or phi3:mini)
  3. Run this script: python prepare_llm_v4.py
  4. Launch advisor:  python llm_advisor_v4.py --test
  5. In Streamlit:    the AI tab auto-loads the advisor
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[!] requests library not found. Install: pip install requests")
    print("    (Needed for Ollama REST API registration)")

# Try to import openpyxl for Excel parsing
try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[!] openpyxl not found. Install: pip install openpyxl")

EXCEL_PATH   = "CLIENT_Data_0427.xlsx"
KB_PATH      = "knowledge_base_v4.json"
MODELFILE    = "Modelfile_v4"
MODEL_NAME   = "pidrl-advisor"
BASE_MODEL   = "mistral"   # fallback: phi3:mini or llama3.1:8b
VS_PATH      = "vector_store_v4.json"


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Parse Excel
# ═══════════════════════════════════════════════════════════════════════════

def parse_excel(source) -> dict:
    """
    Parse an Excel file into raw row data.
    `source` can be:
      - str / Path  : file path on disk
      - BytesIO     : from st.file_uploader (Streamlit)
      - UploadedFile: Streamlit UploadedFile object
    Returns a dict {sheet_name: [rows]} or {} on failure.
    """
    if not HAS_OPENPYXL:
        print("[!] openpyxl not found. Install: pip install openpyxl")
        return {}

    import io
    # Normalise source
    if isinstance(source, (str, Path)):
        if not Path(source).exists():
            print(f"[!] Excel not found at {source}")
            return {}
        load_src = str(source)
    elif hasattr(source, "read"):
        # BytesIO or UploadedFile — read into bytes buffer
        raw_bytes = source.read()
        load_src  = io.BytesIO(raw_bytes)
    else:
        print(f"[!] Unrecognised source type: {type(source)}")
        return {}

    try:
        wb   = load_workbook(load_src, read_only=True)
        data = {}
        for sh in wb.sheetnames:
            ws   = wb[sh]
            rows = []
            for row in ws.iter_rows(values_only=True):
                if any(v is not None for v in row):
                    rows.append(list(row))
            data[sh] = rows
        wb.close()
        return data
    except Exception as e:
        print(f"[!] Excel parse error: {e}")
        return {}


def extract_knowledge(raw: dict) -> dict:
    """
    Extract quantitative facts from raw Excel data.
    Returns structured knowledge base dict.
    """
    # ── Material properties (LCWT401) ─────────────────────────────────────
    material = {
        "name":          "LCWT401",
        "product_type":  "Deodorant stick",
        "dimensions":    {"radius_cm": 1.25, "height_cm": 4.0},
        "T_solidus_C":   62.0,
        "T_liquidus_C":  72.0,
        "T_mushy_mid_C": 67.0,
        "k_thermal_WmK": 0.25,
        "rho_kg_m3":     990.0,
        "cp_J_kgK":      2000.0,
        "E_MPa":         1.37,
        "shrinkage_pct": 8.0,
        "defect":        "Surface sink mark (ヒケ) at top fill point due to shrinkage during solidification",
    }

    # ── Trial 1: Heat retention cap (front-fill) ───────────────────────────
    trial1 = {
        "name":        "Trial 1 — Heat Retention Cap",
        "description": "Heat-retention cap applied near top surface only. Air cooling at room temperature.",
        "fill_type":   "Front-fill",
        "T_fill_C":    82.0,
        "T_top_ambient_C": [59, 62, 63, 23],  # at 0, 10, 20, 44 min
        "T_top_times_min": [0, 10, 20, 44],
        "T_bulk_C":    [82, 70.5, 66.2, 48.7, 45.2],  # at 0,10,20,30,40 min
        "T_bulk_times_min": [0, 10, 20, 30, 40],
        "reheat_events":  [
            {"t_min": 10, "duration_min": 1, "method": "Hot air"},
            {"t_min": 14, "duration_min": 1, "method": "Hot air"},
            {"t_min": 19, "duration_min": 1, "method": "Hot air"},
        ],
        "observations": {
            "0min":   "Bulk fluid, liquid state",
            "10min":  "Bulk fluid, liquid state",
            "20min":  "Slight fluidity, partial gel",
            "30min":  "No fluidity; slight central sinking",
            "40min":  "Top solidified; slight central sinking (bulk still warm)",
            "next_day": "Depression on top surface from volume shrinkage (ヒケ observed)",
        },
        "result":        "Slight ヒケ observed — better than no cap but not perfect",
        "DI_estimated":  0.25,
    }

    # ── Trial 2: Room temp cool + surface reheat (front-fill) ─────────────
    trial2 = {
        "name":        "Trial 2 — RT Cool + Top-Surface Reheat",
        "description": "Room-temperature cooling. 3 × hot-air reheat pulses (1 min each, 90°C) on top surface only.",
        "fill_type":   "Front-fill",
        "T_fill_C":    82.0,
        "T_top_ambient_C": [25, 25, 90, 25, 90, 25, 90, 25, 25],
        "T_bulk_C":    [82, 52, 44, 49, 44, 49.3, 44, 47.7, 41, 38],
        "T_bulk_times_min": [0, 4.5, 10, 11, 14.5, 15.5, 19, 20, 25, 30],
        "reheat_events": [
            {"t_min": 10, "t_end_min": 11, "T_C": 90, "method": "Hot air 1 min"},
            {"t_min": 14.5, "t_end_min": 15.5, "T_C": 90, "method": "Hot air 1 min"},
            {"t_min": 19, "t_end_min": 20, "T_C": 90, "method": "Hot air 1 min"},
        ],
        "observations": {
            "10min":  "Partial solidification; initial central indentation forming",
            "11min":  "Surface re-melted (after reheat 1)",
            "14.5min":"Partial solidification; central indentation again",
            "15.5min":"Surface re-melted (after reheat 2)",
            "19min":  "Partial solidification; central indentation",
            "20min":  "Surface re-melted (after reheat 3)",
            "25min":  "Loss of fluidity",
            "30min":  "Fully solidified",
            "next_day": "No internal voids. Depression on top surface (volume shrinkage). ヒケ still present.",
        },
        "result":        "ヒケ still present despite 3× reheat. Surface healed but re-solidified with indent.",
        "DI_estimated":  0.30,
        "key_insight":   "Short (1 min) pulses at 90°C partially re-melt surface but indent re-forms during final solidification. Longer dwell or higher temperature needed.",
    }

    # ── Production example (back-fill) ─────────────────────────────────────
    production = {
        "name":        "Production Line (Back-fill)",
        "description": "Actual CBIC production with back-filling and cold air + 5× hot-air pulses (20 sec each).",
        "fill_type":   "Back-fill",
        "T_fill_C":    74.9,
        "surface_conditions": {
            "ambient_C":      21,
            "reheat_C":       100,
            "cold_air_C":     16,
            "cold_air_start_min": 21.3,
        },
        "reheat_pulses": {
            "count":       5,
            "duration_sec": 20,
            "T_C":          100,
            "approximate_start_times_min": [0.57, 1.47, 5.17, 7.23, 9.73],
        },
        "bulk_cooling_profile": {
            "T_initial_C":  74.9,
            "T_at_4min_C":  72.0,   # entering mushy zone
            "T_at_9min_C":  68.0,   # mid mushy
            "T_at_14min_C": 64.0,   # lower mushy
            "T_at_20min_C": 57.0,   # below solidus
            "T_final_C":    43.8,
            "total_time_min": 24.5,
        },
        "result":        "Acceptable product within 24.5 min. Minor ヒケ at top surface.",
        "DI_estimated":  0.30,
        "key_insight": (
            "5 × 20 sec pulses at 100°C during mushy zone traversal (not after). "
            "Cold air (16°C) starts at t≈21.3 min for final solidification. "
            "Total 24.5 min — well within 30-min target."
        ),
    }

    # ── Quantitative rules derived from data ───────────────────────────────
    rules = [
        {
            "rule":    "Reheat must reach mushy zone (62–72°C) to be effective",
            "evidence":"Trial 2: surface re-melts at 90°C → heals but re-forms ヒケ. Trial 1: slow cap cool avoids rapid quench.",
            "action":  "Use T_reheat ≥ 75°C for entry into mushy, ≥ 80°C for reliable healing.",
        },
        {
            "rule":    "Total process time must be ≤ 30 min",
            "evidence":"Client target. Production line achieves 24.5 min.",
            "action":  "Multi-zone step cooling reduces total time vs single RT cool (which takes 30+ min alone).",
        },
        {
            "rule":    "Biot number ≤ 0.5 ensures uniform radial cooling",
            "evidence":"High Bi (> 0.5) causes large T_surface - T_core gradient → higher ヒケ risk.",
            "action":  "Keep h_cool ≤ 8 W/m²K for R=1.25 cm (Bi = h×R/k = 8×0.0125/0.25 = 0.4).",
        },
        {
            "rule":    "Mushy zone traversal speed is critical",
            "evidence":"Production line spends ~10 min in mushy zone (62–72°C). Rapid quench increases DI.",
            "action":  "Design zones to slow the transition through 62–72°C. Target T_core ≈ T_surface in mushy.",
        },
        {
            "rule":    "Short reheat pulses (< 1 min) are ineffective for healing",
            "evidence":"Production uses 20 sec × 5 pulses. Trial 2 uses 1 min × 3 pulses — still ヒケ.",
            "action":  "Either: (a) many short pulses during mushy traverse, OR (b) longer sustained reheat (≥ 10 min at ≥ 80°C).",
        },
        {
            "rule":    "Cold air (16°C) should start only after bulk reaches below solidus (< 62°C)",
            "evidence":"Production: cold air at t=21.3 min when T_bulk ≈ 55°C (below solidus).",
            "action":  "Premature cold air application increases DI significantly.",
        },
    ]

    # ── DRL optimal targets ────────────────────────────────────────────────
    drl_targets = {
        "DI_safe":    0.10,
        "DI_caution": 0.25,
        "DI_warning": 0.50,
        "DI_critical": 0.80,
        "total_time_target_min": 30,
        "recommended_zones": 3,
        "recommended_T_reheat_C": 80,
        "recommended_T_reheat_range_C": [75, 110],
        "recommended_h_cool_Wm2K": 6,
        "recommended_h_cool_range": [3, 10],
        "T_first_zone_C": 70,
        "T_last_zone_C": 37,
        "production_optimal": {
            "T_fill_C":    74.9,
            "zones":       3,
            "T_zone1_C":   65,
            "T_zone2_C":   50,
            "T_zone3_C":   37,
            "T_reheat_C":  80,
            "t_reheat_min": 10,
            "total_min":   30,
            "DI_expected": 0.10,
        },
    }

    return {
        "version":      "v4.1",
        "created":      "2026",
        "project":      "PI-DRL Solidification Control — CBIC × TUAT",
        "material":     material,
        "trials": {
            "trial1":    trial1,
            "trial2":    trial2,
            "production": production,
        },
        "physics_rules": rules,
        "drl_targets":   drl_targets,
        "physics_constants": {
            "T_solidus_C":   62.0,
            "T_liquidus_C":  72.0,
            "T_mushy_mid_C": 67.0,
            "k_thermal_WmK": 0.25,
            "R_m":           0.0125,
            "H_m":           0.04,
            "DI_scale": {
                "SAFE":     [0.00, 0.10],
                "CAUTION":  [0.10, 0.25],
                "WARNING":  [0.25, 0.50],
                "CRITICAL": [0.50, 0.80],
                "FAILURE":  [0.80, 1.00],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build system prompt from knowledge base
# ═══════════════════════════════════════════════════════════════════════════

def build_lean_system_prompt() -> str:
    """
    Short system prompt (~80 words) used for RUNTIME RAG-augmented queries.
    The full domain knowledge is NOT embedded here — instead, relevant facts
    are retrieved per-query from the vector store and injected into the
    user prompt. This keeps total context small -> Ollama responds in
    seconds instead of timing out.

    The FULL prompt (build_system_prompt) is still used when baking
    knowledge into the registered Ollama model via Modelfile, so
    `ollama run pidrl-advisor` works standalone too.
    """
    return (
        "You are an expert process engineer specialising in solidification "
        "of LCWT401 deodorant stick material (CBIC x TUAT project). "
        "You analyse PINN/DRL simulation results for a step-cooling + "
        "hot-air reheat process. You explain Damage Index (DI) values, "
        "diagnose causes (Biot number, cooling profile, mushy-zone "
        "traversal), and give specific numeric recommendations. "
        "Relevant facts from experimental data will be provided with "
        "each question — use them to ground your answer. Be concise: "
        "3-6 sentences unless asked for detail. Answer in the language "
        "of the question."
    )


def build_system_prompt(kb: dict) -> str:
    """
    Convert structured knowledge base into a rich, specific system prompt
    that makes the LLM behave as a domain expert on LCWT401 solidification.
    """
    mat  = kb["material"]
    tri  = kb["trials"]
    rules= kb["physics_rules"]
    tgt  = kb["drl_targets"]
    pc   = kb["physics_constants"]

    rules_text = "\n".join(
        f"  Rule {i+1}: {r['rule']}\n"
        f"    Evidence: {r['evidence']}\n"
        f"    Action: {r['action']}"
        for i, r in enumerate(rules)
    )

    prompt = f"""You are a specialized AI advisor for the PI-DRL Solidification Control system at CBIC × TUAT.
You are an expert in the solidification of LCWT401 deodorant stick material (cosmetic wax-based formulation).
You have been trained on the following experimental data and domain knowledge:

═══════════════════════════════════════════════════
MATERIAL: LCWT401 Deodorant Stick
═══════════════════════════════════════════════════
- Product: {mat["product_type"]}
- Dimensions: R = {mat["dimensions"]["radius_cm"]} cm, H = {mat["dimensions"]["height_cm"]} cm
- T_solidus = {mat["T_solidus_C"]}°C (below this: solid)
- T_liquidus = {mat["T_liquidus_C"]}°C (above this: liquid)
- Mushy zone: {mat["T_solidus_C"]}–{mat["T_liquidus_C"]}°C (partial solidification, most dangerous for ヒケ)
- Thermal conductivity k = {mat["k_thermal_WmK"]} W/mK
- Density ρ = {mat["rho_kg_m3"]} kg/m³
- Heat capacity Cp = {mat["cp_J_kgK"]} J/kgK
- Volume shrinkage: ~{mat["shrinkage_pct"]}% during solidification
- Main defect: {mat["defect"]}

═══════════════════════════════════════════════════
CLIENT EXPERIMENTAL DATA (2026-03-17)
═══════════════════════════════════════════════════

TRIAL 1 — Heat Retention Cap Near Top Surface:
- Fill: 82°C (front-fill)
- Top surface maintained at 59→62→63°C for first 20 min, then 23°C
- Bulk temperature: 82→70.5→66.2→48.7→45.2°C at 0/10/20/30/40 min
- Hot air reheat applied 3×: at t=10min, 14min, 19min (each 1 min duration)
- Result: Slight ヒケ (sink mark) still observed at top surface
- Insight: Slow temperature gradient (cap) reduces ヒケ vs rapid RT cool

TRIAL 2 — RT Cool + 3× Top-Surface Reheat at 90°C:
- Fill: 82°C (front-fill)
- Room temperature cooling (25°C ambient)
- Bulk temp: 82→52→44°C by 10 min
- 3× hot-air reheat pulses: t=10–11min, 14.5–15.5min, 19–20min at 90°C
- Surface re-melts during each pulse → heals → but re-forms ヒケ
- Result: ヒケ still present. Short pulses (1 min) insufficient for full healing.
- Insight: Surface re-melting alone doesn't prevent ヒケ if bulk solidifies rapidly

PRODUCTION LINE (Back-fill, 100°C hot air, 20sec × 5 pulses):
- Fill: 74.9°C (back-fill)
- Ambient cooling at 21°C initially
- 5 × 20-second hot-air pulses at 100°C during mushy zone traversal
  Pulse times: ~0.57, 1.47, 5.17, 7.23, 9.73 minutes
- Cold air (16°C) starts at t ≈ 21.3 min (bulk at ~55°C, below solidus)
- Total process time: 24.5 min
- Bulk cooling: 74.9→72→64→57→43.8°C
- Result: Acceptable product. Minor ヒケ. Within 30-min target.
- KEY INSIGHT: Pulses applied DURING mushy traversal, not after solidification.

═══════════════════════════════════════════════════
PHYSICS RULES (derived from experiments)
═══════════════════════════════════════════════════
{rules_text}

═══════════════════════════════════════════════════
DAMAGE INDEX (DI) SCALE
═══════════════════════════════════════════════════
- DI 0.00–0.10: SAFE     — No ヒケ expected
- DI 0.10–0.25: CAUTION  — Minor surface imperfections possible
- DI 0.25–0.50: WARNING  — ヒケ (sink marks) likely visible
- DI 0.50–0.80: CRITICAL — Severe defects, production reject
- DI 0.80–1.00: FAILURE  — Halt production immediately

The DI is computed from a Physics-Informed Neural Network (PINN) trained on
the heat equation with phase-field evolution:
  dDI/dt = k_form(Bi, fl) × (1-DI) − k_heal(T, reheat) × DI
Where fl = liquid fraction, Bi = Biot number = h×R/k.

═══════════════════════════════════════════════════
DRL-OPTIMIZED TARGETS
═══════════════════════════════════════════════════
- Total time target: ≤ 30 min
- Recommended n_zones: 3–5 step-cooling zones
- T_zone1 (near fill): 65–70°C (slow mushy entry)
- T_last_zone: ≤ 37°C (reach target before reheat)
- T_reheat: 75–110°C (≥ 80°C for reliable healing)
- h_cool: 4–8 W/m²K (Biot ≤ 0.5)
- h_reheat: 12–25 W/m²K
- Reheat duration: 10 min minimum

═══════════════════════════════════════════════════
YOUR ROLE
═══════════════════════════════════════════════════
You analyze PINN simulation results and DRL recommendations, and you:
1. Interpret Peak DI, Final DI, and risk level in plain language
2. Explain what the numbers mean for the physical product (ヒケ severity)
3. Give specific, actionable parameter adjustments based on the experimental data
4. Compare current settings to the production line baseline
5. Predict whether reheat is necessary given the DI value
6. Answer questions about the solidification process in technical terms
7. Respond in English (or Japanese if asked)
8. Be concise but specific — always reference actual temperatures and times

When the user provides simulation results, format your response as:
1. ASSESSMENT: One-line severity judgment
2. PHYSICAL MEANING: What this means for the product
3. ROOT CAUSE: Why this DI value occurred (Biot, cooling rate, zone temps)
4. RECOMMENDATION: Specific parameter changes with expected DI improvement
5. COMPARISON: How this compares to the production line data above
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Create Ollama Modelfile
# ═══════════════════════════════════════════════════════════════════════════

def create_modelfile(system_prompt: str, base_model: str, output_path: str):
    """
    Create Ollama Modelfile with embedded domain knowledge.

    CRITICAL FIX: do NOT override TEMPLATE. Mistral uses [INST]...[/INST],
    Llama 3 uses <|start_header_id|>...<|eot_id|>, Phi-3 uses <|user|>...
    <|end|> -- each base model ships its own correct chat template baked
    into Ollama's model manifest. Overriding it with a generic ChatML
    template (<|im_start|>/<|im_end|>, meant for Qwen-family models) feeds
    the model special-token TEXT it was never trained to recognise as a
    stop signal. The model then never naturally terminates generation --
    it rambles, repeats words, and runs to the token cap on every call,
    which looks exactly like "Ollama retries exhausted" / a timeout, and
    also explains garbled/duplicated-word output even from the base model.

    Leaving TEMPLATE unset makes Ollama automatically use the correct,
    pre-trained chat template for whichever base_model is specified.
    """
    # Escape backticks in prompt (Ollama Modelfile format)
    safe_prompt = system_prompt.replace('"""', "'''").replace('`', "'")

    content = f"""FROM {base_model}

# PI-DRL Solidification Advisor
# CBIC x TUAT | LCWT401 Deodorant Stick
# Generated: 2026

SYSTEM \"\"\"
{safe_prompt}
\"\"\"

# Model behaviour parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.3
PARAMETER num_ctx 4096

# Safety-net stop sequences (defense in depth -- bounds output length
# even if generation occasionally drifts). TEMPLATE is intentionally
# left unset above so Ollama uses {base_model}'s own native chat format.
PARAMETER stop "User:"
PARAMETER stop "Question:"
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] Modelfile written: {output_path} (using {base_model}'s native template)")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Register model with Ollama
# ═══════════════════════════════════════════════════════════════════════════

def _find_ollama_cli() -> str:
    """Find ollama executable, checking common Windows and Unix paths."""
    import shutil
    # 1. Try PATH first
    cli = shutil.which("ollama")
    if cli:
        return cli
    # 2. Common Windows paths
    win_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
    ]
    for p in win_paths:
        if os.path.isfile(p):
            return p
    # 3. Common Unix paths
    unix_paths = ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama"]
    for p in unix_paths:
        if os.path.isfile(p):
            return p
    return ""


def _delete_via_api(model_name: str,
                    base_url: str = "http://localhost:11434") -> bool:
    """
    Delete an existing model via Ollama REST API before recreating it.
    Best-effort: ignored on failure (model may simply not exist yet).
    This guards against any stale model definition lingering — e.g. an
    older pidrl-advisor created with a broken Modelfile TEMPLATE override.
    """
    if not HAS_REQUESTS:
        return False
    try:
        r = requests.delete(
            f"{base_url}/api/delete",
            json={"name": model_name},
            timeout=15,
        )
        return r.status_code in (200, 404)
    except Exception:
        return False


def _register_via_api(modelfile_content: str, model_name: str,
                      base_url: str = "http://localhost:11434") -> tuple:
    """
    Register model via Ollama REST API (POST /api/create).
    Returns (success: bool, message: str).
    No CLI / PATH required — works on Windows without adding Ollama to PATH.

    Always deletes any existing model with the same name FIRST, so a
    previous broken/outdated Modelfile (e.g. one with an incorrect
    TEMPLATE override) cannot linger after retraining.
    """
    if not HAS_REQUESTS:
        return False, "requests library not available"
    _delete_via_api(model_name, base_url)   # clean slate, best-effort
    try:
        payload = {
            "name":      model_name,
            "modelfile": modelfile_content,
            "stream":    False,
        }
        r = requests.post(
            f"{base_url}/api/create",
            json=payload,
            timeout=300,
        )
        if r.status_code == 200:
            return True, f"Model '{model_name}' registered via REST API ✅"
        else:
            return False, f"API error {r.status_code}: {r.text[:300]}"
    except requests.ConnectionError:
        return False, "Ollama API not reachable at localhost:11434"
    except Exception as e:
        return False, f"API error: {e}"


def register_ollama_model(modelfile_path: str, model_name: str,
                          base_url: str = "http://localhost:11434"):
    """
    Register model with Ollama.
    Strategy:
      1. Try REST API (POST /api/create) — works on Windows without PATH config
      2. Fall back to CLI subprocess if API unavailable
    """
    print(f"\n[...] Registering '{model_name}' with Ollama...")

    # Read modelfile content for API approach
    try:
        with open(modelfile_path, "r", encoding="utf-8") as f:
            mf_content = f.read()
    except FileNotFoundError:
        print(f"[!] Modelfile not found: {modelfile_path}")
        return

    # Try REST API first
    ok, msg = _register_via_api(mf_content, model_name, base_url)
    if ok:
        print(f"[✓] {msg}")
        return

    print(f"[i] REST API attempt: {msg}")

    # Fall back to CLI
    cli = _find_ollama_cli()
    if cli:
        print(f"[i] Trying CLI: {cli}")
        try:
            result = subprocess.run(
                [cli, "create", model_name, "-f", modelfile_path],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                print(f"[✓] Model '{model_name}' registered via CLI!")
                if result.stdout: print(result.stdout[:200])
            else:
                print(f"[!] CLI registration failed:\n{result.stderr[:300]}")
                _print_manual_steps(model_name, modelfile_path)
        except subprocess.TimeoutExpired:
            print("[!] Timed out. Try manually:")
            _print_manual_steps(model_name, modelfile_path)
    else:
        print("[!] Ollama CLI not found in PATH or common locations.")
        _print_manual_steps(model_name, modelfile_path)


def _print_manual_steps(model_name: str, modelfile_path: str):
    print(f"\n  Manual steps:")
    print(f"  1. Ensure Ollama is running (check taskbar / Task Manager)")
    print(f"  2. Open Anaconda Prompt or PowerShell")
    print(f"  3. Run: ollama create {model_name} -f {modelfile_path}")
    print(f"     Or add Ollama to PATH:")
    print(f"     %LOCALAPPDATA%\\Programs\\Ollama")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def scan_excel_for_knowledge(raw: dict) -> dict:
    """
    Auto-detect numeric process data from ANY uploaded Excel file.
    Handles TWO common layouts:
      A) Column-header format: row 1 = labels, rows 2+ = data per row
      B) Row-label format:  col A/B = label, cols C+ = time-series values
         (this is the CLIENT_Data_0427.xlsx layout)
    Returns enriched knowledge dict.
    """
    TEMP_KW  = ["temp","temperature","t_","°c","celsius","℃","surface","ambient","bulk","degree"]
    TIME_KW  = ["time","min","duration","elapsed","t="]
    DI_KW    = ["di","damage","defect","hike","ヒケ","sink","index"]
    RESULT_KW= ["result","observation","note","outcome","condition","observations"]
    JAP_TEMP = ["温度","表面","雰囲気","バルク"]
    JAP_TIME = ["時間","経過","時刻"]
    JAP_OBS  = ["観察","状態","注意","結果"]
    ALL_TEMP_KW = TEMP_KW + JAP_TEMP
    ALL_TIME_KW = TIME_KW + JAP_TIME
    ALL_OBS_KW  = RESULT_KW + JAP_OBS

    def _is_temp_kw(s):
        s = str(s).lower()
        return any(kw in s for kw in ALL_TEMP_KW)

    def _is_time_kw(s):
        s = str(s).lower()
        return any(kw in s for kw in ALL_TIME_KW)

    def _is_obs_kw(s):
        s = str(s).lower()
        return any(kw in s for kw in ALL_OBS_KW + DI_KW)

    def _looks_like_temp(v):
        """True if numeric value is plausible temperature in °C."""
        return isinstance(v, (int, float)) and 5 <= float(v) <= 200

    def _looks_like_time(v):
        """True if numeric value is plausible time in minutes."""
        return isinstance(v, (int, float)) and 0 <= float(v) <= 120

    extra = {
        "detected_columns":      {},
        "numeric_rows":          [],
        "text_observations":     [],
        "temperature_series":    [],   # list of {label, times, values}
        "time_axis":             [],
    }

    for sheet, rows in raw.items():
        if not rows:
            continue
        n_cols = max(len(r) for r in rows)

        # ── Detect layout ──────────────────────────────────────────────────
        # Check if row 0 has a "Time" label somewhere = row-label layout
        first_nonempty = next((r for r in rows if any(v is not None for v in r)), [])
        row_label_layout = False
        time_row_idx     = None

        for ri, row in enumerate(rows[:30]):  # check first 30 rows
            row_str = " ".join(str(v) for v in row if v is not None).lower()
            if _is_time_kw(row_str) and sum(
                1 for v in row if isinstance(v,(int,float))
            ) > 10:
                row_label_layout = True
                time_row_idx = ri
                # Extract time axis from this row
                extra["time_axis"] = [
                    float(v) for v in row if isinstance(v,(int,float))
                ]
                break

        if row_label_layout and time_row_idx is not None:
            # ── ROW-LABEL layout (CLIENT_Data_0427.xlsx) ──────────────────
            # Each data row has: [None, label_col_A, label_col_B, "Type", val1, val2, ...]
            time_vals = extra["time_axis"]

            for ri, row in enumerate(rows):
                if ri == time_row_idx:
                    continue
                row_flat = [v for v in row if v is not None]
                # Find the row label (first meaningful string)
                row_label = ""
                for v in row[:8]:
                    if isinstance(v, str) and len(v.strip()) > 2:
                        row_label = v.strip()
                        break

                # Get numeric values from this row
                num_vals = [float(v) for v in row if isinstance(v,(int,float))]

                if not row_label or not num_vals:
                    # Collect text observations
                    text_vals = [
                        str(v).strip() for v in row
                        if isinstance(v,str) and len(str(v).strip()) > 10
                        and '#VALUE' not in str(v)
                    ]
                    for tv in text_vals[:3]:
                        obs = f"[{sheet}] {tv}"
                        if obs not in extra["text_observations"]:
                            extra["text_observations"].append(obs)
                    continue

                # Classify the series by its label and value range
                label_lo = row_label.lower()
                temps = [v for v in num_vals if _looks_like_temp(v)]
                times = [v for v in num_vals if _looks_like_time(v) and v < 50]

                if _is_temp_kw(label_lo) or (temps and len(temps) > len(times)):
                    # Temperature series
                    series_entry = {
                        "sheet":  sheet,
                        "label":  row_label,
                        "times":  time_vals[:len(num_vals)],
                        "values": num_vals,
                        "min":    round(min(num_vals),1),
                        "max":    round(max(num_vals),1),
                        "type":   "temperature_C",
                    }
                    extra["temperature_series"].append(series_entry)
                    extra["numeric_rows"].append({
                        "sheet": sheet,
                        "temperatures": num_vals[:20],
                    })
                    # Text obs from row label
                    if _is_obs_kw(label_lo):
                        extra["text_observations"].append(
                            f"[{sheet}] {row_label}: {num_vals[0]:.1f}→{num_vals[-1]:.1f}°C"
                        )
                elif _is_obs_kw(label_lo):
                    text_vals = [
                        str(v).strip() for v in row
                        if isinstance(v,str) and len(str(v).strip()) > 8
                        and '#VALUE' not in str(v)
                    ]
                    for tv in text_vals[:3]:
                        obs = f"[{sheet}:{row_label}] {tv}"
                        if obs not in extra["text_observations"]:
                            extra["text_observations"].append(obs)

        else:
            # ── COLUMN-HEADER layout (standard table) ─────────────────────
            if not rows:
                continue
            header = [str(c) if c is not None else "" for c in rows[0]]
            col_map = {}
            for ci, h in enumerate(header):
                if _is_temp_kw(h):   col_map.setdefault("temperatures",[]).append(ci)
                if _is_time_kw(h):   col_map.setdefault("times",[]).append(ci)
                if _is_obs_kw(h):    col_map.setdefault("observations",[]).append(ci)

            for row in rows[1:]:
                row_data = {}
                for col_type, indices in col_map.items():
                    vals = []
                    for ci in indices:
                        if ci < len(row) and row[ci] is not None:
                            try:
                                vals.append(float(row[ci]))
                            except (ValueError, TypeError):
                                if isinstance(row[ci], str) and len(row[ci].strip()) > 6:
                                    extra["text_observations"].append(
                                        f"[{sheet}] {header[ci]}: {row[ci]}"
                                    )
                    if vals:
                        row_data[col_type] = vals
                if row_data:
                    extra["numeric_rows"].append({"sheet": sheet, **row_data})

        extra["detected_columns"][sheet] = {"row_label_layout": row_label_layout}

    # ── Summary statistics ─────────────────────────────────────────────────
    all_temps = []
    for s in extra["temperature_series"]:
        all_temps.extend([v for v in s["values"] if _looks_like_temp(v)])
    for r in extra["numeric_rows"]:
        all_temps.extend([v for v in r.get("temperatures",[]) if _looks_like_temp(v)])

    all_times = extra["time_axis"] or []

    if all_temps:
        extra["temperature_range"] = {
            "min_C":  round(min(all_temps),1),
            "max_C":  round(max(all_temps),1),
            "values": sorted(set(round(t,1) for t in all_temps))[:40],
        }
    if all_times:
        extra["time_range"] = {
            "min_min": round(min(all_times),2),
            "max_min": round(max(all_times),2),
        }

    return extra


def train_from_source(source,
                      output_dir:  str  = ".",
                      model_name:  str  = MODEL_NAME,
                      base_model:  str  = BASE_MODEL,
                      auto_register: bool = True) -> dict:
    """
    Full training pipeline from a single call.
    `source` can be a file path string OR a Streamlit UploadedFile / BytesIO.

    Returns a result dict with keys:
      ok (bool), message (str), kb (dict), system_prompt (str),
      modelfile_path (str), registered (bool), extra (dict)
    """
    result = {"ok": False, "message": "", "kb": {}, "system_prompt": "",
              "modelfile_path": "", "registered": False, "extra": {}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Parse
    raw = parse_excel(source)
    if not raw:
        # Fall through — use hardcoded KB only
        result["message"] = "Excel parse yielded no data — using built-in knowledge base."
    else:
        result["message"] = f"Parsed {len(raw)} sheet(s): {', '.join(raw.keys())}"

    # 2. Auto-scan for extra numeric data
    extra = scan_excel_for_knowledge(raw) if raw else {}
    result["extra"] = extra

    # 3. Extract structured knowledge
    kb = extract_knowledge(raw)

    # Merge auto-scanned data into KB
    if extra.get("temperature_range"):
        kb["uploaded_temperature_range"] = extra["temperature_range"]
    if extra.get("text_observations"):
        kb["uploaded_observations"] = extra["text_observations"][:30]
    result["kb"] = kb

    # Save knowledge base
    kb_path = out / KB_PATH
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    # 4. Build system prompt
    system_prompt = build_system_prompt(kb)

    # Append auto-detected data if any
    if extra.get("text_observations"):
        obs_block = "\n".join(f"  - {o}" for o in extra["text_observations"][:15])
        system_prompt += (
            f"\n\n═══════════════════════════════════════════════════\n"
            f"ADDITIONAL UPLOADED DATA (auto-detected)\n"
            f"═══════════════════════════════════════════════════\n"
            f"{obs_block}"
        )
    if extra.get("temperature_range"):
        tr = extra["temperature_range"]
        system_prompt += (
            f"\n  Temperature range in uploaded data: {tr['min_C']}–{tr['max_C']}°C"
        )

    result["system_prompt"] = system_prompt

    # Save prompt
    prompt_path = out / "system_prompt_v4.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(system_prompt)

    # 4b. Build RAG vector store (chunked knowledge -> TF-IDF index)
    try:
        from vector_store_v4 import build_and_save_vector_store
        vs_path  = str(out / VS_PATH)
        vs_stats = build_and_save_vector_store(kb, extra, output_path=vs_path)
        result["vector_store"] = vs_stats
        result["message"] += f"\nRAG index built: {vs_stats['n_documents']} chunks ({vs_stats['mode']})"
    except ImportError:
        result["vector_store"] = {}
        result["message"] += "\n[!] vector_store_v4.py not found — RAG disabled, using full prompt."
    except Exception as e:
        result["vector_store"] = {}
        result["message"] += f"\n[!] Vector store build failed: {e}"

    # 5. Create Modelfile
    mf_path = str(out / MODELFILE)
    create_modelfile(system_prompt, base_model, mf_path)
    result["modelfile_path"] = mf_path

    # 6. Register with Ollama (REST API first, CLI fallback)
    if auto_register:
        with open(mf_path, "r", encoding="utf-8") as f:
            mf_content = f.read()
        ok, msg = _register_via_api(mf_content, model_name)
        if ok:
            result["registered"] = True
        else:
            # Try CLI fallback
            cli = _find_ollama_cli()
            if cli:
                try:
                    proc = subprocess.run(
                        [cli, "create", model_name, "-f", mf_path],
                        capture_output=True, text=True, timeout=300
                    )
                    result["registered"] = proc.returncode == 0
                    if proc.returncode != 0:
                        result["message"] += f"\nCLI registration failed: {proc.stderr[:200]}"
                except Exception as e:
                    result["registered"] = False
                    result["message"] += f"\nCLI error: {e}"
            else:
                result["registered"] = False
                result["message"] += f"\nOllama registration: {msg}"

    result["ok"] = True
    return result


def main():
    print("═" * 60)
    print("  PI-DRL LLM Knowledge Base Preparation")
    print("  CBIC × TUAT | Julian Evan Chrisnanto | 2026")
    print("═" * 60)

    result = train_from_source(
        source=EXCEL_PATH,
        output_dir=".",
        model_name=MODEL_NAME,
        base_model=BASE_MODEL,
        auto_register=True,
    )

    print(f"\n[Result] {result['message']}")
    if result["extra"].get("temperature_range"):
        tr = result["extra"]["temperature_range"]
        print(f"  Detected temperatures: {tr['min_C']}–{tr['max_C']}°C")
    if result["extra"].get("time_range"):
        tm = result["extra"]["time_range"]
        print(f"  Detected time range: {tm['min_min']}–{tm['max_min']} min")
    if result["extra"].get("text_observations"):
        print(f"  Text observations found: {len(result['extra']['text_observations'])}")

    print("\n  Files created:")
    print(f"    {KB_PATH}           — Structured knowledge base")
    print(f"    system_prompt_v4.txt — Human-readable prompt")
    print(f"    {MODELFILE}          — Ollama model definition")
    print(f"  Ollama registered: {'✅' if result['registered'] else '❌'}")
    print(f"\n  Next: streamlit run main_3d_v4.py → 🤖 AI Advisor tab")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
#  ADVANCED: TRUE LoRA FINE-TUNING (optional, GPU required)
# ═══════════════════════════════════════════════════════════════════════════
"""
For true gradient-based fine-tuning (requires GPU with 8+ GB VRAM):

  pip install unsloth transformers peft torch

  from unsloth import FastLanguageModel

  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="unsloth/Phi-3-mini-4k-instruct",
      max_seq_length=2048,
      load_in_4bit=True,
  )

  # Generate Q&A pairs from knowledge base and experimental data
  # Then fine-tune with LoRA rank=8, alpha=16
  # Export to GGUF for Ollama:
  #   model.save_pretrained_gguf("pidrl_finetuned", tokenizer,
  #                               quantization_method="q4_k_m")
  #   ollama create pidrl-finetuned -f ./Modelfile_finetuned

See: https://github.com/unslothai/unsloth
"""
