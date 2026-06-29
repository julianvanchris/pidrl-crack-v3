# 🔥 PI-DRL Solidification Control — v4.1

**Physics-Informed Deep Reinforcement Learning for Sink-Mark (ヒケ) Elimination**
CBIC × TUAT | Julian Evan Chrisnanto | Particle & Transfer Laboratory, TUAT | 2026

A Streamlit dashboard that combines a Physics-Informed Neural Network (PINN), a Deep Reinforcement Learning (DRL) optimiser, and a local RAG-augmented LLM advisor to design and validate the cooling/reheat schedule for the **LCWT401 deodorant stick** production line, eliminating top-surface sink marks (ヒケ) while staying within a 30-minute cycle time.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Structure](#3-repository-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Quick Start (3 commands)](#6-quick-start-3-commands)
7. [Step-by-Step: Training the PINN](#7-step-by-step-training-the-pinn)
8. [Step-by-Step: Training the DRL Agent](#8-step-by-step-training-the-drl-agent)
9. [Step-by-Step: Setting Up Ollama](#9-step-by-step-setting-up-ollama)
10. [Step-by-Step: Training the LLM on Client Data (RAG)](#10-step-by-step-training-the-llm-on-client-data-rag)
11. [Running the Dashboard](#11-running-the-dashboard)
12. [Dashboard Tab Guide](#12-dashboard-tab-guide)
13. [Using the AI Advisor](#13-using-the-ai-advisor)
14. [Uploading New Data In-Browser](#14-uploading-new-data-in-browser)
15. [Configuration Reference](#15-configuration-reference)
16. [Troubleshooting](#16-troubleshooting)
17. [File Reference](#17-file-reference)
18. [Physics & Algorithm Notes](#18-physics--algorithm-notes)

---

## 1. What This Project Does

LCWT401 deodorant sticks develop a sink mark (ヒケ) at the top fill point during solidification. This project:

- **Simulates** the cooling process with a physics-informed ODE/PINN model that tracks a **Damage Index (DI)** from 0 (perfect) to 1 (severe defect)
- **Optimises** the cooling-zone schedule and hot-air reheat parameters using a DRL agent (or a fast physics-proxy grid search built into the dashboard)
- **Visualises** the process as a 3D U-turn conveyor belt animation, temperature/DI time-series, and a peridynamic crack-propagation field
- **Advises** via a local LLM (Ollama) trained on your experimental Excel data, using Retrieval-Augmented Generation (RAG) so it answers fast and grounds every answer in your actual trial data — with a safe rule-based fallback if Ollama isn't running

| Target | Value |
|---|---|
| Total cycle time | ≤ 30 min |
| Damage Index (SAFE) | < 0.10 |
| Material | LCWT401 (R=1.25cm, H=4cm) |
| T_solidus / T_liquidus | 62°C / 72°C |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Dashboard                        │
│                      (main_3d_v4.py)                             │
│                                                                   │
│  Sidebar: Process Builder        Tabs:                           │
│  - Scenario presets              🏭 U-Turn Belt (3D animation)   │
│  - Cooling zones                 📈 Temperature & DI charts      │
│  - Hot-air reheat                💥 Crack Propagation (3D)       │
│  - Convection coefficients       📊 Results & Advice             │
│  - DRL Optimiser button          🤖 AI Advisor (LLM + upload)    │
└───────────────────┬───────────────────────────┬──────────────────┘
                     │                           │
         ┌───────────▼──────────┐    ┌──────────▼─────────────┐
         │  Physics ODE Engine   │    │   LLM Advisor Stack     │
         │  (build_timeline)     │    │   (llm_advisor_v4.py)    │
         │  - Analytical DI ODE  │    │                          │
         │  - Mushy-zone tracking│    │  ┌────────────────────┐  │
         │  - Healing during     │    │  │ vector_store_v4.py │  │
         │    reheat              │    │  │ (RAG, TF-IDF)      │  │
         └───────────────────────┘    │  └─────────┬──────────┘  │
                                       │            │             │
         ┌───────────────────────┐    │  ┌─────────▼──────────┐  │
         │  PINN (optional)       │    │  │ Ollama REST API     │  │
         │  train_pinn_*.py       │    │  │ (pidrl-advisor)     │  │
         └───────────────────────┘    │  └─────────┬──────────┘  │
                                       │            │             │
         ┌───────────────────────┐    │  ┌─────────▼──────────┐  │
         │  DRL Agent (optional)  │    │  │ Rule-Based Fallback │  │
         │  train_drl_agent_v4.py │    │  │ (always available)  │  │
         └───────────────────────┘    │  └────────────────────┘  │
                                       └──────────────────────────┘
                     ▲
                     │
         ┌───────────┴───────────┐
         │ prepare_llm_v4.py      │
         │ - Parses Excel data    │
         │ - Builds knowledge JSON│
         │ - Builds RAG index     │
         │ - Registers Ollama model│
         └────────────────────────┘
                     ▲
                     │
            CLIENT_Data_0427.xlsx
            (or any uploaded .xlsx)
```

**Two independent optimisation paths** — you can use either or both:

- **Fast path** (no training required): the dashboard's built-in `build_timeline()` ODE model + sidebar "⚡ Run DRL Optimise" grid search. Works immediately, zero setup.
- **Deep path** (optional): train a full PINN (heat equation + phase field) and a DDPG DRL agent for higher-fidelity results. Takes 10–30 min each.

---

## 3. Repository Structure

```
version4/
├── main_3d_v4.py                       # Streamlit dashboard (the app you run)
├── prepare_llm_v4.py                   # LLM knowledge-base + RAG + Ollama training
├── llm_advisor_v4.py                   # LLM advisor class (Ollama client + RAG + fallback)
├── vector_store_v4.py                  # Local TF-IDF RAG vector store
├── train_pinn_physics_enforced_v4.py   # PINN training (optional)
├── train_drl_agent_v4.py               # DRL/DDPG training (optional)
├── ddpg_agent_v4.py                    # DDPG agent architecture
├── model_wrapper_v4.py                 # PINN inference wrapper
├── crack_predictor_v4.py               # Damage Index computation
├── crack_synthesis_v4.py               # Crack/sink-mark visualisation synthesis
├── verify_v4_outputs.py                # Sanity-checks trained model outputs
├── requirements_v4.txt                 # Python dependencies
├── CLIENT_Data_0427.xlsx               # Client experimental data (required for LLM training)
├── RUN_GUIDE_v4.py                     # Printable quick-reference guide
└── README.md                           # This file
```

---

## 4. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Windows 10/11, macOS, or Linux | Tested on Windows + Anaconda |
| Python | 3.10+ | via Anaconda/Miniconda recommended |
| RAM | 8 GB | 16 GB recommended if running Ollama + training simultaneously |
| GPU | Optional | Speeds up PINN/DRL training; CPU works fine, just slower |
| Disk | ~6 GB free | For Ollama base model (mistral ≈ 4.1 GB) |
| Browser | Chrome or Edge | Plotly animations are not supported in Safari |

---

## 5. Installation

### 5.1 Create the conda environment

```bash
conda create -n env-pinn python=3.10 -y
conda activate env-pinn
```

### 5.2 Install Python dependencies

```bash
cd version4/
pip install -r requirements_v4.txt
```

If `requirements_v4.txt` is missing or incomplete, install manually:

```bash
pip install streamlit plotly numpy scipy pandas openpyxl requests
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU
# OR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU-only
```

Verify GPU availability (optional):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 5.3 Install Ollama (for the AI Advisor)

| OS | Command |
|---|---|
| Windows | Download installer from [ollama.com/download](https://ollama.com/download) and run it |
| macOS | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

Verify:

```bash
ollama --version
```

Pull a base model (pick **one**):

```bash
ollama pull mistral        # 4.1 GB — recommended, best quality
ollama pull phi3:mini      # 2.3 GB — fastest, lighter
ollama pull llama3.1:8b    # 5.0 GB — best reasoning, slowest
```

Confirm it downloaded:

```bash
ollama list
```

> ⚠️ **Windows PATH note:** if `ollama` is not recognised in your terminal, the dashboard and `prepare_llm_v4.py` still work — they talk to Ollama over its REST API (`localhost:11434`), which only requires the **Ollama background service** to be running (check Task Manager for `ollama.exe`), not the CLI in your PATH.

---

## 6. Quick Start (3 commands)

Once installed, every day you only need:

```bash
conda activate env-pinn
cd version4/
streamlit run main_3d_v4.py
```

Open the browser at **http://localhost:8501** (opens automatically).

If you haven't trained the LLM yet, also run once:

```bash
python prepare_llm_v4.py
```

---

## 7. Step-by-Step: Training the PINN

> **Optional.** The dashboard's built-in analytical ODE model works without this. Train the PINN only if you want a higher-fidelity neural temperature field.

```bash
conda activate env-pinn
cd version4/
python train_pinn_physics_enforced_v4.py
```

**What it does:** trains a 4-layer MLP on the 2D heat equation + Stefan (phase-field) condition in three phases — warm-up, physics-enforced, fine-tune.

| | GPU | CPU |
|---|---|---|
| Duration | 15–30 min | 60–120 min |

**Outputs** (in `outputs/pinn_v4/`):
- `pinn_best.pt` — best checkpoint (used by the dashboard if present)
- `pinn_final.pt` — final epoch checkpoint
- `training_log.csv` — loss history
- `pinn_config.json` — architecture config

**Verify:**

```bash
python verify_v4_outputs.py --pinn
```

---

## 8. Step-by-Step: Training the DRL Agent

> **Optional.** The sidebar's "⚡ Run DRL Optimise" button already performs a fast physics-proxy grid search with no training required. Train the full DDPG agent only for long-horizon, higher-quality optimisation.

```bash
python train_drl_agent_v4.py
```

**What it does:** trains a DDPG actor-critic agent against the physics-proxy ODE environment to minimise Damage Index within the 30-min budget.

| State | `[T_surface, T_core, DI, fl, time_elapsed, zone_index, Bi, ΔT]` |
|---|---|
| Action (8D) | `[ΔT_zone, Δduration, ΔT_reheat, Δh_cool]` |
| Reward | `−DI_peak − 0.3·DI_final + time_bonus` (if total ≤ 30 min) |

| | GPU | CPU |
|---|---|---|
| Duration | 10–25 min | 30–60 min |

**Outputs** (in `outputs/drl_v4/`):
- `drl_best.pt` — best agent weights (DI < 0.15 threshold)
- `drl_final.pt` — final weights
- `training_log.csv` — episode history
- `optimal_params.json` — best zone schedule found

**Verify:**

```bash
python verify_v4_outputs.py --drl
```

---

## 9. Step-by-Step: Setting Up Ollama

### 9.1 Confirm Ollama is running

| OS | Check |
|---|---|
| Windows | Task Manager → look for `ollama.exe` (auto-starts as a service) |
| macOS/Linux | `ollama serve &` then keep the terminal open (or it runs as a service depending on install method) |

Either way, confirm with:

```bash
ollama list
```

This should list your downloaded model(s) without error.

### 9.2 Quick standalone test

```bash
ollama run mistral
>>> What is solidification?
```

Type `/bye` to exit. If this streams a coherent answer, Ollama is working correctly.

---

## 10. Step-by-Step: Training the LLM on Client Data (RAG)

This is the key step that makes the **AI Advisor** tab give specific, grounded answers about your process instead of generic LLM responses.

### 10.1 What happens under the hood

```
CLIENT_Data_0427.xlsx
        │
        ▼  parse_excel()                — reads all sheets, both column-header
        │                                  AND row-label (time-series) layouts,
        │                                  English + Japanese keyword detection
        ▼  extract_knowledge()          — structures trials, rules, DI targets
        │
        ▼  chunk_knowledge_base()       — splits into ~30-90 small text chunks
        │                                  (vector_store_v4.py)
        ▼  build_index()                — TF-IDF vectorisation (instant, no
        │                                  embedding model download needed)
        ▼  vector_store_v4.json         — saved RAG index
        │
        ▼  build_system_prompt()        — full knowledge baked into Modelfile
        ▼  build_lean_system_prompt()   — short persona prompt for fast runtime queries
        │
        ▼  Modelfile_v4                — Ollama model definition
        ▼  POST /api/create             — registers "pidrl-advisor" via REST API
                                           (no CLI / PATH dependency)
```

At **query time**, instead of sending the full knowledge base every time (slow), the advisor retrieves only the top-5 most relevant chunks via TF-IDF cosine similarity and injects those into a short prompt — keeping responses fast even on CPU-only Ollama.

### 10.2 Run the training script

```bash
conda activate env-pinn
cd version4/
python prepare_llm_v4.py
```

Make sure `CLIENT_Data_0427.xlsx` is in the same folder first.

**Expected output:**

```
══════════════════════════════════════════════════════════
  PI-DRL LLM Knowledge Base Preparation
══════════════════════════════════════════════════════════
[1] Parsing CLIENT_Data_0427.xlsx...
[2] Extracting domain knowledge...           [OK]
[3] Building system prompt...                [OK]
[4] Building RAG vector store...             [OK]  33 chunks indexed
[5] Creating Modelfile_v4...                  [OK]
[6] Registering pidrl-advisor (REST API)...   [OK]

Next: streamlit run main_3d_v4.py -> AI Advisor tab
```

Duration: **2–5 minutes**.

### 10.3 Verify the model

```bash
ollama list
```
`pidrl-advisor:latest` should now appear.

```bash
python llm_advisor_v4.py --test
python llm_advisor_v4.py --status
python llm_advisor_v4.py --chat "Why does hike form at the top surface?"
```

---

## 11. Running the Dashboard

```bash
conda activate env-pinn
cd version4/
streamlit run main_3d_v4.py
```

```
Local URL:   http://localhost:8501
Network URL: http://192.168.x.x:8501
```

If port 8501 is busy:

```bash
streamlit run main_3d_v4.py --server.port 8502
```

### Try these scenarios in order (sidebar → Scenario Preset):

| Order | Scenario | Expected result |
|---|---|---|
| 1 | ❌ Bad — No reheat (RT blast) | DI ≈ 0.85 **CRITICAL** |
| 2 | ⚠️ Client current — 40 min total | DI ≈ 0.45 **WARNING**, 40 min |
| 3 | ✅ Target — Step cool + reheat (30 min) | DI ≈ 0.15 **CAUTION**, 30 min |
| 4 | 🚀 DRL Optimal (v4) | DI < 0.10 **SAFE**, ≤ 28 min |
| 5 | Click **⚡ Run DRL Optimise** (sidebar) | Auto-finds the best schedule live |

---

## 12. Dashboard Tab Guide

### 🏭 U-Turn Belt
3D animation of the stick travelling through cooling zones, U-turning, then through reheat and final cooling. Press **▶ Play**. Stick colour shifts red→yellow→green as it cools; red crack lines appear when DI ≥ 0.25.

### 📈 Temperature & DI
Three stacked, animated charts: surface/core temperature vs setpoint, ΔT and cooling rate, and the Damage Index curve. Watch DI rise through the mushy zone (62–72°C) and dip during reheat (healing).

### 💥 Crack Propagation
Side-by-side 3D crack fields **before** and **after** reheat. Damage surface coloured blue (intact) → red (critical). Cracks are rendered as a single trunk from the top fill-point branching outward — matching real ヒケ stress patterns. Hidden entirely when DI < 0.25.

### 📊 Results & Advice
Colour-coded risk banner with a DI progress bar, six metric cards, the cooling-zone table, a reheat-physics advisory box, a 6-point process-quality checklist, and a written recommendation referencing the actual client trial data (Trial 1 heat-retention cap, Trial 2 pulsed reheat, production-line baseline).

### 🤖 AI Advisor
See [Section 13](#13-using-the-ai-advisor) below.

---

## 13. Using the AI Advisor

1. Run any simulation — the advisor **auto-analyses** it into 5 sections: ASSESSMENT, PHYSICAL MEANING, ROOT CAUSE, RECOMMENDATION, COMPARISON (vs. your real trial data).
2. The status bar shows:
   - 🟢 **Ollama LLM active** / 🟡 **Rule-based mode** (Ollama offline)
   - 🔥 **Model warm — fast responses** / 🥶 **Loading model (~30–90s first query)**
   - RAG chunk count, knowledge-base status, cache size
3. **Chat box** — ask anything; 4 quick-question buttons are provided. Responses **stream in real time**, token by token, so you see progress even during a slow cold start.
4. Every response shows its **source** (`🤖 LLM: pidrl-advisor` or `📊 Rule-based`) so you always know whether you got a full LLM answer or the deterministic physics fallback.
5. **🔄 Retry** button next to "Detailed Analysis" — useful if Ollama just finished loading.

### Why the first query is slow (and how this is handled)

Loading a multi-gigabyte model from disk into RAM takes 30–90 seconds on CPU — this is normal Ollama behaviour, not a bug. The dashboard mitigates this three ways:

- **Background pre-warming**: a daemon thread force-loads the model the moment the advisor starts, before you ask anything
- **`keep_alive=30m`**: the model stays loaded for 30 minutes between uses instead of Ollama's 5-minute default
- **Streaming + adaptive retry**: cold-start queries get a 90s budget; once the model is confirmed warm, subsequent queries use a 30s budget and respond in seconds. If Ollama is completely unresponsive, it **always** falls back to the rule-based engine — you never see a raw timeout error.

---

## 14. Uploading New Data In-Browser

You don't need the terminal to retrain the LLM on new experimental data.

1. Open the **🤖 AI Advisor** tab
2. Scroll to **📂 Upload & Train from Excel Data** → expand **📤 Upload Excel → Retrain Model**
3. Drag and drop **one or more** `.xlsx` files (multiple files are merged)
4. Choose base model (default `mistral`) and model name (default `pidrl-advisor`)
5. Keep **Auto-register with Ollama** checked
6. Click **⚡ Train / Update Model**
7. Wait ~30–120 seconds for the progress bar
8. Review the summary: temperature series detected, observations learned, RAG chunk count
9. **No page reload needed** — the advisor hot-reloads the knowledge base, RAG index, and re-detects the Ollama model live in the same session

The auto-detector understands both **column-header** tables and **row-label time-series** layouts (the latter is how `CLIENT_Data_0427.xlsx` is structured), and recognises both English and Japanese keywords (温度, 時間, 観察, ヒケ, etc.).

---

## 15. Configuration Reference

Environment variables (optional — sensible defaults are built in):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama REST API endpoint |
| `PIDRL_MODEL` | `pidrl-advisor` | Registered model name to query |
| `PIDRL_FALLBACK` | `mistral` | Base model used if `pidrl-advisor` isn't found |
| `PIDRL_EMBED_MODEL` | `nomic-embed-text` | Optional semantic embedding model (falls back to TF-IDF automatically if not pulled) |

Set them before launching, e.g. on Windows PowerShell:

```powershell
$env:PIDRL_MODEL = "my-custom-advisor"
streamlit run main_3d_v4.py
```

---

## 16. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` on any script | `conda activate env-pinn && pip install -r requirements_v4.txt` |
| `Ollama not found` running `prepare_llm_v4.py` | Install from ollama.com; ensure the **service** is running (Task Manager). Registration uses the REST API, not the CLI, so PATH issues don't block it. |
| `StreamlitValueAboveMaxError` on zone duration | Already fixed in v4.1 (`max_value=35.0`). Re-download `main_3d_v4.py` if seen. |
| AI Advisor stuck on "Rule-based mode" | Ollama isn't reachable. Run `ollama list` to confirm the service is up. |
| `pidrl-advisor` not in `ollama list` | `cd version4/ && ollama create pidrl-advisor -f Modelfile_v4` |
| "Ollama retries exhausted" / long first wait | Normal **cold start** (model loading). Wait for the 🔥 "Model warm" badge; subsequent queries are fast. The dashboard always falls back to rule-based — it never hard-fails. |
| `SyntaxError: f-string expression part cannot include a backslash` | You're on Python < 3.12 with an outdated file. Re-download `main_3d_v4.py` (fixed in this version). |
| Total time > 30 min after DRL Optimise | File may be outdated — verify: `grep "if t_f<1.5" main_3d_v4.py` should return a match. |
| Cracks shown even at low DI | Threshold is DI ≥ 0.25 in the current version; re-download if seen at lower DI. |
| Animations don't play | Use Chrome or Edge — Plotly animations are unsupported in Safari. |
| Japanese Excel file shows 0 detected rows | Use the current `prepare_llm_v4.py` — it detects row-label/time-series layouts and Japanese keywords (温度, 時間, 観察, etc.) automatically. |

---

## 17. File Reference

| File | Role | Required for dashboard to run? |
|---|---|---|
| `main_3d_v4.py` | Streamlit app | ✅ Yes |
| `prepare_llm_v4.py` | LLM/RAG training pipeline | Only for AI Advisor full LLM mode |
| `llm_advisor_v4.py` | LLM advisor runtime class | ✅ Yes (falls back to rules if Ollama absent) |
| `vector_store_v4.py` | RAG vector store | ✅ Yes (used by `llm_advisor_v4.py`) |
| `train_pinn_physics_enforced_v4.py` | PINN training | ❌ Optional |
| `train_drl_agent_v4.py` | DRL training | ❌ Optional |
| `ddpg_agent_v4.py` | DDPG architecture | Only with DRL training |
| `model_wrapper_v4.py` | PINN inference | Only if using trained PINN |
| `crack_predictor_v4.py` | DI computation | Used by training scripts |
| `crack_synthesis_v4.py` | Crack synthesis | Used by training scripts |
| `verify_v4_outputs.py` | Sanity checks | ❌ Optional, for verifying training |
| `requirements_v4.txt` | pip dependencies | ✅ Yes |
| `CLIENT_Data_0427.xlsx` | Experimental data | Only for LLM training step |

---

## 18. Physics & Algorithm Notes

### Damage Index ODE (analytical, unconditionally stable)

```
dDI/dt = k_form × (1 − DI) − k_heal × DI

Analytical step:  DI(t+dt) = DI_eq + (DI − DI_eq) × exp(−(k_form+k_heal)×dt)
                   where DI_eq = k_form / (k_form + k_heal)
```

- `k_form` activates only while liquid fraction `fl > 0.01` (mushy-zone transition); scales with Biot number and cooling severity
- `k_heal` activates above 55°C, 3.5× stronger during active reheat — drives DI toward 0 when T_reheat ≥ 80°C

### DI Risk Scale

| Level | Range |
|---|---|
| ✅ SAFE | 0.00 – 0.10 |
| ⚠️ CAUTION | 0.10 – 0.25 |
| 🔶 WARNING | 0.25 – 0.50 |
| 🚨 CRITICAL | 0.50 – 0.80 |
| 🛑 FAILURE | 0.80 – 1.00 |

### Biot Number

```
Bi = h_cool × R / k_thermal = h_cool × 0.0125 / 0.25
```
Target **Bi ≤ 0.5** (i.e. `h_cool ≤ 10 W/m²K`) for uniform radial cooling.

### DRL Search (sidebar "⚡ Run DRL Optimise")

Evaluates 500+ combinations of `T_reheat × h_reheat × h_cool × n_zones × distribution_strategy`, hard-rejecting any combination exceeding 30.5 min total, scoring the rest as `0.6×DI_peak + 0.4×DI_final`.

---

## Credits

**Julian Evan Chrisnanto** — PhD Student, Graduate School of Bio-Applications and Systems Engineering, Tokyo University of Agriculture and Technology (TUAT)
**Supervisor:** Prof. Wuled Lenggoro, Particle & Transfer Laboratory
**Industry partner:** CBIC
**Project:** PI-DRL Solidification Control, 2026