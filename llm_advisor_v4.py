"""
llm_advisor_v4.py
PI-DRL LLM Advisor — Ollama Integration + Local RAG
CBIC × TUAT | Julian Evan Chrisnanto | 2026

Architecture:
  1. RAG RETRIEVAL: vector_store_v4.py retrieves only relevant knowledge
     chunks per-query (TF-IDF, instant) -> keeps prompts short -> Ollama
     responds in seconds instead of timing out.
  2. PRIMARY: Ollama local model "pidrl-advisor" (REST API, localhost:11434)
     with retry-with-shrinking-timeout strategy (no more dead timeouts).
  3. FALLBACK: Physics-based rule engine from knowledge_base_v4.json
     (always available, zero network dependency).
  4. CACHE: Last N responses cached in-memory for speed.

Usage:
  from llm_advisor_v4 import PILLMAdvisor
  advisor = PILLMAdvisor()
  result  = advisor.analyze(sim_results)
  answer  = advisor.chat("Why is DI still high after reheat?", context=result)
"""

import json
import os
import time
import hashlib
import textwrap
import threading
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from vector_store_v4 import VectorStore
    HAS_VECTOR_STORE = True
except ImportError:
    HAS_VECTOR_STORE = False

try:
    import warnings as _warnings
    # RunnableWithMessageHistory still works fine; silence its LangGraph
    # migration notice so it doesn't spam the app console every query.
    _warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory.*")
    from langchain_ollama import ChatOllama
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import (
        ChatPromptTemplate, MessagesPlaceholder,
        SystemMessagePromptTemplate, HumanMessagePromptTemplate,
    )
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

try:
    from groq import Groq          # cloud LLM backend (free tier) for deployment
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ─── Constants ────────────────────────────────────────────────────────────────
OLLAMA_URL    = os.environ.get("OLLAMA_URL",    "http://localhost:11434")
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"   # strong free model on Groq
MODEL_NAME    = os.environ.get("PIDRL_MODEL",   "pidrl-advisor")
FALLBACK_MODEL= os.environ.get("PIDRL_FALLBACK","mistral")
KB_PATH       = "knowledge_base_v4.json"
VS_PATH       = "vector_store_v4.json"
TIMEOUT_S     = 45          # seconds per Ollama request (reduced - RAG = shorter prompts)
MAX_TOKENS    = 350         # response length cap (reduced for speed)
CACHE_SIZE    = 32          # cached responses
RAG_TOP_K     = 5           # number of retrieved chunks per query

# ─── GPU offload control ────────────────────────────────────────────────────
# CRITICAL FIX for the "[TIMEOUT] cold start exceeded budget" problem on
# AMD Ryzen APUs (e.g. 5750G with integrated Radeon Vega):
#   Ollama's Vulkan backend tries to PARTIALLY offload the model to the
#   integrated GPU's shared memory. On these iGPUs that path is pathologically
#   slow to initialise — measured COLD LOAD = 7.5 MINUTES vs 3 SECONDS on
#   pure CPU for the same 7B Q4 model. Partial offload also runs slower at
#   inference than just using all 16 CPU threads.
#   Setting num_gpu=0 forces CPU-only execution, which on a modern multi-core
#   CPU loads in seconds and generates at a usable rate.
# Override with OLLAMA_NUM_GPU=<layers> if you have a real discrete GPU.
NUM_GPU       = int(os.environ.get("OLLAMA_NUM_GPU", "0"))   # 0 = CPU-only
NUM_CTX       = int(os.environ.get("OLLAMA_NUM_CTX", "4096")) # context window


def _base_options(temperature: float, max_tokens: int) -> dict:
    """Shared Ollama generation options — single source of truth so the
    CPU-only (num_gpu) and context settings apply to every request path
    (generate / chat / stream / warm-up)."""
    return {
        "temperature":    temperature,
        "num_predict":    max_tokens,
        "num_gpu":        NUM_GPU,
        "num_ctx":        NUM_CTX,
        "repeat_penalty": 1.3,
        "top_p":          0.85,
        "stop":           ["User:", "Question:"],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Ollama client
# ═══════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    KEY FIX for '[TIMEOUT] Ollama retries exhausted': the dominant cause of
    timeouts is Ollama COLD START — loading a 4GB model from disk into RAM
    takes 30-90s on CPU before a single token is generated. Once loaded,
    Ollama keeps the model in memory (default 5 min) and responses are fast.

    This client addresses cold start three ways:
      1. keep_alive="30m" on every request — model stays loaded much longer
         between dashboard interactions, avoiding repeat cold starts.
      2. warm_up() — fires a 1-token request to force model load BEFORE
         the user asks anything (called in a background thread on startup).
      3. Adaptive retry budgets — cold-start attempts get a generous first
         timeout (90s); once warm, subsequent attempts use a short budget.
    """

    KEEP_ALIVE = "30m"   # keep model loaded for 30 min of inactivity

    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url  = base_url.rstrip("/")
        self._alive    = None       # cached liveness
        self._warm_models = set()   # models confirmed loaded in this session

    def is_alive(self) -> bool:
        if not HAS_REQUESTS:
            return False
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            self._alive = r.status_code == 200
        except Exception:
            self._alive = False
        return bool(self._alive)

    def list_models(self) -> list:
        if not HAS_REQUESTS:
            return []
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def list_loaded_models(self) -> list:
        """Models CURRENTLY loaded in memory (via /api/ps) — these respond fast."""
        if not HAS_REQUESTS:
            return []
        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def model_exists(self, model_name: str) -> bool:
        models = self.list_models()
        return any(model_name in m for m in models)

    def check_model_health(self, model_name: str) -> dict:
        """
        Inspect a registered model's ACTUAL template/modelfile via
        POST /api/show, and flag the specific bug that caused repeated
        timeouts/garbled output: an old Modelfile that overrides TEMPLATE
        with ChatML markers (<|im_start|>/<|im_end|>) which the base
        model was never trained to recognise as a stop signal.

        This catches the case where the PYTHON FILES were updated with
        the fix, but the OLLAMA MODEL itself was never recreated --
        updating main_3d_v4.py/prepare_llm_v4.py does not retroactively
        fix a model that already exists in Ollama's registry.

        Returns: {"healthy": bool|None, "reason": str, "exists": bool}
        """
        if not HAS_REQUESTS:
            return {"healthy": None, "reason": "requests unavailable", "exists": False}
        try:
            r = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}, timeout=10,
            )
            if r.status_code != 200:
                return {"healthy": None,
                       "reason": f"model not registered (HTTP {r.status_code})",
                       "exists": False}
            data = r.json()
            template  = data.get("template", "")  or ""
            modelfile = data.get("modelfile", "") or ""
            combined  = template + modelfile
            if "<|im_start|>" in combined or "<|im_end|>" in combined:
                return {"healthy": False,
                       "reason": "outdated broken ChatML template detected — retrain required",
                       "exists": True}
            return {"healthy": True, "reason": "template OK", "exists": True}
        except Exception as e:
            return {"healthy": None, "reason": str(e), "exists": False}

    def is_model_warm(self, model_name: str) -> bool:
        """True if this model is already loaded in Ollama's memory."""
        if model_name in self._warm_models:
            return True
        loaded = self.list_loaded_models()
        warm = any(model_name in m for m in loaded)
        if warm:
            self._warm_models.add(model_name)
        return warm

    def warm_up(self, model: str, timeout: float = 240.0) -> bool:
        """
        Force-load a model into memory with a minimal (num_predict=1)
        request. BLOCKING — intended to be called from a background
        thread so it doesn't block the Streamlit UI thread.

        180s default (was 120s) because cold-start time is dominated by
        disk read speed, not generation — on slower disks, machines with
        active antivirus scanning, or larger models (Llama3 8B ≈5GB),
        even a 1-token request needs the full model loaded into RAM first.
        """
        if not HAS_REQUESTS:
            return False
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model, "prompt": "Hi", "stream": False,
                    "keep_alive": self.KEEP_ALIVE,
                    "options": {"num_predict": 1,
                                "num_gpu": NUM_GPU, "num_ctx": NUM_CTX},
                },
                timeout=timeout,
            )
            ok = r.status_code == 200
            if ok:
                self._warm_models.add(model)
            return ok
        except Exception:
            return False

    def generate(self, model: str, prompt: str,
                 system: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = MAX_TOKENS,
                 timeout: float = None,
                 stream: bool = False) -> Optional[str]:
        """
        Send a generate request; return text response or None on failure.
        Returning None (not an error string) lets the caller retry or
        fall back to the rule-based engine cleanly.
        """
        if not HAS_REQUESTS:
            return None
        timeout = timeout or TIMEOUT_S
        payload = {
            "model":      model,
            "prompt":     prompt,
            "stream":     stream,
            "keep_alive": self.KEEP_ALIVE,
            "options":    _base_options(temperature, max_tokens),
        }
        if system:
            payload["system"] = system

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload, timeout=timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            if text:
                self._warm_models.add(model)
            return text if text else None
        except requests.Timeout:
            return None
        except Exception:
            return None

    def chat(self, model: str, messages: list,
             temperature: float = 0.3,
             max_tokens: int = MAX_TOKENS,
             timeout: float = None) -> Optional[str]:
        """Send a chat request; return assistant text or None on failure."""
        if not HAS_REQUESTS:
            return None
        timeout = timeout or TIMEOUT_S
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": self.KEEP_ALIVE,
            "options":    _base_options(temperature, max_tokens),
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload, timeout=timeout,
            )
            r.raise_for_status()
            text = (r.json()
                      .get("message", {})
                      .get("content", "")
                      .strip())
            if text:
                self._warm_models.add(model)
            return text if text else None
        except requests.Timeout:
            return None
        except Exception:
            return None

    def chat_stream(self, model: str, messages: list,
                    temperature: float = 0.3,
                    max_tokens: int = MAX_TOKENS,
                    timeout: float = 180.0):
        """
        Generator yielding response text chunks as they arrive.
        Used for the chat UI so the user sees tokens appear immediately
        instead of staring at a blank spinner during cold start.
        Yields nothing (empty generator) on any failure — caller must
        handle the "no chunks received" case as a failure signal.
        """
        if not HAS_REQUESTS:
            return
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     True,
            "keep_alive": self.KEEP_ALIVE,
            "options":    _base_options(temperature, max_tokens),
        }
        try:
            with requests.post(f"{self.base_url}/api/chat", json=payload,
                              timeout=timeout, stream=True) as r:
                r.raise_for_status()
                got_any = False
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        got_any = True
                        yield chunk
                    if data.get("done"):
                        break
                if got_any:
                    self._warm_models.add(model)
        except Exception:
            return

    def generate_with_retry(self, model: str, prompt: str,
                            system: Optional[str] = None,
                            temperature: float = 0.3,
                            max_tokens: int = MAX_TOKENS,
                            cold_start: bool = True,
                            ) -> Optional[str]:
        """
        Try generate() with cold-start-aware retry budgets.

        Budgets are sized for CPU-only inference (num_gpu=0): a full
        ~350-token answer generates at a few tok/s, so it needs ~60-90s
        even when the model is already warm — the old 30s "warm" budget
        was itself a hidden cause of spurious rule-based fallbacks.

        cold_start=True  (model not yet confirmed warm):
            attempt 1: 240s budget, full response  (covers first-query
                       weight page-in + full generation)
            attempt 2: 120s budget, shorter response
        cold_start=False (model already warm):
            attempt 1: 180s budget, full response
            attempt 2: 90s budget, shorter response
        """
        attempts = ((240, max_tokens), (120, 250)) if cold_start else \
                   ((180, max_tokens), (90, 250))
        for timeout_s, mt in attempts:
            result = self.generate(model, prompt, system=system,
                                   temperature=temperature, max_tokens=mt,
                                   timeout=timeout_s)
            if result:
                return result
        return None

    def chat_with_retry(self, model: str, messages: list,
                        temperature: float = 0.3,
                        max_tokens: int = MAX_TOKENS,
                        cold_start: bool = True,
                        ) -> Optional[str]:
        """Same cold-start-aware retry strategy as generate_with_retry."""
        attempts = ((240, max_tokens), (120, 250)) if cold_start else \
                   ((180, max_tokens), (90, 250))
        for timeout_s, mt in attempts:
            result = self.chat(model, messages, temperature=temperature,
                              max_tokens=mt, timeout=timeout_s)
            if result:
                return result
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Rule-based fallback engine
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedAdvisor:
    """
    Physics-driven advice when Ollama is unavailable.
    Uses the structured knowledge base + DI thresholds.
    """

    def __init__(self, kb: dict):
        self.kb = kb
        self.pc = kb.get("physics_constants", {})
        self.rules = kb.get("physics_rules", [])
        self.tgt   = kb.get("drl_targets", {})

    def _risk_level(self, DI: float) -> str:
        scale = self.pc.get("DI_scale", {})
        for level, (lo, hi) in scale.items():
            if lo <= DI < hi:
                return level
        return "FAILURE" if DI >= 0.80 else "SAFE"

    def analyze(self, sim: dict) -> str:
        DI_pk  = sim.get("DI_peak",   0.0)
        DI_fin = sim.get("DI_final",  0.0)
        t_tot  = sim.get("t_total",   0.0)
        Bi     = sim.get("Bi",        0.0)
        T_reh  = sim.get("T_reheat",  0.0)
        t_reh  = sim.get("t_reheat",  0.0)
        heal   = sim.get("DI_healed", 0.0)
        n_zon  = sim.get("n_zones",   1)
        rl     = self._risk_level(DI_pk)

        # Assessment
        risk_text = {
            "SAFE":     "✅ No ヒケ expected — product quality acceptable.",
            "CAUTION":  "⚠️ Minor surface imperfections possible at top fill point.",
            "WARNING":  "🔶 ヒケ (sink marks) likely visible — process adjustment needed.",
            "CRITICAL": "🚨 Severe surface defects — production reject likely.",
            "FAILURE":  "🛑 Critical failure — halt and redesign cooling schedule.",
        }.get(rl, "Unknown risk level.")

        # Root cause
        causes = []
        if Bi > 0.5:
            causes.append(f"High Biot number ({Bi:.3f} > 0.5) indicates non-uniform radial cooling — large T_surface/T_core gradient accelerates ヒケ formation.")
        if n_zon <= 1:
            causes.append("Single cooling zone does not slow mushy-zone traversal (62–72°C) — rapid quench maximises shrinkage stress.")
        if t_reh == 0:
            causes.append("No reheat applied — accumulated damage (DI) cannot be healed.")
        if t_reh > 0 and T_reh < 62:
            causes.append(f"T_reheat = {T_reh:.0f}°C is below solidus (62°C) — surface softens but cannot re-enter mushy zone for effective healing.")
        if t_tot > 30:
            causes.append(f"Total time ({t_tot:.0f} min) exceeds 30-min target — process is too slow for production throughput.")
        if not causes:
            causes.append("All process parameters within recommended range.")

        # Recommendations
        recs = []
        if DI_pk >= 0.25:
            if Bi > 0.5:
                h_rec = max(2, round(0.5 * 0.25 / 0.0125))
                recs.append(f"Reduce h_cool to ≤ {h_rec} W/m²K (currently Bi={Bi:.2f}) to enforce Bi ≤ 0.5.")
            if n_zon <= 2:
                recs.append("Add 2–3 step-cooling zones: Zone1≈65–70°C / Zone2≈50°C / Zone3≈37°C to slow mushy traversal.")
            if t_reh == 0 or T_reh < 75:
                recs.append("Apply hot-air reheat at T_reheat ≥ 80°C for ≥ 10 min after cooling to heal accumulated damage.")
            if T_reh >= 75 and t_reh < 10:
                recs.append(f"Increase reheat duration from {t_reh:.0f} → ≥ 10 min for complete healing.")
        else:
            recs.append("Current schedule is within acceptable range. Minor optimisation: raise T_reheat to 90°C for additional DI reduction.")

        # Production comparison
        prod_DI   = self.kb["trials"]["production"]["DI_estimated"]
        prod_time = self.kb["trials"]["production"]["bulk_cooling_profile"]["total_time_min"]

        lines = [
            f"ASSESSMENT: {risk_text}",
            f"\nPHYSICAL MEANING:",
            f"  Peak DI = {DI_pk:.3f}, Final DI = {DI_fin:.3f} — " + (
                "surface should be smooth." if DI_pk < 0.10
                else f"expect {'slight' if DI_pk < 0.25 else 'visible'} depression at top fill point."
            ),
            f"  Reheat healed ΔDI = {heal:+.3f} ({(heal/max(DI_pk,0.01)*100):.0f}% of peak)." if heal > 0.005 else
            f"  No healing applied.",
            f"\nROOT CAUSE:",
        ] + [f"  • {c}" for c in causes] + [
            f"\nRECOMMENDATION:",
        ] + [f"  {i+1}. {r}" for i, r in enumerate(recs)] + [
            f"\nCOMPARISON WITH PRODUCTION LINE:",
            f"  Production baseline: DI ≈ {prod_DI:.2f}, total time = {prod_time} min (back-fill 74.9°C, 5×20sec at 100°C).",
            f"  Current simulation: DI = {DI_pk:.3f}, total = {t_tot:.0f} min.",
            f"  {'Current is ' + ('better ✅' if DI_pk < prod_DI else 'worse ⚠️') + f' than production (ΔDI={DI_pk-prod_DI:+.3f}).' if DI_pk != prod_DI else 'Matches production baseline.'}",
        ]

        return "\n".join(lines)

    def answer_question(self, question: str, context: str) -> str:
        """Simple keyword-based Q&A from rules and knowledge."""
        q = question.lower()
        kb = self.kb

        if any(w in q for w in ["hike", "ヒケ", "sink", "mark", "defect"]):
            return (
                "ヒケ (sink marks) form at the top fill point due to volume shrinkage (~8%) "
                "during solidification. The liquid center contracts as it solidifies, pulling "
                "the top surface downward. To prevent: (1) slow the mushy-zone traversal "
                "(62–72°C), (2) apply hot-air reheat at ≥80°C for ≥10 min to re-fill the void, "
                "(3) keep Biot number ≤ 0.5 to minimise radial temperature gradients."
            )
        if any(w in q for w in ["biot", "bi", "convection", "h_cool"]):
            return (
                "The Biot number Bi = h·R/k = h × 0.0125 / 0.25 = 0.05×h.\n"
                "For Bi ≤ 0.5: h ≤ 10 W/m²K. We recommend h_cool ≤ 6 W/m²K (Bi ≤ 0.3).\n"
                "High Bi → large surface/core ΔT → non-uniform shrinkage → ヒケ.\n"
                "Bi ≤ 0.1 (h≤2 W/m²K) gives nearly uniform radial cooling — ideal but slow."
            )
        if any(w in q for w in ["reheat", "heal", "repair"]):
            return (
                "Reheat heals ヒケ by re-entering the mushy zone (62–72°C), allowing the "
                "residual liquid to re-fill the void before final solidification.\n"
                "Minimum effective parameters (from Trial 2 and production data):\n"
                "  • T_reheat ≥ 75°C (surface must reach mushy zone)\n"
                "  • Duration ≥ 10 min (short pulses like 1 min/3× are insufficient)\n"
                "  • Better: T_reheat ≥ 90°C for DI < 0.10\n"
                "  • Production uses 100°C × 20sec × 5 pulses DURING mushy traversal."
            )
        if any(w in q for w in ["zone", "step", "cool", "temperature"]):
            return (
                "Step-cooling zones slow the mushy-zone traversal, reducing peak DI.\n"
                "Recommended schedule:\n"
                "  Zone 1: 65–70°C / 5–6 min  (slow entry into mushy)\n"
                "  Zone 2: 50°C   / 6–7 min   (core passes through solidus)\n"
                "  Zone 3: 37°C   / 5–7 min   (approach target before reheat)\n"
                "  Reheat: 80–90°C / 10 min    (heal surface, DI → <0.10)\n"
                "Total: ≈ 16–20 min cooling + 10 min reheat + 2–4 min final ≤ 30 min."
            )
        if any(w in q for w in ["di ", "damage index", "damage"]):
            return (
                "The Damage Index (DI) is computed from an ODE physics model:\n"
                "  dDI/dt = k_form × (1−DI) − k_heal × DI\n"
                "Where:\n"
                "  k_form ∝ Biot × liquid_fraction (rises during mushy zone)\n"
                "  k_heal ∝ (T−55)/10 × 3.5 (activated when surface > 55°C in reheat)\n"
                "Thresholds: SAFE<0.10, CAUTION<0.25, WARNING<0.50, CRITICAL<0.80."
            )
        if any(w in q for w in ["30 min", "time", "duration", "target"]):
            return (
                "The 30-min production target includes:\n"
                "  Cooling zones: 16–20 min (3–5 zones)\n"
                "  Reheat: 10 min\n"
                "  Final cool to room temp: 2–4 min\n"
                "Production baseline achieves 24.5 min. DRL optimal achieves 27–29 min.\n"
                "Exceeding 30 min reduces throughput significantly."
            )
        # Generic fallback
        return (
            "Based on the LCWT401 experimental data:\n"
            f"{context[:400] if context else 'No simulation context available.'}\n\n"
            "For detailed advice, please ensure Ollama (pidrl-advisor model) is running: "
            "ollama run pidrl-advisor"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LangChain conversational engine (contextual chat memory)
# ═══════════════════════════════════════════════════════════════════════════

class LangChainChatEngine:
    """
    Conversational chat built on LangChain + ChatOllama.

    Why LangChain here: it gives the advisor a proper *contextual memory* of
    the conversation. Each turn, LangChain automatically replays the running
    message history into the model via RunnableWithMessageHistory, so follow-up
    questions like "and what about the second zone?" are understood in context
    without us hand-rolling the history plumbing.

    Design notes:
      • Uses the SAME Ollama model + CPU-only options (num_gpu=0) as the rest
        of the advisor — see NUM_GPU. No second model, no extra RAM.
      • RAG context is injected into the *system* message each turn (fresh per
        query) so the stored chat history stays clean (just Q/A pairs).
      • Streams tokens via .stream() for a live "typing" feel in the UI.
      • Memory is per-session (keyed session_id); we use one "default" session
        per advisor instance and trim it to the last MAX_TURNS exchanges.
    """

    MAX_MESSAGES = 12   # keep last ~6 Q/A exchanges in working memory

    def __init__(self, model: str, base_url: str, system_prompt: str):
        self.model      = model
        self.base_url   = base_url
        self.system     = system_prompt
        self._store: dict = {}   # session_id -> InMemoryChatMessageHistory

        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            num_gpu=NUM_GPU,          # CPU-only — see NUM_GPU rationale above
            num_ctx=NUM_CTX,
            num_predict=MAX_TOKENS,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.3,
            keep_alive="30m",
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system}\n\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])
        chain = prompt | self.llm
        self.runnable = RunnableWithMessageHistory(
            chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_history(self, session_id: str):
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def _trim(self, session_id: str):
        hist = self._store.get(session_id)
        if hist and len(hist.messages) > self.MAX_MESSAGES:
            hist.messages = hist.messages[-self.MAX_MESSAGES:]

    def stream(self, question: str, rag_context: str = "",
               session_id: str = "default"):
        """Yield response text chunks; raises on transport error so the
        caller can fall back. Empty generator (no chunks) also signals
        failure to the caller."""
        cfg = {"configurable": {"session_id": session_id}}
        for chunk in self.runnable.stream(
                {"system": self.system, "context": rag_context, "input": question},
                config=cfg):
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text
        self._trim(session_id)

    def add_exchange(self, user: str, assistant: str, session_id: str = "default"):
        """Manually record a turn that was answered outside LangChain
        (raw stream / rule-based) so conversational memory stays complete."""
        hist = self._get_history(session_id)
        hist.add_user_message(user)
        hist.add_ai_message(assistant)
        self._trim(session_id)

    def clear(self, session_id: str = "default"):
        self._store.pop(session_id, None)

    def update_system(self, system_prompt: str):
        self.system = system_prompt


# ═══════════════════════════════════════════════════════════════════════════
#  Groq cloud backend (free tier) — used in deployment
# ═══════════════════════════════════════════════════════════════════════════

class GroqEngine:
    """
    Cloud LLM backend on Groq's free tier (OpenAI-compatible, very fast).

    Why this exists: the local `pidrl-advisor` (Mistral 7B via Ollama) can't
    run on a free cloud host, so for the deployed site we generate with a
    hosted model instead — keeping the SAME lean system prompt + RAG context,
    so answers stay grounded in the client's experimental data. Activated only
    when GROQ_API_KEY is set; locally (no key) the advisor keeps using Ollama.
    """

    # Tried in order — Groq periodically retires model names, so if the
    # configured one is decommissioned we transparently fall to the next.
    FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                       "llama-3.1-70b-versatile", "llama3-70b-8192"]

    def __init__(self, api_key: str, model: str = GROQ_MODEL_DEFAULT):
        self.model  = model
        self.client = Groq(api_key=api_key)
        self.models = [model] + [m for m in self.FALLBACK_MODELS if m != model]
        self.last_error = ""

    @staticmethod
    def _is_model_error(e) -> bool:
        s = str(e).lower()
        return ("model" in s and any(k in s for k in
                ("decommission", "not found", "does not exist", "invalid",
                 "unavailable", "deprecat")))

    def _messages(self, system, messages):
        return [{"role": "system", "content": system}] + messages

    def stream(self, system: str, messages: list,
               temperature: float = 0.3, max_tokens: int = 700):
        """Yield response text chunks; auto-falls to the next model name if the
        configured one is decommissioned. Raises non-model errors (auth/rate)
        so the caller can record them and fall back to rule-based."""
        last_exc = None
        for m in self.models:
            try:
                resp = self.client.chat.completions.create(
                    model=m, messages=self._messages(system, messages),
                    temperature=temperature, max_tokens=max_tokens, top_p=0.9,
                    stream=True,
                )
            except Exception as e:
                last_exc = e; self.last_error = f"{m}: {e}"
                if self._is_model_error(e):
                    continue
                raise
            self.model = m; self.last_error = ""
            for chunk in resp:
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    yield delta
            return
        if last_exc:
            raise last_exc

    def generate(self, system: str, prompt: str,
                 temperature: float = 0.3, max_tokens: int = 700) -> str:
        """Non-streaming completion → full text (auto model fallback)."""
        last_exc = None
        for m in self.models:
            try:
                resp = self.client.chat.completions.create(
                    model=m, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens, top_p=0.9,
                    stream=False,
                )
                self.model = m; self.last_error = ""
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_exc = e; self.last_error = f"{m}: {e}"
                if self._is_model_error(e):
                    continue
                raise
        if last_exc:
            raise last_exc
        return ""

    def ping(self):
        """One-token connectivity/credential check. Returns (ok, error_str)."""
        try:
            self.generate("You are a test.", "Reply with: ok", max_tokens=2)
            return True, ""
        except Exception as e:
            return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Advisor class
# ═══════════════════════════════════════════════════════════════════════════

class PILLMAdvisor:
    """
    Primary entry point for LLM-based process advice.

    Priority:
      1. Ollama 'pidrl-advisor' model (custom-trained on client data)
      2. Ollama base model (mistral/llama3.1:8b) with injected system prompt
      3. Rule-based fallback engine

    Thread-safe: uses a simple dict cache keyed by MD5 hash of inputs.
    """

    def __init__(self, ollama_url: str = OLLAMA_URL,
                 model_name:   str = MODEL_NAME,
                 kb_path:      str = KB_PATH,
                 vs_path:      str = VS_PATH):

        self.client = OllamaClient(ollama_url)
        self.model  = model_name
        self._cache : dict = {}
        self._history: list = []   # chat history

        # Load knowledge base
        self.kb = self._load_kb(kb_path)

        # Load RAG vector store (TF-IDF retrieval index)
        self.vector_store = None
        self.rag_active    = False
        if HAS_VECTOR_STORE:
            vs = VectorStore()
            if vs.load(vs_path):
                self.vector_store = vs
                self.rag_active   = True
            elif self.kb:
                # No saved index yet — build one in-memory from current KB
                try:
                    from vector_store_v4 import chunk_knowledge_base
                    vs.add_documents(chunk_knowledge_base(self.kb))
                    vs.build_index()
                    self.vector_store = vs
                    self.rag_active   = True
                except Exception as e:
                    print(f"[!] Could not build in-memory RAG index: {e}")

        # Lean system prompt for RAG-augmented runtime queries (short = fast)
        try:
            from prepare_llm_v4 import build_lean_system_prompt
            self.lean_system_prompt = build_lean_system_prompt()
        except ImportError:
            self.lean_system_prompt = (
                "You are an expert process engineer for LCWT401 deodorant "
                "stick solidification (CBIC x TUAT). Be concise and specific."
            )

        # Full system prompt (used only as last-resort fallback if RAG unavailable)
        sp_path = Path("system_prompt_v4.txt")
        self.system_prompt = sp_path.read_text(encoding="utf-8") if sp_path.exists() else ""

        # Build rule-based fallback engine
        self.rule_engine = RuleBasedAdvisor(self.kb) if self.kb else None

        # ── Cloud backend (Groq) — read fresh from env so Streamlit secrets
        #    set just before construction are honoured. When a key is present
        #    this becomes the PRIMARY backend (deployment); without it, the
        #    advisor uses local Ollama exactly as before.
        self.groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
        self.groq_model = os.environ.get("GROQ_MODEL", GROQ_MODEL_DEFAULT)
        self.groq_engine = None
        if HAS_GROQ and self.groq_key:
            try:
                self.groq_engine = GroqEngine(self.groq_key, self.groq_model)
            except Exception as e:
                print(f"[PILLMAdvisor] Groq init failed: {e}")
                self.groq_engine = None

        # Detect mode
        self._mode = self._detect_mode()

        # ── Model health check ────────────────────────────────────────────
        # Detects the specific bug that causes repeated timeouts/garbled
        # output: an OLD registered model with a broken ChatML TEMPLATE
        # override. Updating the .py files does NOT retroactively fix an
        # already-created Ollama model — only re-running the training
        # pipeline (which deletes + recreates it) does. This check
        # surfaces that mismatch immediately instead of silently timing
        # out on every query.
        self.model_health = {"healthy": None, "reason": "not checked", "exists": False}
        if self._mode.startswith("ollama"):
            model_to_check = self._mode.split(":")[1].split("+")[0]
            self.model_health = self.client.check_model_health(model_to_check)
            if self.model_health["healthy"] is False:
                print(f"[PILLMAdvisor] ⚠️ MODEL HEALTH ISSUE: {self.model_health['reason']} "
                      f"— retrain via prepare_llm_v4.py or the Upload & Train panel.")

        # ── Background model warm-up ─────────────────────────────────────
        # The #1 cause of "[TIMEOUT] Ollama retries exhausted" is COLD
        # START: loading a multi-GB model from disk takes 30-90s before
        # any tokens are generated. We fire a non-blocking warm-up request
        # immediately so that by the time the user actually asks a
        # question, the model is likely already loaded into RAM.
        # Skipped entirely if the model is already known to be unhealthy —
        # no point warming up a model that will just produce garbage.
        self._warm_lock = threading.Lock()
        self._is_warm   = False
        if self._mode.startswith("ollama") and self.model_health.get("healthy") is not False:
            model_to_warm = self._mode.split(":")[1].split("+")[0]
            if self.client.is_model_warm(model_to_warm):
                self._is_warm = True
            else:
                threading.Thread(
                    target=self._background_warm_up,
                    args=(model_to_warm,), daemon=True,
                ).start()

        # ── LangChain conversational engine ──────────────────────────────
        # Gives the chat real contextual memory (follow-up questions
        # understood in context). Optional: only built if langchain-ollama
        # is installed and we're in a healthy Ollama mode. Falls back
        # transparently to the raw-requests streaming path otherwise.
        self.lc_engine = None
        self._build_lc_engine()

        rag_status = f"ON ({len(self.vector_store)} chunks)" if self.rag_active else "OFF"
        if self._mode.startswith("groq"):
            warm_status = f"cloud (Groq · {self.groq_model})"
        elif not self._mode.startswith("ollama"):
            warm_status = "n/a (rule-based mode)"
        elif self.model_health.get("healthy") is False:
            warm_status = "⚠️ UNHEALTHY — retrain required"
        else:
            warm_status = "warm" if self._is_warm else "warming up in background..."
        lc_status = "ON" if self.lc_engine else ("unavailable" if not HAS_LANGCHAIN else "off")
        print(f"[PILLMAdvisor] Mode: {self._mode} | RAG: {rag_status} | "
              f"LangChain: {lc_status} | Model: {warm_status}")

    def _build_lc_engine(self):
        """(Re)build the LangChain conversational engine for the current
        Ollama model. Safe no-op if langchain isn't installed, Ollama is
        offline, or the model is known-unhealthy."""
        self.lc_engine = None
        if not (HAS_LANGCHAIN and self._mode.startswith("ollama")
                and self.model_health.get("healthy") is not False):
            return
        try:
            model_to_use = self._mode.split(":")[1].split("+")[0]
            sys_prompt = self.lean_system_prompt if self.rag_active else (
                self.system_prompt or self.lean_system_prompt)
            self.lc_engine = LangChainChatEngine(
                model=model_to_use, base_url=self.client.base_url,
                system_prompt=sys_prompt)
        except Exception as e:
            print(f"[PILLMAdvisor] LangChain engine unavailable: {e}")
            self.lc_engine = None

    def _background_warm_up(self, model: str):
        """Runs in a daemon thread — loads the model into Ollama's memory."""
        ok = self.client.warm_up(model, timeout=240)
        with self._warm_lock:
            self._is_warm = ok
        if ok:
            print(f"[PILLMAdvisor] Model '{model}' warmed up and ready.")
        else:
            print(f"[PILLMAdvisor] Warm-up failed for '{model}' "
                  f"(disk read may be slow — will retry with full budget on first query).")

    @property
    def is_warm(self) -> bool:
        """
        True if the model is confirmed loaded in Ollama's memory.
        Checks the cached flag first (fast path — already confirmed by a
        prior successful call or warm-up). If not yet confirmed, does a
        live, lightweight check against Ollama's /api/ps endpoint — this
        catches cases where the model is already warm for OTHER reasons
        (e.g. the user ran `ollama run mistral` manually moments ago, or
        a previous app session's keep_alive window hasn't expired yet)
        even if THIS advisor instance's own warm-up attempt failed/hasn't
        run yet.
        """
        if self._mode.startswith("groq"):
            return True   # cloud backend is always "ready" — no cold start
        with self._warm_lock:
            if self._is_warm:
                return True
        if not self._mode.startswith("ollama"):
            return False
        model_to_check = self._mode.split(":")[1].split("+")[0]
        live_warm = self.client.is_model_warm(model_to_check)
        if live_warm:
            with self._warm_lock:
                self._is_warm = True
        return live_warm

    # ── Helpers ──────────────────────────────────────────────────────────

    def _load_kb(self, path: str) -> dict:
        p = Path(path)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Could not load knowledge base: {e}")
        return {}

    def _detect_mode(self) -> str:
        # Cloud backend wins when configured (deployment); avoids probing
        # localhost Ollama on a cloud host where it doesn't exist.
        if getattr(self, "groq_engine", None) is not None:
            return f"groq:{self.groq_model}"
        if not self.client.is_alive():
            return "rule-based"
        models = self.client.list_models()
        if any(self.model in m for m in models):
            return f"ollama:{self.model}"
        if any(FALLBACK_MODEL in m for m in models):
            return f"ollama:{FALLBACK_MODEL}+system-prompt"
        if models:
            return f"ollama:{models[0].split(':')[0]}+system-prompt"
        return "rule-based"

    def _cache_key(self, *args) -> str:
        return hashlib.md5(json.dumps(args, sort_keys=True,
                                      default=str).encode()).hexdigest()[:12]

    def _retrieve_context(self, query: str, top_k: int = RAG_TOP_K) -> str:
        """
        RAG retrieval: pull the top_k most relevant knowledge chunks for
        this query and format them as a short context block. This is what
        replaces the old "send the entire 900-word prompt every time"
        approach — much shorter context = much faster Ollama response.
        """
        if not self.rag_active or not self.vector_store:
            return ""
        results = self.vector_store.search(query, top_k=top_k, min_score=0.02)
        if not results:
            return ""
        lines = [f"  - {r['text']}" for r in results]
        return "Relevant facts from experimental data:\n" + "\n".join(lines)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_llm_active(self) -> bool:
        return self._mode.startswith("ollama") or self._mode.startswith("groq")

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(self, sim_results: dict) -> str:
        """
        Analyze PINN/DRL simulation results and return expert advice.
        Uses RAG: retrieves relevant facts from the vector store instead
        of sending the full knowledge base every time -> fast + accurate.

        sim_results keys:
          DI_peak, DI_final, t_total, Bi, T_reheat, t_reheat,
          DI_healed, n_zones, T_fill, zones (list), scenario
        """
        key = self._cache_key("analyze", sim_results)
        if key in self._cache:
            return self._cache[key]

        zones = sim_results.get("zones", [])
        zone_text = " -> ".join(
            f"Zone{i+1}({z['T']:.0f}C/{z['duration']:.0f}min)"
            for i, z in enumerate(zones)
        ) if zones else "N/A"

        risk  = sim_results.get('risk', 'UNKNOWN')
        DI_pk = sim_results.get('DI_peak', 0)

        # RAG retrieval query — built from the situation so it matches
        # semantically with trial descriptions and physics rules.
        rag_query = (
            f"DI {DI_pk:.2f} {risk} Biot {sim_results.get('Bi',0):.2f} "
            f"reheat {sim_results.get('T_reheat',0):.0f}C "
            f"zones {sim_results.get('n_zones',1)} hike sink mark cause fix"
        )
        retrieved = self._retrieve_context(rag_query, top_k=RAG_TOP_K)

        sim_block = (
            f"T_fill={sim_results.get('T_fill', 80):.0f}C | "
            f"Cooling: {zone_text} | "
            f"T_reheat={sim_results.get('T_reheat', 0):.0f}C "
            f"({sim_results.get('t_reheat', 0):.0f}min) | "
            f"h_cool={sim_results.get('h_cool', 6):.1f}W/m2K | "
            f"Bi={sim_results.get('Bi', 0):.3f} | "
            f"Peak_DI={DI_pk:.4f} | Final_DI={sim_results.get('DI_final', 0):.4f} | "
            f"DI_healed={sim_results.get('DI_healed', 0):+.4f} | "
            f"Total_time={sim_results.get('t_total', 0):.1f}min | "
            f"Risk={risk}"
        )

        prompt = (
            f"Simulation result: {sim_block}\n\n"
            f"{retrieved}\n\n"
            f"Give a short analysis with these 5 labelled sections: "
            f"ASSESSMENT (one line), PHYSICAL MEANING, ROOT CAUSE, "
            f"RECOMMENDATION (specific numbers), COMPARISON (vs experimental data above)."
        )

        result = self._query_llm(prompt, fallback_sim=sim_results)
        self._cache[key] = result
        return result

    def chat(self, user_message: str,
             context: str = "",
             sim_results: Optional[dict] = None) -> str:
        """
        Conversational interface with RAG retrieval. Maintains chat history.
        """
        key = self._cache_key("chat", user_message, context[:200])
        if key in self._cache:
            return self._cache[key]

        # RAG: retrieve facts relevant to the user's actual question
        retrieved = self._retrieve_context(user_message, top_k=RAG_TOP_K)

        ctx_parts = []
        if sim_results:
            ctx_parts.append(
                f"Current simulation: DI_peak={sim_results.get('DI_peak',0):.3f} "
                f"[{sim_results.get('risk','?')}], "
                f"total={sim_results.get('t_total',0):.0f}min, "
                f"Bi={sim_results.get('Bi',0):.3f}"
            )
        if retrieved:
            ctx_parts.append(retrieved)

        ctx_str = "\n\n".join(ctx_parts)
        full_prompt = f"{ctx_str}\n\nQuestion: {user_message}" if ctx_str else user_message

        result = self._query_llm(full_prompt, chat_mode=True, fallback_sim=sim_results,
                                 fallback_question=user_message)
        self._cache[key] = result

        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant", "content": result})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return result

    def chat_stream(self, user_message: str,
                    context: str = "",
                    sim_results: Optional[dict] = None):
        """
        Streaming version of chat() — yields text chunks as they arrive
        from Ollama, so the UI can display tokens in real-time via
        st.write_stream(). This dramatically improves perceived latency
        during cold start: instead of a blank spinner for 60-90s, the
        user sees the model "thinking" and then streaming text.

        If Ollama is unavailable or streaming yields nothing at all,
        falls back to the rule-based engine and yields its full answer
        as a single chunk (still works with st.write_stream).

        Sets self._last_source same as chat() for UI source-badge display.
        """
        retrieved = self._retrieve_context(user_message, top_k=RAG_TOP_K)
        ctx_parts = []
        if sim_results:
            ctx_parts.append(
                f"Current simulation: DI_peak={sim_results.get('DI_peak',0):.3f} "
                f"[{sim_results.get('risk','?')}], "
                f"total={sim_results.get('t_total',0):.0f}min, "
                f"Bi={sim_results.get('Bi',0):.3f}"
            )
        if retrieved:
            ctx_parts.append(retrieved)
        ctx_str = "\n\n".join(ctx_parts)
        full_prompt = f"{ctx_str}\n\nQuestion: {user_message}" if ctx_str else user_message

        full_response = ""
        lc_handled    = False   # did the LangChain engine already store this turn?

        # ── PRIMARY (deployment): Groq cloud backend, streaming ───────────
        if self.groq_engine is not None:
            sys_prompt = self.lean_system_prompt if self.rag_active else (
                self.system_prompt or self.lean_system_prompt)
            messages = self._history[-6:] + [{"role": "user", "content": full_prompt}]
            self._last_was_cold_start = False
            try:
                for chunk in self.groq_engine.stream(sys_prompt, messages, max_tokens=700):
                    full_response += chunk
                    yield chunk
            except Exception as e:
                print(f"[!] Groq stream error: {e} — using rule-based fallback")
                full_response = ""
            if full_response.strip():
                self._last_source = f"groq:{self.groq_model}"
            else:
                self._last_source = "rule-based (Groq error)"
                full_response = self._fallback(sim_results, user_message)
                yield full_response

        elif self._mode.startswith("ollama") and self.model_health.get("healthy") is not False:
            mode_parts   = self._mode.split(":")
            model_to_use = mode_parts[1].split("+")[0] if len(mode_parts) > 1 else self.model
            sys_prompt   = self.lean_system_prompt if self.rag_active else (
                self.system_prompt or self.lean_system_prompt
            )
            self._last_was_cold_start = not self.is_warm

            # ── PRIMARY: LangChain contextual engine ──────────────────────
            # Streams with full conversational memory (RunnableWithMessage
            # History). RAG context passed separately so chat memory stays
            # clean. On any error/empty output, we transparently fall
            # through to the raw-requests path below.
            if self.lc_engine is not None:
                try:
                    for chunk in self.lc_engine.stream(user_message, rag_context=ctx_str):
                        full_response += chunk
                        yield chunk
                    if full_response.strip():
                        lc_handled = True
                        self._last_source = f"ollama:{model_to_use} (LangChain)"
                        with self._warm_lock:
                            self._is_warm = True
                except Exception as e:
                    print(f"[!] LangChain stream failed ({e}) — raw-stream fallback")
                    full_response = ""

            # ── SECONDARY: raw-requests streaming ─────────────────────────
            if not full_response.strip():
                messages = [{"role": "system", "content": sys_prompt}]
                messages += self._history[-6:]
                messages.append({"role": "user", "content": full_prompt})
                timeout = 240.0 if self._last_was_cold_start else 180.0
                for chunk in self.client.chat_stream(model_to_use, messages, timeout=timeout):
                    full_response += chunk
                    yield chunk
                if full_response.strip():
                    self._last_source = f"ollama:{model_to_use}"
                    with self._warm_lock:
                        self._is_warm = True

            # ── TERTIARY: rule-based fallback ─────────────────────────────
            if not full_response.strip():
                print("[!] Streaming yielded no tokens — using rule-based fallback")
                self._last_source = "rule-based (Ollama timeout)"
                full_response = self._fallback(sim_results, user_message)
                yield full_response
        else:
            self._last_source = ("rule-based (model needs retraining)"
                                 if self.model_health.get("healthy") is False
                                 else "rule-based (Ollama offline)")
            full_response = self._fallback(sim_results, user_message)
            yield full_response

        # Update history + cache once streaming is complete
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant", "content": full_response})
        if len(self._history) > 20:
            self._history = self._history[-20:]
        # Keep LangChain memory complete even when a non-LangChain path
        # answered this turn, so follow-ups stay contextual.
        if not lc_handled and self.lc_engine is not None:
            try:
                self.lc_engine.add_exchange(user_message, full_response)
            except Exception:
                pass
        key = self._cache_key("chat", user_message, context[:200])
        self._cache[key] = full_response

    def quick_summary(self, DI_peak: float, t_total: float, Bi: float) -> str:
        """
        Single-line advisory for the KPI bar / header badge.
        Returns ≤ 60 chars suitable for display in the Streamlit header.
        """
        if DI_peak <= 0.10:
            return "✅ No ヒケ expected — production ready"
        if DI_peak <= 0.25:
            return "⚠️ Minor surface marks possible — consider raising T_reheat"
        if DI_peak <= 0.50:
            msgs = []
            if Bi > 0.5: msgs.append("reduce h_cool")
            msgs.append("increase T_reheat ≥ 80°C")
            return "🔶 ヒケ likely — " + " & ".join(msgs)
        if DI_peak <= 0.80:
            return "🚨 Severe ヒケ — redesign zone schedule + strong reheat"
        return "🛑 Critical failure — halt production"

    def groq_ping(self):
        """Live connectivity/credential check for the Groq backend.
        Returns (ok: bool, detail: str) — used by the UI to show exactly
        why the cloud LLM is or isn't working on the deployed site."""
        if not HAS_GROQ:
            return False, "groq package not installed"
        if not self.groq_key:
            return False, "GROQ_API_KEY not set"
        if self.groq_engine is None:
            return False, "Groq engine not initialised"
        ok, err = self.groq_engine.ping()
        return ok, (f"connected · model {self.groq_engine.model}" if ok else err)

    def clear_history(self):
        self._history.clear()
        if self.lc_engine is not None:
            self.lc_engine.clear()

    def clear_cache(self):
        self._cache.clear()

    def reload(self, kb_path: str = KB_PATH, vs_path: str = VS_PATH):
        """
        Hot-reload knowledge base, vector store, and re-detect Ollama models.
        Call this after training completes — NO app restart / F5 needed.
        """
        self.kb = self._load_kb(kb_path)
        self.rule_engine = RuleBasedAdvisor(self.kb) if self.kb else None

        if HAS_VECTOR_STORE:
            vs = VectorStore()
            if vs.load(vs_path):
                self.vector_store = vs
                self.rag_active   = True
            elif self.kb:
                try:
                    from vector_store_v4 import chunk_knowledge_base
                    vs.add_documents(chunk_knowledge_base(self.kb))
                    vs.build_index()
                    self.vector_store = vs
                    self.rag_active   = True
                except Exception:
                    pass

        sp_path = Path("system_prompt_v4.txt")
        if sp_path.exists():
            self.system_prompt = sp_path.read_text(encoding="utf-8")

        self._mode = self._detect_mode()
        self.clear_cache()

        # Re-check model health — confirms whether retraining actually
        # fixed a previously-broken (ChatML template) model.
        self.model_health = {"healthy": None, "reason": "not checked", "exists": False}
        if self._mode.startswith("ollama"):
            model_to_check = self._mode.split(":")[1].split("+")[0]
            self.model_health = self.client.check_model_health(model_to_check)

        # Re-check / re-trigger warm-up for the (possibly new) model —
        # skip entirely if the model is still unhealthy.
        if self._mode.startswith("ollama") and self.model_health.get("healthy") is not False:
            model_to_warm = self._mode.split(":")[1].split("+")[0]
            if self.client.is_model_warm(model_to_warm):
                with self._warm_lock:
                    self._is_warm = True
            else:
                with self._warm_lock:
                    self._is_warm = False
                threading.Thread(
                    target=self._background_warm_up,
                    args=(model_to_warm,), daemon=True,
                ).start()

        # Rebuild the LangChain engine for the (possibly new) model and
        # reset its conversational memory.
        self._build_lc_engine()

        print(f"[PILLMAdvisor] Reloaded. Mode: {self._mode} | "
              f"RAG: {len(self.vector_store) if self.vector_store else 0} chunks | "
              f"LangChain: {'ON' if self.lc_engine else 'off'} | "
              f"Warm: {self.is_warm} | Health: {self.model_health.get('healthy')}")

    def status(self) -> dict:
        # In Groq (cloud) mode don't probe localhost Ollama — it isn't there
        # and the 3s timeout would slow every rerun.
        groq_on = self.groq_engine is not None
        alive   = False if groq_on else self.client.is_alive()
        return {
            "mode":          self._mode,
            "ollama_up":     alive,
            "groq":          groq_on,
            "groq_model":    self.groq_model if groq_on else "",
            "has_groq_pkg":  HAS_GROQ,
            "groq_key_set":  bool(self.groq_key),
            "groq_error":    getattr(self.groq_engine, "last_error", "") if groq_on else "",
            "models":        self.client.list_models() if alive else [],
            "kb_loaded":     bool(self.kb),
            "rag_active":    self.rag_active,
            "rag_chunks":    len(self.vector_store) if self.vector_store else 0,
            "cache_size":    len(self._cache),
            "history_len":   len(self._history),
            "last_source":   getattr(self, "_last_source", "n/a"),
            "model_warm":    self.is_warm,
            "was_cold_start":getattr(self, "_last_was_cold_start", None),
            "model_healthy": self.model_health.get("healthy"),
            "health_reason": self.model_health.get("reason", ""),
            "langchain":     self.lc_engine is not None,
        }

    def recheck_model_health(self) -> dict:
        """
        Re-run the template health check — call this after retraining
        (without needing a full advisor.reload()) to confirm the fix
        took effect immediately.
        """
        if self._mode.startswith("ollama"):
            model_to_check = self._mode.split(":")[1].split("+")[0]
            self.model_health = self.client.check_model_health(model_to_check)
        return self.model_health

    # ── Internal query dispatcher ─────────────────────────────────────────

    def _query_llm(self, prompt: str, chat_mode: bool = False,
                   fallback_sim: Optional[dict] = None,
                   fallback_question: str = "") -> str:
        """
        Query Ollama with cold-start-aware retry budgets. NEVER returns a
        raw error/timeout string to the UI — always falls back to the
        rule engine on total failure, so the user always gets a useful
        answer even if Ollama is completely unresponsive.
        """
        # ── PRIMARY (deployment): Groq cloud backend ──────────────────────
        if self.groq_engine is not None:
            sys_prompt = self.lean_system_prompt if self.rag_active else (
                self.system_prompt or self.lean_system_prompt)
            try:
                result = self.groq_engine.generate(sys_prompt, prompt, max_tokens=700)
                if result:
                    self._last_source = f"groq:{self.groq_model}"
                    return result
            except Exception as e:
                print(f"[!] Groq error: {e} — falling back to rule engine")
            self._last_source = "rule-based (Groq error)"
            return self._fallback(fallback_sim, fallback_question)

        if not self._mode.startswith("ollama"):
            self._last_source = "rule-based (Ollama offline)"
            return self._fallback(fallback_sim, fallback_question)

        # Known-broken model (outdated ChatML template) — don't waste the
        # user's time on a 150s timeout we already know will fail.
        if self.model_health.get("healthy") is False:
            self._last_source = "rule-based (model needs retraining)"
            return self._fallback(fallback_sim, fallback_question)

        mode_parts   = self._mode.split(":")
        model_to_use = mode_parts[1].split("+")[0] if len(mode_parts) > 1 else self.model
        # With RAG active we always use the lean prompt (short + fast),
        # regardless of whether the registered model already has the
        # full knowledge baked in — this keeps latency low and consistent.
        sys_prompt = self.lean_system_prompt if self.rag_active else (
            self.system_prompt or self.lean_system_prompt
        )

        is_cold = not self.is_warm
        self._last_was_cold_start = is_cold

        try:
            if chat_mode and self._history:
                messages  = [{"role": "system", "content": sys_prompt}]
                messages += self._history[-6:]    # last 3 turns (keep short)
                messages.append({"role": "user", "content": prompt})
                result = self.client.chat_with_retry(model_to_use, messages,
                                                     cold_start=is_cold)
            else:
                result = self.client.generate_with_retry(
                    model_to_use, prompt, system=sys_prompt, cold_start=is_cold)

            if result:
                self._last_source = f"ollama:{model_to_use}"
                with self._warm_lock:
                    self._is_warm = True   # confirmed warm now
                return result

            # All retry attempts failed (timeout or empty) — fall back
            reason = "cold start exceeded budget" if is_cold else "model unresponsive"
            print(f"[!] Ollama retries exhausted ({reason}) — using rule-based fallback")
            self._last_source = "rule-based (Ollama timeout)"
            return self._fallback(fallback_sim, fallback_question)

        except Exception as e:
            print(f"[!] Ollama error: {e} — falling back to rule engine")
            self._last_source = "rule-based (error)"
            return self._fallback(fallback_sim, fallback_question)

    def _fallback(self, sim_results: Optional[dict] = None,
                  question: str = "") -> str:
        """
        Rule-based fallback. Prefers structured sim_results (passed
        directly by analyze()/chat()) over text-parsing — far more
        reliable than parsing the LLM prompt string.
        """
        if not self.rule_engine:
            return (
                "⚠️ LLM advisor unavailable (Ollama not running + no knowledge base).\n"
                "Run: python prepare_llm_v4.py\n"
                "Then: ollama run pidrl-advisor"
            )
        if sim_results:
            return self.rule_engine.analyze(sim_results)
        if question:
            return self.rule_engine.answer_question(question, "")
        return self.rule_engine.answer_question("general", "")


# ═══════════════════════════════════════════════════════════════════════════
#  Streamlit-friendly caching helper
# ═══════════════════════════════════════════════════════════════════════════

_advisor_singleton: Optional[PILLMAdvisor] = None

def get_advisor() -> PILLMAdvisor:
    """
    Return a cached singleton PILLMAdvisor.
    Use in Streamlit with @st.cache_resource for session-level caching.
    """
    global _advisor_singleton
    if _advisor_singleton is None:
        _advisor_singleton = PILLMAdvisor()
    return _advisor_singleton


# ═══════════════════════════════════════════════════════════════════════════
#  CLI test runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PI-DRL LLM Advisor test")
    parser.add_argument("--test",  action="store_true", help="Run test suite")
    parser.add_argument("--chat",  type=str, default="", help="Ask a question")
    parser.add_argument("--status",action="store_true", help="Show advisor status")
    args = parser.parse_args()

    advisor = PILLMAdvisor()

    if args.status:
        st_dict = advisor.status()
        print("\nAdvisor Status:")
        for k, v in st_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.chat:
        print(f"\n[User] {args.chat}")
        response = advisor.chat(args.chat)
        print(f"\n[Advisor]\n{response}")

    elif args.test:
        print("\n=== Testing analyze() ===")
        test_sim = {
            "DI_peak":   0.42,
            "DI_final":  0.38,
            "DI_healed": 0.05,
            "t_total":   32.0,
            "Bi":        0.72,
            "T_reheat":  65.0,
            "t_reheat":  10.0,
            "h_cool":    14.4,
            "n_zones":   1,
            "T_fill":    80.0,
            "risk":      "WARNING",
            "scenario":  "❌ Bad — No reheat",
            "zones":     [{"T": 23, "duration": 20, "label": "Zone 1"}],
        }
        result = advisor.analyze(test_sim)
        print(result)

        print("\n=== Testing chat() ===")
        q = "Why does the DI stay high even after reheat at 65°C?"
        print(f"[User] {q}")
        print(f"[Advisor]\n{advisor.chat(q, context=result)}")

        print("\n=== Testing quick_summary() ===")
        for DI in [0.05, 0.18, 0.35, 0.65, 0.90]:
            print(f"  DI={DI:.2f}: {advisor.quick_summary(DI, 28, 0.3)}")

    else:
        print("Usage: python llm_advisor_v4.py --test | --status | --chat 'question'")
