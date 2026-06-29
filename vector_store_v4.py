"""
vector_store_v4.py
Lightweight local RAG (Retrieval-Augmented Generation) vector store
CBIC × TUAT | Julian Evan Chrisnanto | 2026

WHY THIS EXISTS:
  Sending the entire 900-word system prompt to Ollama on every single
  query is slow (large context = slow inference on CPU) and is the
  primary cause of "[TIMEOUT] Ollama response timed out."

  This module chunks the knowledge base into small retrievable documents,
  builds a TF-IDF vector index (pure numpy, no extra downloads needed),
  and retrieves only the top-k most relevant chunks for any given query.
  This makes prompts ~5-10x shorter -> Ollama responds in seconds instead
  of timing out.

  Optional upgrade: if an Ollama embedding model (e.g. nomic-embed-text)
  is available, semantic embeddings are used instead of TF-IDF for
  higher retrieval quality. Falls back to TF-IDF automatically.

USAGE:
  from vector_store_v4 import VectorStore, chunk_knowledge_base

  store = VectorStore()
  store.add_documents(chunk_knowledge_base(kb, extra))
  store.build_index()
  store.save("vector_store_v4.json")

  # Later, in the advisor:
  store.load("vector_store_v4.json")
  results = store.search("why does hike form?", top_k=5)
"""

import json
import os
import re
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("PIDRL_EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT_S = 8


# ═══════════════════════════════════════════════════════════════════════════
#  Tokenisation (English + Japanese aware)
# ═══════════════════════════════════════════════════════════════════════════

_STOPWORDS = set("""
a an the is are was were be been being of in on at to for with by from
and or but if then else this that these those it its as which who whom
whose what when where why how all any both each few more most other
some such no nor not only own same so than too very can will just
""".split())


def tokenize(text: str) -> list:
    """Tokenise mixed English/Japanese text. Keeps alnum + CJK runs."""
    text = str(text).lower()
    tokens = re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════════════
#  Vector Store
# ═══════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    Minimal local vector store for RAG retrieval.
    Default mode: TF-IDF (instant, zero extra dependencies).
    Optional mode: Ollama embeddings (better semantic match, requires
                   `ollama pull nomic-embed-text`).
    """

    def __init__(self):
        self.documents:   list = []   # [{"text":..., "metadata":...}]
        self.vocab:       dict = {}   # token -> column index
        self.idf:         np.ndarray = None
        self.doc_vectors: np.ndarray = None
        self.mode:        str  = "tfidf"
        self.embed_dim:   int  = None

    # ── Building ─────────────────────────────────────────────────────────

    def add_documents(self, docs: list):
        """docs: list of {"text": str, "metadata": dict}"""
        self.documents.extend(docs)

    def clear(self):
        self.documents   = []
        self.vocab       = {}
        self.idf          = None
        self.doc_vectors = None

    def _build_tfidf(self):
        token_lists = [tokenize(d["text"]) for d in self.documents]
        vocab_set = set()
        for toks in token_lists:
            vocab_set.update(toks)
        self.vocab = {tok: i for i, tok in enumerate(sorted(vocab_set))}
        V = max(len(self.vocab), 1)
        N = max(len(self.documents), 1)

        df = np.zeros(V)
        tf_matrix = np.zeros((N, V))
        for i, toks in enumerate(token_lists):
            counts  = Counter(toks)
            doc_len = max(len(toks), 1)
            for tok, c in counts.items():
                j = self.vocab[tok]
                tf_matrix[i, j] = c / doc_len
            for tok in set(toks):
                df[self.vocab[tok]] += 1

        idf = np.log((N + 1) / (df + 1)) + 1.0
        self.idf = idf
        tfidf = tf_matrix * idf[None, :]
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = tfidf / norms
        self.mode = "tfidf"
        self.embed_dim = V

    def _vectorize_query_tfidf(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        V = len(self.vocab)
        vec = np.zeros(V)
        counts  = Counter(toks)
        doc_len = max(len(toks), 1)
        for tok, c in counts.items():
            if tok in self.vocab:
                j = self.vocab[tok]
                vec[j] = (c / doc_len) * self.idf[j]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _try_ollama_embed(self, text: str):
        if not HAS_REQUESTS:
            return None
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text[:2000]},
                timeout=EMBED_TIMEOUT_S,
            )
            if r.status_code == 200:
                emb = r.json().get("embedding")
                if emb:
                    return np.array(emb, dtype=np.float64)
        except Exception:
            pass
        return None

    def build_index(self, use_ollama_embeddings: bool = False):
        """
        Build the retrieval index.
        use_ollama_embeddings: try semantic embeddings first (requires
        the embedding model to be pulled); silently falls back to TF-IDF
        on any failure so this is always safe to call.
        """
        if not self.documents:
            self.doc_vectors = np.zeros((0, 1))
            return

        if use_ollama_embeddings:
            embs = []
            ok = True
            for d in self.documents:
                e = self._try_ollama_embed(d["text"])
                if e is None:
                    ok = False
                    break
                embs.append(e)
            if ok and embs:
                arr = np.array(embs)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.doc_vectors = arr / norms
                self.mode = "ollama_embed"
                self.embed_dim = arr.shape[1]
                return
        # Default / fallback
        self._build_tfidf()

    # ── Searching ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list:
        """Return top_k most relevant documents for the query."""
        if self.doc_vectors is None or len(self.documents) == 0:
            return []

        if self.mode == "ollama_embed":
            qvec = self._try_ollama_embed(query)
            if qvec is None:
                # Embedding service unavailable now — fall back for this query
                if not self.vocab:
                    self._build_tfidf()
                qvec = self._vectorize_query_tfidf(query)
                sims = self._cosine_against_tfidf_fallback(qvec)
            else:
                qn = np.linalg.norm(qvec)
                qvec = qvec / qn if qn > 0 else qvec
                if qvec.shape[0] != self.doc_vectors.shape[1]:
                    return []
                sims = self.doc_vectors @ qvec
        else:
            qvec = self._vectorize_query_tfidf(query)
            if qvec.shape[0] != self.doc_vectors.shape[1]:
                return []
            sims = self.doc_vectors @ qvec

        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= min_score:
                continue
            results.append({
                "text":     self.documents[idx]["text"],
                "metadata": self.documents[idx].get("metadata", {}),
                "score":    score,
            })
        return results

    def _cosine_against_tfidf_fallback(self, qvec):
        # Used only if mode is ollama_embed but embedding call failed —
        # we don't have tfidf doc_vectors built, so just return zeros
        return np.zeros(len(self.documents))

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "documents":   self.documents,
            "vocab":       self.vocab,
            "idf":         self.idf.tolist() if self.idf is not None else None,
            "doc_vectors": self.doc_vectors.tolist() if self.doc_vectors is not None else None,
            "mode":        self.mode,
            "embed_dim":   self.embed_dim,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents   = data["documents"]
            self.vocab       = data["vocab"]
            self.idf         = np.array(data["idf"]) if data["idf"] is not None else None
            self.doc_vectors = np.array(data["doc_vectors"]) if data["doc_vectors"] is not None else None
            self.mode        = data.get("mode", "tfidf")
            self.embed_dim   = data.get("embed_dim")
            return True
        except Exception as e:
            print(f"[VectorStore] Load failed: {e}")
            return False

    def __len__(self):
        return len(self.documents)

    def stats(self) -> dict:
        return {
            "n_documents": len(self.documents),
            "mode":        self.mode,
            "embed_dim":   self.embed_dim,
            "vocab_size":  len(self.vocab),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Knowledge-base chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_knowledge_base(kb: dict, extra: dict = None) -> list:
    """
    Convert a structured knowledge_base_v4.json dict (+ optional auto-scanned
    'extra' data from uploaded Excel files) into small retrievable chunks.

    Each chunk: {"text": str, "metadata": {"source": str, "type": str}}
    Chunks are kept short (1-3 sentences) for precise retrieval.
    """
    chunks = []

    # ── Material properties ─────────────────────────────────────────────
    mat = kb.get("material", {})
    if mat:
        dims = mat.get("dimensions", {})
        chunks.append({
            "text": (
                f"Material {mat.get('name','LCWT401')}: "
                f"R={dims.get('radius_cm')}cm, H={dims.get('height_cm')}cm. "
                f"T_solidus={mat.get('T_solidus_C')}C, "
                f"T_liquidus={mat.get('T_liquidus_C')}C, "
                f"mushy zone mid={mat.get('T_mushy_mid_C')}C. "
                f"k={mat.get('k_thermal_WmK')}W/mK, rho={mat.get('rho_kg_m3')}kg/m3. "
                f"Main defect: {mat.get('defect')}"
            ),
            "metadata": {"source": "material", "type": "properties"},
        })

    # ── Trials ───────────────────────────────────────────────────────────
    for trial_key, trial in kb.get("trials", {}).items():
        name    = trial.get("name", trial_key)
        desc    = trial.get("description", "")
        result  = trial.get("result", "")
        insight = trial.get("key_insight", "")

        chunks.append({
            "text": f"{name}: {desc} Result: {result}",
            "metadata": {"source": trial_key, "type": "trial_summary"},
        })
        if insight:
            chunks.append({
                "text": f"{name} key insight: {insight}",
                "metadata": {"source": trial_key, "type": "insight"},
            })

        # Reheat events as their own chunk
        reheat_events = trial.get("reheat_events") or trial.get("reheat_pulses")
        if reheat_events:
            chunks.append({
                "text": f"{name} reheat schedule: {json.dumps(reheat_events, ensure_ascii=False)[:300]}",
                "metadata": {"source": trial_key, "type": "reheat_schedule"},
            })

        obs = trial.get("observations", {})
        if obs:
            for t_key, o_text in obs.items():
                chunks.append({
                    "text": f"{name} at {t_key}: {o_text}",
                    "metadata": {"source": trial_key, "type": "observation"},
                })

    # ── Physics rules ────────────────────────────────────────────────────
    for rule in kb.get("physics_rules", []):
        chunks.append({
            "text": f"Rule: {rule['rule']}. Evidence: {rule['evidence']}. Action: {rule['action']}",
            "metadata": {"source": "physics_rules", "type": "rule"},
        })

    # ── DRL targets ──────────────────────────────────────────────────────
    tgt = kb.get("drl_targets", {})
    if tgt:
        chunks.append({
            "text": (
                f"DRL optimal targets: DI safe<{tgt.get('DI_safe')}, "
                f"caution<{tgt.get('DI_caution')}, warning<{tgt.get('DI_warning')}. "
                f"Total time target <= {tgt.get('total_time_target_min')} min. "
                f"Recommended {tgt.get('recommended_zones')} zones. "
                f"T_reheat range {tgt.get('recommended_T_reheat_range_C')}C, "
                f"recommended {tgt.get('recommended_T_reheat_C')}C. "
                f"h_cool range {tgt.get('recommended_h_cool_range')} W/m2K."
            ),
            "metadata": {"source": "drl_targets", "type": "targets"},
        })
        prod = tgt.get("production_optimal", {})
        if prod:
            chunks.append({
                "text": f"Production-optimal schedule: {json.dumps(prod, ensure_ascii=False)[:300]}",
                "metadata": {"source": "drl_targets", "type": "production_optimal"},
            })

    # ── Physics constants / DI scale ────────────────────────────────────
    pc = kb.get("physics_constants", {})
    if pc.get("DI_scale"):
        scale_text = "; ".join(f"{k}: {v}" for k, v in pc["DI_scale"].items())
        chunks.append({
            "text": f"Damage Index (DI) scale: {scale_text}",
            "metadata": {"source": "physics_constants", "type": "DI_scale"},
        })

    # ── Uploaded data (from Excel scan) ─────────────────────────────────
    if extra:
        for obs in extra.get("text_observations", [])[:80]:
            chunks.append({
                "text": obs,
                "metadata": {"source": "uploaded", "type": "observation"},
            })
        for s in extra.get("temperature_series", [])[:60]:
            chunks.append({
                "text": (
                    f"[{s['sheet'][:30]}] {s['label'][:60]}: "
                    f"temperature ranged {s['min']}-{s['max']}C "
                    f"over {len(s['values'])} data points"
                ),
                "metadata": {"source": "uploaded", "type": "temperature_series"},
            })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: build + save in one call
# ═══════════════════════════════════════════════════════════════════════════

def build_and_save_vector_store(
    kb: dict,
    extra: dict = None,
    output_path: str = "vector_store_v4.json",
    use_ollama_embeddings: bool = False,
) -> dict:
    """
    One-call helper: chunk KB -> build index -> save to disk.
    Returns stats dict.
    """
    store = VectorStore()
    store.add_documents(chunk_knowledge_base(kb, extra))
    store.build_index(use_ollama_embeddings=use_ollama_embeddings)
    store.save(output_path)
    return store.stats()


if __name__ == "__main__":
    # Quick self-test
    test_kb = {
        "material": {"name": "LCWT401", "dimensions": {"radius_cm": 1.25, "height_cm": 4.0},
                     "T_solidus_C": 62, "T_liquidus_C": 72, "defect": "Sink mark"},
        "trials": {
            "trial1": {"name": "Trial 1", "description": "Heat cap test",
                       "result": "Slight hike observed",
                       "key_insight": "Slow cooling reduces hike"},
        },
        "physics_rules": [
            {"rule": "Reheat must reach mushy zone", "evidence": "Trial data",
             "action": "Use T_reheat >= 75C"},
        ],
        "drl_targets": {"DI_safe": 0.10, "total_time_target_min": 30,
                        "recommended_zones": 3, "recommended_T_reheat_C": 80,
                        "recommended_T_reheat_range_C": [75, 110],
                        "recommended_h_cool_range": [3, 10]},
    }
    stats = build_and_save_vector_store(test_kb, output_path="/tmp/test_vs.json")
    print("Build stats:", stats)

    store = VectorStore()
    store.load("/tmp/test_vs.json")
    results = store.search("why does hike form at the top surface", top_k=3)
    print("\nSearch results for 'why does hike form':")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:100]}")
