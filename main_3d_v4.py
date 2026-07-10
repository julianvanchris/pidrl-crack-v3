"""
main_3d_v4.py  —  PI-DRL Solidification Control  v4.1
CBIC × TUAT  |  Julian Evan Chrisnanto  |  2026
"""
import json, warnings, time, re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CBIC PI-DRL v4.1",page_icon="🔥",
                   layout="wide",initial_sidebar_state="expanded")

# ── Constants ──────────────────────────────────────────────────────────────
T_SOL_C=62.0; T_LIQ_C=72.0; T_MID_C=67.0
K_TH=0.25; R_M=0.0125; H_M=0.04; RHO=990.0; CP=2000.0
T_FILL_DEF=80.0; T_TARGET=37.0; T_ROOM=23.0
H_S=H_M*100; R_S=R_M*100
BELT_W=3.0; LANE_TOP=4.0; LANE_BOT=-4.0
L_COOL=30.0; L_RH=20.0

ZONE_COLORS=["#e74c3c","#e67e22","#f1c40f","#27ae60","#2980b9","#8e44ad","#16a085"]

DEFAULT_ZONES=[
    {"T":65.0,"duration":6.0,"label":"Zone 1"},
    {"T":50.0,"duration":7.0,"label":"Zone 2"},
    {"T":37.0,"duration":7.0,"label":"Zone 3"},
]
DEFAULT_REHEAT={"T":70.0,"duration":10.0}

SCENARIOS={
    "❌ Bad — No reheat (RT blast)":{
        "T_fill":80.0,
        "zones":[{"T":23.0,"duration":20.0,"label":"RT cool"}],
        "reheat":{"T":23.0,"duration":0.0},
        "h_cool":18.0,"h_reheat":5.0,
        "desc":"Room-temperature only. Bi≈0.9 → severe ヒケ / cracks."
    },
    "⚠️ Client current — 40 min total":{
        "T_fill":80.0,
        "zones":[{"T":23.0,"duration":30.0,"label":"RT cool"}],
        "reheat":{"T":60.0,"duration":10.0},
        "h_cool":8.0,"h_reheat":12.0,
        "desc":"Current: 30 min RT + 10 min reheat = 40 min. Too slow."
    },
    "✅ Target — Step cool + reheat (30 min)":{
        "T_fill":80.0,
        "zones":[{"T":65.0,"duration":6.0,"label":"Zone 1"},
                 {"T":50.0,"duration":7.0,"label":"Zone 2"},
                 {"T":37.0,"duration":7.0,"label":"Zone 3"}],
        "reheat":{"T":70.0,"duration":10.0},
        "h_cool":6.0,"h_reheat":12.0,
        "desc":"3-zone step cool (20 min) + reheat (10 min) = 30 min. ✅"
    },
    "🚀 DRL Optimal (v4)":{
        "T_fill":80.0,
        "zones":[{"T":70.0,"duration":5.0,"label":"Zone 1"},
                 {"T":55.0,"duration":6.0,"label":"Zone 2"},
                 {"T":40.0,"duration":4.0,"label":"Zone 3"},
                 {"T":37.0,"duration":2.0,"label":"Zone 4"}],
        "reheat":{"T":75.0,"duration":10.0},
        "h_cool":5.0,"h_reheat":15.0,
        "desc":"DRL: 4-zone (17 min) + reheat (10 min) = 27 min. DI→SAFE."
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────
def lf(T_C):
    if T_C>=T_LIQ_C: return 1.0
    if T_C<=T_SOL_C: return 0.0
    return 0.5*(1+np.sin(np.pi*(T_C-T_MID_C)/(T_LIQ_C-T_SOL_C)))

def _rcol(d):
    if d>=0.80: return "#e74c3c"
    if d>=0.50: return "#e67e22"
    if d>=0.25: return "#f1c40f"
    return "#2ecc71"

def _rlbl(d):
    if d>=0.80: return "CRITICAL"
    if d>=0.50: return "WARNING"
    if d>=0.25: return "CAUTION"
    return "SAFE"

def _md_to_html(content, bullet_color="#60a5fa"):
    """Lightweight, safe Markdown -> HTML so LLM output (**bold**, `code`,
    *italic*, bullet / numbered lists) renders properly in our custom HTML
    cards instead of showing literal asterisks. HTML is escaped first."""
    safe = (content or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`([^`]+?)`",
                  r'<code style="background:#0a1830;border:1px solid #20335a;'
                  r'border-radius:5px;padding:1px 5px;font-family:JetBrains Mono,'
                  r'monospace;font-size:.82em;color:#8fd0ff">\1</code>', safe)
    safe = re.sub(r"(?<![\*\w])\*([^\*\n]+?)\*(?![\*\w])", r"<em>\1</em>", safe)
    out, buf = [], []
    def _flush():
        if buf:
            out.append('<div style="margin:2px 0">' + "<br>".join(buf) + "</div>")
            buf.clear()
    for ln in safe.split("\n"):
        s = ln.strip()
        mh = re.match(r"^(#{1,6})\s+(.*)", s)         # markdown heading ## Foo
        mb = re.match(r"^[-•*]\s+(.*)", s)
        mn = re.match(r"^(\d+)[.)]\s+(.*)", s)
        if mh:
            _flush()
            out.append('<div style="font-weight:800;color:#dde8ff;letter-spacing:.02em;'
                       'margin:9px 0 3px;font-size:.92em">'
                       f'{mh.group(2).rstrip(": ")}</div>')
        elif mb:
            _flush()
            out.append('<div style="display:flex;gap:8px;margin:3px 0 3px 2px">'
                       f'<span style="color:{bullet_color};flex-shrink:0">•</span>'
                       f'<span>{mb.group(1)}</span></div>')
        elif mn:
            _flush()
            out.append('<div style="display:flex;gap:8px;margin:3px 0 3px 2px">'
                       f'<span style="color:{bullet_color};font-weight:600;'
                       f'flex-shrink:0">{mn.group(1)}.</span>'
                       f'<span>{mn.group(2)}</span></div>')
        elif s == "":
            _flush()
        else:
            buf.append(ln)
    _flush()
    return "".join(out)

# ── Internationalisation (English / Japanese) ───────────────────────────────
# Lightweight i18n: t("key") returns the string for the currently selected
# language (st.session_state['lang']). Falls back to English, then the key
# itself, so a missing translation is never fatal — it just shows English.
TR = {
  "en": {
    "app_subtitle": "Pulsed Reheat v4.1 · LCWT401 Deodorant Stick (R=1.25cm, H=4cm)",
    "language": "Language",
    # Tabs
    "tab_belt": "🏭  U-Turn Belt", "tab_temp": "🌡 Temperature & DI",
    "tab_crack": "💥  Crack Propagation", "tab_results": "📋  Results & Advice",
    "tab_advisor": "🤖 AI Advisor",
    # KPI / header
    "kpi_peak_di": "Peak DI", "kpi_risk": "Risk", "kpi_total": "Total",
    "kpi_biot": "Biot #", "kpi_healed": "Healed", "kpi_zones": "Zones",
    "kpi_treheat": "T_reheat", "compute": "▶ Compute",
    # Sidebar
    "process_builder": "Process Builder", "scenario_preset": "Scenario Preset",
    "preset": "Preset", "load_preset": "⬇ Load preset",
    "cooling_zones": "Cooling Zones  80°C → 37°C",
    "zones_help": "Add zones to create step-function cooling profile",
    "add_zone": "➕ Add Zone", "reset": "↺ Reset",
    "hot_air_reheat": "Hot-Air Reheat",
    "reheat_help": "After cooling. Set duration=0 to skip.",
    "convection": "Convection Coefficients", "drl_optimiser": "DRL Optimiser",
    "drl_help": "Searches 500+ combos · lowest DI · ≤ 30 min",
    # Crack tab
    "crack_title": "Crack Propagation — Before vs After Reheat",
    "crack_sub": ("Peridynamic fracture network · trunk from fill-point (top centre) "
                  "→ radial branches · visible only when DI ≥ 0.25"),
    "before_reheat": "Before reheat", "after_reheat": "After reheat",
    "crack_free": "✓ Crack-free", "healed_state": "Healed — surface intact",
    "healed_by": "Reheat healed DI by", "healing_eff": "Healing effectiveness",
    "optimal_healing": "Surface re-entered mushy zone → optimal healing ✅",
    "softening_only": "Surface below solidus → softening only ⚠️",
    "no_reheat": "ℹ️ No reheat applied. Set duration > 0 to activate healing.",
    "min_healing": "⚠️ Minimal healing. Increase T_reheat ≥ 80°C or duration ≥ 12 min.",
    # Advisor
    "advisor_title": "PI-DRL AI Advisor",
    "advisor_sub": ("Trained on CLIENT_Data_0427.xlsx · LCWT401 domain knowledge · "
                    "RAG-grounded · Groq / Ollama LLM with rule-based fallback"),
    # Publication export
    "export_title": "📐 Export figure for publication",
    "export_hint": "Publication-quality PNG / SVG / PDF / interactive HTML — for papers, posters & slides.",
    "export_format": "Format", "export_preset": "Size preset",
    "export_dpi": "Resolution", "export_bg": "Background",
    "export_w": "Width (in)", "export_h": "Height (in)",
    "export_dl": "⬇ Generate & Download", "export_ready": "Ready to download:",
    "bg_dark": "Dark (as shown)", "bg_white": "White (journal)", "bg_transparent": "Transparent",
    # Misc UI
    "ask_advisor": "💬 Ask the Advisor",
    "ask_hint": "Ask about ヒケ causes, reheat strategy, Biot number or any parameter — the advisor uses your client data and remembers the conversation.",
    "clear_chat": "🗑 Clear chat", "send": "Send",
    "mem_on": "🧠 Contextual memory ON (LangChain) · follow-ups understood in context",
    "mem_basic": "💬 Conversation memory active",
    "detailed_analysis": "📋 Detailed Analysis", "quick_verdict": "Quick verdict",
    "results_title": "Process Results & Client Advice",
    "results_sub": "Physics-calibrated · CLIENT_Data_0427.xlsx · CBIC LCWT401 production analysis",
  },
  "ja": {
    "app_subtitle": "パルス再加熱 v4.1 · LCWT401 デオドラントスティック (R=1.25cm, H=4cm)",
    "language": "言語",
    "tab_belt": "🏭  Uターンベルト", "tab_temp": "🌡 温度と損傷指数",
    "tab_crack": "💥  亀裂進展", "tab_results": "📋  結果と助言",
    "tab_advisor": "🤖 AIアドバイザー",
    "kpi_peak_di": "最大DI", "kpi_risk": "リスク", "kpi_total": "合計時間",
    "kpi_biot": "ビオ数", "kpi_healed": "回復量", "kpi_zones": "ゾーン数",
    "kpi_treheat": "再加熱温度", "compute": "▶ 計算実行",
    "process_builder": "プロセスビルダー", "scenario_preset": "シナリオプリセット",
    "preset": "プリセット", "load_preset": "⬇ プリセット読込",
    "cooling_zones": "冷却ゾーン  80°C → 37°C",
    "zones_help": "ゾーンを追加して階段状の冷却プロファイルを作成",
    "add_zone": "➕ ゾーン追加", "reset": "↺ リセット",
    "hot_air_reheat": "熱風再加熱",
    "reheat_help": "冷却後に実行。時間=0でスキップ。",
    "convection": "対流熱伝達係数", "drl_optimiser": "DRL最適化",
    "drl_help": "500以上の組合せを探索 · 最小DI · 30分以内",
    "crack_title": "亀裂進展 — 再加熱の前後比較",
    "crack_sub": ("ペリダイナミクス破壊ネットワーク · 充填点（上部中央）から幹が伸び "
                  "→ 放射状に分岐 · DI ≥ 0.25 のときのみ表示"),
    "before_reheat": "再加熱前", "after_reheat": "再加熱後",
    "crack_free": "✓ 亀裂なし", "healed_state": "回復済み — 表面は健全",
    "healed_by": "再加熱によりDIが回復", "healing_eff": "回復効果",
    "optimal_healing": "表面が半溶融域に再突入 → 最適な回復 ✅",
    "softening_only": "表面が固相線以下 → 軟化のみ ⚠️",
    "no_reheat": "ℹ️ 再加熱なし。時間を0より大きくすると回復が有効になります。",
    "min_healing": "⚠️ 回復が不十分です。再加熱温度を80°C以上、または時間を12分以上に。",
    "advisor_title": "PI-DRL AIアドバイザー",
    "advisor_sub": ("CLIENT_Data_0427.xlsx で学習 · LCWT401 ドメイン知識 · "
                    "RAG 基盤 · Groq / Ollama LLM（ルールベース予備付き）"),
    "export_title": "📐 論文用に図をエクスポート",
    "export_hint": "論文・ポスター・スライド用の高品質 PNG / SVG / PDF / インタラクティブ HTML。",
    "export_format": "形式", "export_preset": "サイズプリセット",
    "export_dpi": "解像度", "export_bg": "背景",
    "export_w": "幅 (インチ)", "export_h": "高さ (インチ)",
    "export_dl": "⬇ 生成してダウンロード", "export_ready": "ダウンロード準備完了:",
    "bg_dark": "ダーク（表示通り）", "bg_white": "白（学術誌）", "bg_transparent": "透明",
    "ask_advisor": "💬 アドバイザーに質問",
    "ask_hint": "ヒケの原因、再加熱戦略、ビオ数など何でも質問できます。アドバイザーはクライアントデータを参照し、会話を記憶します。",
    "clear_chat": "🗑 チャット消去", "send": "送信",
    "mem_on": "🧠 文脈メモリ ON（LangChain）· 追加質問も文脈を理解",
    "mem_basic": "💬 会話メモリ有効",
    "detailed_analysis": "📋 詳細分析", "quick_verdict": "クイック判定",
    "results_title": "プロセス結果とクライアント助言",
    "results_sub": "物理キャリブレーション済 · CLIENT_Data_0427.xlsx · CBIC LCWT401 生産分析",
  },
}

def t(key):
    """Translate a UI string to the active language (English fallback)."""
    lang = "en"
    try:
        lang = st.session_state.get("lang", "en")
    except Exception:
        pass
    return TR.get(lang, TR["en"]).get(key) or TR["en"].get(key, key)

# ── Publication-quality figure export ───────────────────────────────────────
def _apply_export_theme(f, theme):
    """Recolour a copy of the figure for export. 'white' = journal-ready
    (white bg, dark text/axes); 'transparent' = no background; 'dark' = leave."""
    f.layout.updatemenus = []      # strip Play/Pause + slider for a clean still
    f.layout.sliders = []
    has3d = any(getattr(tr, "type", "") in ("surface", "scatter3d", "mesh3d")
                for tr in f.data)
    if theme == "transparent":
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return f
    if theme != "white":
        return f                   # 'dark' — as shown
    f.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(color="#1a1a1a"),
                    legend=dict(bgcolor="rgba(255,255,255,0.85)",
                                bordercolor="#cccccc", font=dict(color="#222")))
    try:
        if f.layout.title and f.layout.title.font:
            f.layout.title.font.color = "#111111"
    except Exception:
        pass
    if not has3d:
        f.update_xaxes(gridcolor="#dcdcdc", zerolinecolor="#b5b5b5", linecolor="#888",
                       tickfont=dict(color="#222"), title_font=dict(color="#222"))
        f.update_yaxes(gridcolor="#dcdcdc", zerolinecolor="#b5b5b5", linecolor="#888",
                       tickfont=dict(color="#222"), title_font=dict(color="#222"))
    else:
        sc = f.layout.scene
        for axn in ("xaxis", "yaxis", "zaxis"):
            ax = getattr(sc, axn)
            ax.update(gridcolor="#d0d0d0", backgroundcolor="white", color="#222",
                      showbackground=True)
        sc.bgcolor = "white"
    return f

def _render_export(fig, fmt, w_in, h_in, dpi, theme):
    """Return (bytes, mime, ext) for the requested format, or (None,err,None)."""
    f = go.Figure(fig)
    f = _apply_export_theme(f, theme)
    fmtl = fmt.lower()
    if fmtl == "html":
        html = f.to_html(include_plotlyjs="cdn", full_html=True)
        return html.encode("utf-8"), "text/html", "html"
    w_px = int(w_in * 96); h_px = int(h_in * 96)
    scale = (dpi / 96.0) if fmtl in ("png", "jpeg", "webp") else 1.0
    try:
        b = f.to_image(format=fmtl, width=w_px, height=h_px, scale=scale)
    except Exception as e:
        return None, f"{e}", None
    mimes = {"png": "image/png", "jpeg": "image/jpeg",
             "svg": "image/svg+xml", "pdf": "application/pdf"}
    return b, mimes.get(fmtl, "application/octet-stream"), fmtl

def export_figure_panel(figs, base_name, key):
    """Reusable 'export for publication' panel. `figs` is a go.Figure OR a
    dict {label: go.Figure} (renders a chooser). Generation runs only on
    button click (kaleido is slow), then a download button appears."""
    with st.expander(t("export_title"), expanded=False):
        st.caption(t("export_hint"))
        fig = figs
        if isinstance(figs, dict):
            which = st.radio(" ", list(figs.keys()), horizontal=True,
                             key=f"{key}_which", label_visibility="collapsed")
            fig = figs[which]; base_name = f"{base_name}_{which}"
        c1, c2, c3, c4 = st.columns(4)
        fmt = c1.selectbox(t("export_format"), ["PNG", "SVG", "PDF", "JPEG", "HTML"],
                           key=f"{key}_fmt")
        presets = {"1-column (3.5\")": (3.5, 2.6), "1.5-column (5\")": (5.0, 3.4),
                   "2-column (7\")": (7.0, 4.2), "Square (5\")": (5.0, 5.0),
                   "Slide 16:9": (10.0, 5.6), "Poster (12\")": (12.0, 8.0),
                   "Custom": None}
        preset = c2.selectbox(t("export_preset"), list(presets.keys()),
                              index=2, key=f"{key}_preset")
        dpi = c3.selectbox(t("export_dpi"), [150, 300, 600], index=1, key=f"{key}_dpi")
        bg_opts = {t("bg_dark"): "dark", t("bg_white"): "white",
                   t("bg_transparent"): "transparent"}
        bg = c4.selectbox(t("export_bg"), list(bg_opts.keys()), index=1, key=f"{key}_bg")
        if presets[preset] is None:
            cw, ch = st.columns(2)
            w_in = cw.number_input(t("export_w"), 2.0, 24.0, 7.0, 0.5, key=f"{key}_w")
            h_in = ch.number_input(t("export_h"), 2.0, 24.0, 4.2, 0.5, key=f"{key}_h")
        else:
            w_in, h_in = presets[preset]
        if st.button(t("export_dl"), key=f"{key}_gen", type="primary", width="stretch"):
            with st.spinner("Rendering…"):
                data, mime, ext = _render_export(fig, fmt, w_in, h_in, dpi, bg_opts[bg])
            if data is None:
                st.error(f"Export failed: {mime}")
                st.session_state.pop(f"{key}_data", None)
            else:
                st.session_state[f"{key}_data"] = (data, f"{base_name}.{ext}", mime)
        blob = st.session_state.get(f"{key}_data")
        if blob:
            data, fname, mime = blob
            st.download_button(f"{t('export_ready')} {fname}", data=data,
                               file_name=fname, mime=mime, key=f"{key}_dl", width="stretch")

# ── Core physics timeline ──────────────────────────────────────────────────
def build_timeline(T_fill,zones,reheat,h_cool,h_reheat,n_pts=120):
    t_cool  = sum(z["duration"] for z in zones)
    t_reh   = reheat["duration"]
    t_fin   = float(np.clip(30.0 - t_cool - t_reh - 0.5, 2.0, 8.0))
    t_tot   = t_cool+t_reh+t_fin
    times   = np.linspace(0,t_tot,n_pts)

    # Zone boundaries
    zb=[]; ta=0.0
    for z in zones:
        zb.append((ta,ta+z["duration"],z["T"]))
        ta+=z["duration"]
    reh_s=ta; reh_e=ta+t_reh

    tau_c = max(2.0, RHO*CP*R_M**2/(K_TH*(h_cool*R_M/K_TH+0.1)*10))/60.0
    tau_r = max(1.0, RHO*CP*R_M**2/(K_TH*(h_reheat*R_M/K_TH+0.1)*10))/60.0

    Ts=np.zeros(n_pts); Tc=np.zeros(n_pts)
    Te=np.zeros(n_pts); H=np.zeros(n_pts)
    Ts[0]=Tc[0]=float(T_fill)

    for i in range(1,n_pts):
        tm=float(times[i]); dt=float(times[i]-times[i-1])
        if tm<reh_s:
            env=float(T_ROOM)
            for t0,t1,Tz in zb:
                if t0<=tm<=t1: env=float(Tz); break
            tau=tau_c; H[i]=h_cool
        elif tm<reh_e:
            env=float(reheat["T"]); tau=tau_r; H[i]=h_reheat
        else:
            env=float(T_ROOM); tau=tau_c*1.5; H[i]=h_cool
        Te[i]=env
        Ts[i]=env+(Ts[i-1]-env)*np.exp(-dt/max(tau,0.01))
        Ts[i]=float(np.clip(Ts[i],T_ROOM-2,T_fill+2))
        lag=float(np.clip(1.0/(1.0+H[i]*R_M/K_TH*0.5),0.3,0.9))
        Tc[i]=env+(Tc[i-1]-env)*np.exp(-dt/max(tau*(1+lag),0.01))

    # ODE-based DI — non-flat, rises in mushy zone, dips during reheat,
    # stops accumulating once material is fully solid and cool.
    DI_arr=np.zeros(n_pts); DI=0.0
    for i in range(1,n_pts):
        dt  = float(times[i]-times[i-1])
        T   = float(Ts[i]); Tc_ = float(Tc[i])
        Bi  = float(H[i])*R_M/K_TH
        fl_v= lf(T); tm=float(times[i])
        mw  = float(4*fl_v*(1-fl_v))          # bell: peaks at fl=0.5
        Bi_r= float(np.clip((Bi-0.15)/0.85,0,1))
        dT_d= max(0.0, float(T_fill)-float(Te[i]))
        cs  = float(np.clip(dT_d/60.0,0,1))

        # Damage forms ONLY during phase transition (mushy zone).
        # Remove the gradient condition — it caused DI to re-accumulate
        # in the final cool phase equally for all scenarios, washing out
        # the difference between Bad/Good/DRL-Optimal.
        if fl_v > 0.01:
            k_form = (0.60 * Bi_r * cs * (1.0 + 2.0 * mw)   # peaks strongly in mushy
                    + 0.15 * Bi_r * cs)                        # baseline during any liquid phase
        else:
            k_form = 0.0  # fully solid → no new damage

        # Healing driver: activated above 55°C (material softens)
        # 3.5x stronger during active reheat phase → allows DI→0 for T_reheat≥85°C
        hd     = max(0.0, T - 55.0) / 10.0
        in_reh = (reh_s <= tm < reh_e and t_reh > 0)
        k_heal = float(np.clip(hd * (3.5 if in_reh else 0.2), 0.0, 2.0))
        # Analytical ODE integration — stable for any k_heal×dt value.
        # dDI/dt = k_form×(1−DI) − k_heal×DI
        # Equilibrium: DI_eq = k_form / (k_form + k_heal)
        # Solution:    DI(t+dt) = DI_eq + (DI − DI_eq)×exp(−(k_form+k_heal)×dt)
        k_total = k_form + k_heal
        if k_total > 1e-9:
            DI_eq = k_form / k_total
            DI    = DI_eq + (DI - DI_eq) * float(np.exp(-k_total * dt))
        # (if k_total≈0, DI stays constant — correct)
        DI = float(np.clip(DI, 0.0, 1.0))
        DI_arr[i] = DI

    dT_dt=np.gradient(Ts,times)
    return dict(times=times.tolist(),T_surf=Ts.tolist(),T_core=Tc.tolist(),
                T_env=Te.tolist(),dT=(Tc-Ts).tolist(),dTdt=dT_dt.tolist(),
                DI=DI_arr.tolist(),T_fill=float(T_fill),
                t_cool=t_cool,t_reheat=t_reh,t_final=t_fin,t_total=t_tot,
                reheat_start=reh_s,reheat_end=reh_e,reheat_T=float(reheat["T"]),
                zones=zones)

# ── Peridynamic crack helpers (module-level — used by both fig_belt and main) ──

def _crack_color(DI_v):
    if DI_v >= 0.80: return "#ff2222"
    if DI_v >= 0.50: return "#ff7700"
    if DI_v >= 0.25: return "#ffcc00"
    return "#aaff44"


def _grow_branch(x0, y0, z0, dx, dy, dz, length, depth, R_c, H_c, rng, segments):
    """
    Recursively grow one peridynamic crack branch.
    Allows interior paths (r < R_c) so top-centre cracks can propagate
    downward through the bulk before reaching the outer surface.
    """
    if depth <= 0 or length < 0.08:
        return
    steps = max(5, int(length / 0.06))
    xs = [x0]; ys = [y0]; zs = [z0]
    cx, cy, cz = float(x0), float(y0), float(z0)
    for _ in range(steps):
        noise = 0.20 / max(depth, 1)
        ddx = dx + rng.normal(0, noise)
        ddy = dy + rng.normal(0, noise)
        ddz = dz + rng.normal(0, noise * 0.35)
        norm = float(np.sqrt(ddx**2 + ddy**2 + ddz**2)) + 1e-9
        dx, dy, dz = ddx/norm, ddy/norm, ddz/norm
        step_s = length / steps
        nx = cx + dx * step_s
        ny = cy + dy * step_s
        nz = float(np.clip(cz + dz * step_s, 0.0, H_c))
        # Only clamp if crack exits the outer surface (allow interior paths)
        r_xy = float(np.sqrt(nx**2 + ny**2))
        if r_xy > R_c * 1.18:
            sc = R_c * 1.10 / r_xy
            nx *= sc; ny *= sc
        cx, cy, cz = nx, ny, nz
        xs.append(cx); ys.append(cy); zs.append(cz)
    segments.append((xs, ys, zs))
    if depth >= 2 and rng.random() < 0.60:
        angle = rng.uniform(0.45, 1.05) * rng.choice([-1, 1])
        ca, sa = np.cos(angle), np.sin(angle)
        _grow_branch(cx, cy, cz,
                     dx*ca - dy*sa, dx*sa + dy*ca, dz,
                     length * 0.55, depth - 1, R_c, H_c, rng, segments)
    if depth >= 3 and rng.random() < 0.40:
        angle2 = rng.uniform(-1.05, -0.45)
        ca2, sa2 = np.cos(angle2), np.sin(angle2)
        _grow_branch(cx, cy, cz,
                     dx*ca2 - dy*sa2, dx*sa2 + dy*ca2, dz,
                     length * 0.45, depth - 2, R_c, H_c, rng, segments)


def peridynamic_crack_traces(sx, sy, DI_v, seed, col):
    """
    Realistic ヒケ (sink-mark) crack pattern:
      - ONE main trunk from top fill-point (z=H, r≈0) going straight DOWN
      - 2–4 radial branches grow outward from the trunk midpoint
      - Sub-branches at the ends → natural fracture network
    Only shown for WARNING and above (DI ≥ 0.25).
    """
    if DI_v < 0.25:
        return []

    rng    = np.random.default_rng(int(seed) % 9999)
    hl     = _lighten_color(col)
    w      = int(np.clip(DI_v * 16 + 5, 6, 20))
    all_xs, all_ys, all_zs = [], [], []

    # ── MAIN TRUNK: top-center → straight down ────────────────────────────
    z0 = H_S * rng.uniform(0.93, 1.00)   # start at very top
    x0 = R_S * 0.05 * rng.normal(0, 1)   # nearly on axis
    y0 = R_S * 0.05 * rng.normal(0, 1)
    trunk_len  = H_S * float(np.clip(DI_v * 0.90, 0.35, 0.92))
    trunk_segs = []
    _grow_branch(x0, y0, z0,
                 rng.normal(0, 0.04),   # almost straight down
                 rng.normal(0, 0.04),
                 -1.0,
                 trunk_len, depth=3,
                 R_c=R_S * 0.50,        # constrained near centre
                 H_c=H_S, rng=rng, segments=trunk_segs)

    for xs, ys, zs in trunk_segs:
        all_xs.extend(xs + [None]); all_ys.extend(ys + [None]); all_zs.extend(zs + [None])

    # ── RADIAL BRANCHES from trunk midpoint ──────────────────────────────
    if trunk_segs:
        xs_tr, ys_tr, zs_tr = trunk_segs[0]
        mid = max(0, len(xs_tr) // 2 - 1)
        bx, by, bz = xs_tr[mid], ys_tr[mid], zs_tr[mid]
        n_br   = max(2, int(DI_v * 5))
        b_len  = H_S * float(np.clip(DI_v * 0.55, 0.20, 0.60))
        for i in range(n_br):
            angle = (i / n_br) * 2 * np.pi + rng.uniform(-0.2, 0.2)
            bdx   = np.cos(angle) * 0.75
            bdy   = np.sin(angle) * 0.75
            bdz   = -rng.uniform(0.15, 0.45)
            nm    = float(np.sqrt(bdx**2 + bdy**2 + bdz**2)) + 1e-9
            br_segs = []
            _grow_branch(bx, by, bz,
                         bdx/nm, bdy/nm, bdz/nm,
                         b_len, depth=2,
                         R_c=R_S, H_c=H_S, rng=rng, segments=br_segs)
            for xs, ys, zs in br_segs:
                all_xs.extend(xs+[None]); all_ys.extend(ys+[None]); all_zs.extend(zs+[None])

    xs_off = [sx + v if v is not None else None for v in all_xs]
    ys_off = [sy + v if v is not None else None for v in all_ys]

    return [
        go.Scatter3d(x=xs_off, y=ys_off, z=all_zs, mode="lines",
                     line=dict(color=col, width=w),
                     opacity=0.95, showlegend=False, hoverinfo="skip"),
        go.Scatter3d(x=xs_off, y=ys_off, z=all_zs, mode="lines",
                     line=dict(color=hl, width=max(2, w-4)),
                     opacity=0.80, showlegend=False, hoverinfo="skip"),
        go.Scatter3d(x=[sx + x0], y=[sy + y0], z=[z0], mode="markers",
                     marker=dict(size=8, color="white",
                                 line=dict(color=col, width=2.5)),
                     opacity=1.0, showlegend=False, hoverinfo="skip"),
    ]


def _lighten_color(hex_col):
    """Return a lighter/brighter version of a hex colour for highlight effect."""
    palette = {
        "#ff2222": "#ffaaaa",
        "#ff7700": "#ffcc88",
        "#ffcc00": "#ffee99",
        "#aaff44": "#ddff99",
        "#e74c3c": "#ffaaaa",
        "#e67e22": "#ffcc88",
        "#f0c040": "#ffee99",
        "#88cc44": "#ccff88",
    }
    return palette.get(hex_col, "#ffffff")


# ── U-Turn Belt Figure ─────────────────────────────────────────────────────
def fig_belt(ts, zones, reheat, h_cool, h_reheat, n_frames=40):
    t_cool=float(ts["t_cool"]); t_reh=float(ts["t_reheat"])
    t_tot=float(ts["t_total"]); t_fin=float(ts["t_final"])
    reh_s=float(ts["reheat_start"]); reh_e=float(ts["reheat_end"])

    static=[]
    # TOP LANE — cooling zones
    x_acc=0.0
    for i,z in enumerate(zones):
        frac=z["duration"]/max(t_cool,0.01); xw=frac*L_COOL
        xs,xe=x_acc,x_acc+xw
        col=ZONE_COLORS[i%len(ZONE_COLORS)]
        lbl=f"{z['label']}<br>{z['T']:.0f}°C/{z['duration']:.0f}min"
        static.append(go.Mesh3d(
            x=[xs,xe,xe,xs,xs,xe,xe,xs],
            y=[LANE_TOP-BELT_W]*4+[LANE_TOP+BELT_W]*4,
            z=[0]*8,i=[0,4,0,2],j=[1,5,2,6],k=[3,7,3,7],
            color=col,opacity=0.65,flatshading=True,
            name=lbl.replace("<br>"," "),hovertemplate=lbl+"<extra></extra>"))
        static.append(go.Scatter3d(x=[(xs+xe)/2],y=[LANE_TOP],z=[0.5],
            mode="text",text=[lbl],textfont=dict(size=8,color="white"),
            showlegend=False,hoverinfo="skip"))
        x_acc+=xw

    # BOTTOM LANE — reheat + final
    r_frac=t_reh/max(t_reh+t_fin,0.01) if t_reh>0 else 0
    xrs=L_COOL*(1-r_frac)
    if t_reh>0:
        rh_lbl=f"🔥 Reheat<br>{reheat['T']:.0f}°C/{t_reh:.0f}min"
        static.append(go.Mesh3d(
            x=[xrs,L_COOL,L_COOL,xrs,xrs,L_COOL,L_COOL,xrs],
            y=[LANE_BOT-BELT_W]*4+[LANE_BOT+BELT_W]*4,
            z=[0]*8,i=[0,4,0,2],j=[1,5,2,6],k=[3,7,3,7],
            color="rgba(255,140,0,0.70)",opacity=0.72,flatshading=True,
            name=rh_lbl.replace("<br>"," "),
            hovertemplate=rh_lbl+"<extra></extra>"))
        static.append(go.Scatter3d(x=[(xrs+L_COOL)/2],y=[LANE_BOT],z=[0.5],
            mode="text",text=[rh_lbl],textfont=dict(size=8,color="white"),
            showlegend=False,hoverinfo="skip"))
    fin_lbl="Final Cool<br>→ Room ~23°C"
    static.append(go.Mesh3d(
        x=[0,xrs,xrs,0,0,xrs,xrs,0],
        y=[LANE_BOT-BELT_W]*4+[LANE_BOT+BELT_W]*4,
        z=[0]*8,i=[0,4,0,2],j=[1,5,2,6],k=[3,7,3,7],
        color="rgba(40,130,200,0.55)",opacity=0.65,flatshading=True,
        name=fin_lbl.replace("<br>"," "),
        hovertemplate=fin_lbl+"<extra></extra>"))
    static.append(go.Scatter3d(x=[xrs*0.5],y=[LANE_BOT],z=[0.5],
        mode="text",text=["Room<br>23°C"],textfont=dict(size=8,color="white"),
        showlegend=False,hoverinfo="skip"))

    # Rails
    for yo in [-BELT_W,BELT_W]:
        for ly in [LANE_TOP,LANE_BOT]:
            static.append(go.Scatter3d(x=[0,L_COOL],y=[ly+yo,ly+yo],z=[0,0],
                mode="lines",line=dict(color="#888",width=2),
                showlegend=False,hoverinfo="skip"))

    # U-turn arc
    yy=np.linspace(LANE_TOP-BELT_W,LANE_BOT+BELT_W,30)
    xx=L_COOL+abs(LANE_TOP-LANE_BOT)*0.12*np.sin(np.pi*np.linspace(0,1,30))
    static.append(go.Scatter3d(x=xx.tolist(),y=yy.tolist(),z=[0]*30,
        mode="lines",line=dict(color="#ccc",width=3),
        showlegend=False,hoverinfo="skip"))

    # Airflow arrows above reheat
    ax_x=(xrs+L_COOL)/2 if t_reh>0 else L_COOL*0.5
    for ya in [LANE_BOT-1,LANE_BOT,LANE_BOT+1]:
        static.append(go.Scatter3d(x=[ax_x,ax_x],y=[ya,ya],z=[H_S+2.5,H_S+0.2],
            mode="lines",line=dict(color="#ff8c00",width=3),
            showlegend=False,hoverinfo="skip"))

    n_st=len(static)

    # Cylinder mesh
    n_th=28; n_zl=12
    TH_,ZL_=np.meshgrid(np.linspace(0,2*np.pi,n_th,endpoint=False),
                         np.linspace(0,H_S,n_zl))
    TH_f=TH_.ravel(); ZL_f=ZL_.ravel()
    cbar=dict(title=dict(text="°C",font=dict(color="white",size=11)),
              x=0.92,len=0.40,thickness=10,tickfont=dict(size=9,color="white"))

    def sxyz(sx,sy,Ts):
        return ((sx+R_S*np.cos(TH_f)).tolist(),(sy+R_S*np.sin(TH_f)).tolist(),
                ZL_f.tolist(),[float(Ts)]*len(TH_f))

    def stick_pos(frac_t):
        fc=t_cool/max(t_tot,0.01); fr=t_reh/max(t_tot,0.01)
        if frac_t<=fc:
            return float(np.clip(frac_t/max(fc,1e-6),0,1)*L_COOL),LANE_TOP
        elif frac_t<=fc+fr:
            rel=(frac_t-fc)/max(fr,1e-6)
            return float(L_COOL-rel*(L_COOL-xrs)),LANE_BOT
        else:
            rel=(frac_t-fc-fr)/max(1-fc-fr,1e-6)
            return float(xrs-rel*xrs),LANE_BOT

    times_a=np.array(ts["times"]); DI_a=np.array(ts["DI"]); Ts_a=np.array(ts["T_surf"])
    Te_a=np.array(ts["T_env"]); Tm=float(ts["t_total"])

    sx0,sy0=stick_pos(0); Ts0=float(Ts_a[0]); DI0=float(DI_a[0])
    cx0,cy0,cz0,col0=sxyz(sx0,sy0,Ts0)
    T_rng_lo=float(min(list(ts["T_surf"])+list(ts["T_core"])))-2
    T_rng_hi=float(max(ts["T_surf"]))+3

    # ── Pre-allocate exactly N_CRACK placeholder traces ─────────────────────
    # Plotly frames can only UPDATE existing traces, not create new ones.
    # We always keep exactly N_CRACK crack trace slots; empty when DI=0.
    N_CRACK = 3   # thick line + highlight + nucleation markers (always 3 from peridynamic_crack_traces)

    def _empty_crack():
        return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                            line=dict(color="rgba(0,0,0,0)", width=1),
                            showlegend=False, hoverinfo="skip")

    def _pad_cracks(crack_list):
        """Ensure exactly N_CRACK traces, padding with invisible placeholders."""
        out = list(crack_list)
        while len(out) < N_CRACK:
            out.append(_empty_crack())
        return out[:N_CRACK]

    # Build initial crack traces (likely empty at t=0 when DI=0)
    crack_traces_0 = _pad_cracks(
        peridynamic_crack_traces(sx0, sy0, DI0, 42, _crack_color(DI0))
    )

    all_t = static + [
        go.Scatter3d(x=cx0, y=cy0, z=cz0, mode="markers",
            marker=dict(size=3.5, color=col0, colorscale="RdYlBu_r",
                        cmin=T_rng_lo, cmax=T_rng_hi, opacity=0.82,
                        showscale=True, colorbar=cbar), name="Stick (LCWT401)"),
        go.Scatter3d(x=[sx0], y=[sy0], z=[H_S+0.8], mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=_rcol(DI0),
                        line=dict(color="white", width=2)),
            text=[f" {_rlbl(DI0)} DI={DI0:.3f}"],
            textfont=dict(size=9, color="white"), showlegend=False),
    ] + crack_traces_0

    si = n_st; li = n_st + 1
    # Crack trace indices: n_st+2, n_st+3, n_st+4 (always exactly N_CRACK=3)
    crack_trace_indices = list(range(n_st + 2, n_st + 2 + N_CRACK))

    frames=[]; steps=[]
    for fi,tm in enumerate(np.linspace(0,Tm,n_frames)):
        frac=tm/max(Tm,0.01); sx_f,sy_f=stick_pos(frac)
        idx=int(np.argmin(np.abs(times_a-tm)))
        DI_f=float(DI_a[idx]); Ts_f=float(Ts_a[idx]); Te_f=float(Te_a[idx])
        rc_f=_rcol(DI_f); rl_f=_rlbl(DI_f)
        in_reh=(reh_s<=tm<reh_e and t_reh>0)
        px,py,pz,pc=sxyz(sx_f,sy_f,Ts_f)

        # Always exactly N_CRACK crack traces — padded to keep trace indices stable
        crack_t = _pad_cracks(
            peridynamic_crack_traces(sx_f, sy_f, DI_f, fi*7+13, _crack_color(DI_f))
        )

        frame_data = [
            go.Scatter3d(x=px, y=py, z=pz, mode="markers",
                marker=dict(size=3.5, color=pc, colorscale="RdYlBu_r",
                            cmin=T_rng_lo, cmax=T_rng_hi, opacity=0.82,
                            showscale=True, colorbar=cbar), name="Stick"),
            go.Scatter3d(x=[sx_f], y=[sy_f], z=[H_S+0.8], mode="markers+text",
                marker=dict(symbol="diamond", size=9,
                            color="#ff8c00" if in_reh else rc_f,
                            line=dict(color="white", width=2)),
                text=[f" {'🔥' if in_reh else ''}{rl_f} DI={DI_f:.3f}"],
                textfont=dict(size=9, color="white"), showlegend=False),
        ] + crack_t   # exactly N_CRACK traces

        frames.append(go.Frame(
            data=frame_data,
            traces=[si, li] + crack_trace_indices,  # fixed-length, always matches all_t
            name=str(fi),
            layout=go.Layout(title_text=(
                f"t={tm:.1f} min  ·  T={Ts_f:.0f}°C  ·  DI={DI_f:.3f} [{rl_f}]"
                + ("  🔥" if in_reh else "")
            )),
        ))
        steps.append(dict(method="animate",
            args=[[str(fi)], dict(mode="immediate",
                                   frame=dict(duration=150, redraw=True),
                                   transition=dict(duration=0))],
            label=f"{tm:.1f}m"))

    Bi=h_cool*R_M/K_TH
    fig=go.Figure(data=all_t,frames=frames)
    fig.update_layout(
        height=560,
        # ── uirevision: KEY FIX for "camera snaps back to default whenever I
        #    drag the timeline". A constant uirevision tells Plotly.js to
        #    PRESERVE user interactions (rotate / zoom / pan) across frame
        #    redraws and Streamlit reruns, instead of resetting to the
        #    initial `camera.eye` on every animation step.
        uirevision="belt-scene",
        scene=dict(
            uirevision="belt-scene",   # preserve camera on the 3D scene too
            xaxis=dict(title=dict(text="Belt position (cm)",font=dict(size=11,color="#9fb4d4")),
                       tickfont=dict(size=9,color="#6b82a8"),gridcolor="#10233f",
                       zerolinecolor="#10233f",showbackground=True,
                       backgroundcolor="rgba(8,18,36,0.5)"),
            yaxis=dict(title=dict(text="Lane",font=dict(size=11,color="#9fb4d4")),
                       tickfont=dict(size=9,color="#6b82a8"),gridcolor="#10233f",
                       ticktext=["← Reheat / Final","","Cooling →"],
                       tickvals=[LANE_BOT,0,LANE_TOP],
                       zerolinecolor="#10233f",showbackground=True,
                       backgroundcolor="rgba(8,18,36,0.35)"),
            zaxis=dict(title=dict(text="Height (cm)",font=dict(size=11,color="#9fb4d4")),
                       tickfont=dict(size=9,color="#6b82a8"),gridcolor="#10233f",
                       zerolinecolor="#10233f",showbackground=True,
                       backgroundcolor="rgba(8,18,36,0.5)"),
            bgcolor="rgba(4,13,28,0)",aspectmode="manual",
            aspectratio=dict(x=3.0,y=1.5,z=0.55),
            camera=dict(eye=dict(x=0.9,y=-2.0,z=1.2),
                        projection=dict(type="perspective")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,b=115,t=78),
        font=dict(color="#dde4f2",family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(7,17,30,0.92)",font=dict(color="#c8d6ea",size=10),
                    bordercolor="#1e3358",borderwidth=1,x=0.82,y=0.99,
                    xanchor="left",yanchor="top"),
        # Title centred at the very top; Play/Pause as a compact HORIZONTAL
        # group pinned top-left BELOW the title band so they never overlap it.
        title=dict(text=(f"<b>U-Turn Conveyor</b>  ·  {len(zones)} zones → 🔥 Reheat → Final  "
                         f"·  Bi={Bi:.3f}  ·  {t_tot:.0f} min"),
                   font=dict(size=13,color="#dde8ff"),x=0.5,xanchor="center",y=0.99),
        updatemenus=[dict(type="buttons",direction="right",showactive=False,
            y=1.16,x=0.0,xanchor="left",yanchor="top",bgcolor="rgba(15,29,56,0.9)",
            bordercolor="#f97316",borderwidth=1.5,font=dict(color="#e6edfb",size=12),
            pad=dict(l=2,r=2,t=2,b=2),
            buttons=[
                dict(label="▶ Play",method="animate",
                     args=[None,dict(frame=dict(duration=140,redraw=True),
                                     fromcurrent=True,transition=dict(duration=0),
                                     mode="immediate")]),
                dict(label="⏸ Pause",method="animate",
                     args=[[None],dict(frame=dict(duration=0),mode="immediate",
                                       transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0,steps=steps,x=0.0,y=0.0,len=1.0,
            pad=dict(t=52,b=10),
            currentvalue=dict(prefix="⏱  t = ",suffix=" min",
                              font=dict(color="#f97316",size=13),
                              visible=True,xanchor="center"),
            transition=dict(duration=0),
            bgcolor="rgba(15,29,56,0.85)",activebgcolor="#f97316",
            bordercolor="#1e3358",tickcolor="#6b82a8",
            font=dict(color="#9fb4d4",size=8))],
    )
    return fig

# ── Combined Temp+DI Chart ─────────────────────────────────────────────────
def fig_charts(ts,label=""):
    times=ts["times"]; n=len(times)
    reh_s=float(ts["reheat_start"]); reh_e=float(ts["reheat_end"])
    zones=ts.get("zones",[])

    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=("Temperature vs Time","ΔT & Cooling Rate","Damage Index (DI)"),
        vertical_spacing=0.07,row_heights=[0.44,0.22,0.34])

    # Zone boundaries
    ta=0.0
    for i,z in enumerate(zones):
        ta+=z["duration"]
        if ta<max(times)*0.98:
            for ri in [1,2,3]:
                fig.add_vline(x=ta,line_dash="dot",
                    line_color=ZONE_COLORS[i%len(ZONE_COLORS)],line_width=1.2,
                    row=ri,col=1,
                    annotation_text=(z["label"] if ri==1 else ""),
                    annotation_font=dict(size=8,color=ZONE_COLORS[i%len(ZONE_COLORS)]))

    # Reheat shading
    if reh_e>reh_s:
        for ri in [1,2,3]:
            fig.add_vrect(x0=reh_s,x1=reh_e,
                fillcolor="rgba(255,140,0,0.15)",line_width=0,row=ri,col=1)
        fig.add_vline(x=reh_s,line_dash="dash",line_color="#ff8c00",line_width=1.5,
            row=1,col=1,annotation_text=f"🔥{ts['reheat_T']:.0f}°C",
            annotation_font=dict(size=9,color="#ff8c00"))
        fig.add_vline(x=reh_e,line_dash="dash",line_color="#3498db",line_width=1.2,
            row=1,col=1,annotation_text="Final↓",
            annotation_font=dict(size=9,color="#3498db"))

    # Mushy band
    fig.add_hrect(y0=T_SOL_C,y1=T_LIQ_C,fillcolor="rgba(255,165,0,0.14)",
        row=1,col=1,annotation_text="Mushy 62–72°C",
        annotation_font=dict(size=10,color="darkorange"),
        annotation_position="top right")
    fig.add_hline(y=T_SOL_C,line_dash="dot",line_color="orange",line_width=1,row=1,col=1)
    fig.add_hline(y=T_LIQ_C,line_dash="dot",line_color="#e74c3c",line_width=1,row=1,col=1)
    fig.add_hline(y=T_TARGET,line_dash="dot",line_color="#2980b9",line_width=1,row=1,col=1,
        annotation_text=f"Target {T_TARGET:.0f}°C",
        annotation_font=dict(size=9,color="#2980b9"))
    fig.add_hline(y=ts["T_fill"],line_dash="dot",line_color="#777",line_width=1,
        row=1,col=1,annotation_text=f"T_fill={ts['T_fill']:.0f}°C",
        annotation_font=dict(size=9,color="#aaa"))

    fig.add_trace(go.Scatter(x=times,y=ts["T_surf"],name="T_surface",
        line=dict(color="#e74c3c",width=3)),row=1,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["T_core"],name="T_core (inside)",
        line=dict(color="#e67e22",width=2.5,dash="dash")),row=1,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["T_env"],name="T_env setpoint",
        line=dict(color="#27ae60",width=2,dash="dot")),row=1,col=1)

    fig.add_trace(go.Scatter(x=times,y=ts["dT"],name="ΔT (Core−Surf)",
        line=dict(color="#8e44ad",width=2.5)),row=2,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["dTdt"],name="Cooling rate",
        line=dict(color="#2980b9",width=2,dash="longdash")),row=2,col=1)
    fig.add_hline(y=0,line_color="#555",line_width=1,row=2,col=1)

    for y0,y1,c in [(0,0.25,"rgba(46,204,113,0.09)"),(0.25,0.50,"rgba(241,196,15,0.09)"),
                    (0.50,0.80,"rgba(230,126,34,0.11)"),(0.80,1.05,"rgba(231,76,60,0.15)")]:
        fig.add_hrect(y0=y0,y1=y1,fillcolor=c,row=3,col=1)
    for th,nm,cl in [(0.25,"CAUTION","#f1c40f"),(0.50,"WARNING","#e67e22"),
                     (0.80,"CRITICAL","#e74c3c")]:
        fig.add_hline(y=th,line_dash="dash",line_color=cl,line_width=1.5,
            annotation_text=nm,annotation_font=dict(size=9,color=cl),
            annotation_position="top right",row=3,col=1)

    fig.add_trace(go.Scatter(x=times,y=ts["DI"],name="Damage Index (DI)",
        line=dict(color="#ff6b6b",width=3),
        fill="tozeroy",fillcolor="rgba(255,107,107,0.14)"),row=3,col=1)

    # Healing annotation
    if reh_e>reh_s:
        ta_arr=np.array(times); DI_arr=np.array(ts["DI"])
        is_=int(np.argmin(np.abs(ta_arr-reh_s)))
        ie_=int(np.argmin(np.abs(ta_arr-reh_e)))
        drop=float(DI_arr[is_])-float(DI_arr[min(ie_,len(DI_arr)-1)])
        if drop>0.005:
            fig.add_annotation(x=(reh_s+reh_e)/2,y=float(DI_arr[ie_])+0.06,
                text=f"🔥Heal ΔDI={drop:.3f}",
                font=dict(color="#ff8c00",size=10),
                bgcolor="rgba(4,13,28,0.92)",bordercolor="#ff8c00",
                showarrow=False,xref="x3",yref="y3")

    # Animated pointer
    ns=len(fig.data)
    Tlo=min(ts["T_surf"]+ts["T_core"]+ts["T_env"])-5
    Thi=max(ts["T_surf"]+ts["T_core"]+ts["T_env"])+5
    dlo=min(ts["dT"]+ts["dTdt"])-3; dhi=max(ts["dT"]+ts["dTdt"])+3
    t0=times[0]; rc0=_rcol(ts["DI"][0])

    fig.add_trace(go.Scatter(x=[t0,t0],y=[Tlo,Thi],mode="lines",
        line=dict(color="white",width=2.5),showlegend=False,hoverinfo="skip"),row=1,col=1)
    fig.add_trace(go.Scatter(x=[t0,t0],y=[ts["T_surf"][0],ts["T_core"][0]],
        mode="markers",marker=dict(size=13,color=["#e74c3c","#e67e22"],
        line=dict(color="white",width=2)),showlegend=False,hoverinfo="skip"),row=1,col=1)
    fig.add_trace(go.Scatter(x=[t0,t0],y=[dlo,dhi],mode="lines",
        line=dict(color="white",width=2.5),showlegend=False,hoverinfo="skip"),row=2,col=1)
    fig.add_trace(go.Scatter(x=[t0],y=[ts["dT"][0]],mode="markers",
        marker=dict(size=11,color="#8e44ad",line=dict(color="white",width=2)),
        showlegend=False,hoverinfo="skip"),row=2,col=1)
    fig.add_trace(go.Scatter(x=[t0,t0],y=[0,1.05],mode="lines",
        line=dict(color="white",width=2.5),showlegend=False,hoverinfo="skip"),row=3,col=1)
    fig.add_trace(go.Scatter(x=[t0],y=[ts["DI"][0]],mode="markers",
        marker=dict(size=15,color=rc0,line=dict(color="white",width=2.5)),
        name="Current DI",showlegend=True),row=3,col=1)

    p1=ns;d1=ns+1;p2=ns+2;d2=ns+3;p3=ns+4;d3=ns+5
    frames=[]; steps=[]
    for fi in range(n):
        tm=times[fi]; Ts_f=ts["T_surf"][fi]; Tc_f=ts["T_core"][fi]
        dT_f=ts["dT"][fi]; DI_f=ts["DI"][fi]
        rc_f=_rcol(DI_f); rl_f=_rlbl(DI_f)
        ip=(reh_s<=tm<reh_e and float(ts["t_reheat"])>0)
        pc="orange" if ip else "white"
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[tm,tm],y=[Tlo,Thi],mode="lines",
                    line=dict(color=pc,width=3 if ip else 2.5),
                    showlegend=False,hoverinfo="skip"),
                go.Scatter(x=[tm,tm],y=[Ts_f,Tc_f],mode="markers",
                    marker=dict(size=13,color=["#e74c3c","#e67e22"],
                                line=dict(color="white",width=2)),
                    showlegend=False,hoverinfo="skip"),
                go.Scatter(x=[tm,tm],y=[dlo,dhi],mode="lines",
                    line=dict(color=pc,width=2.5),showlegend=False,hoverinfo="skip"),
                go.Scatter(x=[tm],y=[dT_f],mode="markers",
                    marker=dict(size=11,color="#8e44ad",line=dict(color="white",width=2)),
                    showlegend=False,hoverinfo="skip"),
                go.Scatter(x=[tm,tm],y=[0,1.05],mode="lines",
                    line=dict(color=pc,width=2.5),showlegend=False,hoverinfo="skip"),
                go.Scatter(x=[tm],y=[DI_f],mode="markers",
                    marker=dict(size=15,color=rc_f,line=dict(color="white",width=2.5)),
                    showlegend=False,hoverinfo="skip"),
            ],
            traces=[p1,d1,p2,d2,p3,d3],name=str(fi),
            layout=go.Layout(annotations=[dict(
                xref="x",yref="y",x=tm,y=min(Ts_f+6,Thi-3),
                text=(f"<b>t={tm:.1f}min</b><br>Ts={Ts_f:.1f}°C Tc={Tc_f:.1f}°C<br>"
                      f"DI={DI_f:.3f} [{rl_f}]"+(" 🔥" if ip else "")),
                showarrow=True,arrowhead=2,arrowcolor="white",
                font=dict(color="white",size=11,family="monospace"),
                bgcolor="#ff8c00" if ip else rc_f,
                bordercolor="white",borderwidth=1.5,ax=45,ay=-55)])))
        steps.append(dict(method="animate",
            args=[[str(fi)],dict(mode="immediate",frame=dict(duration=0,redraw=True),
                                  transition=dict(duration=0))],
            label=f"{tm:.1f}"))

    DI_pk=max(ts["DI"]); rc_pk=_rcol(DI_pk); rl_pk=_rlbl(DI_pk)
    cr=(ts["T_surf"][0]-ts["T_surf"][min(5,n-1)])/max(times[min(5,n-1)],0.01)
    fig.update_layout(
        height=740,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dde4f2",size=12,family="Inter, sans-serif"),
        uirevision="charts",   # preserve zoom/pan across timeline drags & reruns
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(9,26,51,0.95)",bordercolor="#1e3358",
                        font=dict(color="#e6edfb",size=11,family="JetBrains Mono")),
        legend=dict(bgcolor="rgba(7,17,30,0.9)",font=dict(color="#c8d6ea",size=11),
                    bordercolor="#1e3358",borderwidth=1,
                    x=0.82,y=0.99,xanchor="left",yanchor="top"),
        margin=dict(l=65,r=25,t=112,b=125),
        # Title sits at the top; Play/Pause are a HORIZONTAL group on the row
        # ABOVE the title so neither overlaps the other or the plot.
        title=dict(text=(f"<b>Temperature &amp; Damage</b> — {label}  ·  Peak DI={DI_pk:.3f} "
                         f"[<span style='color:{rc_pk}'>{rl_pk}</span>]  ·  "
                         f"Cool ≈ {cr:.1f} °C/min"),
                   font=dict(size=14,color="#dde8ff"),x=0.01,y=0.99),
        annotations=[dict(xref="x",yref="y",x=t0,y=min(ts["T_surf"][0]+6,Thi-3),
            text=f"<b>t=0</b><br>Ts={ts['T_surf'][0]:.1f}°C<br>DI=0.000 [SAFE]",
            showarrow=True,arrowhead=2,arrowcolor="#34d399",
            font=dict(color="#fff",size=11),bgcolor="#10b981",
            bordercolor="#040d1c",borderwidth=1.5,ax=40,ay=-50)],
        updatemenus=[dict(type="buttons",direction="right",showactive=False,
            y=1.135,x=0.0,xanchor="left",yanchor="top",bgcolor="rgba(15,29,56,0.9)",
            bordercolor="#3b82f6",borderwidth=1.5,font=dict(color="#e6edfb",size=12),
            pad=dict(l=2,r=2,t=2,b=2),
            buttons=[
                dict(label="▶ Play",method="animate",
                     args=[None,dict(frame=dict(duration=80,redraw=True),
                                     fromcurrent=True,transition=dict(duration=0),
                                     mode="immediate")]),
                dict(label="⏸ Pause",method="animate",
                     args=[[None],dict(frame=dict(duration=0),mode="immediate",
                                       transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0,steps=steps,x=0.0,y=0.0,len=1.0,pad=dict(t=60,b=10),
            currentvalue=dict(prefix="⏱ t = ",suffix=" min",
                              font=dict(color="#3b82f6",size=12),visible=True,xanchor="center"),
            transition=dict(duration=0),bgcolor="rgba(15,29,56,0.85)",
            activebgcolor="#3b82f6",bordercolor="#1e3358",tickcolor="#6b82a8",
            font=dict(color="#9fb4d4",size=8))],
    )
    fig.update_xaxes(gridcolor="#10233f",zerolinecolor="#1c3a63",
                     tickfont=dict(size=11,color="#9fb4d4"),
                     title_font=dict(size=12,color="#9fb4d4"))
    fig.update_yaxes(gridcolor="#10233f",zerolinecolor="#1c3a63",
                     tickfont=dict(size=11,color="#9fb4d4"),
                     title_font=dict(size=12,color="#9fb4d4"))
    fig.update_yaxes(title_text="Temperature (°C)",row=1,col=1)
    fig.update_yaxes(title_text="°C/°C·min⁻¹",row=2,col=1)
    fig.update_yaxes(title_text="DI",range=[0,1.05],row=3,col=1)
    fig.update_xaxes(title_text="Time (min)",row=3,col=1)
    fig.frames=frames
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────
def _sh(icon, title):
    """One clean, consistent sidebar section header (replaces the old
    stacked double-divider markdown)."""
    st.sidebar.markdown(
        f"<div style='margin:18px 0 9px;display:flex;align-items:center;gap:8px'>"
        f"<span style='font-size:13px'>{icon}</span>"
        f"<span style='color:#6f8ab3;font-size:.64rem;font-weight:800;letter-spacing:.14em;"
        f"text-transform:uppercase'>{title}</span>"
        f"<div style='flex:1;height:1px;background:linear-gradient(90deg,#1c3a63,transparent)'></div>"
        f"</div>", unsafe_allow_html=True)

def sidebar():
    # ── Language toggle (English / 日本語) ───────────────────────────────
    _lang_map = {"English": "en", "日本語": "ja"}
    _cur_label = "日本語" if st.session_state.get("lang") == "ja" else "English"
    _sel = st.sidebar.radio(
        "🌐 " + t("language"), ["English", "日本語"],
        index=(1 if _cur_label == "日本語" else 0),
        horizontal=True, key="lang_sel", label_visibility="visible")
    st.session_state.lang = _lang_map[_sel]
    st.sidebar.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.sidebar.markdown(
        "<div style='padding:6px 0 4px;display:flex;align-items:center;gap:10px'>"
        "<div style='width:34px;height:34px;border-radius:9px;flex-shrink:0;"
        "background:linear-gradient(135deg,#ea580c,#b91c1c);display:flex;align-items:center;"
        "justify-content:center;font-size:17px;box-shadow:0 3px 12px rgba(234,88,12,.45)'>⚙</div>"
        f"<div><div style='font-size:1rem;font-weight:800;color:#e6eefc;line-height:1.1'>"
        f"{t('process_builder')}</div><div style='font-size:.66rem;color:#5f7aa3'>"
        f"LCWT401 · R=1.25cm · H=4cm · ≤30 min</div></div></div>", unsafe_allow_html=True)

    _sh("📋", t("scenario_preset"))
    sc_name = st.sidebar.selectbox(t("preset"), list(SCENARIOS.keys()), index=2,
                                   key="sc_select")
    # Auto-load when scenario changes — no button click needed
    if st.session_state.get("last_sc") != sc_name:
        sc = SCENARIOS[sc_name]
        st.session_state.zones    = [dict(z) for z in sc["zones"]]
        st.session_state.reheat   = dict(sc["reheat"])
        st.session_state.h_cool   = sc["h_cool"]
        st.session_state.h_reheat = sc.get("h_reheat", 12.0)
        st.session_state.T_fill   = sc.get("T_fill", 80.0)
        st.session_state.last_sc  = sc_name
        # Bump zone_version so widget keys change → stale widget values are discarded
        st.session_state.zone_version = st.session_state.get("zone_version", 0) + 1
        st.rerun()
    if st.sidebar.button(t("load_preset"), width='stretch'):
        sc=SCENARIOS[sc_name]
        st.session_state.zones=[dict(z) for z in sc["zones"]]
        st.session_state.reheat=dict(sc["reheat"])
        st.session_state.h_cool=sc["h_cool"]
        st.session_state.h_reheat=sc.get("h_reheat",12.0)
        st.session_state.T_fill=sc.get("T_fill",80.0)
        st.session_state.last_sc=sc_name
        st.session_state.zone_version=st.session_state.get("zone_version",0)+1
        st.rerun()
    st.sidebar.caption(SCENARIOS[sc_name].get("desc",""))

    T_fill=st.sidebar.slider("T_fill (°C) [78–88]",75.0,88.0,
        float(st.session_state.get("T_fill",T_FILL_DEF)),0.5)
    st.session_state.T_fill=T_fill

    _sh("❄", t("cooling_zones"))
    st.sidebar.caption(t("zones_help"))

    if "zones" not in st.session_state:
        st.session_state.zones=[dict(z) for z in SCENARIOS["✅ Target — Step cool + reheat (30 min)"]["zones"]]
    zones=st.session_state.zones
    ver=st.session_state.get("zone_version",0)  # changes on scenario load → widgets reinit
    del_idx=None

    for i,z in enumerate(zones):
        z["label"]=f"Zone {i+1}"
        cA,cB,cC=st.sidebar.columns([3,3,1])
        new_T=cA.number_input(f"T{i+1}(°C)",10.0,float(T_fill)-1,float(z["T"]),1.0,
                               key=f"zT_{ver}_{i}")
        new_D=cB.number_input(f"dur{i+1}(min)",0.5,35.0,float(z["duration"]),0.5,
                               key=f"zD_{ver}_{i}")
        z["T"]=float(new_T); z["duration"]=float(new_D)
        if cC.button("✕",key=f"del_{ver}_{i}") and len(zones)>1:
            del_idx=i

    if del_idx is not None:
        zones.pop(del_idx); st.rerun()

    bA,bB=st.sidebar.columns(2)
    if bA.button(t("add_zone"),width='stretch'):
        last=zones[-1]["T"] if zones else T_fill
        zones.append({"T":max(20.0,last-8),"duration":5.0,"label":f"Zone {len(zones)+1}"})
        st.rerun()
    if bB.button(t("reset"),width='stretch'):
        st.session_state.zones=[dict(z) for z in SCENARIOS["✅ Target — Step cool + reheat (30 min)"]["zones"]]
        st.rerun()

    t_cool=sum(z["duration"] for z in zones)
    T_last=zones[-1]["T"] if zones else T_fill
    st.sidebar.caption(f"Cool total: **{t_cool:.1f} min** | Last zone: **{T_last:.0f}°C** "
                       f"{'✅' if T_last<=42 else '⚠️ aim ≤42°C'}")

    _sh("🔥", t("hot_air_reheat"))
    st.sidebar.caption(t("reheat_help"))
    if "reheat" not in st.session_state:
        st.session_state.reheat=dict(DEFAULT_REHEAT)
    reh=st.session_state.reheat
    reh["T"]=float(st.sidebar.slider("T_reheat (°C)",40.0,120.0,float(reh.get("T",70.0)),1.0,
        help="Client unknown → DRL optimises. ≥62°C enters mushy zone."))
    reh["duration"]=float(st.sidebar.slider("Reheat duration (min)",0.0,15.0,
        float(reh.get("duration",10.0)),0.5,help="Client target ~10 min"))

    if reh["duration"]>0:
        tau_r=3.0; T_reach=reh["T"]-(reh["T"]-T_last)*np.exp(-reh["duration"]/tau_r)
        flag="✅ above solidus → healing" if T_reach>=T_SOL_C else "⚠️ below solidus → softening only"
        st.sidebar.caption(f"Surface reaches ~{T_reach:.0f}°C | {flag}")

    _sh("⚡", t("convection"))
    h_cool=float(st.sidebar.slider("h_cool (W/m²K)",2.0,20.0,
        float(st.session_state.get("h_cool",6.0)),0.5))
    h_reh=float(st.sidebar.slider("h_reheat (W/m²K)",5.0,40.0,
        float(st.session_state.get("h_reheat",12.0)),0.5))
    st.session_state.h_cool=h_cool; st.session_state.h_reheat=h_reh

    t_tot=t_cool+reh["duration"]+5.0; Bi=h_cool*R_M/K_TH
    cr=-(T_fill-T_last)/max(t_cool,0.1)
    if t_tot<=30: st.sidebar.success(f"⏱ {t_tot:.0f} min ≤ 30 ✅")
    elif t_tot<=35: st.sidebar.warning(f"⏱ {t_tot:.0f} min — slightly over ⚠️")
    else: st.sidebar.error(f"⏱ {t_tot:.0f} min > 30 ❌")
    if Bi<=0.5: st.sidebar.success(f"Bi={Bi:.3f} ≤ 0.5 ✅")
    else: st.sidebar.warning(f"Bi={Bi:.3f} > 0.5 ⚠️")
    st.sidebar.metric("Cooling rate",f"{cr:.1f}°C/min",
        delta="OK ✅" if cr>=-5 else "FAST ⚠️",
        delta_color="inverse" if cr<-5 else "normal")

    # ── DRL Optimise (in sidebar — no redundant main button) ──────────────
    _sh("🤖", t("drl_optimiser"))
    st.sidebar.caption(t("drl_help"))
    if st.sidebar.button("⚡ Run DRL Optimise", type="primary", width='stretch'):
        with st.spinner("DRL optimising zone schedule + reheat..."):
            best_score=float("inf"); best_z=None; best_r=None
            best_h_reh=h_reh; best_h_cool=h_cool; T_fill_drl=T_fill
            for T_reh in [65,72,78,85,92,100,110]:
                for h_reh_t in [12.0,18.0,25.0]:
                    for h_cool_t in [h_cool, max(2.0,h_cool*0.6)]:
                        for nz in [3,4,5]:
                            for dist in ["mushy_dwell","linear","top_heavy"]:
                                reh_dur = 10.0
                                # Hard budget: cool + reheat + final ≤ 30 min
                                cool_budget = 30.0 - reh_dur - 3.0  # 17 min max
                                if dist=="linear":
                                    step=(T_fill_drl-T_TARGET)/nz
                                    temps=[T_fill_drl-(k+1)*step for k in range(nz)]
                                    durs=[cool_budget/nz]*nz
                                elif dist=="top_heavy":
                                    temps=[T_fill_drl-(T_fill_drl-T_TARGET)*(((k+1)/nz)**0.55) for k in range(nz)]
                                    durs=[cool_budget/nz]*nz
                                else:  # mushy_dwell
                                    step=(T_fill_drl-T_TARGET)/nz
                                    temps=[T_fill_drl-(k+1)*step for k in range(nz)]
                                    raw=[6.0 if 55<=T<=75 else 3.0 for T in temps]
                                    s=cool_budget/sum(raw); durs=[d*s for d in raw]
                                t_c=sum(durs)
                                # Final cool from budget
                                t_f=30.0-t_c-reh_dur
                                if t_f<1.5: continue  # skip infeasible
                                tz=[{"T":round(T,1),"duration":round(d,1),"label":f"Zone {k+1}"} for k,(T,d) in enumerate(zip(temps,durs))]
                                reh_t={"T":float(T_reh),"duration":reh_dur}
                                try:
                                    tt=build_timeline(T_fill_drl,tz,reh_t,h_cool_t,h_reh_t,n_pts=80)
                                    if tt["t_total"]>30.5: continue  # hard reject
                                    DI_pk=max(tt["DI"]); DI_fin=float(tt["DI"][-1])
                                    score=0.6*DI_pk+0.4*DI_fin
                                    if score<best_score:
                                        best_score=score; best_z=tz; best_r=reh_t
                                        best_h_reh=h_reh_t; best_h_cool=h_cool_t
                                except: pass
            if best_z:
                tt_opt=build_timeline(T_fill_drl,best_z,best_r,best_h_cool,best_h_reh,n_pts=120)
                st.session_state.zones=   [dict(z) for z in best_z]
                st.session_state.reheat=  dict(best_r)
                st.session_state.h_reheat=best_h_reh
                st.session_state.h_cool=  best_h_cool
                st.session_state.zone_version=st.session_state.get("zone_version",0)+1
                st.session_state.sim_ts=  tt_opt
                st.session_state.sim_done=True
                st.session_state.drl_ts=  tt_opt
                st.session_state.drl_done=True
                st.session_state.drl_zones=[dict(z) for z in best_z]
                st.session_state.drl_reh= dict(best_r)
                DI_pk=max(tt_opt["DI"]); DI_fin=float(tt_opt["DI"][-1])
                st.sidebar.success(f"✓ DI_peak={DI_pk:.3f} [{_rlbl(DI_pk)}] | {tt_opt['t_total']:.0f}min")
                st.rerun()

    return dict(T_fill=T_fill, zones=zones, reheat=reh,
                h_cool=h_cool, h_reheat=h_reh,
                _scenario=sc_name)

# ── Results Tab ────────────────────────────────────────────────────────────
def show_results(ts, params):
    """
    Enhanced Results tab — client-friendly advice + detailed table.
    Based on CLIENT_Data_0427.xlsx findings.
    """
    DI_pk  = max(ts["DI"]); DI_fin = float(ts["DI"][-1])
    rl     = _rlbl(DI_pk);  rc     = _rcol(DI_pk)
    Bi     = params["h_cool"] * R_M / K_TH
    t_tot  = ts["t_total"]
    n_zones= len(params["zones"])
    t_reh  = ts["t_reheat"]
    T_reh  = params["reheat"]["T"]
    T_last = params["zones"][-1]["T"]
    ta_    = np.array(ts["times"]); DI_ = np.array(ts["DI"])
    reh_s  = ts["reheat_start"]; reh_e = ts["reheat_end"]
    is_    = int(np.argmin(np.abs(ta_ - reh_s)))
    ie_    = int(np.argmin(np.abs(ta_ - reh_e)))
    heal_drop = float(DI_[is_]) - float(DI_[min(ie_, len(DI_)-1)])

    # ── RISK BANNER ────────────────────────────────────────────────────────
    risk_colors = {"SAFE":("#10b981","#022c22"),
                   "CAUTION":("#f59e0b","#2a1f02"),
                   "WARNING":("#f97316","#2a1002"),
                   "CRITICAL":("#ef4444","#2a0202")}
    rc, rbg = risk_colors.get(rl, ("#888","#111"))
    risk_icons = {"SAFE":"✅","CAUTION":"⚠️","WARNING":"🔶","CRITICAL":"🚨"}
    ri = risk_icons.get(rl,"🔴")
    # DI progress bar
    pct = int(DI_pk * 100)
    bar_col = rc
    st.markdown(f"""
    <div style="background:{rbg};border:1px solid {rc}33;border-left:4px solid {rc};
                border-radius:12px;padding:18px 22px;margin-bottom:16px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:2rem">{ri}</span>
          <div>
            <div style="font-size:1.5rem;font-weight:800;color:{rc}">{rl}</div>
            <div style="font-size:.8rem;color:#5a7199;margin-top:1px">
              Peak DI = <b style="color:{rc};font-family:'JetBrains Mono',monospace">{DI_pk:.4f}</b>
              &nbsp;·&nbsp; Final DI = <b style="font-family:'JetBrains Mono',monospace">{DI_fin:.4f}</b>
              &nbsp;·&nbsp; Total = <b>{t_tot:.1f} min</b>
              &nbsp;·&nbsp; {"✅ Within target" if t_tot<=30 else "❌ Exceeds 30 min"}
            </div>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:.72rem;color:#3d5478;margin-bottom:4px;text-transform:uppercase;letter-spacing:.08em">Damage Index</div>
          <div style="width:140px;background:#0f2040;border-radius:20px;height:10px;overflow:hidden">
            <div style="width:{pct}%;background:linear-gradient(90deg,{rc}99,{rc});height:100%;border-radius:20px;transition:width .6s"></div>
          </div>
          <div style="font-size:.72rem;color:#5a7199;margin-top:3px">{pct}% of maximum</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KEY METRICS ROW ────────────────────────────────────────────────────
    def _card(label, value, icon, border_col, note="", note_ok=None):
        nc = "#10b981" if note_ok else ("#ef4444" if note_ok is False else "#5a7199")
        note_html = f'<div style="font-size:.72rem;color:{nc};margin-top:3px">{note}</div>' if note else ""
        return (
            f'<div style="background:#080f1e;border:1px solid #0f2040;border-top:3px solid {border_col};'
            f'border-radius:10px;padding:14px 16px;height:100%">'
            f'<div style="font-size:.68rem;color:#3d5478;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:6px">{icon} {label}</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:#dde4f2;'
            f'font-family:&quot;JetBrains Mono&quot;,monospace">{value}</div>'
            f'{note_html}</div>'
        )

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(_card("Peak DI",    f"{DI_pk:.3f}", "🎯", rc,
                      f"{rl}", DI_pk<0.25), unsafe_allow_html=True)
    c2.markdown(_card("Total time", f"{t_tot:.0f} min", "⏱", "#3b82f6",
                      "≤30 ✅" if t_tot<=30 else ">30 ❌", t_tot<=30), unsafe_allow_html=True)
    c3.markdown(_card("Zones",      str(n_zones), "❄", "#8b5cf6",
                      f"Last={T_last:.0f}°C", T_last<=42), unsafe_allow_html=True)
    c4.markdown(_card("Biot #",     f"{Bi:.3f}", "📐", "#06b6d4",
                      "≤0.5 ✅" if Bi<=0.5 else ">0.5 ⚠️", Bi<=0.5), unsafe_allow_html=True)
    c5.markdown(_card("DI Healed",  f"{heal_drop:+.3f}", "🔥", "#f97316",
                      "Effective ✅" if heal_drop>0.02 else "Minimal", heal_drop>0.02 or t_reh==0),
                unsafe_allow_html=True)
    c6.markdown(_card("T_reheat",   f"{T_reh:.0f}°C" if t_reh>0 else "None", "🌡", "#f59e0b",
                      "Active ✅" if t_reh>0 else "Off"), unsafe_allow_html=True)

    st.markdown("<div style='margin:14px 0 6px;height:1px;background:#0f2040'></div>",
                unsafe_allow_html=True)

    # ── COOLING ZONE TABLE ─────────────────────────────────────────────────
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### ❄️ Cooling Zone Schedule")
        rows = []
        t_acc = 0.0
        for z in params["zones"]:
            in_mushy = T_SOL_C <= z["T"] <= T_LIQ_C
            rows.append({
                "Zone":       z["label"],
                "T_set (°C)": f"{z['T']:.0f}",
                "Duration":   f"{z['duration']:.1f} min",
                "Cumul.":     f"{t_acc + z['duration']:.1f} min",
                "Note":       "🔶 Mushy zone" if in_mushy else
                              ("✅ Below solidus" if z["T"]<T_SOL_C else "🔵 Above mushy"),
            })
            t_acc += z["duration"]
        st.dataframe(rows, width='stretch', hide_index=True)

    with right:
        st.markdown("#### 🔥 Reheat Advisory")
        if t_reh <= 0:
            st.info("No reheat applied.\n\n"
                    f"Current DI={DI_pk:.3f}. "
                    "Add reheat if DI > 0.20 to reduce ヒケ (sink marks).")
        else:
            T_enter  = T_last
            tau_r    = 3.0
            T_reach  = T_reh - (T_reh - T_enter) * np.exp(-t_reh / tau_r)
            if T_reach >= T_LIQ_C:
                heat_st = "🔴 Full re-melt — maximum sink-mark healing"
                col_h   = "#e74c3c"
            elif T_reach >= T_SOL_C:
                heat_st = "🟠 Mushy zone reached — partial healing"
                col_h   = "#e67e22"
            else:
                heat_st = "🟡 Below solidus — softening only"
                col_h   = "#f1c40f"
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:10px;padding:14px'>"
                f"<b>T_reheat:</b> {T_reh:.0f}°C &nbsp;|&nbsp; "
                f"<b>Duration:</b> {t_reh:.0f} min<br>"
                f"<b>Surface reaches:</b> ~{T_reach:.0f}°C<br>"
                f"<b>Effect:</b> <span style='color:{col_h}'>{heat_st}</span><br>"
                f"<b>DI healed:</b> {heal_drop:+.3f} "
                f"({'✅ Effective' if heal_drop>0.02 else '⚠️ Minimal'})"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── THRESHOLD CHECKLIST ────────────────────────────────────────────────
    st.markdown("#### ✔️ Process Quality Checklist")
    checks = [
        ("Total time ≤ 30 min",        t_tot <= 30,      f"{t_tot:.0f} min",     "≤ 30"),
        ("Biot ≤ 0.5 (uniform cool)",  Bi <= 0.5,        f"{Bi:.3f}",            "≤ 0.5"),
        ("Last zone ≤ 42°C",           T_last <= 42,     f"{T_last:.0f}°C",      "≤ 42"),
        ("Peak DI ≤ 0.25 (SAFE)",      DI_pk <= 0.25,    f"{DI_pk:.3f}",         "< 0.25"),
        ("No ヒケ expected",           DI_pk <= 0.15,    f"DI={DI_pk:.3f}",      "< 0.15"),
        ("Reheat heals DI",            heal_drop>0.01 or t_reh==0,
                                                          f"ΔDI={heal_drop:.3f}", "> 0.01"),
    ]
    check_df = {"Criterion":[c[0] for c in checks],
                "Pass?":[    "✅" if c[1] else "❌" for c in checks],
                "Current":[  c[2] for c in checks],
                "Target":[   c[3] for c in checks]}
    st.dataframe(check_df, width='stretch', hide_index=True)

    # ── CLIENT RECOMMENDATION BOX ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Recommendation for CBIC Production")
    if DI_pk <= 0.10:
        msg = (f"✅ **Excellent** — Current schedule achieves DI={DI_pk:.3f}. "
               f"No ヒケ expected. This {n_zones}-zone profile at {t_tot:.0f} min "
               f"{'with' if t_reh>0 else 'without'} reheat is production-ready.")
        col_box = "#1a4a2e"
    elif DI_pk <= 0.25:
        msg = (f"✅ **Good** — DI={DI_pk:.3f} [SAFE]. Minor surface imperfections "
               f"may appear. Consider increasing T_reheat to ≥ 90°C or adding "
               f"one more cooling zone near the mushy zone (62–72°C range) to "
               f"slow solidification. Estimated improvement: ΔDI≈−0.05.")
        col_box = "#1a3a1a"
    elif DI_pk <= 0.50:
        msg = (f"⚠️ **Warning** — DI={DI_pk:.3f}. Visible ヒケ likely at top surface. "
               f"Recommended actions: (1) Reduce h_cool to ≤ 6 W/m²K. "
               f"(2) Set Zone 1 temperature to 70–75°C (near mushy zone) to slow "
               f"surface cooling. (3) Increase T_reheat to 90–100°C. "
               f"(4) Run DRL Optimise in sidebar for automatic schedule.")
        col_box = "#3a2a0a"
    else:
        msg = (f"🚨 **Critical** — DI={DI_pk:.3f}. Severe cracking / ヒケ expected. "
               f"Process redesign required: (1) Use heat-retention cap near top "
               f"surface (like Trial 1: 59–63°C zone for 20 min). "
               f"(2) Increase T_fill to ≤ 82°C. "
               f"(3) Set h_cool ≤ 4 W/m²K to avoid rapid surface quench. "
               f"(4) Apply 90°C reheat × 3 pulses × 1 min (Trial 2 protocol).")
        col_box = "#3a0a0a"
    st.markdown(
        f"<div style='background:{col_box};border-radius:10px;"
        f"padding:16px 20px;font-size:0.97em;line-height:1.6'>{msg}</div>",
        unsafe_allow_html=True,
    )

def _hydrate_env_from_secrets():
    """Copy GROQ_API_KEY / GROQ_MODEL / OLLAMA_URL from st.secrets into the
    environment BEFORE the advisor is built (Streamlit Cloud only loads
    secrets into st.secrets, not os.environ). Tolerant of the common
    formatting mistakes: any letter case, and a `[groq]` section."""
    import os
    try:
        sec = st.secrets
    except Exception:
        return {}
    seen = {}
    def _set(k, v):
        v = str(v).strip()
        if v:
            os.environ[k] = v
    # Top-level keys, case-insensitive (handles groq_api_key, Groq_Api_Key, …)
    try:
        for key in list(sec.keys()):
            ku = str(key).upper()
            seen[str(key)] = True
            if ku in ("GROQ_API_KEY", "GROQ_MODEL", "OLLAMA_URL"):
                _set(ku, sec[key])
    except Exception:
        pass
    # Optional [groq] section: api_key = "...", model = "..."
    try:
        if "groq" in sec:
            g = sec["groq"]
            if g.get("api_key"): _set("GROQ_API_KEY", g["api_key"])
            if g.get("model"):   _set("GROQ_MODEL", g["model"])
    except Exception:
        pass
    return seen


def main():
    # ─── Cloud secrets → env (deployment) ──────────────────────────────────
    _secret_keys_seen = _hydrate_env_from_secrets()
    # Developer/diagnostic mode: append ?debug=1 to the URL. Hides all
    # setup/LLM/training panels from clients while keeping them for us.
    try:
        _admin = str(st.query_params.get("debug", "")) == "1"
    except Exception:
        _admin = False

    # ─── CSS ─────────────────────────────────────────────────────────────────
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    /* ═══ PI-DRL design system v2 — "Precision Lab" ════════════════════════ */
    :root{
      --bg0:#060c1a; --bg1:#0a1426; --surface:#0e1c34; --surface2:#10233f;
      --line:#172c4d; --line2:#244372; --hair:rgba(255,255,255,.04);
      --ink:#eaf1ff; --ink2:#aabfe4; --muted:#6a82ad;
      --brand:#5b8cff; --brand2:#84a9ff; --brandDk:#2f5fe0;
      --thermal:#fb923c; --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
      --r:14px; --r-sm:10px; --shadow:0 10px 34px rgba(0,0,0,.40);
    }
    html,body,.stApp{
      background:
        radial-gradient(1300px 680px at 82% -10%, rgba(91,140,255,.12), transparent 58%),
        radial-gradient(1000px 560px at 4% 2%, rgba(251,146,60,.07), transparent 55%),
        linear-gradient(180deg,#081123 0%, #060c1a 55%, #05091580 100%)!important;
      background-attachment:fixed!important;
      color:var(--ink)!important;font-family:Inter,system-ui,sans-serif!important;
      -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
      font-feature-settings:"cv02","cv03","cv04","ss01";
    }
    .block-container{padding-top:1.0rem!important;padding-bottom:2rem!important;max-width:1560px!important}
    h1,h2,h3,h4,h5{letter-spacing:-.01em!important}
    /* headings rendered by st.markdown('#### …') */
    .stMarkdown h4{color:var(--ink)!important;font-weight:800!important;font-size:1.02rem!important}
    *{scrollbar-width:thin;scrollbar-color:#244372 transparent}
    ::-webkit-scrollbar{width:9px;height:9px}
    ::-webkit-scrollbar-thumb{background:linear-gradient(#244372,#172c4d);border-radius:8px;border:2px solid transparent;background-clip:padding-box}
    ::-webkit-scrollbar-thumb:hover{background:#33599a;background-clip:padding-box}
    ::-webkit-scrollbar-track{background:transparent}
    /* ── Sidebar: frosted control deck with a left accent rail ── */
    [data-testid="stSidebar"]{
      background:linear-gradient(180deg,#0a1730 0%, #07101f 100%)!important;
      border-right:1px solid var(--line)!important;width:312px!important;
      box-shadow:inset -1px 0 0 var(--hair), 2px 0 30px rgba(0,0,0,.45)!important}
    [data-testid="stSidebar"]>div{padding-top:.4rem!important}
    [data-testid="stSidebar"] *{font-size:.82rem}
    [data-testid="stSidebar"]::-webkit-scrollbar{width:6px}
    [data-testid="stSidebar"] .stRadio [role="radiogroup"]{gap:6px!important}
    /* ── Tabs: segmented glass control with gradient active pill ── */
    .stTabs [data-baseweb="tab-list"]{background:linear-gradient(180deg,rgba(14,28,52,.75),rgba(8,17,30,.6))!important;
      border:1px solid var(--line)!important;border-radius:16px!important;
      padding:6px!important;gap:4px!important;backdrop-filter:blur(10px);
      box-shadow:inset 0 1px 0 var(--hair)}
    .stTabs [data-baseweb="tab"]{color:var(--ink2)!important;font-size:.83rem!important;
      font-weight:600!important;padding:9px 20px!important;border-radius:11px!important;
      border:none!important;transition:all .2s cubic-bezier(.2,.8,.2,1)!important}
    .stTabs [aria-selected="true"]{color:#fff!important;font-weight:700!important;
      background:linear-gradient(135deg,#3461e6,#5b8cff)!important;
      box-shadow:0 6px 20px rgba(91,140,255,.5),inset 0 1px 0 rgba(255,255,255,.25)!important}
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){
      color:var(--brand2)!important;background:rgba(91,140,255,.1)!important}
    .stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important}
    /* ── Buttons ── */
    .stButton>button,.stFormSubmitButton>button{border-radius:11px!important;font-weight:600!important;
      font-size:.82rem!important;transition:all .17s cubic-bezier(.2,.8,.2,1)!important;
      border:1px solid transparent!important;letter-spacing:.01em!important}
    .stButton>button[kind="primary"],.stFormSubmitButton>button{
      background:linear-gradient(135deg,#3461e6,#5b8cff)!important;color:#fff!important;
      box-shadow:0 6px 18px rgba(48,95,224,.45),inset 0 1px 0 rgba(255,255,255,.22)!important;
      padding:10px 22px!important;border:1px solid rgba(132,169,255,.35)!important}
    .stButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{
      box-shadow:0 10px 30px rgba(91,140,255,.6)!important;transform:translateY(-1.5px)!important;
      filter:brightness(1.07)!important}
    .stButton>button[kind="primary"]:active,.stFormSubmitButton>button:active{transform:translateY(0)!important}
    .stButton>button:not([kind="primary"]){background:rgba(14,28,52,.85)!important;
      border-color:var(--line2)!important;color:var(--ink2)!important}
    .stButton>button:not([kind="primary"]):hover{border-color:var(--brand)!important;
      color:var(--brand2)!important;background:rgba(91,140,255,.09)!important;
      transform:translateY(-1.5px)!important}
    /* ── Metric cards ── */
    [data-testid="stMetric"],div[data-testid="metric-container"]{
      background:linear-gradient(150deg,rgba(16,35,63,.7),rgba(9,18,32,.78))!important;
      border:1px solid var(--line)!important;border-radius:var(--r)!important;
      padding:13px 17px!important;transition:all .2s!important;
      box-shadow:inset 0 1px 0 var(--hair)}
    [data-testid="stMetric"]:hover,div[data-testid="metric-container"]:hover{
      border-color:var(--line2)!important;box-shadow:var(--shadow)!important;
      transform:translateY(-2px)}
    [data-testid="stMetricLabel"],div[data-testid="metric-container"] label{
      color:var(--muted)!important;font-size:.62rem!important;font-weight:700!important;
      letter-spacing:.13em!important;text-transform:uppercase!important}
    [data-testid="stMetricValue"],div[data-testid="metric-container"] [data-testid="metric-value"]{
      color:var(--ink)!important;font-family:"JetBrains Mono",monospace!important;
      font-size:1.18rem!important;font-weight:700!important;font-variant-numeric:tabular-nums}
    /* ── Inputs ── */
    .stNumberInput input,.stTextInput input,.stTextArea textarea{
      background:rgba(9,18,32,.9)!important;border:1px solid var(--line)!important;
      color:var(--ink)!important;border-radius:var(--r-sm)!important;
      font-family:"JetBrains Mono",monospace!important;font-size:.82rem!important;
      padding:9px 12px!important;transition:all .16s!important}
    .stNumberInput input:focus,.stTextInput input:focus,.stTextArea textarea:focus{
      border-color:var(--brand)!important;box-shadow:0 0 0 3px rgba(91,140,255,.22)!important}
    .stTextInput input::placeholder{color:#4d6699!important}
    .stNumberInput button{background:rgba(14,28,52,.9)!important;border-color:var(--line)!important;color:var(--muted)!important}
    .stNumberInput button:hover{background:#1a3056!important;color:var(--brand2)!important}
    .stSelectbox>div>div,[data-baseweb="select"]>div{background:rgba(9,18,32,.9)!important;border:1px solid var(--line)!important;border-radius:var(--r-sm)!important}
    .stSelectbox>div>div:hover{border-color:var(--line2)!important}
    .stRadio [role="radio"]{transition:all .15s!important}
    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] [role="slider"]{box-shadow:0 0 0 5px rgba(91,140,255,.22)!important}
    .stSlider [data-testid="stThumbValue"]{background:var(--brand)!important;color:#fff!important;
      border-radius:7px!important;font-family:"JetBrains Mono",monospace!important}
    /* ── Containers / alerts ── */
    .stDataFrame{border-radius:12px!important;overflow:hidden!important;border:1px solid var(--line)!important}
    .stSuccess{background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(9,18,32,.4))!important;border:1px solid rgba(16,185,129,.3)!important;border-radius:12px!important}
    .stWarning{background:linear-gradient(135deg,rgba(245,158,11,.12),rgba(9,18,32,.4))!important;border:1px solid rgba(245,158,11,.3)!important;border-radius:12px!important}
    .stInfo{background:linear-gradient(135deg,rgba(91,140,255,.1),rgba(9,18,32,.4))!important;border:1px solid rgba(91,140,255,.25)!important;border-radius:12px!important}
    .stError{background:linear-gradient(135deg,rgba(239,68,68,.12),rgba(9,18,32,.4))!important;border:1px solid rgba(239,68,68,.3)!important;border-radius:12px!important}
    .stCaption,[data-testid="stCaptionContainer"]{color:var(--muted)!important;font-size:.7rem!important}
    hr{border-color:var(--line)!important;margin:10px 0!important}
    .stSpinner>div{border-top-color:var(--brand)!important}
    .stExpander{border:1px solid var(--line)!important;border-radius:var(--r)!important;
      background:linear-gradient(150deg,rgba(14,28,52,.5),rgba(8,17,30,.55))!important;
      backdrop-filter:blur(8px);box-shadow:inset 0 1px 0 var(--hair)}
    .stExpander summary{font-weight:600!important;color:var(--ink2)!important}
    .stExpander summary:hover{color:var(--brand2)!important}
    /* ── Plotly chart: framed, floating panel ── */
    [data-testid="stPlotlyChart"]{border:1px solid var(--line)!important;border-radius:16px!important;
      overflow:hidden!important;
      background:radial-gradient(700px 300px at 80% -10%,rgba(91,140,255,.06),transparent),rgba(7,15,28,.6)!important;
      box-shadow:var(--shadow),inset 0 1px 0 var(--hair)!important;padding:4px!important}
    .modebar{background:transparent!important}
    /* ── File uploader ── */
    .stFileUploader{border-radius:var(--r)!important}
    [data-testid="stFileUploaderDropzone"]{background:rgba(9,18,32,.6)!important;
      border:1.5px dashed var(--line2)!important;border-radius:var(--r)!important;transition:all .2s!important}
    [data-testid="stFileUploaderDropzone"]:hover{border-color:var(--brand)!important;background:rgba(91,140,255,.06)!important}
    .stProgress>div>div>div{background:linear-gradient(90deg,#3461e6,#5b8cff,#fb923c)!important}
    /* ── Reusable utility cards ── */
    .pdrl-card{background:linear-gradient(150deg,rgba(16,35,63,.6),rgba(9,18,32,.78));
      border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow),inset 0 1px 0 var(--hair)}
    /* ── Animations ── */
    @keyframes fadeIn{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
    @keyframes pulseGlow{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.5)}50%{box-shadow:0 0 0 7px rgba(52,211,153,0)}}
    @keyframes shimmer{0%{background-position:-400px 0}100%{background-position:400px 0}}
    @keyframes floatIn{from{opacity:0;transform:translateY(10px) scale(.99)}to{opacity:1;transform:none}}
    @keyframes typingBounce{0%,60%,100%{transform:translateY(0);opacity:.45}30%{transform:translateY(-6px);opacity:1}}
    </style>""", unsafe_allow_html=True)

    # ─── SESSION INIT ─────────────────────────────────────────────────────────
    for k,v in [("sim_done",False),("sim_ts",None),
                 ("drl_done",False),("drl_ts",None),
                 ("last_param_hash",""),("zone_version",0)]:
        if k not in st.session_state: st.session_state[k]=v

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────
    params = sidebar()

    # ─── AUTO-COMPUTE ─────────────────────────────────────────────────────────
    import hashlib, json as _json
    _hs = _json.dumps({
        "T_fill": params["T_fill"],
        "zones":  [(z["T"],z["duration"]) for z in params["zones"]],
        "reheat": (params["reheat"]["T"], params["reheat"]["duration"]),
        "h_cool": params["h_cool"], "h_reheat": params["h_reheat"],
    }, sort_keys=True)
    _hash = hashlib.md5(_hs.encode()).hexdigest()[:8]
    if _hash != st.session_state.last_param_hash:
        with st.spinner(""):
            _ts = build_timeline(params["T_fill"],params["zones"],params["reheat"],
                                 params["h_cool"],params["h_reheat"],n_pts=120)
            st.session_state.sim_ts=_ts; st.session_state.sim_done=True
            st.session_state.last_param_hash=_hash

    ats = st.session_state.sim_ts if st.session_state.sim_done else None

    # ─── LIVE DATA ────────────────────────────────────────────────────────────
    DI_live   = max(ats["DI"])         if ats else 0.0
    t_live    = ats["t_total"]         if ats else 0.0
    Bi_live   = params["h_cool"]*R_M/K_TH
    rl_live   = _rlbl(DI_live)
    rc_live   = _rcol(DI_live)
    heal_live = 0.0
    if ats and ats["t_reheat"]>0:
        ta_=np.array(ats["times"]); DI_=np.array(ats["DI"])
        is_=int(np.argmin(np.abs(ta_-ats["reheat_start"])))
        ie_=int(np.argmin(np.abs(ta_-ats["reheat_end"])))
        heal_live=float(DI_[is_])-float(DI_[min(ie_,len(DI_)-1)])

    # ─── HEADER ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(180deg,#050f20,#040d1c);
                border-bottom:1px solid #0d1e38;padding:14px 0 12px;margin-bottom:2px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <div style="width:40px;height:40px;background:linear-gradient(135deg,#ea580c,#b91c1c);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:20px;box-shadow:0 3px 14px rgba(234,88,12,.5);flex-shrink:0">🔥</div>
        <div style="flex:1">
          <div style="font-size:1.4rem;font-weight:800;
                      background:linear-gradient(135deg,#dde8ff,#93c5fd);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1">
            PI-DRL Solidification Control</div>
          <div style="font-size:.78rem;color:#ea580c;margin-top:1px;font-weight:500">
            {t('app_subtitle')}</div>
        </div>
        <div style="display:flex;gap:6px;flex-shrink:0;align-items:center">
          <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
                       color:#60a5fa;padding:4px 10px;border-radius:14px;font-size:.67rem;font-weight:700;letter-spacing:.05em">CBIC × TUAT</span>
          <span style="background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.2);
                       color:#a78bfa;padding:4px 10px;border-radius:14px;font-size:.67rem;font-weight:700">2026</span>
          <span style="background:{rc_live}22;border:1px solid {rc_live}44;color:{rc_live};
                       padding:4px 12px;border-radius:14px;font-size:.67rem;font-weight:800">{rl_live}</span>
        </div>
      </div>
      <!-- KPI strip -->
      <div style="display:flex;gap:6px">
        {"".join([
          f'<div style="flex:1;background:#07111e;border:1px solid #0d1e38;border-top:2px solid {col};border-radius:7px;padding:8px 12px">'
          f'<div style="font-size:.6rem;color:#2d3f5a;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px">{lbl}</div>'
          f'<div style="font-size:.95rem;font-weight:700;color:{col};font-family:JetBrains Mono,monospace">{val}</div>'
          f'</div>'
          for lbl,val,col in [
            (t('kpi_peak_di'), f"{DI_live:.3f}",  rc_live),
            (t('kpi_risk'),    rl_live,           rc_live),
            (t('kpi_total'),   f"{t_live:.0f} min","#3b82f6" if t_live<=30 else "#ef4444"),
            (t('kpi_biot'),    f"{Bi_live:.3f}",  "#10b981" if Bi_live<=0.5 else "#f59e0b"),
            (t('kpi_healed'),  f"{heal_live:+.3f}","#10b981" if heal_live>0.02 else "#4a6288"),
            (t('kpi_zones'),   str(len(params["zones"])), "#8b5cf6"),
            (t('kpi_treheat'), f"{params['reheat']['T']:.0f}°C" if params['reheat']['duration']>0 else "None","#f97316"),
          ]
        ])}
      </div>
    </div>""", unsafe_allow_html=True)

    # ─── PROCESS FLOW DIAGRAM ─────────────────────────────────────────────────
    zones     = params["zones"]
    reheat    = params["reheat"]
    t_cool    = sum(z["duration"] for z in zones)
    t_reh     = reheat["duration"]
    t_fin     = float(np.clip(30-t_cool-t_reh-0.5,2,8))
    t_tot_est = t_cool+t_reh+t_fin
    ZONE_COLS = ["#ef4444","#f97316","#f59e0b","#10b981","#06b6d4","#8b5cf6","#ec4899"]

    def _flow_box(label, sub, col, w_pct):
        return (
            f'<div style="flex:{w_pct};min-width:0;background:{col}18;border:1px solid {col}44;'
            f'border-top:3px solid {col};border-radius:7px;padding:7px 10px;text-align:center">'
            f'<div style="font-size:.68rem;font-weight:700;color:{col};white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis">{label}</div>'
            f'<div style="font-size:.62rem;color:#3d5478;margin-top:1px">{sub}</div></div>'
        )

    arrow = '<div style="color:#1a3056;font-size:.9rem;display:flex;align-items:center;flex-shrink:0;padding:0 2px">→</div>'
    boxes = [_flow_box("📍 Fill",f"{params['T_fill']:.0f}°C","#f97316",0.7)]
    for i,z in enumerate(zones):
        col = ZONE_COLS[i%len(ZONE_COLS)]
        boxes.append(arrow)
        boxes.append(_flow_box(f"❄ {z['label']}",f"{z['T']:.0f}°C / {z['duration']:.0f}m",col,1))
    if t_reh>0:
        boxes.append(arrow)
        boxes.append(_flow_box("🔥 Reheat",f"{reheat['T']:.0f}°C / {t_reh:.0f}m","#f97316",1))
    boxes.append(arrow)
    boxes.append(_flow_box("🏠 Room","23°C","#3b82f6",0.8))
    ok_col = "#10b981" if t_tot_est<=30 else "#ef4444"
    boxes.append(f'<div style="flex-shrink:0;background:{ok_col}15;border:1px solid {ok_col}44;'
                 f'border-radius:7px;padding:7px 10px;text-align:center;margin-left:4px">'
                 f'<div style="font-size:.68rem;font-weight:700;color:{ok_col}">⏱ Total</div>'
                 f'<div style="font-size:.72rem;font-family:JetBrains Mono,monospace;color:{ok_col};font-weight:800">'
                 f'{t_tot_est:.0f} min {"✓" if t_tot_est<=30 else "✗"}</div></div>')

    st.markdown(
        f'<div style="display:flex;align-items:stretch;gap:4px;'
        f'background:#040d1c;border:1px solid #0d1e38;border-radius:8px;'
        f'padding:8px;margin-bottom:6px;overflow-x:auto">{"".join(boxes)}</div>',
        unsafe_allow_html=True)

    # ─── ACTION BUTTON ────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        run_sim = st.button("▶ Compute", type="primary", width='stretch')
    with col_info:
        if ats:
            st.markdown(
                f'<div style="background:#07111e;border:1px solid #0d1e38;border-radius:7px;'
                f'padding:7px 14px;font-size:.75rem;color:#3d5478;margin-top:2px">'
                f'Simulation ready · Peak DI=<b style="color:{rc_live}">{DI_live:.4f}</b> '
                f'[<b style="color:{rc_live}">{rl_live}</b>] · '
                f'{t_live:.0f} min total · Bi={Bi_live:.3f} · '
                f'{"✅ Within 30-min target" if t_live<=30 else "❌ Exceeds 30-min target"}'
                f'</div>', unsafe_allow_html=True)

    if run_sim:
        with st.spinner("Simulating..."):
            _ts2 = build_timeline(params["T_fill"],params["zones"],params["reheat"],
                                  params["h_cool"],params["h_reheat"],n_pts=120)
            st.session_state.sim_ts=_ts2; st.session_state.sim_done=True
            st.session_state.last_param_hash=_hash
        st.rerun()

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ─── TABS ─────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        t("tab_belt"), t("tab_temp"), t("tab_crack"),
        t("tab_results"), t("tab_advisor"),
    ])

    # ══ TAB 1: Belt ════════════════════════════════════════════════════════════
    with tab1:
        ts_b = ats if ats else build_timeline(params["T_fill"],params["zones"],
            params["reheat"],params["h_cool"],params["h_reheat"],n_pts=60)
        belt = fig_belt(ts_b,params["zones"],params["reheat"],
                        params["h_cool"],params["h_reheat"],n_frames=40)
        # Stable key + figure uirevision => camera rotation/zoom is preserved
        # across timeline drags AND Streamlit reruns (no more snap-to-default).
        st.plotly_chart(belt, width='stretch', key="belt3d",
                        config={"displayModeBar":True,"displaylogo":False,
                                "scrollZoom":True,
                                "modeBarButtonsToRemove":["pan3d","tableRotation"]})
        st.markdown(
            '<div style="background:#07111e;border:1px solid #0d1e38;border-radius:7px;'
            'padding:7px 14px;font-size:.74rem;color:#2d3f5a">'
            '<b style="color:#60a5fa">→ Top lane</b>: Cooling zones &nbsp;'
            '<b style="color:#9ca3af">↻ U-turn</b>: Right end &nbsp;'
            '<b style="color:#f97316">← Bottom lane</b>: Reheat + Final cooling &nbsp;|&nbsp;'
            '🔴 Crack lines appear when DI ≥ 0.25 &nbsp;|&nbsp;'
            '▶ <b>Play</b> animates fully in-browser'
            '</div>', unsafe_allow_html=True)
        export_figure_panel(belt, "uturn_belt", "exp_belt")

        if ats:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Peak DI",    f"{DI_live:.3f}", delta=rl_live,
                      delta_color="inverse" if DI_live>=0.25 else "normal")
            c2.metric("Min DI",     f"{min(ats['DI']):.3f}")
            c3.metric("Total",      f"{t_live:.0f} min",
                      delta="✅" if t_live<=30 else "❌",
                      delta_color="normal" if t_live<=30 else "inverse")
            c4.metric("Biot #",     f"{Bi_live:.3f}",
                      delta="✅ uniform" if Bi_live<=0.5 else "⚠️ non-uniform",
                      delta_color="normal" if Bi_live<=0.5 else "inverse")
            c5.metric("DI Healed",  f"{heal_live:+.3f}",
                      delta="✅ effective" if heal_live>0.02 else "—")

    # ══ TAB 2: Temperature & DI ═════════════════════════════════════════════════
    with tab2:
        if ats:
            ts_use=ats; lbl_use=params.get("_scenario","Simulation")
        else:
            with st.spinner("Preview..."):
                ts_use=build_timeline(params["T_fill"],params["zones"],params["reheat"],
                                      params["h_cool"],params["h_reheat"],n_pts=120)
            lbl_use="Preview"
        charts=fig_charts(ts_use,lbl_use)
        st.plotly_chart(charts,width='stretch')
        st.markdown(
            '<div style="background:#07111e;border:1px solid #0d1e38;border-radius:7px;'
            'padding:7px 14px;font-size:.74rem;color:#2d3f5a">'
            '🟠 <b style="color:#f97316">Orange shading</b> = reheat window &nbsp;|&nbsp;'
            'Dashed lines = zone boundaries &nbsp;|&nbsp;'
            'DI dips during reheat = healing effect &nbsp;|&nbsp;'
            '▶ / ⏸ fully browser-side — no page reload'
            '</div>', unsafe_allow_html=True)
        export_figure_panel(charts, "temperature_DI", "exp_temp")

    # ══ TAB 3: Cracks ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:12px'>
          <div style='width:40px;height:40px;border-radius:11px;flex-shrink:0;
            background:linear-gradient(135deg,#dc2626,#7f1d1d);display:flex;
            align-items:center;justify-content:center;font-size:20px;
            box-shadow:0 4px 16px rgba(220,38,38,.4)'>💥</div>
          <div><div style='font-size:1.15rem;font-weight:800;
            background:linear-gradient(135deg,#fecaca,#fca5a5);-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;line-height:1.15'>{t("crack_title")}</div>
          <div style='font-size:.72rem;color:#5f7aa3;margin-top:1px'>{t("crack_sub")}</div>
          </div></div>""", unsafe_allow_html=True)

        ts_cr = ats if ats else ts_use
        ta_   = np.array(ts_cr["times"]); DI_   = np.array(ts_cr["DI"])
        reh_s = float(ts_cr["reheat_start"]); reh_e = float(ts_cr["reheat_end"])
        has_reheat = float(ts_cr["t_reheat"]) > 0
        # "Before" = peak damage formed during cooling (up to end of reheat
        # window). "After" = the HEALED residual at the end of reheat — this
        # is what the user sees as the final condition. So when reheat (e.g.
        # a DRL-optimised schedule) heals DI below the visible threshold,
        # the After panel is correctly crack-free.
        ie_reh = int(np.argmin(np.abs(ta_ - reh_e))) if has_reheat else len(DI_)-1
        ib     = int(np.argmax(DI_[:max(1, ie_reh+1)]))
        ia     = ie_reh if has_reheat else len(DI_)-1
        DIb    = float(DI_[ib]); DIa=float(DI_[ia])
        tb_    = float(ta_[ib]); ta2_=float(ta_[ia])
        hdrop  = DIb-DIa

        # Cracks become visible only at CAUTION+ (DI ≥ 0.25), matching the
        # risk thresholds. So a well-healed / DRL-optimised state (DI < 0.25)
        # renders crack-FREE — the fracture network follows the actual
        # condition instead of always drawing cracks.
        CRACK_TH = 0.25

        def crack3d(DI_v, ttl):
            H_c=H_M*100; R_c=R_M*100
            n_th=64; n_zl=36
            th_s=np.linspace(0,2*np.pi,n_th); z_s=np.linspace(0,H_c,n_zl)
            TH_,ZS_=np.meshgrid(th_s,z_s)
            z_norm=ZS_/H_c
            has_crack = DI_v >= CRACK_TH
            rng_sf=np.random.default_rng(int(DI_v*1000)%9999)
            # Damage field φ on the surface — concentrated near the top
            # fill-point and scaled by DI so colour reads as severity.
            n_hot=max(1,int(DI_v*6))
            hot_th=rng_sf.uniform(0,2*np.pi,n_hot); hot_z=rng_sf.uniform(0.78,1.0,n_hot)
            phi_sf=np.zeros_like(TH_)
            phi_sf+=DI_v*np.clip(z_norm-0.45,0,1)*1.7
            for ha,hz in zip(hot_th,hot_z):
                d_th=np.abs(np.arctan2(np.sin(TH_-ha),np.cos(TH_-ha)))
                d_z=np.abs(z_norm-hz)
                phi_sf+=DI_v*1.15*np.exp(-(d_th**2/0.32+d_z**2/0.055))
            phi_sf=np.clip(phi_sf,0,1)
            Xs=R_c*np.cos(TH_); Ys=R_c*np.sin(TH_)
            f3=go.Figure()
            # High-contrast, more opaque colourscale → far better visibility
            f3.add_trace(go.Surface(x=Xs,y=Ys,z=ZS_,surfacecolor=phi_sf,
                colorscale=[[0.00,"rgba(30,64,140,0.72)"],[0.16,"rgba(16,150,165,0.78)"],
                             [0.38,"rgba(120,200,75,0.84)"],[0.58,"rgba(245,200,40,0.92)"],
                             [0.78,"rgba(249,115,22,0.97)"],[1.00,"rgba(225,28,28,1.0)"]],
                cmin=0,cmax=1,showscale=True,
                colorbar=dict(title=dict(text="φ (damage)",font=dict(color="#cfe0f5",size=10)),
                              tickfont=dict(color="#9fb4d4",size=8),x=1.02,len=0.62,thickness=11,
                              outlinewidth=0),
                lighting=dict(ambient=0.65,diffuse=0.85,specular=0.18,roughness=0.55),
                opacity=0.92,name="Damage φ",hoverinfo="skip"))
            if has_crack:
                # Crack density & extent scale with how far above threshold
                sev=float(np.clip((DI_v-CRACK_TH)/(1-CRACK_TH),0,1))   # 0..1 over CAUTION→FAILURE
                rng2=np.random.default_rng(42)
                n_nuc=max(2,int(2+sev*13)); blen=float(np.clip(1.2+sev*6,1.2,7))
                segs=[]; nxl,nyl,nzl=[],[],[]
                for _ in range(n_nuc):
                    r_n=R_c*rng2.uniform(0,0.28); th_n=rng2.uniform(0,2*np.pi)
                    z_n=H_c*rng2.uniform(0.85,1)
                    xn=r_n*np.cos(th_n); yn=r_n*np.sin(th_n)
                    nxl.append(xn); nyl.append(yn); nzl.append(z_n)
                    out_th=rng2.uniform(0,2*np.pi); out_f=rng2.uniform(0.25,0.65)
                    pdx=np.cos(out_th)*out_f; pdy=np.sin(out_th)*out_f
                    pdz=-rng2.uniform(0.65,0.98)
                    nm=float(np.sqrt(pdx**2+pdy**2+pdz**2))+1e-9
                    _grow_branch(xn,yn,z_n,pdx/nm,pdy/nm,pdz/nm,blen,
                                 depth=int(3+sev*3),R_c=R_c,H_c=H_c,rng=rng2,segments=segs)
                xm,ym,zm=[],[],[]
                for (xs,ys,zs) in segs:
                    xm.extend(xs+[None]); ym.extend(ys+[None]); zm.extend(zs+[None])
                cc=_crack_color(DI_v); hl=_lighten_color(cc)
                w=int(np.clip(6+sev*16,6,22))
                # outer glow + bright core for a vivid, readable crack
                f3.add_trace(go.Scatter3d(x=xm,y=ym,z=zm,mode="lines",
                    line=dict(color=hl,width=w+6),opacity=0.30,
                    showlegend=False,hoverinfo="skip"))
                f3.add_trace(go.Scatter3d(x=xm,y=ym,z=zm,mode="lines",
                    line=dict(color=cc,width=w),opacity=0.98,name="Cracks"))
                f3.add_trace(go.Scatter3d(x=xm,y=ym,z=zm,mode="lines",
                    line=dict(color="#ffffff",width=max(1,w-6)),opacity=0.55,
                    showlegend=False,hoverinfo="skip"))
                f3.add_trace(go.Scatter3d(x=nxl,y=nyl,z=nzl,mode="markers",
                    marker=dict(size=7,color="#fff5cc",
                                line=dict(color=cc,width=2.5)),
                    name="Nucleation",opacity=1))
            else:
                # Healed / SAFE — clean surface + a clear crack-free badge
                f3.add_trace(go.Scatter3d(x=[0],y=[0],z=[H_c*1.06],
                    mode="markers+text",
                    marker=dict(size=22,color="#10b981",symbol="circle",
                                line=dict(color="#ffffff",width=2)),
                    text=["✓"],textposition="middle center",
                    textfont=dict(size=15,color="#ffffff"),
                    name=t("crack_free"),hoverinfo="skip"))
            rl_=_rlbl(DI_v); rc_=_rcol(DI_v)
            f3.update_layout(
                height=460,paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dde4f2",family="Inter, sans-serif"),
                uirevision="crack",
                title=dict(text=f"<b>{ttl}</b>  ·  DI={DI_v:.3f}  [{rl_}]",
                           font=dict(size=12,color=rc_),x=0.5,xanchor="center",y=0.97),
                scene=dict(
                    uirevision="crack",
                    xaxis=dict(title=dict(text="X (cm)",font=dict(size=9,color="#9fb4d4")),
                               range=[-2,2],tickfont=dict(size=7,color="#6b82a8"),
                               gridcolor="#10233f",showbackground=True,
                               backgroundcolor="rgba(8,18,36,0.4)"),
                    yaxis=dict(title=dict(text="Y (cm)",font=dict(size=9,color="#9fb4d4")),
                               range=[-2,2],tickfont=dict(size=7,color="#6b82a8"),
                               gridcolor="#10233f",showbackground=True,
                               backgroundcolor="rgba(8,18,36,0.4)"),
                    zaxis=dict(title=dict(text="H (cm)",font=dict(size=9,color="#9fb4d4")),
                               range=[0,H_c+0.6],tickfont=dict(size=7,color="#6b82a8"),
                               gridcolor="#10233f",showbackground=True,
                               backgroundcolor="rgba(8,18,36,0.4)"),
                    bgcolor="rgba(4,13,28,0)",aspectmode="manual",
                    aspectratio=dict(x=1,y=1,z=2.0),
                    camera=dict(eye=dict(x=0.8,y=-1.8,z=1.6))),
                margin=dict(l=0,r=0,t=46,b=0),
                legend=dict(bgcolor="rgba(7,17,30,.85)",font=dict(color="#c8d6ea",size=8),
                            x=0.78,y=0.98,bordercolor="#1e3358",borderwidth=1))
            return f3

        _cfg = {"displayModeBar":True,"displaylogo":False,"scrollZoom":True,
                "modeBarButtonsToRemove":["pan3d","tableRotation"]}
        _fig_before = crack3d(DIb, t("before_reheat"))
        _fig_after  = crack3d(DIa, t("after_reheat"))
        cl,cr = st.columns(2)
        with cl:
            st.markdown(f"<div style='font-size:.84rem;font-weight:700;color:#e6eefc;"
                        f"margin-bottom:6px'>{t('before_reheat')} — "
                        f"<span style='color:#f97316'>t={tb_:.1f}min</span></div>",
                        unsafe_allow_html=True)
            st.plotly_chart(_fig_before,width='stretch',key="crack_before",config=_cfg)
        with cr:
            st.markdown(f"<div style='font-size:.84rem;font-weight:700;color:#e6eefc;"
                        f"margin-bottom:6px'>{t('after_reheat')} — "
                        f"<span style='color:#10b981'>t={ta2_:.1f}min</span></div>",
                        unsafe_allow_html=True)
            st.plotly_chart(_fig_after,width='stretch',key="crack_after",config=_cfg)
        export_figure_panel({t("before_reheat"): _fig_before, t("after_reheat"): _fig_after},
                            "crack_propagation", "exp_crack")

        if hdrop>0.005:
            eff=int(hdrop/max(DIb,0.01)*100)
            healed_clean = (DIa < CRACK_TH <= DIb)
            extra = (f' &nbsp;·&nbsp; <b style="color:#34d399">{t("crack_free")}</b>'
                     if healed_clean else "")
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(16,185,129,.12),'
                f'rgba(7,17,30,.6));border:1px solid rgba(16,185,129,.3);'
                f'border-radius:12px;padding:14px 20px;display:flex;align-items:center;'
                f'gap:14px;box-shadow:0 6px 22px rgba(0,0,0,.3)">'
                f'<span style="font-size:1.6rem">🔥</span>'
                f'<div><div style="font-size:.9rem;font-weight:700;color:#34d399">'
                f'{t("healed_by")} {hdrop:.3f} ({DIb:.3f} → {DIa:.3f})</div>'
                f'<div style="font-size:.78rem;color:#7e96bd;margin-top:3px">'
                f'{t("healing_eff")}: <b style="color:#34d399">{eff}%</b> &nbsp;·&nbsp;'
                f'{t("optimal_healing") if params["reheat"]["T"]>=62 else t("softening_only")}'
                f'{extra}</div></div></div>', unsafe_allow_html=True)
        elif params["reheat"]["duration"]==0:
            st.info(t("no_reheat"))
        else:
            st.warning(t("min_healing"))

    # ══ TAB 4: Results ═══════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
            f'<div style="width:40px;height:40px;border-radius:11px;flex-shrink:0;'
            f'background:linear-gradient(135deg,#3461e6,#5b8cff);display:flex;'
            f'align-items:center;justify-content:center;font-size:20px;'
            f'box-shadow:0 4px 16px rgba(48,95,224,.45)">📊</div>'
            f'<div><div style="font-size:1.15rem;font-weight:800;'
            f'background:linear-gradient(135deg,#dde8ff,#84a9ff);-webkit-background-clip:text;'
            f'-webkit-text-fill-color:transparent;line-height:1.15">{t("results_title")}</div>'
            f'<div style="font-size:.72rem;color:#6a82ad;margin-top:1px">{t("results_sub")}'
            f'</div></div></div>', unsafe_allow_html=True)
        if ats:
            show_results(ats,params)
        else:
            st.info("▶ Click **Compute** or select a scenario — results appear automatically.")

    # ══ TAB 5: AI Advisor ═══════════════════════════════════════════════════════
    with tab5:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
            '<div style="width:40px;height:40px;border-radius:11px;flex-shrink:0;'
            'background:linear-gradient(135deg,#f97316,#dc2626);display:flex;'
            'align-items:center;justify-content:center;font-size:20px;'
            'box-shadow:0 4px 16px rgba(234,88,12,.45)">\U0001F916</div>'
            '<div><div style="font-size:1.15rem;font-weight:800;'
            'background:linear-gradient(135deg,#dde8ff,#93c5fd);-webkit-background-clip:text;'
            f'-webkit-text-fill-color:transparent;line-height:1.15">{t("advisor_title")}</div>'
            f'<div style="font-size:.72rem;color:#5f7aa3;margin-top:1px">{t("advisor_sub")}'
            f'</div></div></div>', unsafe_allow_html=True)

        # ── Load advisor (cached, but version-keyed so retraining doesn't
        #    need a page reload — bumping advisor_version creates a fresh
        #    cache entry while _load_advisor.clear() drops the old one) ──
        if "advisor_version" not in st.session_state:
            st.session_state.advisor_version = 0

        @st.cache_resource(show_spinner=False)
        def _load_advisor(_version: int):
            try:
                from llm_advisor_v4 import PILLMAdvisor
                return PILLMAdvisor()
            except ImportError:
                return None

        advisor = _load_advisor(st.session_state.advisor_version)

        # ── Per-session conversation state ─────────────────────────────────
        # CRITICAL: the advisor above is @st.cache_resource — a SINGLE object
        # shared by every visitor/session on the server. Storing chat history
        # on it would leak one user's conversation to everyone (the "PC2 sees
        # PC1's chat" bug). So each browser session keeps its OWN history list
        # and id here in st.session_state, which is per-session and starts
        # empty whenever the site is opened afresh.
        if "chat_sid" not in st.session_state:
            import uuid
            st.session_state.chat_sid = uuid.uuid4().hex
        if "chat_hist" not in st.session_state:
            st.session_state.chat_hist = []

        # ── Status bar ─────────────────────────────────────────────────────
        if advisor is None:
            st.error("⚠️ llm_advisor_v4.py not found. "
                     "Copy it to the version4/ folder and restart the app.")
        else:
            st_info = advisor.status()
            _groq   = st_info.get("groq")
            _llm_up = _groq or st_info["ollama_up"]
            if _groq:
                mode_col, mode_icon = "#34d399", "🟢"
                active_label   = "Groq Cloud LLM active"
                model_name_show = st_info.get("groq_model", "groq")
            elif st_info["ollama_up"]:
                mode_col, mode_icon = "#10b981", "🟢"
                active_label   = "Ollama LLM active"
                model_name_show = st_info["mode"].replace("ollama:", "")
            else:
                mode_col, mode_icon = "#f59e0b", "🟡"
                active_label   = "Rule-based mode (no LLM backend)"
                model_name_show = "rule-based"

            # Only nudge to start Ollama when there's NO backend at all
            _ollama_hint = (
                '<div style="margin-left:auto"><span style="background:rgba(239,68,68,.1);'
                'border:1px solid rgba(239,68,68,.2);color:#f87171;padding:4px 10px;'
                'border-radius:6px;font-size:.7rem">Set GROQ_API_KEY, or run: ollama run pidrl-advisor</span></div>'
                if not _llm_up else ""
            )

            # Backend badge
            _warm_badge = ""
            if _groq:
                _warm_badge = (
                    '<span style="background:rgba(52,211,153,.1);border:1px solid '
                    'rgba(52,211,153,.25);color:#34d399;padding:2px 8px;border-radius:5px;'
                    'font-size:.66rem;margin-left:8px">⚡ Cloud — fast responses</span>'
                )
            elif st_info["ollama_up"]:
                if st_info["model_warm"]:
                    _warm_badge = (
                        '<span style="background:rgba(16,185,129,.1);border:1px solid '
                        'rgba(16,185,129,.25);color:#34d399;padding:2px 8px;border-radius:5px;'
                        'font-size:.66rem;margin-left:8px">🔥 Model warm — fast responses</span>'
                    )
                else:
                    _warm_badge = (
                        '<span style="background:rgba(245,158,11,.1);border:1px solid '
                        'rgba(245,158,11,.25);color:#f59e0b;padding:2px 8px;border-radius:5px;'
                        'font-size:.66rem;margin-left:8px">🥶 Loading model (~30-90s first query)</span>'
                    )

            _lc_badge = (
                '<b style="color:#a78bfa">\U0001F9E0 LangChain</b> &nbsp;·&nbsp;'
                if st_info.get("langchain") else "")
            if _admin:
                # Full technical status bar — developers only (?debug=1)
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,rgba(11,32,64,.55),'
                    f'rgba(7,17,30,.8));border:1px solid #13294a;'
                    f'border-radius:12px;padding:12px 18px;margin-bottom:12px;'
                    f'display:flex;align-items:center;gap:14px;flex-wrap:wrap;'
                    f'box-shadow:0 6px 22px rgba(0,0,0,.3)">'
                    f'<span style="font-size:1.1rem;{"animation:pulseGlow 2s infinite;border-radius:50%" if _llm_up else ""}">{mode_icon}</span>'
                    f'<div>'
                    f'<div style="font-size:.82rem;font-weight:700;color:{mode_col}">'
                    f'{active_label}{_warm_badge}</div>'
                    f'<div style="font-size:.72rem;color:#5f7aa3;margin-top:2px">'
                    f'Model: <b style="color:#9fb4d4">{model_name_show}</b> &nbsp;·&nbsp;'
                    f'KB: <b style="color:#9fb4d4">{"✅" if st_info["kb_loaded"] else "❌"}</b> &nbsp;·&nbsp;'
                    f'RAG: <b style="color:{"#34d399" if st_info["rag_active"] else "#5a7199"}">'
                    f'{("✅ "+str(st_info["rag_chunks"])+" chunks") if st_info["rag_active"] else "off"}</b> &nbsp;·&nbsp;'
                    f'{_lc_badge}Cache: {st_info["cache_size"]}'
                    f'</div></div>{_ollama_hint}</div>', unsafe_allow_html=True)
            else:
                # Clean, client-facing status pill — no technical jargon.
                _online = bool(_llm_up)
                _c = "#34d399" if _online else "#5f7aa3"
                _txt = "AI Advisor — online" if _online else "AI Advisor — ready"
                st.markdown(
                    f'<div style="display:inline-flex;align-items:center;gap:9px;'
                    f'background:rgba(52,211,153,.08);border:1px solid {_c}44;'
                    f'border-radius:30px;padding:6px 16px;margin-bottom:12px">'
                    f'<span style="width:9px;height:9px;border-radius:50%;background:{_c};'
                    f'{"animation:pulseGlow 2s infinite" if _online else ""};flex-shrink:0"></span>'
                    f'<span style="font-size:.78rem;font-weight:700;color:{_c}">{_txt}</span>'
                    f'<span style="font-size:.68rem;color:#5f7aa3">· grounded in your experimental data</span>'
                    f'</div>', unsafe_allow_html=True)

            if _admin and st_info["ollama_up"] and not st_info["model_warm"] and st_info.get("model_healthy") is not False:
                st.caption("⏳ Model is loading into memory in the background "
                          "(happens once — subsequent queries respond in seconds). "
                          "You can keep using the dashboard while it loads.")

            # ── Backend diagnostics — developers only (?debug=1) ───────────
            if _admin and _groq:
                # Live connectivity check, run once per session (cached).
                if "groq_ping" not in st.session_state:
                    with st.spinner("Checking Groq connection…"):
                        st.session_state["groq_ping"] = advisor.groq_ping()
                _ok, _detail = st.session_state["groq_ping"]
                if _ok:
                    st.caption(f"✅ Groq backend: {_detail}")
                else:
                    st.error(
                        f"⚠️ **Groq is configured but the call failed** — answers are "
                        f"falling back to rule-based.\n\n**Reason:** `{_detail}`\n\n"
                        "Fix: check the API key is valid, or set a current model via the "
                        "`GROQ_MODEL` secret (e.g. `llama-3.1-8b-instant`).")
                    if st.button("🔄 Recheck Groq", key="groq_recheck"):
                        st.session_state.pop("groq_ping", None); st.rerun()
            elif _admin and not st_info["ollama_up"]:
                # Deployed with no LLM backend at all → rule-based only.
                _pkg = "✅" if st_info.get("has_groq_pkg") else "❌ not installed"
                _key = "✅ set" if st_info.get("groq_key_set") else "❌ not set"
                _seen = ", ".join(sorted(_secret_keys_seen.keys())) or "(none)"
                st.warning(
                    "🤖 **The advisor is running in rule-based mode (no live LLM).** "
                    "To turn on the free Groq LLM on this deployed site: open "
                    "**⋮ → Settings → Secrets** and add `GROQ_API_KEY = \"gsk_…\"` "
                    "(get a free key at console.groq.com), then the app reboots automatically.\n\n"
                    f"Diagnostics — groq package: {_pkg} · GROQ_API_KEY: {_key} · "
                    f"secret keys Streamlit sees: `{_seen}`")

            # ── Model health banner — developers only (Ollama-specific) ────
            if _admin and st_info.get("model_healthy") is False:
                st.markdown(
                    '<div style="background:rgba(239,68,68,.08);border:2px solid rgba(239,68,68,.35);'
                    'border-radius:10px;padding:14px 18px;margin-bottom:12px">'
                    '<div style="display:flex;align-items:flex-start;gap:12px">'
                    '<span style="font-size:1.4rem">🚨</span>'
                    '<div style="flex:1">'
                    '<div style="font-size:.88rem;font-weight:700;color:#f87171">'
                    'Model needs retraining — this is why queries keep timing out</div>'
                    '<div style="font-size:.78rem;color:#c8a0a0;margin-top:4px;line-height:1.6">'
                    'The registered <b>pidrl-advisor</b> model still has an outdated chat '
                    'template from before the fix. Updating the Python files alone does '
                    '<b>not</b> retroactively fix an already-created Ollama model — it must '
                    'be recreated. Click below to fix it automatically (uses your existing '
                    'knowledge base, no re-upload needed).'
                    '</div></div></div></div>', unsafe_allow_html=True)

                fix_col1, fix_col2 = st.columns([1, 3])
                if fix_col1.button("🔧 Fix Now (Retrain)", type="primary", key="auto_fix_btn", width="stretch"):
                    with st.spinner("Recreating model with corrected template..."):
                        try:
                            from prepare_llm_v4 import (build_system_prompt, create_modelfile,
                                                        _register_via_api, MODELFILE as MF_NAME)
                            # Reuse existing knowledge base — no re-upload needed
                            sp = build_system_prompt(advisor.kb) if advisor.kb else advisor.lean_system_prompt
                            base_model_guess = st_info["mode"].replace("ollama:", "").split("+")[0]
                            create_modelfile(sp, base_model_guess, MF_NAME)
                            with open(MF_NAME, "r", encoding="utf-8") as f:
                                mf_txt = f.read()
                            # Always register under the configured advisor
                            # model name (not the base model name)
                            model_name_to_fix = "pidrl-advisor"
                            reg_ok, reg_msg = _register_via_api(mf_txt, model_name_to_fix)
                            advisor.reload()
                            advisor.recheck_model_health()
                            _load_advisor.clear()
                            st.session_state.advisor_version += 1
                            if reg_ok:
                                st.success("✅ Model recreated with corrected template! Responses should work now.")
                            else:
                                st.warning(f"Recreation attempted but: {reg_msg}")
                        except Exception as e:
                            st.error(f"Auto-fix failed: {e}. Try running `python prepare_llm_v4.py` from the terminal instead.")
                    time.sleep(0.8)
                    st.rerun()
                fix_col2.caption(f"Detected issue: {st_info.get('health_reason','')}")

            # ── Auto-analysis when simulation is ready ─────────────────────
            if ats:
                # Build sim_results dict for advisor
                ta_  = np.array(ats["times"]); DI_  = np.array(ats["DI"])
                reh_s= ats["reheat_start"];    reh_e= ats["reheat_end"]
                is_  = int(np.argmin(np.abs(ta_-reh_s)))
                ie_  = int(np.argmin(np.abs(ta_-reh_e)))
                heal = float(DI_[is_])-float(DI_[min(ie_,len(DI_)-1)])
                DI_pk= max(ats["DI"]); DI_fn=float(ats["DI"][-1])
                Bi_v = params["h_cool"]*R_M/K_TH

                sim_ctx = {
                    "DI_peak":   DI_pk,
                    "DI_final":  DI_fn,
                    "DI_healed": heal,
                    "t_total":   ats["t_total"],
                    "Bi":        Bi_v,
                    "T_reheat":  params["reheat"]["T"],
                    "t_reheat":  ats["t_reheat"],
                    "h_cool":    params["h_cool"],
                    "n_zones":   len(params["zones"]),
                    "T_fill":    params["T_fill"],
                    "risk":      _rlbl(DI_pk),
                    "scenario":  params.get("_scenario","Custom"),
                    "zones":     params["zones"],
                }

                # Quick one-line summary in header area
                quick = advisor.quick_summary(DI_pk, ats["t_total"], Bi_v)
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,rgba(91,140,255,.12),'
                    f'rgba(9,18,32,.5));border:1px solid rgba(91,140,255,.28);'
                    f'border-radius:12px;padding:11px 16px;'
                    f'font-size:.84rem;color:#bcd0ff;margin-bottom:12px">'
                    f'💡 <b>{t("quick_verdict")}:</b> {quick}</div>',
                    unsafe_allow_html=True)

                # Full analysis
                col_h1, col_h2 = st.columns([4, 1])
                col_h1.markdown(f"#### {t('detailed_analysis')}")
                analysis_key = f"llm_analysis_{id(ats)}_{DI_pk:.4f}"

                force_retry = col_h2.button("🔄 Retry", key="llm_retry_analysis",
                                            width='stretch',
                                            help="Re-run analysis (e.g. if Ollama just started)")
                if force_retry and analysis_key in st.session_state:
                    del st.session_state[analysis_key]

                if analysis_key not in st.session_state:
                    if _groq:
                        spinner_msg = "⚡ Querying Groq LLM (RAG-accelerated)..."
                    elif st_info["ollama_up"] and not st_info["model_warm"]:
                        spinner_msg = "🥶 Loading model (~30-90s, first time only)..."
                    elif st_info["ollama_up"]:
                        spinner_msg = "🤖 Querying LLM (RAG-accelerated, ~5-15s)..."
                    else:
                        spinner_msg = "📊 Computing rule-based analysis..."
                    with st.spinner(spinner_msg):
                        st.session_state[analysis_key] = advisor.analyze(sim_ctx)
                        st.session_state[f"{analysis_key}_source"] = advisor.status()["last_source"]

                analysis_text = st.session_state[analysis_key]
                analysis_source = st.session_state.get(f"{analysis_key}_source", "unknown")

                # Source badge
                _is_groq_src  = analysis_source.startswith("groq")
                is_llm_source = analysis_source.startswith("ollama") or _is_groq_src
                src_col  = "#34d399" if is_llm_source else "#f59e0b"
                src_icon = ("⚡" if _is_groq_src else "🤖") if is_llm_source else "📊"
                src_text = (analysis_source.replace("groq:", "Groq: ").replace("ollama:", "LLM: ")
                            if is_llm_source else
                            "Rule-based" + (" (timed out)" if "timeout" in analysis_source else ""))
                st.markdown(
                    f'<div style="font-size:.7rem;color:{src_col};margin-bottom:6px">'
                    f'{src_icon} Source: {src_text}</div>', unsafe_allow_html=True)

                # Render analysis in styled card
                sections = {"ASSESSMENT": "#3b82f6",
                            "PHYSICAL MEANING": "#10b981",
                            "ROOT CAUSE": "#f59e0b",
                            "RECOMMENDATION": "#8b5cf6",
                            "COMPARISON": "#06b6d4"}
                html_parts = ['<div style="background:linear-gradient(135deg,'
                              'rgba(11,32,64,.5),rgba(7,17,30,.78));'
                              'border:1px solid #13294a;'
                              'border-radius:14px;padding:18px 22px;font-size:.83rem;'
                              'line-height:1.75;font-family:Inter,sans-serif;'
                              'box-shadow:0 8px 28px rgba(0,0,0,.3)">']
                for raw in analysis_text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    # Identify a section label regardless of how the LLM marked
                    # it: "## ASSESSMENT", "**ASSESSMENT:**", "ASSESSMENT:" …
                    # Strip leading #/* and trailing :/*/# before matching.
                    clean   = re.sub(r"^[#*\s]+", "", line).rstrip(":*# ").strip()
                    clean_u = clean.upper()
                    sec, body = None, ""
                    for _s in sections:
                        if clean_u == _s or clean_u.startswith(_s + ":") or clean_u.startswith(_s + " "):
                            sec  = _s
                            body = clean[len(_s):].lstrip(":  ").strip()
                            break
                    if sec is not None:
                        col = sections[sec]
                        html_parts.append(
                            f'<div style="margin:12px 0 4px;font-size:.7rem;font-weight:700;'
                            f'color:{col};letter-spacing:.1em;text-transform:uppercase">'
                            f'▸ {sec}</div>')
                        if body:
                            html_parts.append(
                                f'<div style="color:#c8d6ea">{_md_to_html(body, col)}</div>')
                    else:
                        html_parts.append(
                            f'<div style="color:#a8bcd6;padding-left:10px">'
                            f'{_md_to_html(line)}</div>')
                html_parts.append("</div>")
                st.markdown("".join(html_parts), unsafe_allow_html=True)

            else:
                st.info("▶ Run a simulation first — the AI Advisor will automatically "
                        "analyse the results.")

            # ── Chat interface — proper chat-bubble UI ──────────────────────
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin:2px 0 6px">'
                f'<div style="font-size:1.0rem;font-weight:800;color:#eaf1ff">{t("ask_advisor")}</div>'
                f'<div style="flex:1;height:1px;background:linear-gradient(90deg,#244372,transparent)"></div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:.74rem;color:#6a82ad;margin-bottom:10px">'
                f'{t("ask_hint")}'
                '</div>', unsafe_allow_html=True)

            def _format_msg(content):
                return _md_to_html(content)

            def _bubble_html(role, content, source_label=None, cursor=False):
                """Render one chat bubble \u2014 modern AI-assistant aesthetic.
                User = right / blue gradient; assistant = left / glassy dark
                card with avatar, name tag, markdown body and source pill."""
                cur = ('<span style="display:inline-block;width:7px;height:15px;'
                       'background:#60a5fa;border-radius:2px;margin-left:2px;'
                       'animation:blink 1s steps(2) infinite;vertical-align:middle">'
                       '</span>') if cursor else ""
                if role == "user":
                    return (
                        '<div style="display:flex;justify-content:flex-end;gap:10px;'
                        'margin:14px 0;animation:fadeIn .25s ease">'
                        '<div style="max-width:76%;background:linear-gradient(135deg,#3b82f6,#1d4ed8);'
                        'color:#fff;padding:11px 16px;'
                        'border-radius:18px 18px 5px 18px;font-size:.85rem;line-height:1.6;'
                        'box-shadow:0 4px 18px rgba(37,99,235,.32);'
                        'border:1px solid rgba(147,197,253,.25)">'
                        f'{_format_msg(content)}</div>'
                        '<div style="width:34px;height:34px;border-radius:50%;flex-shrink:0;'
                        'background:linear-gradient(135deg,#1e293b,#0f1d38);'
                        'border:1px solid #2a3f63;display:flex;align-items:center;'
                        'justify-content:center;font-size:15px">\U0001F464</div>'
                        '</div>'
                    )
                # assistant
                src_html = ""
                if source_label:
                    _is_groq = source_label.startswith("groq")
                    is_llm = source_label.startswith("ollama") or _is_groq
                    sc  = "#34d399" if is_llm else "#f59e0b"
                    bg  = "rgba(52,211,153,.1)" if is_llm else "rgba(245,158,11,.1)"
                    txt = ((source_label.replace("groq:", "⚡ ") if _is_groq
                            else source_label.replace("ollama:", "\U0001F916 ")) if is_llm else (
                        "\U0001F4CA Rule-based" +
                        (" (timed out)" if "timeout" in source_label else
                         " (retrain needed)" if "retraining" in source_label else "")))
                    src_html = (
                        f'<div style="display:inline-flex;align-items:center;gap:4px;'
                        f'margin-top:7px;padding:2px 9px;border-radius:20px;background:{bg};'
                        f'border:1px solid {sc}55;font-size:.62rem;color:{sc};'
                        f'font-weight:600;letter-spacing:.02em">{txt}</div>')
                return (
                    '<div style="display:flex;justify-content:flex-start;gap:10px;'
                    'margin:14px 0;animation:fadeIn .25s ease">'
                    '<div style="width:34px;height:34px;border-radius:50%;flex-shrink:0;'
                    'background:linear-gradient(135deg,#f97316,#dc2626);'
                    'display:flex;align-items:center;justify-content:center;font-size:16px;'
                    'box-shadow:0 3px 12px rgba(234,88,12,.4)">\U0001F916</div>'
                    '<div style="max-width:80%">'
                    '<div style="font-size:.64rem;color:#5a7199;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin:0 0 3px 3px">'
                    'PI-DRL Advisor</div>'
                    '<div style="background:linear-gradient(135deg,#0f1d38,#0a1424);'
                    'border:1px solid #1e3358;color:#e6edfb;padding:12px 16px;'
                    'border-radius:5px 18px 18px 18px;font-size:.85rem;line-height:1.7;'
                    'box-shadow:0 4px 20px rgba(0,0,0,.35)">'
                    f'{_format_msg(content)}{cur}</div>'
                    f'{src_html}</div></div>'
                )

            def _thinking_bubble(label):
                """Animated 'advisor is thinking' bubble (bouncing dots)."""
                dots = "".join(
                    f'<span style="width:7px;height:7px;border-radius:50%;'
                    f'background:#84a9ff;display:inline-block;'
                    f'animation:typingBounce 1.2s infinite {d}s"></span>'
                    for d in (0.0, 0.15, 0.30))
                return (
                    '<div style="display:flex;gap:10px;margin:14px 0;animation:fadeIn .2s ease">'
                    '<div style="width:34px;height:34px;border-radius:50%;flex-shrink:0;'
                    'background:linear-gradient(135deg,#f97316,#dc2626);display:flex;'
                    'align-items:center;justify-content:center;font-size:16px;'
                    'box-shadow:0 3px 12px rgba(234,88,12,.4)">\U0001F916</div>'
                    '<div><div style="font-size:.64rem;color:#5a7199;font-weight:700;'
                    'letter-spacing:.08em;text-transform:uppercase;margin:0 0 3px 3px">'
                    'PI-DRL Advisor</div>'
                    '<div style="background:linear-gradient(135deg,#0f1d38,#0a1424);'
                    'border:1px solid #1e3358;border-radius:5px 18px 18px 18px;'
                    'padding:14px 18px;display:inline-flex;align-items:center;gap:7px">'
                    f'{dots}</div>'
                    f'<div style="font-size:.62rem;color:#5a7199;margin:5px 0 0 4px">{label}</div>'
                    '</div></div>')

            # ── Render existing chat thread (scrollable container) ──────────
            #    Source of truth is the PER-SESSION list (not the shared
            #    advisor), so each visitor only ever sees their own thread.
            history_pairs = []
            hist = st.session_state.chat_hist
            for i in range(0, len(hist) - (len(hist) % 2), 2):
                history_pairs.append((hist[i], hist[i+1] if i+1 < len(hist) else None))

            if history_pairs:
                thread_html = ['<div style="max-height:460px;overflow-y:auto;padding:10px 14px;'
                               'background:radial-gradient(900px 300px at 80% -10%,'
                               'rgba(37,99,235,.06),transparent),rgba(3,9,26,.6);'
                               'border:1px solid #13294a;border-radius:14px;'
                               'margin-bottom:10px;box-shadow:inset 0 2px 20px rgba(0,0,0,.35)">']
                for user_msg, asst_msg in history_pairs:
                    thread_html.append(_bubble_html("user", user_msg["content"]))
                    if asst_msg:
                        thread_html.append(_bubble_html("assistant", asst_msg["content"]))
                thread_html.append('</div>')
                st.markdown("".join(thread_html), unsafe_allow_html=True)
            else:
                _empty_txt = (
                    "シミュレーションについて何でも質問してください — ヒケの原因、再加熱の戦略、"
                    "ビオ数、DIのしきい値など。<br>アドバイザーは会話を記憶します。"
                    if st.session_state.get("lang") == "ja" else
                    "Ask anything about your simulation — ヒケ causes, reheat strategy, "
                    "Biot number, DI thresholds.<br>The advisor remembers the conversation.")
                st.markdown(
                    '<div style="text-align:center;padding:30px 24px;color:#46618c;'
                    'font-size:.8rem;background:radial-gradient(600px 200px at 50% 0%,'
                    'rgba(37,99,235,.06),transparent),rgba(3,9,26,.6);'
                    'border:1px solid #13294a;border-radius:14px;margin-bottom:10px">'
                    '<div style="font-size:1.6rem;margin-bottom:6px">\U0001F4AC</div>'
                    f'{_empty_txt}'
                    '</div>', unsafe_allow_html=True)

            # \u2500\u2500 Suggested questions \u2014 clicking SENDS immediately (sets
            #    pending_q). Before any chat: starter presets. After a chat:
            #    contextual follow-ups generated from the last answer. \u2500\u2500
            if history_pairs:
                _lu = history_pairs[-1][0]["content"]
                _la = history_pairs[-1][1]["content"] if history_pairs[-1][1] else ""
                _fk = f"fups_{len(hist)}_{st.session_state.get('lang','en')}"
                if _fk not in st.session_state:
                    with st.spinner(""):
                        st.session_state[_fk] = advisor.suggest_followups(
                            _lu, _la, lang=st.session_state.get("lang", "en"))
                suggestions = st.session_state[_fk]
                _fl = ("\u7d9a\u3051\u3066\u8cea\u554f" if st.session_state.get("lang")=="ja"
                       else "Follow-up questions")
                st.markdown(f"<div style='font-size:.66rem;color:#5f7aa3;font-weight:700;"
                            f"letter-spacing:.06em;margin:2px 0 5px'>\U0001F4A1 {_fl}</div>",
                            unsafe_allow_html=True)
            elif st.session_state.get("lang") == "ja":
                suggestions = [
                    "\u306a\u305c\u4e0a\u9762\u3067\u30d2\u30b1\u304c\u767a\u751f\u3059\u308b\u306e\u3067\u3059\u304b\uff1f",
                    "DI<0.10\u306b\u3059\u308b\u305f\u3081\u306e\u518d\u52a0\u71b1\u6e29\u5ea6\u306f\uff1f",
                    "\u30d3\u30aa\u6570\u306f\u30d2\u30b1\u306b\u3069\u3046\u5f71\u97ff\u3057\u307e\u3059\u304b\uff1f",
                    "\u751f\u7523\u30e9\u30a4\u30f3\u306e\u30c7\u30fc\u30bf\u3068\u6bd4\u8f03\u3057\u3066\u304f\u3060\u3055\u3044",
                ]
            else:
                suggestions = [
                    "Why is \u30d2\u30b1 forming at the top surface?",
                    "What T_reheat do I need for DI < 0.10?",
                    "How does Biot number affect \u30d2\u30b1?",
                    "Compare my result to production line data",
                ]
            s_cols = st.columns(len(suggestions) if suggestions else 1)
            for i, (col, sq) in enumerate(zip(s_cols, suggestions)):
                if col.button(sq[:34]+"\u2026" if len(sq)>34 else sq,
                              key=f"suggest_{len(hist)}_{i}", width='stretch'):
                    st.session_state["pending_q"] = sq
                    st.rerun()

            # Bind the box to its OWN session-state key (not a transient
            # value=) so the text survives the form-submit rerun — fixes
            # "preset → Send did nothing / chat cleared".
            with st.form("chat_composer", clear_on_submit=True, border=False):
                cc1, cc2 = st.columns([9, 1])
                user_q = cc1.text_input(
                    "Your question:", key="llm_chatbox",
                    placeholder=f"{t('send')} →  " + (
                        "質問を入力… (Enterで送信)" if st.session_state.get("lang")=="ja"
                        else "Type your question…  (press Enter to send)"),
                    label_visibility="collapsed")
                sent = cc2.form_submit_button("➤", type="primary",
                                              width='stretch',
                                              help=t("send"))

            lc_on   = st_info.get("langchain") or _groq
            mem_txt = t("mem_on") if lc_on else t("mem_basic")
            mc1, mc2 = st.columns([3, 1])
            mc1.markdown(
                f'<div style="font-size:.66rem;color:#6a82ad;padding-top:6px">{mem_txt}</div>',
                unsafe_allow_html=True)
            if mc2.button(t("clear_chat"), key="llm_clear", width='stretch'):
                # Clear this session's OWN history only (not the shared cache
                # object) so we never touch another visitor's conversation.
                st.session_state.chat_hist = []
                advisor.clear_history(st.session_state.chat_sid)
                # Only drop cached analyses/follow-ups — never delete the
                # composer's own widget key (llm_chatbox), which is already
                # instantiated this run and would raise if modified/deleted.
                for k in list(st.session_state.keys()):
                    if k.startswith(("llm_analysis_", "fups_")): del st.session_state[k]
                st.session_state.pop("pending_q", None)
                st.rerun()

            ask_q = ""
            if st.session_state.get("pending_q"):
                ask_q = st.session_state.pop("pending_q")
            elif sent and user_q.strip():
                ask_q = user_q.strip()

            if ask_q:
                ctx_str = st.session_state.get(
                    f"llm_analysis_{id(ats) if ats else 0}_{DI_live:.4f}", "")
                sim_c = sim_ctx if ats else None

                # ── Live exchange — shown as bubbles directly below the
                #    composer, identical styling to the thread above, so it
                #    reads as a natural continuation once the page reruns.
                st.markdown(_bubble_html("user", ask_q), unsafe_allow_html=True)

                _ja = st.session_state.get("lang") == "ja"
                if _groq:
                    think_lbl = "Groq\u3067\u8003\u3048\u3066\u3044\u307e\u3059\u2026" if _ja else "Thinking with Groq\u2026"
                elif st_info["ollama_up"] and not st_info["model_warm"]:
                    think_lbl = ("\u30e2\u30c7\u30eb\u3092\u8aad\u307f\u8fbc\u307f\u4e2d\uff08\u521d\u56de\u300130\u301c90\u79d2\uff09\u2026" if _ja
                                 else "Loading model (first query, 30-90s)\u2026")
                elif st_info["ollama_up"]:
                    think_lbl = "\u8003\u3048\u3066\u3044\u307e\u3059\u2026" if _ja else "Thinking\u2026"
                else:
                    think_lbl = "\u8a08\u7b97\u4e2d\u2026" if _ja else "Computing\u2026"

                typing_ph = st.empty()
                typing_ph.markdown(_thinking_bubble(think_lbl), unsafe_allow_html=True)

                bubble_ph = st.empty()
                accumulated = ""
                last_render = 0.0
                chunk_count = 0

                for chunk in advisor.chat_stream(ask_q,
                                                 context=ctx_str[:600],
                                                 sim_results=sim_c,
                                                 history=st.session_state.chat_hist,
                                                 session_id=st.session_state.chat_sid,
                                                 lang=st.session_state.get("lang", "en")):
                    accumulated += chunk
                    chunk_count += 1
                    if chunk_count == 1:
                        typing_ph.empty()      # first token → drop the thinking dots
                    now = time.time()
                    if now - last_render > 0.10 or chunk_count <= 3:
                        bubble_ph.markdown(_bubble_html("assistant", accumulated, cursor=True),
                                           unsafe_allow_html=True)
                        last_render = now

                chat_source = advisor.status()["last_source"]
                bubble_ph.markdown(_bubble_html("assistant", accumulated, source_label=chat_source),
                                   unsafe_allow_html=True)
                typing_ph.empty()
                st.rerun()   # re-render so this exchange joins the persistent thread above

            # ── Developer-only panels below (data upload / Ollama training /
            #    setup help). Hidden from clients; visible with ?debug=1.
            if not _admin:
                return

            # ── Data Upload & Training Panel ──────────────────────────────
            st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
            st.markdown("#### 📂 Upload & Train from Excel Data")
            st.markdown(
                '<div style="font-size:.75rem;color:#2d3f5a;margin-bottom:10px">'
                'Upload any process Excel file (.xlsx) to instantly update the AI knowledge base '
                'and retrain the Ollama model. The system auto-detects temperature columns, '
                'time series, and text observations.'
                '</div>', unsafe_allow_html=True)

            with st.expander("📤 Upload Excel → Retrain Model", expanded=not st_info["kb_loaded"]):
                uploaded_files = st.file_uploader(
                    "Choose Excel file(s) — drag & drop multiple",
                    type=["xlsx","xls"],
                    accept_multiple_files=True,
                    key="llm_xlsx_upload",
                    help="Upload CLIENT_Data_0427.xlsx or any process data. "
                         "Multiple files merged. Supports row-label & column-header layouts.",
                )

                col_u1, col_u2 = st.columns([2, 2])
                base_model_choice = col_u1.selectbox(
                    "Base LLM model",
                    ["mistral", "phi3:mini", "llama3.1:8b", "llama3.2:3b", "gemma2:2b"],
                    index=0, key="llm_base_model",
                    help="mistral: best quality (4GB) | phi3:mini: fastest (2.3GB)",
                )
                model_name_choice = col_u2.text_input(
                    "Custom model name", value="pidrl-advisor",
                    key="llm_model_name",
                    help="Name for the trained Ollama model",
                )
                auto_register = st.checkbox(
                    "Auto-register with Ollama after training",
                    value=True, key="llm_auto_register",
                    help="Uses REST API at localhost:11434 — no PATH config needed on Windows.",
                )

                if uploaded_files:
                    pills = " &nbsp; ".join(
                        f'<span style="background:#07111e;border:1px solid #1c3566;'
                        f'border-radius:5px;padding:2px 8px;font-size:.72rem;color:#60a5fa">'
                        f'📄 {f.name} ({f.size//1024}KB)</span>'
                        for f in uploaded_files)
                    st.markdown(pills, unsafe_allow_html=True)

                train_btn = st.button(
                    f"⚡ Train / Update Model from {len(uploaded_files) if uploaded_files else 0} File(s)",
                    type="primary", width='stretch', key="llm_train_btn",
                    disabled=not bool(uploaded_files),
                )
                if not uploaded_files:
                    st.info("👆 Upload one or more Excel files above to enable training.")

                if train_btn and uploaded_files:
                    progress_bar = st.progress(0, text="Starting…")
                    try:
                        from prepare_llm_v4 import (parse_excel, scan_excel_for_knowledge,
                            extract_knowledge, build_system_prompt, create_modelfile,
                            _register_via_api, _find_ollama_cli, MODELFILE as MF_NAME)
                    except ImportError:
                        st.error("⚠️ prepare_llm_v4.py not found in version4/ folder."); st.stop()

                    import time as _t, json as _j, subprocess as _sp

                    # 1. Parse all files
                    combined_raw = {}
                    for fi, uf in enumerate(uploaded_files):
                        progress_bar.progress(5+int(30*(fi+1)/len(uploaded_files)),
                                             text=f"📊 Parsing {uf.name}…")
                        for sh, rows in parse_excel(uf).items():
                            combined_raw[f"{uf.name[:20]}::{sh}"] = rows

                    # 2. Extract knowledge
                    progress_bar.progress(40, text="🧠 Extracting knowledge…")
                    kb    = extract_knowledge(combined_raw)
                    extra = scan_excel_for_knowledge(combined_raw)
                    if extra.get("temperature_range"):
                        kb["uploaded_temperature_range"] = extra["temperature_range"]
                    if extra.get("text_observations"):
                        kb["uploaded_observations"] = extra["text_observations"][:40]
                    if extra.get("temperature_series"):
                        kb["uploaded_series_count"] = len(extra["temperature_series"])
                    with open("knowledge_base_v4.json","w",encoding="utf-8") as f:
                        _j.dump(kb, f, indent=2, ensure_ascii=False)

                    # 2b. Build RAG vector store (TF-IDF index of chunked knowledge)
                    progress_bar.progress(48, text="🔍 Building RAG index…")
                    rag_n_chunks = 0
                    try:
                        from vector_store_v4 import build_and_save_vector_store
                        vs_stats = build_and_save_vector_store(kb, extra, output_path="vector_store_v4.json")
                        rag_n_chunks = vs_stats.get("n_documents", 0)
                    except Exception as e:
                        print(f"[!] RAG build failed: {e}")

                    # 3. Build prompt (full version baked into Modelfile;
                    #    a separate LEAN prompt is used at runtime via RAG)
                    progress_bar.progress(60, text="📝 Building system prompt…")
                    sp = build_system_prompt(kb)
                    if extra.get("temperature_series"):
                        sp += "\n\n═══ UPLOADED TEMPERATURE SERIES ═══\n"
                        for s in extra["temperature_series"][:8]:
                            sp += f"  {s['label'][:50]}: {s['min']}→{s['max']}°C ({len(s['values'])} pts)\n"
                    if extra.get("text_observations"):
                        sp += "\n\nFURTHER OBSERVATIONS:\n" + "\n".join(f"  - {o}" for o in extra["text_observations"][:15])
                    with open("system_prompt_v4.txt","w",encoding="utf-8") as f:
                        f.write(sp)

                    # 4. Create Modelfile
                    progress_bar.progress(75, text="🔧 Creating Modelfile…")
                    create_modelfile(sp, base_model_choice, MF_NAME)

                    # 5. Register (REST API first, then CLI fallback)
                    progress_bar.progress(88, text="🤖 Registering with Ollama…")
                    with open(MF_NAME,"r",encoding="utf-8") as f:
                        mf_txt = f.read()
                    reg_ok, reg_msg = _register_via_api(mf_txt, model_name_choice)
                    if not reg_ok:
                        cli = _find_ollama_cli()
                        if cli:
                            proc = _sp.run([cli,"create",model_name_choice,"-f",MF_NAME],
                                          capture_output=True, text=True, timeout=300)
                            reg_ok  = proc.returncode == 0
                            reg_msg = "CLI ✅" if reg_ok else proc.stderr[:150]

                    progress_bar.progress(100, text="✅ Done!")
                    _t.sleep(0.3); progress_bar.empty()

                    # 6. Show results
                    n_s = len(extra.get("temperature_series",[])); n_o = len(extra.get("text_observations",[]))
                    tr  = extra.get("temperature_range",{})
                    st.markdown(
                        f'<div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);'
                        f'border-radius:8px;padding:14px 18px">' 
                        f'<div style="font-size:.88rem;font-weight:700;color:#34d399;margin-bottom:8px">'
                        f'✅ {len(uploaded_files)} file(s) processed</div>'
                        f'<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:.76rem">'
                        f'<div>Temp series: <b style="color:#60a5fa">{n_s}</b></div>'
                        f'<div>RAG chunks: <b style="color:#a78bfa">{rag_n_chunks}</b></div>'
                        f'<div>Observations: <b style="color:#d8e2f0">{n_o}</b></div>'
                        f'<div>Range: <b style="color:#d8e2f0">{tr.get("min_C","—")}–{tr.get("max_C","—")}°C</b></div>'
                        f'<div>Model: <b style="color:#a78bfa">{model_name_choice}</b></div>'
                        f'<div>Ollama: <b style="color:{"#34d399" if reg_ok else "#f59e0b"}">'
                        f'{"✅ Registered" if reg_ok else "⚠️ "+reg_msg[:50]}</b></div>'
                        f'</div></div>', unsafe_allow_html=True)

                    if extra.get("temperature_series"):
                        with st.expander(f"🌡️ {n_s} temperature series extracted"):
                            for s in extra["temperature_series"][:10]:
                                st.markdown(f'<div style="font-size:.73rem;color:#7a9cc0;padding:3px 0;'
                                           f'border-bottom:1px solid #0d1e38">'
                                           f'<b style="color:#60a5fa">{s["label"][:55]}</b> — '
                                           f'{s["min"]}→{s["max"]}°C  ({len(s["values"])} pts)</div>',
                                           unsafe_allow_html=True)
                    if extra.get("text_observations"):
                        with st.expander(f"📖 {n_o} observations learned"):
                            for obs in extra["text_observations"][:20]:
                                st.markdown(f'<div style="font-size:.72rem;color:#7a9cc0;padding:2px 0">'
                                           f'{obs}</div>', unsafe_allow_html=True)
                    if not reg_ok:
                        st.warning(f"Ollama not reachable. Ensure it is running, then:\n"
                                  f"```\nollama create {model_name_choice} -f Modelfile_v4\n```")

                    # ── Hot-reload: NO page reload (F5) needed ────────────
                    # Knowledge base, vector store, and system prompt were
                    # just rewritten to disk. Reload them into the LIVE
                    # advisor instance right now, then bump the cache
                    # version so the next rerun picks up a fully fresh
                    # PILLMAdvisor (re-detects Ollama models too).
                    try:
                        advisor.reload()
                    except Exception as e:
                        print(f"[!] advisor.reload() failed: {e}")

                    _load_advisor.clear()  # drop old cached instance
                    st.session_state.advisor_version += 1

                    # Invalidate any previously cached analysis so the
                    # next view re-runs with the NEW knowledge.
                    for k in list(st.session_state.keys()):
                        if k.startswith("llm_analysis_"):
                            del st.session_state[k]

                    st.success(
                        f"🤖 **{model_name_choice}** is now active — "
                        f"knowledge base, RAG index, and Ollama model all "
                        f"updated **live**. No page reload needed!"
                    )
                    time_module = __import__("time")
                    time_module.sleep(0.6)
                    st.rerun()
            # ── Setup instructions ─────────────────────────────────────────
            if not st_info["ollama_up"]:
                with st.expander("⚙️ How to enable full LLM mode", expanded=True):
                    st.markdown("""
                    **Step 1 — Install Ollama**
                    ```bash
                    # macOS / Linux
                    curl -fsSL https://ollama.com/install.sh | sh
                    # Windows: download from https://ollama.com/download
                    ```

                    **Step 2 — Pull base model**
                    ```bash
                    ollama pull mistral          # recommended (4 GB)
                    # or: ollama pull phi3:mini  # lighter (2.3 GB)
                    ```

                    **Step 3 — Upload Excel above** (or run from terminal)
                    ```bash
                    cd version4/
                    python prepare_llm_v4.py    # uses CLIENT_Data_0427.xlsx
                    ```

                    **Step 4 — Restart the Streamlit app**
                    ```bash
                    streamlit run main_3d_v4.py
                    ```
                    The advisor switches to **LLM mode** automatically.
                    """)
            # else:
            #     with st.expander("🛠️ Getting garbled output, repeated words, or constant timeouts?",
            #                      expanded=False):
            #         st.markdown("""
            #         **Root cause:** an earlier version of this app registered the
            #         `pidrl-advisor` model with an incorrect chat template
            #         (`<|im_start|>` ChatML tokens), which Mistral/Llama3/Phi-3
            #         were never trained to recognise as a stop signal. The model
            #         then rambles indefinitely — producing duplicated words and
            #         what looks like a timeout, because generation never
            #         naturally terminates.

            #         **Fix — retrain once** (current version no longer overrides
            #         the template; it lets Ollama use each base model's own
            #         correct format):
            #         ```bash
            #         cd version4/
            #         python prepare_llm_v4.py
            #         ```
            #         Or use **Upload & Train from Excel Data** below — either
            #         path automatically deletes the old model definition and
            #         recreates it cleanly. No manual `ollama rm` needed.

            #         **After retraining**, responses should be coherent and the
            #         🔥 "Model warm" badge above should turn green within
            #         30–90 seconds (first query only — subsequent queries are
            #         fast).
            #         """)


if __name__=="__main__":
    main()
