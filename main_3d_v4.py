"""
main_3d_v4.py  —  PI-DRL Solidification Control  v4.1
CBIC × TUAT  |  Julian Evan Chrisnanto  |  2026
"""
import json, warnings, time, re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PI-DRL Solidification Control",
                   page_icon=":material/thermostat:",
                   layout="wide",initial_sidebar_state="auto")

# ── Constants ──────────────────────────────────────────────────────────────
T_SOL_C=62.0; T_LIQ_C=72.0; T_MID_C=67.0
K_TH=0.25; R_M=0.0125; H_M=0.04; RHO=990.0; CP=2000.0
T_FILL_DEF=80.0; T_TARGET=37.0; T_ROOM=23.0
H_S=H_M*100; R_S=R_M*100
BELT_W=3.0; LANE_TOP=4.0; LANE_BOT=-4.0
L_COOL=30.0; L_RH=20.0

# ── Design system: "cold room" ─────────────────────────────────────────────
# The interface is cool and monochrome; every warm hue on screen is a real
# temperature. One thermal ramp (fixed °C domain) colours the stick, the belt
# plates, the hero ribbon and the charts, so the same °C always looks the same.
INK = dict(night="#111926", deck="#172131", deck2="#1F2A3C",
           frost="#E8EDF4", frost2="#B0BACB", frost3="#7E8AA0",
           rule="rgba(205,219,240,0.10)", rule2="rgba(205,219,240,0.18)")
FONT = "Archivo, 'BIZ UDPGothic', 'Helvetica Neue', Arial, sans-serif"
RISK_COL = {"SAFE":"#7CCBA6", "CAUTION":"#E9C46A",
            "WARNING":"#EF9E59", "CRITICAL":"#E2573C"}

HEAT_STOPS = [(15,"#2E5C9E"),(30,"#5F9BD3"),(45,"#BBD7E6"),(60,"#F0DDAE"),
              (72,"#EEA35C"),(85,"#DD6A3A"),(100,"#B8382A")]
HEAT_MIN, HEAT_MAX = HEAT_STOPS[0][0], HEAT_STOPS[-1][0]
HEAT_SCALE = [[(T-HEAT_MIN)/(HEAT_MAX-HEAT_MIN), c] for T,c in HEAT_STOPS]
DMG_SCALE  = [[0,"#7CCBA6"],[0.30,"#BBD7E6"],[0.55,"#E9C46A"],
              [0.78,"#EF9E59"],[1,"#E2573C"]]

def _hex2rgb(h):
    h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))

def _rgba(h, a):
    r,g,b=_hex2rgb(h); return f"rgba({r},{g},{b},{a})"

def _heat(T):
    """Temperature (°C) -> hex colour on the shared thermal ramp."""
    T=float(np.clip(T,HEAT_MIN,HEAT_MAX))
    for (t0,c0),(t1,c1) in zip(HEAT_STOPS,HEAT_STOPS[1:]):
        if T<=t1:
            f=(T-t0)/(t1-t0); a,b=_hex2rgb(c0),_hex2rgb(c1)
            return "#%02x%02x%02x" % tuple(round(x+(y-x)*f) for x,y in zip(a,b))
    return HEAT_STOPS[-1][1]

_AX = dict(gridcolor=INK["rule"], linecolor=INK["rule2"], zeroline=False,
           tickfont=dict(color=INK["frost3"], size=11),
           title=dict(font=dict(color=INK["frost2"], size=12)))
_AX3 = dict(gridcolor="rgba(205,219,240,0.06)", showbackground=False,
            zeroline=False, showspikes=False,
            tickfont=dict(color=INK["frost3"], size=10),
            title=dict(font=dict(color=INK["frost3"], size=11)))
pio.templates["pidrl"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK["frost2"], size=12),
    title=dict(font=dict(family=FONT, color=INK["frost"], size=15)),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[INK["frost"],"#5F9BD3","#EEA35C","#7CCBA6","#F0DDAE","#DD6A3A"],
    xaxis=_AX, yaxis=_AX,
    scene=dict(xaxis=_AX3, yaxis=_AX3, zaxis=_AX3, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=INK["deck"], bordercolor=INK["rule2"],
                    font=dict(family=FONT, color=INK["frost"], size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK["frost2"], size=11)),
))
pio.templates.default = "plotly_dark+pidrl"

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
    "🏭 CBIC actual — Cool + simultaneous top reheat":{
        "T_fill":80.0,
        "zones":[{"T":60.0,"duration":6.0,"label":"Zone 1"},
                 {"T":45.0,"duration":7.0,"label":"Zone 2"},
                 {"T":30.0,"duration":7.0,"label":"Zone 3 (sub-RT air)"}],
        "reheat":{"T":78.0,"duration":8.0,"mode":"simultaneous"},
        "h_cool":8.0,"h_reheat":18.0,
        "desc":("Matches CBIC data: bulk cooled continuously (cold air ≤ RT) "
                "while ONLY the top surface is reheated at the same time. No "
                "separate reheat block → shorter cycle, top ヒケ suppressed."),
    },
    "🏭 CBIC production — Pulsed top reheat (5×20s)":{
        "T_fill":80.0,
        "zones":[{"T":21.0,"duration":7.0,"label":"Cold air (RT)"},
                 {"T":21.0,"duration":7.0,"label":"Cold air (RT)"},
                 {"T":16.0,"duration":6.0,"label":"Sub-RT air"}],
        "reheat":{"T":100.0,"duration":0.0,"mode":"pulsed",
                  "pulses":5,"pulse_sec":20,"pulse_window":6.0},
        "h_cool":10.0,"h_reheat":25.0,
        "melt":67.0,"late_cool_T":16.0,
        "desc":("CBIC production line: bulk cooled with cold air (≤RT, → 16°C "
                "late in the cycle); the top surface is reheated in 5 × 20 s "
                "toggled hot-air bursts (100°C) that re-melt the skin during "
                "solidification → surface cracks suppressed."),
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────
def lf(T_C, t_sol=T_SOL_C, t_liq=T_LIQ_C, t_mid=T_MID_C):
    # Liquid fraction across the mushy zone. Bounds default to the module
    # constants but can be overridden from a user melting point (see melt_band).
    if T_C>=t_liq: return 1.0
    if T_C<=t_sol: return 0.0
    return 0.5*(1+np.sin(np.pi*(T_C-t_mid)/(t_liq-t_sol)))

def melt_band(melt):
    """Melting point -> (solidus, liquidus, mid). A 10°C mushy band centred on
    the melting point (matches the default 62/72 when melt=67)."""
    if melt is None: return T_SOL_C, T_LIQ_C, T_MID_C
    melt=float(melt)
    return melt-5.0, melt+5.0, melt

def _rlbl(d):
    if d>=0.80: return "CRITICAL"
    if d>=0.50: return "WARNING"
    if d>=0.25: return "CAUTION"
    return "SAFE"

def _rcol(d):
    return RISK_COL[_rlbl(d)]

def _rname(d):
    """Risk level in the active UI language, sentence case."""
    return t("risk_"+_rlbl(d).lower())

def _md_to_html(content, bullet_color=INK["frost3"]):
    """Lightweight, safe Markdown -> HTML so LLM output (**bold**, `code`,
    *italic*, bullet / numbered lists) renders properly in our custom HTML
    cards instead of showing literal asterisks. HTML is escaped first."""
    safe = (content or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#E8EDF4;font-weight:600">\1</strong>', safe)
    safe = re.sub(r"`([^`]+?)`",
                  r'<code style="background:#1F2A3C;border-radius:4px;padding:1px 5px;'
                  r'font-size:.86em;color:#E8EDF4">\1</code>', safe)
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
            out.append('<div style="font-weight:600;color:#E8EDF4;'
                       'margin:12px 0 4px;font-size:.95em">'
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
    "language": "Language",
    "brand_desc": "Solidification control",
    # Hero
    "outlook": "Top surface outlook",
    "risk_safe": "Safe", "risk_caution": "Caution",
    "risk_warning": "Warning", "risk_critical": "Critical",
    "v_safe": "Flawless top surface expected.",
    "v_caution": "Minor surface marks are possible.",
    "v_warning": "Sink marks are likely on the top surface.",
    "v_critical": "Severe surface defects are expected.",
    "v_detail": "Peak damage index {di:.3f}. Sink marks begin to show at 0.25.",
    "v_safe_hint": "This recipe meets the quality target.",
    "v_safe_margin": "Top surface within limits, with a small margin.",
    "v_safe_margin_hint": "Raise the hot air or slow the first zone to widen the margin.",
    "v_caution_hint": "Strengthen the top reheat, or try pulsed bursts in step 4.",
    "v_warning_hint": "Slow the first cooling zone and add top-surface reheat.",
    "v_critical_hint": "Redesign the zone schedule, or run Optimise in step 7.",
    "rb_label": "Surface temperature through the cycle",
    "rb_reheat": "Hot air on the top surface",
    "rb_final": "Final cooling", "rb_reheat_seg": "Reheat",
    "zone_n": "Zone {n}",
    "live_hint": "Results update as you change the setup on the left.",
    "ro_di": "Peak damage index", "ro_cycle": "Cycle time",
    "ro_biot": "Biot number", "ro_heal": "Healed by reheat",
    "ro_reheat": "Top reheat",
    "st_within": "within the 30 min target", "st_over": "{d:.0f} min over target",
    "st_uniform": "even cooling", "st_uneven": "uneven cooling",
    "st_effective": "effective", "st_minimal": "minimal", "st_none": "no reheat",
    "st_limit": "limit 0.25",
    "reheat_val_seq": "{T:.0f} °C, {d:.0f} min",
    "reheat_val_sim": "{T:.0f} °C, {d:.0f} min",
    "reheat_val_pul": "{n} × {s} s",
    "reheat_note_seq": "after cooling", "reheat_note_sim": "during cooling",
    "reheat_note_pul": "bursts at {T:.0f} °C",
    "reheat_off": "Off", "min": "min",
    # Tabs
    "tab_belt": "Production line", "tab_temp": "Thermal history",
    "tab_crack": "Surface integrity", "tab_results": "Report", "tab_advisor": "Advisor",
    "sf_belt": ("Follow one stick along the U-turn conveyor. Press Play or drag the "
                "timeline; the colour of the stick is its temperature."),
    "sf_temp": ("Surface and core temperature over the cycle, with the damage index "
                "below. Shaded bands mark hot air on the top surface."),
    "sf_crack": ("The stick at its most damaged moment and after reheat. Cracks are drawn "
                 "only where the damage index reaches 0.25."),
    "sf_report": "The recommendation, the checks this recipe must pass, and the recipe itself.",
    "sf_advisor": ("Ask about this run. Answers draw on CBIC's trial and production data "
                   "and keep the context of the conversation."),
    # Sidebar
    "process_builder": "Process setup",
    "sb_sub": "LCWT401 stick, 1.25 cm radius, 4 cm tall",
    "step_start": "Starting point", "step_fill": "Fill",
    "preset": "Scenario", "load_preset": "Reload scenario",
    "fill_label": "Fill temperature, °C",
    "cooling_zones": "Cooling zones",
    "zones_help": "Each zone holds the air at a set temperature for a set time.",
    "zone_T": "Zone {n}, °C", "zone_min": "Minutes", "remove_zone": "Remove zone {n}",
    "add_zone": "Add zone", "reset": "Reset zones",
    "cool_total": "Cooling {t:.1f} min. Last zone {T:.0f} °C{flag}.",
    "cool_flag": ", aim for 42 °C or less",
    "hot_air_reheat": "Top reheat",
    "reheat_help": "Set the time to zero to switch reheat off.",
    "reheat_mode": "When the hot air runs",
    "reheat_seq": "After cooling",
    "reheat_sim": "During cooling, top only",
    "reheat_pulsed": "Pulsed bursts, as on the line",
    "reheat_mode_help": ("After cooling: the whole stick cools, then hot air reheats it. "
                         "During cooling: the core keeps cooling while only the top is warmed. "
                         "Pulsed bursts: short hot-air bursts on the top, as on CBIC's "
                         "production line."),
    "reheat_T": "Hot-air temperature, °C",
    "reheat_dur": "Reheat time, min",
    "reheat_window": "Top reheat time, min",
    "reheat_window_help": ("How long the top is warmed while the core keeps cooling. It runs "
                           "alongside cooling, so the cycle does not get longer."),
    "reheat_sim_note": "The core keeps cooling, so the cycle does not get longer.",
    "reach": "The surface reaches about {T:.0f} °C, {flag}.",
    "reach_ok": "enough to re-melt and heal", "reach_soft": "which only softens it",
    "pulse_count": "Number of bursts",
    "pulse_count_help": "How many hot-air bursts reach the top surface. CBIC uses about 5.",
    "pulse_dur": "Burst length, s",
    "pulse_dur_help": "Length of each hot-air burst. CBIC uses about 20 s.",
    "pulse_win": "Spread bursts over, min",
    "pulse_win_help": ("Bursts are spread across the start of cooling, while the top skin "
                       "forms and re-melting matters most."),
    "pulse_note": ("{n} bursts of {s} s, {on:.1f} min of hot air in total. Each burst "
                   "re-melts the top skin while the core keeps cooling."),
    "material_ambient": "Material and air",
    "melting_point": "Melting point, °C",
    "melting_point_help": ("The material's melting point. It sets the mushy band where sink "
                           "marks and cracks form."),
    "mushy_note": "Mushy band {lo:.0f}–{hi:.0f} °C.",
    "late_cool": "Late cooling air, °C",
    "late_cool_help": ("The air temperature at the end of the cycle. CBIC drops the line to "
                       "about 16 °C, below room temperature."),
    "convection": "Air flow",
    "h_cool": "Cooling air h, W/m²K", "h_reheat": "Hot air h, W/m²K",
    "sb_cycle": "Cycle about {t:.0f} min, target 30",
    "sb_biot": "Biot {b:.2f}, even below 0.5",
    "sb_rate": "Average cooling {r:.1f} °C/min, gentle above −5",
    "drl_optimiser": "Optimise",
    "drl_help": ("Searches over 500 recipes for the lowest damage within 30 minutes, for each "
                 "reheat timing. The result for your current timing is applied."),
    "drl_run": "Find the best recipe",
    "drl_spin": "Searching recipes for all three reheat timings…",
    "drl_done": "Applied the best recipe for {mode}: damage index {di:.3f}, {t:.0f} min.",
    "drl_compare": "Best result for each timing",
    "mode_seq": "after cooling", "mode_sim": "during cooling", "mode_pul": "pulsed bursts",
    "drl_na": "no feasible recipe",
    # Figures
    "play": "Play", "pause": "Pause", "time": "Time",
    "belt_x": "Belt position, cm",
    "lane_cool": "Cooling", "lane_cool_top": "Cooling and top reheat",
    "lane_final": "Final cooling", "lane_reheat_final": "Reheat and final cooling",
    "belt_frame": "{t:.1f} min, surface {Ts:.0f} °C, damage index {di:.3f}",
    "belt_reheating": "Reheating",
    "belt_hot_air": "Hot air {T:.0f} °C",
    "belt_bursts": "{n} bursts of {s} s at {T:.0f} °C",
    "zone_plate": "{z}<br>{T:.0f} °C, {d:.0f} min",
    "final_plate": "Final cooling<br>{T:.0f} °C",
    "ch_temp": "Temperature", "ch_grad": "Core minus surface, and cooling rate",
    "ch_di": "Damage index",
    "tr_surface": "Surface", "tr_core": "Core", "tr_air": "Air setpoint",
    "tr_dT": "Core minus surface", "tr_rate": "Cooling rate", "tr_di": "Damage index",
    "ax_temp": "°C", "ax_grad": "°C, °C/min", "ax_di": "Index", "ax_time": "Time, min",
    "ch_title": "Peak damage index {di:.3f}, {risk}",
    "ch_mushy": "Mushy band {lo:.0f}–{hi:.0f} °C",
    "ch_target": "Target {T:.0f} °C", "ch_fill": "Fill {T:.0f} °C",
    "ch_reheat": "Hot air {T:.0f} °C", "ch_heal": "Reheat heals {d:.3f}",
    "ch_point": ("<b>{t:.1f} min</b><br>Surface {Ts:.1f} °C, core {Tc:.1f} °C"
                 "<br>Damage index {di:.3f}"),
    "damage": "Damage", "di_short": "damage index",
    # Surface integrity
    "before_reheat": "Most damaged", "after_reheat": "After reheat",
    "end_cycle": "End of cycle",
    "crack_free": "Crack-free",
    "healed_by": "Reheat lowered the damage index by",
    "healing_eff": "Healing",
    "optimal_healing": "The surface re-entered the mushy band, so healing is strong.",
    "softening_only": "The surface stayed below the solidus, so it only softened.",
    "no_reheat": "No reheat in this recipe. Add top reheat in step 4 to heal the surface.",
    "min_healing": "Healing is small. Raise the hot air to 80 °C or more, or reheat for longer.",
    # Report
    "rep_rec": "Recommendation", "rep_checks": "Checks", "rep_recipe": "Recipe",
    "rep_reheat": "Reheat effect",
    "chk_time": "Cycle within 30 minutes", "chk_biot": "Biot number 0.5 or less",
    "chk_last": "Last zone at 42 °C or less", "chk_di": "Peak damage index below 0.25",
    "chk_hike": "No sink marks, below 0.15", "chk_heal": "Reheat lowers the damage",
    "rec_excellent": ("Production ready. With a peak damage index of {di:.3f}, no sink "
                      "marks are expected. The {n}-zone recipe runs in {t:.0f} minutes."),
    "rec_good": ("Safe, with a small margin (peak damage index {di:.3f}). To widen it, raise "
                 "the hot air to 90 °C or more, or hold one cooling zone at 62–72 °C to slow "
                 "solidification."),
    "rec_warning": ("Sink marks are likely on the top surface (peak damage index {di:.3f}). "
                    "Lower the cooling air h to 6 W/m²K or less, hold the first zone at "
                    "70–75 °C, and raise the hot air to 90–100 °C. Optimise, in step 7, can "
                    "search this for you."),
    "rec_critical": ("Severe sink marks and cracks are expected (peak damage index {di:.3f}). "
                     "Keep the top warm with a heat-retaining cap at 59–63 °C for 20 minutes, "
                     "set the cooling air h to 4 W/m²K or less, and apply three 1-minute "
                     "reheats at 90 °C, as in CBIC Trial 2."),
    "col_zone": "Zone", "col_air": "Air, °C", "col_min": "Minutes", "col_end": "Ends at",
    "col_phase": "Air sits",
    "ph_mushy": "in the mushy band", "ph_below": "below the solidus",
    "ph_above": "above the liquidus",
    "rh_mode": "Timing", "rh_air": "Hot air", "rh_reach": "Surface reaches",
    "rh_heal": "Damage healed", "rh_none": "No reheat in this recipe.",
    "rh_full": "full re-melt", "rh_mushy": "mushy band, partial healing",
    "rh_soft": "below the solidus, softening only",
    "run_first": "Results appear here once the simulation has run.",
    # Advisor
    "adv_online": "Advisor online", "adv_ready": "Advisor ready",
    "detailed_analysis": "Analysis of this run", "retry": "Refresh analysis",
    "source": "Source", "rule_based": "rule-based engine",
    "sec_assessment": "Assessment", "sec_physical": "Physical meaning",
    "sec_root": "Root cause", "sec_reco": "Recommendation", "sec_compare": "Comparison",
    "ask_advisor": "Ask about this run",
    "ask_hint": "For example sink-mark causes, reheat strategy or the Biot number.",
    "chat_empty": "No questions yet. Pick a suggestion below or type your own.",
    "followups": "Follow-up questions",
    "chat_ph": "Type a question and press Enter",
    "clear_chat": "Clear conversation", "send": "Send",
    "mem_on": "The advisor keeps the context of this conversation.",
    "mem_basic": "The advisor keeps the context of this conversation.",
    "you": "You", "advisor": "Advisor",
    # Export
    "export_title": "Export this figure",
    "export_hint": "PNG, SVG, PDF or interactive HTML, sized for papers, posters and slides.",
    "export_format": "Format", "export_preset": "Size",
    "export_dpi": "Resolution", "export_bg": "Background",
    "export_w": "Width, in", "export_h": "Height, in",
    "export_dl": "Generate file", "export_ready": "Download",
    "bg_dark": "Dark, as shown", "bg_white": "White, for journals",
    "bg_transparent": "Transparent",
  },
  "ja": {
    "language": "言語",
    "brand_desc": "固化制御",
    "outlook": "天面の見通し",
    "risk_safe": "安全", "risk_caution": "注意", "risk_warning": "警告", "risk_critical": "危険",
    "v_safe": "天面はきれいに仕上がる見込みです。",
    "v_caution": "天面にわずかな跡が出る可能性があります。",
    "v_warning": "天面にヒケが出る可能性が高いです。",
    "v_critical": "重大な表面欠陥が予想されます。",
    "v_detail": "最大損傷指数 {di:.3f}。0.25 を超えるとヒケが現れ始めます。",
    "v_safe_hint": "この条件は品質目標を満たしています。",
    "v_safe_margin": "天面は許容範囲内ですが、余裕は小さめです。",
    "v_safe_margin_hint": "熱風を上げるか、第1ゾーンを緩やかにして余裕を広げてください。",
    "v_caution_hint": "上面の再加熱を強めるか、手順4でパルス加熱を試してください。",
    "v_warning_hint": "第1冷却ゾーンを緩やかにし、上面の再加熱を加えてください。",
    "v_critical_hint": "ゾーン構成を見直すか、手順7の最適化を実行してください。",
    "rb_label": "サイクル中の表面温度",
    "rb_reheat": "上面への熱風",
    "rb_final": "最終冷却", "rb_reheat_seg": "再加熱",
    "zone_n": "ゾーン{n}",
    "live_hint": "左側の設定を変えると、結果はすぐに更新されます。",
    "ro_di": "最大損傷指数", "ro_cycle": "サイクル時間", "ro_biot": "ビオ数",
    "ro_heal": "再加熱による回復", "ro_reheat": "上面再加熱",
    "st_within": "目標30分以内", "st_over": "目標を{d:.0f}分超過",
    "st_uniform": "均一な冷却", "st_uneven": "不均一な冷却",
    "st_effective": "効果あり", "st_minimal": "わずか", "st_none": "再加熱なし",
    "st_limit": "上限 0.25",
    "reheat_val_seq": "{T:.0f} ℃、{d:.0f} 分", "reheat_val_sim": "{T:.0f} ℃、{d:.0f} 分",
    "reheat_val_pul": "{n} × {s} 秒",
    "reheat_note_seq": "冷却後", "reheat_note_sim": "冷却と同時",
    "reheat_note_pul": "{T:.0f} ℃ のバースト",
    "reheat_off": "なし", "min": "分",
    "tab_belt": "生産ライン", "tab_temp": "温度履歴", "tab_crack": "表面品質",
    "tab_results": "レポート", "tab_advisor": "アドバイザー",
    "sf_belt": ("Uターンコンベア上の1本のスティックを追います。再生するか、タイムラインを"
                "動かしてください。スティックの色は温度を表します。"),
    "sf_temp": ("サイクル中の表面と内部の温度、その下に損傷指数を示します。網掛けは上面に"
                "熱風が当たっている時間です。"),
    "sf_crack": ("最も損傷が大きい時点と再加熱後のスティックです。亀裂は損傷指数が0.25に"
                 "達した部分にのみ描画されます。"),
    "sf_report": "推奨事項、この条件が満たすべき確認項目、そして条件の詳細です。",
    "sf_advisor": ("この結果について質問できます。回答はCBICの試験・生産データに基づき、"
                   "会話の文脈を保持します。"),
    "process_builder": "プロセス設定",
    "sb_sub": "LCWT401 スティック、半径1.25 cm、高さ4 cm",
    "step_start": "出発点", "step_fill": "充填",
    "preset": "シナリオ", "load_preset": "シナリオを再読込",
    "fill_label": "充填温度（℃）",
    "cooling_zones": "冷却ゾーン",
    "zones_help": "各ゾーンは、空気を設定温度で一定時間保ちます。",
    "zone_T": "ゾーン{n}（℃）", "zone_min": "分", "remove_zone": "ゾーン{n}を削除",
    "add_zone": "ゾーンを追加", "reset": "ゾーンをリセット",
    "cool_total": "冷却 {t:.1f} 分。最終ゾーン {T:.0f} ℃{flag}。",
    "cool_flag": "（42 ℃以下が目安）",
    "hot_air_reheat": "上面の再加熱",
    "reheat_help": "時間を0にすると再加熱は無効になります。",
    "reheat_mode": "熱風のタイミング",
    "reheat_seq": "冷却後", "reheat_sim": "冷却と同時（上面のみ）",
    "reheat_pulsed": "パルス加熱（生産ライン）",
    "reheat_mode_help": ("冷却後：スティック全体を冷却してから熱風で再加熱します。"
                         "冷却と同時：内部を冷却しながら上面のみを温めます。"
                         "パルス加熱：CBIC生産ラインと同様に、上面へ短い熱風を断続的に当てます。"),
    "reheat_T": "熱風温度（℃）", "reheat_dur": "再加熱時間（分）",
    "reheat_window": "上面の再加熱時間（分）",
    "reheat_window_help": ("内部を冷却しながら上面を温める時間です。冷却と並行するため、"
                           "サイクルは長くなりません。"),
    "reheat_sim_note": "内部は冷却を続けるため、サイクルは長くなりません。",
    "reach": "表面は約 {T:.0f} ℃ に達し、{flag}。",
    "reach_ok": "再溶融して回復します", "reach_soft": "軟化のみにとどまります",
    "pulse_count": "バースト回数",
    "pulse_count_help": "上面に当てる熱風バーストの回数。CBICは約5回です。",
    "pulse_dur": "バースト時間（秒）",
    "pulse_dur_help": "1回の熱風バーストの長さ。CBICは約20秒です。",
    "pulse_win": "バーストを分散する時間（分）",
    "pulse_win_help": ("上面の表皮ができる冷却初期にバーストを分散します。この時期の再溶融が"
                       "最も重要です。"),
    "pulse_note": ("{s}秒のバーストを{n}回、熱風は合計{on:.1f}分。各バーストが上面の表皮を"
                   "再溶融し、内部は冷却を続けます。"),
    "material_ambient": "材料と空気",
    "melting_point": "融点（℃）",
    "melting_point_help": "材料の融点。ヒケや亀裂が生じるマッシー帯を決めます。",
    "mushy_note": "マッシー帯 {lo:.0f}〜{hi:.0f} ℃。",
    "late_cool": "後半の冷却空気（℃）",
    "late_cool_help": ("サイクル終盤の空気温度。CBICはラインを約16 ℃（室温以下）まで"
                       "下げます。"),
    "convection": "気流",
    "h_cool": "冷却空気 h（W/m²K）", "h_reheat": "熱風 h（W/m²K）",
    "sb_cycle": "サイクル約{t:.0f}分（目標30分）",
    "sb_biot": "ビオ数 {b:.2f}（0.5未満で均一）",
    "sb_rate": "平均冷却 {r:.1f} ℃/分（−5より緩やかが目安）",
    "drl_optimiser": "最適化",
    "drl_help": ("再加熱のタイミングごとに、30分以内で損傷が最小となる条件を500通り以上から"
                 "探索します。現在のタイミングの結果が適用されます。"),
    "drl_run": "最適な条件を探す",
    "drl_spin": "3つの再加熱タイミングで条件を探索しています…",
    "drl_done": "「{mode}」の最適条件を適用しました。損傷指数 {di:.3f}、{t:.0f} 分。",
    "drl_compare": "タイミング別の最良結果",
    "mode_seq": "冷却後", "mode_sim": "冷却と同時", "mode_pul": "パルス加熱",
    "drl_na": "実行可能な条件なし",
    "play": "再生", "pause": "一時停止", "time": "時間",
    "belt_x": "ベルト位置（cm）",
    "lane_cool": "冷却", "lane_cool_top": "冷却と上面再加熱",
    "lane_final": "最終冷却", "lane_reheat_final": "再加熱と最終冷却",
    "belt_frame": "{t:.1f} 分、表面 {Ts:.0f} ℃、損傷指数 {di:.3f}",
    "belt_reheating": "再加熱中",
    "belt_hot_air": "熱風 {T:.0f} ℃",
    "belt_bursts": "{T:.0f} ℃・{s}秒のバースト×{n}",
    "zone_plate": "{z}<br>{T:.0f} ℃、{d:.0f} 分",
    "final_plate": "最終冷却<br>{T:.0f} ℃",
    "ch_temp": "温度", "ch_grad": "内部と表面の温度差、冷却速度", "ch_di": "損傷指数",
    "tr_surface": "表面", "tr_core": "内部", "tr_air": "空気設定値",
    "tr_dT": "内部−表面", "tr_rate": "冷却速度", "tr_di": "損傷指数",
    "ax_temp": "℃", "ax_grad": "℃、℃/分", "ax_di": "指数", "ax_time": "時間（分）",
    "ch_title": "最大損傷指数 {di:.3f}、{risk}",
    "ch_mushy": "マッシー帯 {lo:.0f}〜{hi:.0f} ℃",
    "ch_target": "目標 {T:.0f} ℃", "ch_fill": "充填 {T:.0f} ℃",
    "ch_reheat": "熱風 {T:.0f} ℃", "ch_heal": "再加熱で {d:.3f} 回復",
    "ch_point": "<b>{t:.1f} 分</b><br>表面 {Ts:.1f} ℃、内部 {Tc:.1f} ℃<br>損傷指数 {di:.3f}",
    "damage": "損傷", "di_short": "損傷指数",
    "before_reheat": "損傷が最大の時点", "after_reheat": "再加熱後",
    "end_cycle": "サイクル終了時",
    "crack_free": "亀裂なし",
    "healed_by": "再加熱で損傷指数が低下：",
    "healing_eff": "回復率",
    "optimal_healing": "表面がマッシー帯に再び入ったため、回復は十分です。",
    "softening_only": "表面は固相線を下回ったままのため、軟化のみです。",
    "no_reheat": "この条件には再加熱がありません。手順4で上面の再加熱を加えると表面が回復します。",
    "min_healing": "回復はわずかです。熱風を80 ℃以上にするか、再加熱時間を延ばしてください。",
    "rep_rec": "推奨事項", "rep_checks": "確認項目", "rep_recipe": "条件",
    "rep_reheat": "再加熱の効果",
    "chk_time": "サイクル30分以内", "chk_biot": "ビオ数0.5以下",
    "chk_last": "最終ゾーン42 ℃以下", "chk_di": "最大損傷指数0.25未満",
    "chk_hike": "ヒケなし（0.15未満）", "chk_heal": "再加熱で損傷が低下",
    "rec_excellent": ("量産可能です。最大損傷指数は {di:.3f} で、ヒケは発生しない見込みです。"
                      "{n}ゾーンの条件で {t:.0f} 分です。"),
    "rec_good": ("安全ですが余裕は小さめです（最大損傷指数 {di:.3f}）。余裕を広げるには、"
                 "熱風を90 ℃以上にするか、62〜72 ℃に保つ冷却ゾーンを1つ加えて凝固を"
                 "緩やかにしてください。"),
    "rec_warning": ("天面にヒケが出る可能性が高いです（最大損傷指数 {di:.3f}）。冷却空気 h を"
                    "6 W/m²K以下にし、第1ゾーンを70〜75 ℃に保ち、熱風を90〜100 ℃に上げて"
                    "ください。手順7の最適化で自動探索もできます。"),
    "rec_critical": ("重大なヒケと亀裂が予想されます（最大損傷指数 {di:.3f}）。上面を59〜63 ℃の"
                     "保温キャップで20分保ち、冷却空気 h を4 W/m²K以下にし、CBIC試験2と同様に"
                     "90 ℃で1分間の再加熱を3回行ってください。"),
    "col_zone": "ゾーン", "col_air": "空気（℃）", "col_min": "分", "col_end": "終了時刻",
    "col_phase": "空気の位置",
    "ph_mushy": "マッシー帯内", "ph_below": "固相線以下", "ph_above": "液相線以上",
    "rh_mode": "タイミング", "rh_air": "熱風", "rh_reach": "表面到達温度",
    "rh_heal": "損傷の回復", "rh_none": "この条件には再加熱がありません。",
    "rh_full": "完全に再溶融", "rh_mushy": "マッシー帯、部分的に回復",
    "rh_soft": "固相線以下、軟化のみ",
    "run_first": "シミュレーション実行後に結果が表示されます。",
    "adv_online": "アドバイザー稼働中", "adv_ready": "アドバイザー準備完了",
    "detailed_analysis": "この結果の分析", "retry": "分析を更新",
    "source": "情報源", "rule_based": "ルールベース",
    "sec_assessment": "評価", "sec_physical": "物理的な意味", "sec_root": "根本原因",
    "sec_reco": "推奨事項", "sec_compare": "比較",
    "ask_advisor": "この結果について質問",
    "ask_hint": "例：ヒケの原因、再加熱の戦略、ビオ数など。",
    "chat_empty": "まだ質問はありません。下の候補を選ぶか、質問を入力してください。",
    "followups": "続けて質問",
    "chat_ph": "質問を入力して Enter",
    "clear_chat": "会話を消去", "send": "送信",
    "mem_on": "アドバイザーはこの会話の文脈を保持します。",
    "mem_basic": "アドバイザーはこの会話の文脈を保持します。",
    "you": "あなた", "advisor": "アドバイザー",
    "export_title": "この図をエクスポート",
    "export_hint": "論文・ポスター・スライド向けの PNG、SVG、PDF、インタラクティブ HTML。",
    "export_format": "形式", "export_preset": "サイズ", "export_dpi": "解像度",
    "export_bg": "背景", "export_w": "幅（インチ）", "export_h": "高さ（インチ）",
    "export_dl": "ファイルを生成", "export_ready": "ダウンロード",
    "bg_dark": "ダーク（表示どおり）", "bg_white": "白（学術誌向け）", "bg_transparent": "透明",
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

# Display names and descriptions for the scenario presets. The SCENARIOS keys
# stay as they are because they are stored in session state.
SCENARIO_TEXT = {
  "❌ Bad — No reheat (RT blast)": (
      "No reheat, room-temperature blast", "再加熱なし（室温の強風冷却）",
      "Room-temperature air only, with no reheat. Biot about 0.9, so sink marks and "
      "cracks are severe.",
      "室温の空気のみで再加熱なし。ビオ数は約0.9で、ヒケと亀裂が顕著です。"),
  "⚠️ Client current — 40 min total": (
      "Current line, 40 min", "現行ライン（40分）",
      "Today's line: 30 min at room temperature, then 10 min reheat. 40 min in total, "
      "which is too slow.",
      "現行ライン：室温で30分、その後10分の再加熱。合計40分で時間がかかりすぎます。"),
  "✅ Target — Step cool + reheat (30 min)": (
      "Target: stepped cooling and reheat", "目標：段階冷却と再加熱",
      "Three cooling steps over 20 min, then 10 min reheat. 30 min in total.",
      "20分で3段階に冷却し、その後10分再加熱。合計30分。"),
  "🚀 DRL Optimal (v4)": (
      "DRL optimum (v4)", "DRL最適（v4）",
      "Four cooling steps over 17 min, then 10 min reheat. 27 min in total, and safe.",
      "17分で4段階に冷却し、その後10分再加熱。合計27分で安全です。"),
  "🏭 CBIC actual — Cool + simultaneous top reheat": (
      "CBIC trial: top reheat during cooling", "CBIC試験：冷却中の上面再加熱",
      "From CBIC's trial: the core cools in cold air while only the top surface is "
      "warmed at the same time.",
      "CBIC試験より：内部を冷風で冷却しながら、上面のみを同時に温めます。"),
  "🏭 CBIC production — Pulsed top reheat (5×20s)": (
      "CBIC production: pulsed top reheat, 5 × 20 s", "CBIC生産：上面パルス加熱 5×20秒",
      "From CBIC's production line: cold air that drops to 16 °C late in the cycle, and "
      "five 20 s hot-air bursts at 100 °C on the top surface.",
      "CBIC生産ラインより：サイクル後半に16 ℃まで下がる冷風と、上面への100 ℃・20秒の"
      "熱風バースト5回。"),
}

def _sc_text(key, desc=False):
    """Scenario display name (or description) in the active language."""
    v = SCENARIO_TEXT.get(key)
    if not v:
        return SCENARIOS.get(key, {}).get("desc", "") if desc else key
    ja = st.session_state.get("lang") == "ja"
    return v[(3 if ja else 2) if desc else (1 if ja else 0)]

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
def build_timeline(T_fill,zones,reheat,h_cool,h_reheat,n_pts=120,
                   melt=None,late_cool_T=None):
    # ── Reheat mode ────────────────────────────────────────────────────────
    #  "sequential"  : classic — cool the whole bulk, THEN switch the entire
    #                  environment to hot air for a reheat block, then final cool.
    #  "simultaneous": bulk cooled CONTINUOUSLY while ONLY the top surface is
    #                  reheated concurrently (one window from the start).
    #  "pulsed"      : matches the CBIC production line — the top surface is
    #                  reheated in N short TOGGLED bursts (e.g. 5 × 20 s at
    #                  100°C) across the early solidification window, re-melting
    #                  the skin each time so the sink mark never sets, while the
    #                  bulk cools continuously.
    # melt        : material melting point (°C) → mushy band via melt_band().
    # late_cool_T : ambient the FINAL-cool phase relaxes toward (client drops
    #               the line to ~16°C below room temperature late in the cycle).
    mode   = str(reheat.get("mode","sequential")).lower()
    t_cool  = sum(z["duration"] for z in zones)
    t_reh   = float(reheat["duration"])
    n_pulse = int(reheat.get("pulses",0) or 0)
    pulse_s = float(reheat.get("pulse_sec",0) or 0.0)
    pulsed  = (mode == "pulsed") and n_pulse > 0 and pulse_s > 0.0
    simul   = (mode == "simultaneous") and t_reh > 0.0
    surf_reheat = simul or pulsed        # bulk cools; surface reheated concurrently
    t_sol,t_liq,t_mid = melt_band(melt)
    late_T  = float(T_ROOM if late_cool_T is None else late_cool_T)

    # Zone boundaries (the core cooling schedule — identical for all modes)
    zb=[]; ta=0.0
    for z in zones:
        zb.append((ta,ta+z["duration"],z["T"]))
        ta+=z["duration"]

    # Pulse window: N bursts of pulse_s seconds spread across the early cooling.
    pulse_win = float(min(reheat.get("pulse_window",6.0), t_cool)) if pulsed else 0.0
    def _surf_hot(tm):
        if simul:  return reh_s <= tm < reh_e
        if pulsed:
            if tm >= pulse_win: return False
            slot = pulse_win/max(n_pulse,1)
            k = int(tm//slot)
            return (tm - k*slot) < (pulse_s/60.0)
        return False

    if surf_reheat:
        # Surface reheat runs concurrently with cooling; the bulk never sees hot
        # air, so it keeps solidifying while the skin is held soft → the surface
        # sets LATER, warmer than the already-cooled core (inverted gradient →
        # no skin pulled over a liquid pocket → sink mark suppressed). No
        # appended reheat block → shorter cycle.
        if pulsed:
            reh_s = 0.0; reh_e = pulse_win
        else:
            reh_s = 0.0; reh_e = min(t_reh, t_cool)
        t_fin = float(np.clip(30.0 - t_cool - 0.5, 2.0, 8.0))
        t_tot = t_cool + t_fin
    else:
        reh_s = ta; reh_e = ta + t_reh
        t_fin = float(np.clip(30.0 - t_cool - t_reh - 0.5, 2.0, 8.0))
        t_tot = t_cool + t_reh + t_fin

    # Pulsed bursts are short (~20 s); resolve them with enough time steps.
    if pulsed:
        n_pts = max(n_pts, int(t_tot/max(pulse_s/60.0/2.0,0.02)) + 1, 260)
    times   = np.linspace(0,t_tot,n_pts)

    tau_c = max(2.0, RHO*CP*R_M**2/(K_TH*(h_cool*R_M/K_TH+0.1)*10))/60.0
    tau_r = max(1.0, RHO*CP*R_M**2/(K_TH*(h_reheat*R_M/K_TH+0.1)*10))/60.0
    # The top SKIN that forms the sink mark is a thin layer with small thermal
    # mass, so it re-melts fast under hot air — use a short skin time constant
    # while the surface is actively reheated (lets a ~20 s burst re-melt it).
    tau_skin = 0.30

    def _zone_env(tm):
        env=float(T_ROOM)
        for t0,t1,Tz in zb:
            if t0<=tm<=t1: env=float(Tz); break
        return env

    Ts=np.zeros(n_pts); Tc=np.zeros(n_pts)
    Te=np.zeros(n_pts); H=np.zeros(n_pts)
    Ts[0]=Tc[0]=float(T_fill)

    for i in range(1,n_pts):
        tm=float(times[i]); dt=float(times[i]-times[i-1])
        if surf_reheat:
            # Core ALWAYS follows the cooling schedule (never reheated); the
            # final phase relaxes toward the (possibly sub-room) late ambient.
            if tm <= t_cool:
                env_core=_zone_env(tm); tau_core=tau_c; H_core=h_cool
            else:
                env_core=late_T; tau_core=tau_c*1.5; H_core=h_cool
            # Surface: hot air only during the reheat window / pulse bursts.
            if _surf_hot(tm):
                env_surf=float(reheat["T"])
                tau_surf=(min(tau_r,tau_skin) if pulsed else tau_r); H_surf=h_reheat
            else:
                env_surf=env_core; tau_surf=tau_core; H_surf=H_core
        else:
            if tm<reh_s:
                env=_zone_env(tm); tau=tau_c; H_surf=h_cool
            elif tm<reh_e:
                env=float(reheat["T"]); tau=tau_r; H_surf=h_reheat
            else:
                env=late_T; tau=tau_c*1.5; H_surf=h_cool
            env_surf=env_core=env; tau_surf=tau_core=tau; H_core=H_surf
        H[i]=H_surf                     # DI uses the SURFACE Biot (top-face damage)
        Te[i]=env_surf
        Ts[i]=env_surf+(Ts[i-1]-env_surf)*np.exp(-dt/max(tau_surf,0.01))
        Ts[i]=float(np.clip(Ts[i],min(T_ROOM,late_T)-2,T_fill+2))
        lag=float(np.clip(1.0/(1.0+H_core*R_M/K_TH*0.5),0.3,0.9))
        Tc[i]=env_core+(Tc[i-1]-env_core)*np.exp(-dt/max(tau_core*(1+lag),0.01))

    # ODE-based DI — non-flat, rises in mushy zone, dips during reheat,
    # stops accumulating once material is fully solid and cool.
    DI_arr=np.zeros(n_pts); DI=0.0
    for i in range(1,n_pts):
        dt  = float(times[i]-times[i-1])
        T   = float(Ts[i]); Tc_ = float(Tc[i])
        Bi  = float(H[i])*R_M/K_TH
        fl_v= lf(T, t_sol, t_liq, t_mid); tm=float(times[i])
        mw  = float(4*fl_v*(1-fl_v))          # bell: peaks at fl=0.5
        Bi_r= float(np.clip((Bi-0.15)/0.85,0,1))
        dT_d= max(0.0, float(T_fill)-float(Te[i]))
        cs  = float(np.clip(dT_d/60.0,0,1))

        # Inverted thermal gradient (anti-sink). A top-surface sink mark (ヒケ)
        # forms when the SURFACE solidifies into a skin while the CORE beneath
        # is still liquid/warmer — the shrinking interior pulls the skin down.
        # When the surface is held HOTTER than the core (Ts > Tc), which only
        # happens under simultaneous top-surface reheat, that mechanism reverses:
        # the surface stays soft and solidifies last, so no skin is pulled over a
        # liquid pocket. inv_grad is exactly 0 for all normal cooling (Ts ≤ Tc),
        # so it never changes the sequential/pure-cooling scenarios.
        inv_grad = float(np.clip((T - Tc_) / 20.0, 0.0, 1.0))

        # Damage forms ONLY during phase transition (mushy zone). Suppressed
        # when the surface is warmer than the core (inv_grad).
        if fl_v > 0.01:
            k_form = ((0.60 * Bi_r * cs * (1.0 + 2.0 * mw)   # peaks strongly in mushy
                    + 0.15 * Bi_r * cs)                        # baseline during any liquid phase
                    * (1.0 - 0.9 * inv_grad))
        else:
            k_form = 0.0  # fully solid → no new damage

        # Healing driver: activated above 55°C (material softens)
        # 3.5x stronger during active reheat phase → allows DI→0 for T_reheat≥85°C
        # Plus an anti-sink bonus while the surface is held hotter than the core.
        hd     = max(0.0, T - 55.0) / 10.0
        in_reh = _surf_hot(tm) if surf_reheat else (reh_s <= tm < reh_e and t_reh > 0)
        k_heal = float(np.clip(hd * (3.5 if in_reh else 0.2) + 1.5 * inv_grad, 0.0, 2.5))
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
    if pulsed:      t_reheat_out = n_pulse*pulse_s/60.0     # total surface-reheat on-time
    elif simul:     t_reheat_out = reh_e - reh_s
    else:           t_reheat_out = t_reh
    _rmode = "pulsed" if pulsed else ("simultaneous" if simul else "sequential")
    return dict(times=times.tolist(),T_surf=Ts.tolist(),T_core=Tc.tolist(),
                T_env=Te.tolist(),dT=(Tc-Ts).tolist(),dTdt=dT_dt.tolist(),
                DI=DI_arr.tolist(),T_fill=float(T_fill),
                t_cool=t_cool,t_reheat=t_reheat_out,t_final=t_fin,t_total=t_tot,
                reheat_start=reh_s,reheat_end=reh_e,reheat_T=float(reheat["T"]),
                reheat_mode=_rmode,
                n_pulse=(n_pulse if pulsed else 0),
                pulse_sec=(pulse_s if pulsed else 0.0),
                pulse_window=(pulse_win if pulsed else 0.0),
                melt=(float(melt) if melt is not None else t_mid),
                late_cool_T=late_T,
                mushy=(t_sol,t_liq),
                zones=zones)

# ── Peridynamic crack helpers (module-level — used by both fig_belt and main) ──

def _crack_color(DI_v):
    return _rcol(DI_v)


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
                     marker=dict(size=6, color=INK["frost"],
                                 line=dict(color=col, width=2)),
                     opacity=1.0, showlegend=False, hoverinfo="skip"),
    ]


def _lighten_color(hex_col, k=0.55):
    """Mix a hex colour toward white (crack highlight / glow core)."""
    r,g,b=_hex2rgb(hex_col)
    return "#%02x%02x%02x" % tuple(round(c+(255-c)*k) for c in (r,g,b))


# ── U-Turn Belt Figure ─────────────────────────────────────────────────────
# Box triangulation shared by every plate and hot-air hood.
_BOX_I=[0,0,4,4,0,0,3,3,0,0,1,1]
_BOX_J=[1,2,5,6,1,5,2,6,3,7,2,6]
_BOX_K=[2,3,6,7,5,4,6,7,7,4,6,5]
_LIT =dict(ambient=0.62,diffuse=0.55,specular=0.12,roughness=0.85,fresnel=0.05)
_GLOW=dict(ambient=0.95,diffuse=0.35,specular=0.05,roughness=1.0,fresnel=0.0)
_LIGHT_POS=dict(x=60,y=-120,z=400)

def _slab(x0,x1,y0,y1,z0,z1,color,opacity=1.0,hover=None,lighting=_LIT):
    """Axis-aligned box as a lit Mesh3d (belt plates, hot-air hoods)."""
    return go.Mesh3d(x=[x0,x1,x1,x0,x0,x1,x1,x0], y=[y0,y0,y1,y1,y0,y0,y1,y1],
                     z=[z0,z0,z0,z0,z1,z1,z1,z1], i=_BOX_I, j=_BOX_J, k=_BOX_K,
                     color=color, opacity=opacity, flatshading=True,
                     lighting=lighting, lightposition=_LIGHT_POS, showlegend=False,
                     hovertemplate=(hover+"<extra></extra>") if hover else None,
                     hoverinfo=None if hover else "skip")

# Stick = shaded cylinder (side + top cap), vertex-coloured by temperature.
_N_TH, _N_ZL = 40, 14
def _stick_faces():
    I,J,K=[],[],[]
    for iz in range(_N_ZL-1):
        for it in range(_N_TH):
            a=iz*_N_TH+it; b=iz*_N_TH+(it+1)%_N_TH; c=a+_N_TH; d=b+_N_TH
            I+=[a,a]; J+=[b,d]; K+=[d,c]
    C=_N_TH*_N_ZL; top=(_N_ZL-1)*_N_TH
    for it in range(_N_TH):
        I.append(C); J.append(top+it); K.append(top+(it+1)%_N_TH)
    return I,J,K
_STICK_I,_STICK_J,_STICK_K=_stick_faces()
_TH=np.tile(np.linspace(0,2*np.pi,_N_TH,endpoint=False),_N_ZL)
_ZL=np.repeat(np.linspace(0,H_S,_N_ZL),_N_TH)

def _stick(sx,sy,Ts,Tc,cbar=None):
    """The top surface (天面) takes the surface temperature, the body the
    core temperature, so hot top / cool body reads directly as colour."""
    x=np.append(sx+R_S*np.cos(_TH),sx); y=np.append(sy+R_S*np.sin(_TH),sy)
    z=np.append(_ZL,H_S)
    temps=np.append(float(Tc)+(float(Ts)-float(Tc))*(_ZL/H_S)**2.2,float(Ts))
    return go.Mesh3d(x=x.tolist(),y=y.tolist(),z=z.tolist(),
        i=_STICK_I,j=_STICK_J,k=_STICK_K,intensity=temps.tolist(),
        colorscale=HEAT_SCALE,cmin=HEAT_MIN,cmax=HEAT_MAX,opacity=0.8,
        flatshading=False,lightposition=_LIGHT_POS,
        lighting=dict(ambient=0.5,diffuse=0.75,specular=0.45,roughness=0.35,fresnel=0.3),
        showscale=cbar is not None,colorbar=cbar,hoverinfo="skip",showlegend=False)

def fig_belt(ts, zones, reheat, h_cool, h_reheat, n_frames=40):
    t_cool=float(ts["t_cool"]); t_reh=float(ts["t_reheat"])
    t_tot=float(ts["t_total"]); t_fin=float(ts["t_final"])
    reh_s=float(ts["reheat_start"]); reh_e=float(ts["reheat_end"])
    # Concurrent surface reheat (simultaneous OR pulsed): hot air on the TOP
    # surface while the bulk keeps cooling, so there is no separate reheat leg.
    _rmode=ts.get("reheat_mode")
    simul=(_rmode=="simultaneous" and t_reh>0)
    pulsed=(_rmode=="pulsed" and int(ts.get("n_pulse",0))>0)
    surf=simul or pulsed
    n_pulse=int(ts.get("n_pulse",0)); pulse_sec=float(ts.get("pulse_sec",0))
    pulse_win=float(ts.get("pulse_window",0.0)) or reh_e
    T_air=float(reheat["T"]); hot=_heat(T_air)
    late=float(ts.get("late_cool_T",T_ROOM))
    y0t,y1t=LANE_TOP-BELT_W,LANE_TOP+BELT_W
    y0b,y1b=LANE_BOT-BELT_W,LANE_BOT+BELT_W
    PZ=-0.32            # plate thickness, cm
    hood_z=H_S+1.6      # hot-air nozzle bar height above the belt, cm

    static=[]
    def _label(x,y,z,txt,col=INK["frost2"],size=11):
        static.append(go.Scatter3d(x=[x],y=[y],z=[z],mode="text",text=[txt],
            textfont=dict(size=size,color=col,family=FONT),
            showlegend=False,hoverinfo="skip"))
    def _hood(x0,x1,lane_y,hover):
        # A slim nozzle bar over the lane centre; jets fan out onto the tops.
        static.append(_slab(x0,x1,lane_y-0.5,lane_y+0.5,hood_z,hood_z+0.16,
                            hot,0.85,hover,_GLOW))
        for f in (np.linspace(0.2,0.8,3) if (x1-x0)>3 else (0.5,)):
            xj=x0+(x1-x0)*float(f)
            for ya in (lane_y-1.3,lane_y,lane_y+1.3):
                static.append(go.Scatter3d(x=[xj,xj],y=[lane_y,ya],z=[hood_z-0.02,H_S+0.25],
                    mode="lines",line=dict(color=_rgba(hot,0.75),width=3),
                    showlegend=False,hoverinfo="skip"))

    # Top lane: cooling-zone plates, each coloured by its air temperature
    x_acc=0.0
    for i,z in enumerate(zones):
        xw=z["duration"]/max(t_cool,0.01)*L_COOL; xs,xe=x_acc,x_acc+xw
        txt=t("zone_plate").format(z=t("zone_n").format(n=i+1),T=z["T"],d=z["duration"])
        static.append(_slab(xs+0.06,xe-0.06,y0t,y1t,PZ,0,_heat(z["T"]),0.96,
                            txt.replace("<br>",", ")))
        _label((xs+xe)/2,y1t+1.5,0,txt)
        x_acc=xe

    ftxt=t("final_plate").format(T=late)
    if surf:
        xrs=L_COOL
        if pulsed:
            win_x=pulse_win/max(t_cool,0.01)*L_COOL; slot=win_x/max(n_pulse,1)
            bw=float(np.clip((pulse_sec/60.0)/max(t_cool,0.01)*L_COOL*3,slot*0.35,slot*0.9))
            for k in range(n_pulse):
                _hood(k*slot,k*slot+bw,LANE_TOP,
                      f"{k+1}/{n_pulse}, {int(pulse_sec)} s, {T_air:.0f} °C")
            _label(win_x/2,LANE_TOP,hood_z+1.0,
                   t("belt_bursts").format(n=n_pulse,s=int(pulse_sec),T=T_air),
                   _lighten_color(hot,0.35))
        else:
            xr=reh_e/max(t_cool,0.01)*L_COOL
            _hood(0,xr,LANE_TOP,t("belt_hot_air").format(T=T_air))
            _label(xr/2,LANE_TOP,hood_z+1.0,t("belt_hot_air").format(T=T_air),
                   _lighten_color(hot,0.35))
        static.append(_slab(0.06,L_COOL-0.06,y0b,y1b,PZ,0,_heat(late),0.96,
                            ftxt.replace("<br>",", ")))
        _label(L_COOL/2,y0b-1.5,0,ftxt)
    else:
        r_frac=t_reh/max(t_reh+t_fin,0.01) if t_reh>0 else 0
        xrs=L_COOL*(1-r_frac)
        if t_reh>0:
            rtxt=t("belt_hot_air").format(T=T_air)
            # The glowing plate marks the reheat leg. No overhead bar here: from
            # the camera it would project onto the cooling lane and mislead.
            static.append(_slab(xrs+0.06,L_COOL-0.06,y0b,y1b,PZ,0,hot,0.96,rtxt,_GLOW))
            _label((xrs+L_COOL)/2,y0b-1.5,0,rtxt)
        static.append(_slab(0.06,max(xrs-0.06,0.07),y0b,y1b,PZ,0,_heat(late),0.96,
                            ftxt.replace("<br>",", ")))
        _label(xrs/2,y0b-1.5,0,ftxt)

    # Rails and the U-turn
    rail=_rgba(INK["frost3"],0.45)
    for ly in (y0t,y1t,y0b,y1b):
        static.append(go.Scatter3d(x=[0,L_COOL],y=[ly,ly],z=[0.02,0.02],mode="lines",
            line=dict(color=rail,width=2),showlegend=False,hoverinfo="skip"))
    th=np.linspace(0,np.pi,40)
    for r,dep in ((LANE_TOP+BELT_W,1.6),(LANE_TOP-BELT_W,0.5)):
        static.append(go.Scatter3d(x=(L_COOL+dep*np.sin(th)).tolist(),
            y=(r*np.cos(th)).tolist(),z=[0.02]*len(th),mode="lines",
            line=dict(color=rail,width=2),showlegend=False,hoverinfo="skip"))

    n_st=len(static)
    cbar=dict(title=dict(text="°C",font=dict(size=11,color=INK["frost3"])),
              thickness=8,len=0.42,x=0.99,y=0.58,outlinewidth=0,
              tickvals=[20,40,60,80,100],tickfont=dict(size=10,color=INK["frost3"]))

    def stick_pos(frac_t):
        fc=t_cool/max(t_tot,0.01)
        if surf:
            # No separate reheat leg: cooling on the top lane, then final cool
            # on the return lane (reheat happens concurrently on the top lane).
            if frac_t<=fc:
                return float(np.clip(frac_t/max(fc,1e-6),0,1)*L_COOL),LANE_TOP
            rel=(frac_t-fc)/max(1-fc,1e-6)
            return float(L_COOL-rel*L_COOL),LANE_BOT
        fr=t_reh/max(t_tot,0.01)
        if frac_t<=fc:
            return float(np.clip(frac_t/max(fc,1e-6),0,1)*L_COOL),LANE_TOP
        elif frac_t<=fc+fr:
            rel=(frac_t-fc)/max(fr,1e-6)
            return float(L_COOL-rel*(L_COOL-xrs)),LANE_BOT
        else:
            rel=(frac_t-fc-fr)/max(1-fc-fr,1e-6)
            return float(xrs-rel*xrs),LANE_BOT

    times_a=np.array(ts["times"]); DI_a=np.array(ts["DI"]); Ts_a=np.array(ts["T_surf"])
    Tc_a=np.array(ts["T_core"]); Tm=float(ts["t_total"])

    # Plotly frames can only UPDATE existing traces, so keep exactly N_CRACK
    # crack slots (empty while the damage index is below 0.25).
    N_CRACK = 3
    def _empty_crack():
        return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                            line=dict(color="rgba(0,0,0,0)", width=1),
                            showlegend=False, hoverinfo="skip")
    def _pad_cracks(crack_list):
        out = list(crack_list)
        while len(out) < N_CRACK:
            out.append(_empty_crack())
        return out[:N_CRACK]
    def _marker(sx,sy,DI,in_reh):
        col=hot if in_reh else _rcol(DI)
        word=t("belt_reheating") if in_reh else _rname(DI)
        return go.Scatter3d(x=[sx],y=[sy],z=[H_S+0.9],mode="markers+text",
            marker=dict(size=5,color=col,line=dict(color=INK["night"],width=1)),
            text=[f"{word} {DI:.2f}"],textposition="top center",
            textfont=dict(size=11,color=INK["frost"],family=FONT),
            showlegend=False,hoverinfo="skip")
    def _in_reh(tm):
        if pulsed:
            slot=pulse_win/max(n_pulse,1)
            return tm<pulse_win and (tm-int(tm//slot)*slot)<pulse_sec/60.0
        return reh_s<=tm<reh_e and t_reh>0

    sx0,sy0=stick_pos(0); Ts0=float(Ts_a[0]); DI0=float(DI_a[0])
    all_t = static + [_stick(sx0,sy0,Ts0,float(Tc_a[0]),cbar),
                      _marker(sx0,sy0,DI0,_in_reh(0.0))] + _pad_cracks(
        peridynamic_crack_traces(sx0, sy0, DI0, 42, _crack_color(DI0)))
    si = n_st; li = n_st + 1
    crack_trace_indices = list(range(n_st + 2, n_st + 2 + N_CRACK))

    frames=[]; steps=[]
    for fi,tm in enumerate(np.linspace(0,Tm,n_frames)):
        sx_f,sy_f=stick_pos(tm/max(Tm,0.01))
        idx=int(np.argmin(np.abs(times_a-tm)))
        DI_f=float(DI_a[idx]); Ts_f=float(Ts_a[idx]); Tc_f=float(Tc_a[idx])
        ir=_in_reh(tm)
        frame_data=[_stick(sx_f,sy_f,Ts_f,Tc_f,cbar), _marker(sx_f,sy_f,DI_f,ir)] + \
            _pad_cracks(peridynamic_crack_traces(sx_f,sy_f,DI_f,fi*7+13,_crack_color(DI_f)))
        frames.append(go.Frame(data=frame_data, traces=[si, li] + crack_trace_indices,
            name=str(fi), layout=go.Layout(title_text=t("belt_frame").format(
                t=tm,Ts=Ts_f,di=DI_f))))
        steps.append(dict(method="animate",
            args=[[str(fi)], dict(mode="immediate", frame=dict(duration=150, redraw=True),
                                  transition=dict(duration=0))],
            label=f"{tm:.1f}"))

    lane_top=t("lane_cool_top") if surf else t("lane_cool")
    lane_bot=t("lane_final") if surf else t("lane_reheat_final")
    fig=go.Figure(data=all_t,frames=frames)
    fig.update_layout(
        height=620, showlegend=False,
        # Constant uirevision keeps the user's camera through timeline drags
        # and Streamlit reruns instead of snapping back to the default view.
        uirevision="belt-scene",
        title=dict(text=t("belt_frame").format(t=0.0,Ts=Ts0,di=DI0),x=0.0,
                   xanchor="left",y=0.985,font=dict(size=14,color=INK["frost"])),
        margin=dict(l=0,r=0,t=44,b=70),
        scene=dict(
            uirevision="belt-scene",
            xaxis=dict(title=dict(text=t("belt_x")),range=[-1.5,L_COOL+3],showgrid=False),
            yaxis=dict(title=dict(text=""),tickvals=[LANE_BOT,LANE_TOP],
                       ticktext=[lane_bot,lane_top],tickfont=dict(size=11,color=INK["frost2"]),
                       range=[LANE_BOT-BELT_W-3,LANE_TOP+BELT_W+3],showgrid=False),
            zaxis=dict(title=dict(text=""),showticklabels=False,showbackground=True,
                       backgroundcolor="rgba(23,33,49,0.55)",range=[PZ-0.3,hood_z+1.6],
                       showgrid=False),
            aspectmode="manual",aspectratio=dict(x=3.0,y=1.55,z=0.55),
            camera=dict(eye=dict(x=1.05,y=-1.85,z=0.95),center=dict(x=0.04,y=0,z=-0.14),
                        projection=dict(type="perspective"))),
        updatemenus=[dict(type="buttons",direction="right",showactive=False,
            x=0.0,y=0.0,xanchor="left",yanchor="top",pad=dict(t=14,r=8),
            bgcolor=INK["deck2"],bordercolor=INK["rule2"],borderwidth=1,
            font=dict(color=INK["frost"],size=12,family=FONT),
            buttons=[
                dict(label=t("play"),method="animate",
                     args=[None,dict(frame=dict(duration=140,redraw=True),
                                     fromcurrent=True,transition=dict(duration=0),
                                     mode="immediate")]),
                dict(label=t("pause"),method="animate",
                     args=[[None],dict(frame=dict(duration=0),mode="immediate",
                                       transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0,steps=steps,x=0.16,len=0.84,y=0.0,yanchor="top",
            pad=dict(t=14,b=0),
            currentvalue=dict(prefix=t("time")+"  ",suffix=" "+t("min"),visible=True,
                              xanchor="right",offset=6,
                              font=dict(color=INK["frost"],size=12,family=FONT)),
            transition=dict(duration=0),bgcolor=INK["deck2"],activebgcolor=INK["frost"],
            bordercolor=INK["rule2"],borderwidth=1,tickcolor=INK["rule2"],
            ticklen=3,minorticklen=0,font=dict(color="rgba(0,0,0,0)",size=1))],
    )
    return fig

# ── Thermal history (temperature, gradient, damage index) ─────────────────
def _reheat_spans(ts):
    """Time spans with hot air on the surface: one window, or each burst."""
    if ts.get("reheat_mode")=="pulsed" and int(ts.get("n_pulse",0))>0:
        n=int(ts["n_pulse"]); slot=float(ts["pulse_window"])/n
        d=float(ts["pulse_sec"])/60.0
        return [(k*slot,k*slot+d) for k in range(n)]
    if float(ts.get("t_reheat",0))>0 and ts["reheat_end"]>ts["reheat_start"]:
        return [(float(ts["reheat_start"]),float(ts["reheat_end"]))]
    return []

def fig_charts(ts,label=""):
    times=ts["times"]; n=len(times)
    zones=ts.get("zones",[])
    T_air=float(ts.get("reheat_T",0)); hot=_heat(T_air)
    spans=_reheat_spans(ts)

    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=(t("ch_temp"),t("ch_grad"),t("ch_di")),
        vertical_spacing=0.075,row_heights=[0.46,0.22,0.32])
    fig.update_annotations(font=dict(size=12,color=INK["frost2"],family=FONT),
                           x=0,xanchor="left")

    # Zone boundaries (quiet), zone names only where there is room
    ta=0.0
    for i,z in enumerate(zones):
        x0=ta; ta+=z["duration"]
        if ta<max(times)*0.98:
            for ri in (1,2,3):
                fig.add_vline(x=ta,line_dash="dot",line_color=INK["rule2"],
                              line_width=1,row=ri,col=1)
        if z["duration"]>=2.5:
            fig.add_annotation(x=(x0+ta)/2,xref="x",y=0.99,yref="y domain",
                text=t("zone_n").format(n=i+1),showarrow=False,yanchor="top",
                font=dict(size=10,color=INK["frost3"]))

    # Hot air on the surface: each burst (pulsed) or the reheat window
    for a,b in spans:
        for ri in (1,2,3):
            fig.add_vrect(x0=a,x1=b,fillcolor=_rgba(hot,0.24),line_width=0,row=ri,col=1)
    if spans:
        fig.add_annotation(x=spans[0][0],xref="x",y=0.02,yref="y domain",
            text=t("ch_reheat").format(T=T_air),showarrow=False,xanchor="left",
            yanchor="bottom",font=dict(size=10,color=_lighten_color(hot,0.3)))

    # Mushy band (from the melting point), target and fill temperatures
    _msol,_mliq = ts.get("mushy",(T_SOL_C,T_LIQ_C))
    fig.add_hrect(y0=_msol,y1=_mliq,fillcolor=_rgba("#F0DDAE",0.09),line_width=0,row=1,col=1)
    fig.add_annotation(x=1,xref="x domain",y=_mliq,yref="y",
        text=t("ch_mushy").format(lo=_msol,hi=_mliq),showarrow=False,
        xanchor="right",yanchor="bottom",font=dict(size=10,color="#F0DDAE"))
    fig.add_hline(y=T_TARGET,line_dash="dot",line_color=_rgba("#5F9BD3",0.7),
        line_width=1,row=1,col=1,annotation_text=t("ch_target").format(T=T_TARGET),
        annotation_font=dict(size=10,color="#5F9BD3"),annotation_position="bottom right")
    fig.add_hline(y=ts["T_fill"],line_dash="dot",line_color=INK["rule2"],
        line_width=1,row=1,col=1,annotation_text=t("ch_fill").format(T=ts["T_fill"]),
        annotation_font=dict(size=10,color=INK["frost3"]),annotation_position="top right")

    fig.add_trace(go.Scatter(x=times,y=ts["T_surf"],name=t("tr_surface"),
        line=dict(color=INK["frost"],width=2.6)),row=1,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["T_core"],name=t("tr_core"),
        line=dict(color="#5F9BD3",width=2,dash="dash")),row=1,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["T_env"],name=t("tr_air"),
        line=dict(color=INK["frost3"],width=1.3,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["dT"],name=t("tr_dT"),
        line=dict(color="#F0DDAE",width=2)),row=2,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["dTdt"],name=t("tr_rate"),
        line=dict(color="#5F9BD3",width=1.6,dash="longdash")),row=2,col=1)
    fig.add_hline(y=0,line_color=INK["rule2"],line_width=1,row=2,col=1)

    for y0,y1,key in ((0,0.25,"SAFE"),(0.25,0.5,"CAUTION"),
                      (0.5,0.8,"WARNING"),(0.8,1.05,"CRITICAL")):
        fig.add_hrect(y0=y0,y1=y1,fillcolor=_rgba(RISK_COL[key],0.06),
                      line_width=0,row=3,col=1)
    for th_,key in ((0.25,"CAUTION"),(0.5,"WARNING"),(0.8,"CRITICAL")):
        c=RISK_COL[key]
        fig.add_hline(y=th_,line_dash="dash",line_color=_rgba(c,0.6),line_width=1,
            annotation_text=t("risk_"+key.lower()),annotation_font=dict(size=10,color=c),
            annotation_position="top right",row=3,col=1)
    fig.add_trace(go.Scatter(x=times,y=ts["DI"],name=t("tr_di"),
        line=dict(color="#EEA35C",width=2.6),fill="tozeroy",
        fillcolor=_rgba("#EEA35C",0.12)),row=3,col=1)

    if spans:
        ta_arr=np.array(times); DI_arr=np.array(ts["DI"])
        is_=int(np.argmin(np.abs(ta_arr-spans[0][0])))
        ie_=int(np.argmin(np.abs(ta_arr-spans[-1][1])))
        drop=float(DI_arr[is_])-float(DI_arr[min(ie_,len(DI_arr)-1)])
        if drop>0.005:
            fig.add_annotation(x=(spans[0][0]+spans[-1][1])/2,y=float(DI_arr[ie_])+0.07,
                text=t("ch_heal").format(d=drop),font=dict(color=_lighten_color(hot,0.3),size=11),
                bgcolor=INK["deck"],bordercolor=_rgba(hot,0.6),borderwidth=1,
                showarrow=False,xref="x3",yref="y3")

    # Moving read-out pointer (6 traces updated by the frames)
    ns=len(fig.data)
    Tlo=min(ts["T_surf"]+ts["T_core"]+ts["T_env"])-5
    Thi=max(ts["T_surf"]+ts["T_core"]+ts["T_env"])+5
    dlo=min(ts["dT"]+ts["dTdt"])-3; dhi=max(ts["dT"]+ts["dTdt"])+3
    line_c=_rgba(INK["frost"],0.7)
    def _ptr(tm,Ts_f,Tc_f,dT_f,DI_f,ip):
        lc=hot if ip else line_c
        mk=dict(line=dict(color=INK["night"],width=1.5))
        return [
            go.Scatter(x=[tm,tm],y=[Tlo,Thi],mode="lines",line=dict(color=lc,width=1.5),
                       showlegend=False,hoverinfo="skip"),
            go.Scatter(x=[tm,tm],y=[Ts_f,Tc_f],mode="markers",
                       marker=dict(size=10,color=[INK["frost"],"#5F9BD3"],**mk),
                       showlegend=False,hoverinfo="skip"),
            go.Scatter(x=[tm,tm],y=[dlo,dhi],mode="lines",line=dict(color=lc,width=1.5),
                       showlegend=False,hoverinfo="skip"),
            go.Scatter(x=[tm],y=[dT_f],mode="markers",marker=dict(size=9,color="#F0DDAE",**mk),
                       showlegend=False,hoverinfo="skip"),
            go.Scatter(x=[tm,tm],y=[0,1.05],mode="lines",line=dict(color=lc,width=1.5),
                       showlegend=False,hoverinfo="skip"),
            go.Scatter(x=[tm],y=[DI_f],mode="markers",marker=dict(size=11,color=_rcol(DI_f),**mk),
                       showlegend=False,hoverinfo="skip"),
        ]
    def _note(tm,Ts_f,Tc_f,DI_f,ip):
        return dict(xref="x",yref="y",x=tm,y=min(Ts_f+6,Thi-3),
            text=t("ch_point").format(t=tm,Ts=Ts_f,Tc=Tc_f,di=DI_f),
            showarrow=True,arrowhead=0,arrowwidth=1,arrowcolor=INK["frost3"],
            font=dict(color=INK["frost"],size=11,family=FONT),align="left",
            bgcolor=INK["deck"],bordercolor=hot if ip else INK["rule2"],borderwidth=1,
            borderpad=6,ax=48,ay=-52)
    for tr,r in zip(_ptr(times[0],ts["T_surf"][0],ts["T_core"][0],ts["dT"][0],
                         ts["DI"][0],False),(1,1,2,2,3,3)):
        fig.add_trace(tr,row=r,col=1)
    # Frames replace layout.annotations, so every frame carries the static ones.
    base_ann=list(fig.layout.annotations)

    def _ip(tm): return any(a<=tm<b for a,b in spans)
    frames=[]; steps=[]
    for fi in range(n):
        tm=times[fi]; Ts_f=ts["T_surf"][fi]; Tc_f=ts["T_core"][fi]
        dT_f=ts["dT"][fi]; DI_f=ts["DI"][fi]; ip=_ip(tm)
        frames.append(go.Frame(data=_ptr(tm,Ts_f,Tc_f,dT_f,DI_f,ip),
            traces=list(range(ns,ns+6)),name=str(fi),
            layout=go.Layout(annotations=base_ann+[_note(tm,Ts_f,Tc_f,DI_f,ip)])))
        steps.append(dict(method="animate",
            args=[[str(fi)],dict(mode="immediate",frame=dict(duration=0,redraw=True),
                                  transition=dict(duration=0))],
            label=f"{tm:.1f}"))

    DI_pk=max(ts["DI"])
    fig.update_layout(
        height=780, uirevision="charts", hovermode="x unified",
        title=dict(text=t("ch_title").format(di=DI_pk,risk=_rname(DI_pk)),x=0,
                   xanchor="left",y=0.99,font=dict(size=14,color=INK["frost"])),
        legend=dict(orientation="h",x=1,xanchor="right",y=1.04,yanchor="bottom"),
        margin=dict(l=56,r=16,t=86,b=78),
        annotations=base_ann+[_note(times[0],ts["T_surf"][0],ts["T_core"][0],
                                    ts["DI"][0],_ip(times[0]))],
        updatemenus=[dict(type="buttons",direction="right",showactive=False,
            x=0.0,y=0.0,xanchor="left",yanchor="top",pad=dict(t=40,r=8),
            bgcolor=INK["deck2"],bordercolor=INK["rule2"],borderwidth=1,
            font=dict(color=INK["frost"],size=12,family=FONT),
            buttons=[
                dict(label=t("play"),method="animate",
                     args=[None,dict(frame=dict(duration=80,redraw=True),
                                     fromcurrent=True,transition=dict(duration=0),
                                     mode="immediate")]),
                dict(label=t("pause"),method="animate",
                     args=[[None],dict(frame=dict(duration=0),mode="immediate",
                                       transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0,steps=steps,x=0.16,len=0.84,y=0.0,yanchor="top",
            pad=dict(t=40,b=0),
            currentvalue=dict(prefix=t("time")+"  ",suffix=" "+t("min"),visible=True,
                              xanchor="right",offset=6,
                              font=dict(color=INK["frost"],size=12,family=FONT)),
            transition=dict(duration=0),bgcolor=INK["deck2"],activebgcolor=INK["frost"],
            bordercolor=INK["rule2"],borderwidth=1,tickcolor=INK["rule2"],
            ticklen=3,minorticklen=0,font=dict(color="rgba(0,0,0,0)",size=1))],
    )
    fig.update_yaxes(title_text=t("ax_temp"),row=1,col=1)
    fig.update_yaxes(title_text=t("ax_grad"),row=2,col=1)
    fig.update_yaxes(title_text=t("ax_di"),range=[0,1.05],row=3,col=1)
    fig.update_xaxes(title_text=t("ax_time"),row=3,col=1)
    fig.frames=frames
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────
MARK_SVG = ("<svg class='mark' viewBox='0 0 20 30' aria-hidden='true'>"
            "<defs><linearGradient id='pdm' x1='0' y1='0' x2='0' y2='1'>"
            "<stop offset='0' stop-color='#DD6A3A'/><stop offset='.45' stop-color='#F0DDAE'/>"
            "<stop offset='1' stop-color='#5F9BD3'/></linearGradient></defs>"
            "<rect x='3' y='1' width='14' height='28' rx='7' fill='url(#pdm)'/></svg>")

_STEP = [0]
def _sh(title):
    """Numbered step header. The sidebar is a real sequence (scenario, fill,
    zones, reheat, material, air flow, optimise), so the numbers carry meaning."""
    _STEP[0] += 1
    st.sidebar.markdown(f"<div class='step'><b>{_STEP[0]}</b><span>{title}</span></div>",
                        unsafe_allow_html=True)

def sidebar():
    _STEP[0] = 0
    # Language first (read the widget state so this whole run uses it)
    _lang_map = {"English": "en", "日本語": "ja"}
    if "lang_sel" in st.session_state:
        st.session_state.lang = _lang_map.get(st.session_state.lang_sel, "en")
    st.sidebar.markdown(
        f"<div class='sb-head'><div class='sb-title'>{t('process_builder')}</div>"
        f"<div class='sb-sub'>{t('sb_sub')}</div></div>", unsafe_allow_html=True)
    _sel = st.sidebar.radio(t("language"), ["English", "日本語"],
        index=(1 if st.session_state.get("lang") == "ja" else 0),
        horizontal=True, key="lang_sel")
    st.session_state.lang = _lang_map[_sel]

    # 1 ─ Starting point
    _sh(t("step_start"))
    sc_name = st.sidebar.selectbox(t("preset"), list(SCENARIOS.keys()), index=2,
                                   key="sc_select", format_func=_sc_text)
    # Auto-load when scenario changes — no button click needed
    if st.session_state.get("last_sc") != sc_name:
        sc = SCENARIOS[sc_name]
        st.session_state.zones    = [dict(z) for z in sc["zones"]]
        st.session_state.reheat   = dict(sc["reheat"])
        st.session_state.h_cool   = sc["h_cool"]
        st.session_state.h_reheat = sc.get("h_reheat", 12.0)
        st.session_state.T_fill   = sc.get("T_fill", 80.0)
        st.session_state.melt     = sc.get("melt", 67.0)
        st.session_state.late_cool_T = sc.get("late_cool_T", 23.0)
        st.session_state.last_sc  = sc_name
        st.session_state.pop("drl_report", None)
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
        st.session_state.melt=sc.get("melt",67.0)
        st.session_state.late_cool_T=sc.get("late_cool_T",23.0)
        st.session_state.last_sc=sc_name
        st.session_state.pop("drl_report", None)
        st.session_state.zone_version=st.session_state.get("zone_version",0)+1
        st.rerun()
    st.sidebar.caption(_sc_text(sc_name, desc=True))

    # 2 ─ Fill
    _sh(t("step_fill"))
    T_fill=st.sidebar.slider(t("fill_label"),75.0,88.0,
        float(st.session_state.get("T_fill",T_FILL_DEF)),0.5)
    st.session_state.T_fill=T_fill

    # 3 ─ Cooling zones
    _sh(t("cooling_zones"))
    st.sidebar.caption(t("zones_help"))
    if "zones" not in st.session_state:
        st.session_state.zones=[dict(z) for z in SCENARIOS["✅ Target — Step cool + reheat (30 min)"]["zones"]]
    zones=st.session_state.zones
    ver=st.session_state.get("zone_version",0)  # changes on scenario load → widgets reinit
    del_idx=None
    for i,z in enumerate(zones):
        z["label"]=f"Zone {i+1}"
        cA,cB,cC=st.sidebar.columns([3,3,1], vertical_alignment="bottom")
        new_T=cA.number_input(t("zone_T").format(n=i+1),10.0,float(T_fill)-1,float(z["T"]),1.0,
                               key=f"zT_{ver}_{i}")
        new_D=cB.number_input(t("zone_min"),0.5,35.0,float(z["duration"]),0.5,
                               key=f"zD_{ver}_{i}")
        z["T"]=float(new_T); z["duration"]=float(new_D)
        if cC.button("✕",key=f"del_{ver}_{i}",help=t("remove_zone").format(n=i+1)) and len(zones)>1:
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
    st.sidebar.caption(t("cool_total").format(t=t_cool,T=T_last,
                                              flag="" if T_last<=42 else t("cool_flag")))

    # 4 ─ Top reheat
    _sh(t("hot_air_reheat"))
    if "reheat" not in st.session_state:
        st.session_state.reheat=dict(DEFAULT_REHEAT)
    reh=st.session_state.reheat
    # Reheat timing: after cooling (sequential), during cooling on the top
    # only (simultaneous), or pulsed bursts (CBIC production line).
    _mode_opts = [t("reheat_seq"), t("reheat_sim"), t("reheat_pulsed")]
    _cur_mode  = reh.get("mode","sequential")
    _mode_idx  = {"sequential":0,"simultaneous":1,"pulsed":2}.get(_cur_mode,0)
    _mode_sel  = st.sidebar.radio(t("reheat_mode"), _mode_opts, index=_mode_idx,
        help=t("reheat_mode_help"))
    reh["mode"] = ("pulsed" if _mode_sel==t("reheat_pulsed")
                   else "simultaneous" if _mode_sel==t("reheat_sim") else "sequential")
    _simul  = reh["mode"]=="simultaneous"
    _pulsed = reh["mode"]=="pulsed"

    reh["T"]=float(st.sidebar.slider(t("reheat_T"),40.0,120.0,float(reh.get("T",70.0)),1.0))
    if _pulsed:
        # Toggled surface reheat (CBIC production line): N short hot-air bursts.
        reh["pulses"]=int(st.sidebar.slider(t("pulse_count"),1,10,
            int(reh.get("pulses",5)),1,help=t("pulse_count_help")))
        reh["pulse_sec"]=int(st.sidebar.slider(t("pulse_dur"),5,60,
            int(reh.get("pulse_sec",20)),5,help=t("pulse_dur_help")))
        reh["pulse_window"]=float(st.sidebar.slider(t("pulse_win"),2.0,15.0,
            float(reh.get("pulse_window",6.0)),0.5,help=t("pulse_win_help")))
        reh["duration"]=0.0
        _on=reh["pulses"]*reh["pulse_sec"]/60.0
        st.sidebar.caption(t("pulse_note").format(n=reh["pulses"],s=int(reh["pulse_sec"]),on=_on))
    else:
        reh["duration"]=float(st.sidebar.slider(
            t("reheat_window") if _simul else t("reheat_dur"),0.0,15.0,
            float(reh.get("duration",10.0)),0.5,
            help=(t("reheat_window_help") if _simul else t("reheat_help"))))
        if reh["duration"]>0:
            tau_r=3.0; T_reach=reh["T"]-(reh["T"]-T_last)*np.exp(-reh["duration"]/tau_r)
            st.sidebar.caption(t("reach").format(T=T_reach,
                flag=t("reach_ok") if T_reach>=T_SOL_C else t("reach_soft")))
        if _simul:
            st.sidebar.caption(t("reheat_sim_note"))

    # 5 ─ Material and air
    _sh(t("material_ambient"))
    melt=float(st.sidebar.slider(t("melting_point"),40.0,100.0,
        float(st.session_state.get("melt",67.0)),1.0,help=t("melting_point_help")))
    st.session_state.melt=melt
    st.sidebar.caption(t("mushy_note").format(lo=melt-5,hi=melt+5))
    late_cool_T=float(st.sidebar.slider(t("late_cool"),5.0,30.0,
        float(st.session_state.get("late_cool_T",23.0)),1.0,help=t("late_cool_help")))
    st.session_state.late_cool_T=late_cool_T

    # 6 ─ Air flow, with a compact live status of the recipe
    _sh(t("convection"))
    h_cool=float(st.sidebar.slider(t("h_cool"),2.0,20.0,
        float(st.session_state.get("h_cool",6.0)),0.5))
    h_reh=float(st.sidebar.slider(t("h_reheat"),5.0,40.0,
        float(st.session_state.get("h_reheat",12.0)),0.5))
    st.session_state.h_cool=h_cool; st.session_state.h_reheat=h_reh
    # Same cycle-length rule as build_timeline: overlapping reheat adds no time.
    _add=0.0 if reh["mode"] in ("simultaneous","pulsed") else reh["duration"]
    t_tot=t_cool+_add+float(np.clip(30.0-t_cool-_add-0.5,2.0,8.0))
    Bi=h_cool*R_M/K_TH
    cr=-(T_fill-T_last)/max(t_cool,0.1)
    _rows=[(t_tot<=30, t("sb_cycle").format(t=t_tot)),
           (Bi<=0.5,   t("sb_biot").format(b=Bi)),
           (cr>=-5,    t("sb_rate").format(r=cr))]
    st.sidebar.markdown("<div class='sb-status'>"+"".join(
        f"<div><i style='background:{RISK_COL['SAFE'] if ok else RISK_COL['CAUTION']}'></i>"
        f"{txt}</div>" for ok,txt in _rows)+"</div>", unsafe_allow_html=True)

    # 7 ─ Optimise (DRL search over all three reheat timings)
    _sh(t("drl_optimiser"))
    st.sidebar.caption(t("drl_help"))
    if st.sidebar.button(t("drl_run"), type="primary", width='stretch'):
        with st.spinner(t("drl_spin")):
            T_fill_drl=T_fill; reh_dur=10.0
            # Optimise the schedule for ALL THREE reheat timings independently,
            # so we can report each option's best. The applied result uses the
            # timing currently selected on the radio — the DRL run never flips
            # the user's radio choice.
            best={"sequential":{"score":float("inf")},
                  "simultaneous":{"score":float("inf")},
                  "pulsed":{"score":float("inf")}}
            def _reh_grid(mode):
                if mode=="pulsed":     # toggled bursts — lean grid (pulses/T/window)
                    for T_reh in (95.0,105.0):
                        for npul in (3,5,7):
                            for win in (5.0,):
                                yield {"T":T_reh,"duration":0.0,"mode":"pulsed",
                                       "pulses":npul,"pulse_sec":20,"pulse_window":win}
                else:
                    for T_reh in (65.0,72.0,78.0,85.0,92.0,100.0,110.0):
                        yield {"T":T_reh,"duration":reh_dur,"mode":mode}
            for mode in ("sequential","simultaneous","pulsed"):
                overlap = mode in ("simultaneous","pulsed")   # reheat overlaps cooling
                cool_budget = (30.0-3.0) if overlap else (30.0-reh_dur-3.0)
                _hr_grid = [18.0,25.0] if mode=="pulsed" else [12.0,18.0,25.0]
                _nz_grid = [3,4] if mode=="pulsed" else [3,4,5]
                _dist_grid = ["mushy_dwell","top_heavy"] if mode=="pulsed" else ["mushy_dwell","linear","top_heavy"]
                for reh_base in _reh_grid(mode):
                    for h_reh_t in _hr_grid:
                        for h_cool_t in [h_cool, max(2.0,h_cool*0.6)]:
                            for nz in _nz_grid:
                                for dist in _dist_grid:
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
                                    t_f=(30.0-t_c) if overlap else (30.0-t_c-reh_dur)
                                    if t_f<1.5: continue  # skip infeasible
                                    tz=[{"T":round(T,1),"duration":round(d,1),"label":f"Zone {k+1}"} for k,(T,d) in enumerate(zip(temps,durs))]
                                    reh_t=dict(reh_base)
                                    try:
                                        tt=build_timeline(T_fill_drl,tz,reh_t,h_cool_t,h_reh_t,
                                                          n_pts=80,melt=melt,late_cool_T=late_cool_T)
                                        if tt["t_total"]>30.5: continue  # hard reject
                                        DI_pk=max(tt["DI"]); DI_fin=float(tt["DI"][-1])
                                        score=0.6*DI_pk+0.4*DI_fin
                                        if score<best[mode]["score"]:
                                            best[mode]={"score":score,"zones":tz,"reh":reh_t,
                                                        "h_reh":h_reh_t,"h_cool":h_cool_t,
                                                        "di":DI_pk,"ttot":float(tt["t_total"])}
                                    except: pass

            # Apply the optimum for the CURRENTLY selected timing (keeps the
            # radio put). Fall back to any timing that found a solution otherwise.
            _cur=reh.get("mode","sequential")
            apply=best[_cur] if best[_cur].get("zones") else None
            if apply is None:
                for other in ("simultaneous","pulsed","sequential"):
                    if best[other].get("zones"): apply=best[other]; _cur=other; break
            if apply and apply.get("zones"):
                best_z=apply["zones"]; best_r=apply["reh"]
                best_h_reh=apply["h_reh"]; best_h_cool=apply["h_cool"]
                tt_opt=build_timeline(T_fill_drl,best_z,best_r,best_h_cool,best_h_reh,
                                      n_pts=120,melt=melt,late_cool_T=late_cool_T)
                st.session_state.zones=   [dict(z) for z in best_z]
                st.session_state.reheat=  dict(best_r)   # mode=_cur → radio stays put
                st.session_state.h_reheat=best_h_reh
                st.session_state.h_cool=  best_h_cool
                st.session_state.zone_version=st.session_state.get("zone_version",0)+1
                st.session_state.sim_ts=  tt_opt
                st.session_state.sim_done=True
                st.session_state.drl_ts=  tt_opt
                st.session_state.drl_done=True
                st.session_state.drl_zones=[dict(z) for z in best_z]
                st.session_state.drl_reh= dict(best_r)
                # Kept in session state so the comparison survives the rerun.
                st.session_state.drl_report={
                    "mode":_cur,"di":max(tt_opt["DI"]),"t":tt_opt["t_total"],
                    "best":{m:({"di":best[m]["di"],"t":best[m]["ttot"]}
                               if best[m].get("zones") else None) for m in best}}
                st.rerun()
    rep=st.session_state.get("drl_report")
    if rep:
        names={"sequential":t("mode_seq"),"simultaneous":t("mode_sim"),"pulsed":t("mode_pul")}
        rows="".join(
            f"<tr class='{'cur' if m==rep['mode'] else ''}'><td>{names[m]}</td>"
            + (f"<td>{b['di']:.3f}</td><td>{b['t']:.0f} {t('min')}</td>" if b
               else f"<td colspan='2'>{t('drl_na')}</td>") + "</tr>"
            for m,b in rep["best"].items())
        st.sidebar.markdown(
            f"<div class='drl-rep'><p>{t('drl_done').format(mode=names[rep['mode']],di=rep['di'],t=rep['t'])}</p>"
            f"<div class='drl-cap'>{t('drl_compare')}</div><table>{rows}</table></div>",
            unsafe_allow_html=True)

    return dict(T_fill=T_fill, zones=zones, reheat=reh,
                h_cool=h_cool, h_reheat=h_reh,
                melt=melt, late_cool_T=late_cool_T,
                _scenario=sc_name)

# ── Report tab ─────────────────────────────────────────────────────────────
def show_results(ts, params):
    """Recommendation first, then the checks and the recipe behind it."""
    DI_pk  = max(ts["DI"])
    Bi     = params["h_cool"] * R_M / K_TH
    t_tot  = ts["t_total"]
    n_zones= len(params["zones"])
    t_reh  = ts["t_reheat"]
    T_reh  = params["reheat"]["T"]
    T_last = params["zones"][-1]["T"]
    ta_    = np.array(ts["times"]); DI_ = np.array(ts["DI"])
    spans  = _reheat_spans(ts)
    if spans:
        is_ = int(np.argmin(np.abs(ta_ - spans[0][0])))
        ie_ = int(np.argmin(np.abs(ta_ - spans[-1][1])))
        heal_drop = float(DI_[is_]) - float(DI_[min(ie_, len(DI_)-1)])
    else:
        heal_drop = 0.0
    sol, liq = ts.get("mushy", (T_SOL_C, T_LIQ_C))

    rec_key = ("rec_excellent" if DI_pk <= 0.10 else "rec_good" if DI_pk < 0.25
               else "rec_warning" if DI_pk <= 0.50 else "rec_critical")
    rec = t(rec_key).format(di=DI_pk, n=n_zones, t=t_tot)

    checks = [(t("chk_time"), t_tot <= 30,  f"{t_tot:.0f} {t('min')}"),
              (t("chk_biot"), Bi <= 0.5,    f"{Bi:.2f}"),
              (t("chk_last"), T_last <= 42, f"{T_last:.0f} °C"),
              (t("chk_di"),   DI_pk < 0.25, f"{DI_pk:.3f}"),
              (t("chk_hike"), DI_pk < 0.15, f"{DI_pk:.3f}"),
              (t("chk_heal"), heal_drop > 0.01 or not spans, f"{heal_drop:+.3f}")]
    n_ok = sum(1 for c in checks if c[1])
    chk_html = "".join(
        f"<li class='{'ok' if ok else 'no'}'><i>{'✓' if ok else '✕'}</i>"
        f"<span>{lbl}</span><b>{val}</b></li>" for lbl, ok, val in checks)

    rows = []; acc = 0.0
    for i, z in enumerate(params["zones"]):
        acc += z["duration"]
        ph = (t("ph_mushy") if sol <= z["T"] <= liq
              else t("ph_below") if z["T"] < sol else t("ph_above"))
        rows.append(f"<tr><td><i style='background:{_heat(z['T'])}'></i>"
                    f"{t('zone_n').format(n=i+1)}</td><td>{z['T']:.0f}</td>"
                    f"<td>{z['duration']:.1f}</td><td>{acc:.1f} {t('min')}</td><td>{ph}</td></tr>")
    table = (f"<table class='rp-table'><thead><tr><th>{t('col_zone')}</th>"
             f"<th>{t('col_air')}</th><th>{t('col_min')}</th><th>{t('col_end')}</th>"
             f"<th>{t('col_phase')}</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")

    mode = ts.get("reheat_mode", "sequential")
    mode_name = {"sequential": t("mode_seq"), "simultaneous": t("mode_sim"),
                 "pulsed": t("mode_pul")}.get(mode, mode)
    if not spans:
        rh = f"<p class='rp-muted'>{t('rh_none')}</p>"
    else:
        T_reach = T_reh - (T_reh - T_last) * np.exp(-t_reh / 3.0)
        state = (t("rh_full") if T_reach >= liq else
                 t("rh_mushy") if T_reach >= sol else t("rh_soft"))
        air = (t("belt_bursts").format(n=ts["n_pulse"], s=int(ts["pulse_sec"]), T=T_reh)
               if mode == "pulsed" else f"{T_reh:.0f} °C, {t_reh:.0f} {t('min')}")
        rh = ("<dl class='rp-dl'>"
              f"<dt>{t('rh_mode')}</dt><dd>{mode_name}</dd>"
              f"<dt>{t('rh_air')}</dt><dd>{air}</dd>"
              f"<dt>{t('rh_reach')}</dt><dd>{T_reach:.0f} °C, {state}</dd>"
              f"<dt>{t('rh_heal')}</dt><dd>{heal_drop:+.3f}</dd></dl>")

    st.markdown(f"<div class='rp-rec' style='--risk:{_rcol(DI_pk)}'>"
                f"<div class='rp-k'>{t('rep_rec')}</div><p>{rec}</p></div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.25], gap="large")
    c1.markdown(f"<div class='rp-k'>{t('rep_checks')} <span>{n_ok} / {len(checks)}</span></div>"
                f"<ul class='rp-checks'>{chk_html}</ul>", unsafe_allow_html=True)
    c2.markdown(f"<div class='rp-k'>{t('rep_recipe')}</div>{table}"
                f"<div class='rp-k rp-gap'>{t('rep_reheat')}</div>{rh}",
                unsafe_allow_html=True)

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


# ── Hero: outlook, thermal ribbon, read-outs ───────────────────────────────
def _ribbon_html(ts):
    """Surface temperature through the cycle as one colour band (same ramp as
    every other view), hot-air marks above it and the zone segments below."""
    tt=np.asarray(ts["times"],float); Ts=np.asarray(ts["T_surf"],float)
    T=float(tt[-1]) or 1.0
    idx=np.unique(np.linspace(0,len(tt)-1,56).astype(int))
    stops=",".join(f"{_heat(Ts[i])} {100*tt[i]/T:.2f}%" for i in idx)
    heat="".join(f"<i style='left:{100*a/T:.2f}%;width:{max(100*(b-a)/T,0.5):.2f}%'></i>"
                 for a,b in _reheat_spans(ts))
    segs=[]; acc=0.0
    for i,z in enumerate(ts["zones"]):
        segs.append((acc,acc+z["duration"],t("zone_n").format(n=i+1),f"{z['T']:.0f} °C"))
        acc+=z["duration"]
    if ts.get("reheat_mode")=="sequential" and float(ts["t_reheat"])>0:
        segs.append((acc,float(ts["reheat_end"]),t("rb_reheat_seg"),f"{ts['reheat_T']:.0f} °C"))
        acc=float(ts["reheat_end"])
    segs.append((acc,T,t("rb_final"),f"{float(ts.get('late_cool_T',T_ROOM)):.0f} °C"))
    zones="".join(f"<div style='left:{100*a/T:.2f}%;width:{100*(b-a)/T:.2f}%'>"
                  f"<span>{n}</span><b>{v}</b></div>" for a,b,n,v in segs if b>a)
    scale=",".join(f"{c} {100*(Tv-HEAT_MIN)/(HEAT_MAX-HEAT_MIN):.0f}%" for Tv,c in HEAT_STOPS)
    return (f"<div class='ribbon'><div class='rb-cap'><span>{t('rb_label')}</span>"
            f"<span class='rb-scale'>{HEAT_MIN} °C<i style='background:linear-gradient(90deg,{scale})'>"
            f"</i>{HEAT_MAX} °C</span></div>"
            f"<div class='rb-heat' title='{t('rb_reheat')}'>{heat}</div>"
            f"<div class='rb-band' style='background:linear-gradient(90deg,{stops})'></div>"
            f"<div class='rb-zones'>{zones}</div>"
            f"<div class='rb-axis'><span>0 {t('min')}</span><span>{T:.0f} {t('min')}</span></div></div>")

def _readouts_html(ts, params, heal):
    DI=max(ts["DI"]); t_tot=float(ts["t_total"]); Bi=params["h_cool"]*R_M/K_TH
    mode=ts.get("reheat_mode","sequential"); T_air=params["reheat"]["T"]
    spans=_reheat_spans(ts)
    if mode=="pulsed" and spans:
        rv=t("reheat_val_pul").format(n=int(ts["n_pulse"]),s=int(ts["pulse_sec"]))
        rn=t("reheat_note_pul").format(T=T_air)
    elif spans:
        key="sim" if mode=="simultaneous" else "seq"
        rv=t("reheat_val_"+key).format(T=T_air,d=float(ts["t_reheat"]))
        rn=t("reheat_note_"+key)
    else:
        rv=t("reheat_off"); rn=t("st_none")
    caution=RISK_COL["CAUTION"]
    cells=[  # label, value, note, value colour, note colour
        (t("ro_di"), f"{DI:.3f}", t("st_limit"), _rcol(DI), None),
        (t("ro_cycle"), f"{t_tot:.0f}<small>{t('min')}</small>",
         t("st_within") if t_tot<=30 else t("st_over").format(d=t_tot-30),
         None, None if t_tot<=30 else caution),
        (t("ro_biot"), f"{Bi:.2f}", t("st_uniform") if Bi<=0.5 else t("st_uneven"),
         None, None if Bi<=0.5 else caution),
        (t("ro_heal"), f"{heal:+.3f}" if spans else "–",
         t("st_effective") if heal>0.02 else (t("st_minimal") if spans else t("st_none")),
         None, None),
        (t("ro_reheat"), rv, rn, None, None),
    ]
    out=[]
    for lbl,val,note,vc,nc in cells:
        vs=f" style='color:{vc}'" if vc else ""
        ns=f" style='color:{nc}'" if nc else ""
        out.append(f"<div class='ro'><div class='ro-l'>{lbl}</div><div class='ro-v'{vs}>{val}</div>"
                   f"<div class='ro-n'{ns}>{note}</div></div>")
    return "<div class='readouts'>"+"".join(out)+"</div>"

def _src_label(src):
    """Plain description of the engine that answered."""
    src = src or ""
    if src.startswith(("groq:", "ollama:")):
        eng, model = src.split(":", 1)
        return f"{eng.capitalize()}, {model}"
    return t("rule_based")

def _hero_html(DI, ts, params, heal):
    risk=_rlbl(DI).lower()
    vkey="safe_margin" if risk=="safe" and DI>=0.15 else risk
    return (f"<div class='mast'><div class='mast-brand'>{MARK_SVG}<span class='wm'>PI-DRL</span>"
            f"<span class='ds'>{t('brand_desc')}</span></div>"
            f"<div class='mast-client'>CBIC × TUAT</div></div>"
            f"<div class='hero-k'>{t('outlook')}"
            f"<span class='chip' style='color:{_rcol(DI)}'>{_rname(DI)}</span></div>"
            f"<div class='verdict' role='heading' aria-level='1'>{t('v_'+vkey)}</div>"
            f"<p class='verdict-sub'>{t('v_detail').format(di=DI)} "
            f"<span>{t('v_'+vkey+'_hint')}</span></p>"
            f"<p class='live'>{t('live_hint')}</p>"
            + _ribbon_html(ts) + _readouts_html(ts, params, heal))

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
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=BIZ+UDPGothic:wght@400;700&display=swap');
/* PI-DRL design system, "cold room": the chrome is cool and monochrome; the
   only warm colour on screen is a real temperature. */
:root{
  --night:#111926; --deck:#172131; --deck2:#1F2A3C;
  --frost:#E8EDF4; --frost2:#B0BACB; --frost3:#7E8AA0;
  --rule:rgba(205,219,240,.10); --rule2:rgba(205,219,240,.18);
  --safe:#7CCBA6; --caution:#E9C46A; --warning:#EF9E59; --critical:#E2573C;
  --heat:#DD6A3A; --cool:#5F9BD3;
  --font:Archivo,"BIZ UDPGothic","Helvetica Neue",Arial,sans-serif;
}
html,body,.stApp{color:var(--frost);font-family:var(--font)!important;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.stApp{background:
  url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='g'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .045 0'/></filter><rect width='180' height='180' filter='url(%23g)'/></svg>"),
  radial-gradient(1100px 520px at 74% -12%, rgba(221,106,58,.10), transparent 62%),
  radial-gradient(900px 640px at -6% 110%, rgba(95,155,211,.08), transparent 60%),
  #111926!important;background-attachment:fixed!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stDecoration"]{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{max-width:1440px!important;
  padding:2.2rem 3rem 5rem!important}
.stApp p,.stApp li,.stApp label,.stApp input,.stApp textarea,.stApp button,
.stApp td,.stApp th{font-family:var(--font)!important}
[data-testid="stMarkdownContainer"] p{color:var(--frost2);line-height:1.6}
.stMarkdown h4{font-family:var(--font)!important;font-stretch:115%;font-weight:600!important;
  font-size:1.05rem!important;color:var(--frost)!important;letter-spacing:-.005em}
*{scrollbar-width:thin;scrollbar-color:#2a3650 transparent}
::selection{background:rgba(232,237,244,.22)}
:focus-visible{outline:2px solid var(--frost)!important;outline-offset:2px!important}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--deck)!important;border-right:1px solid var(--rule)!important}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:.25rem}
[data-testid="stSidebar"] label p{font-size:.84rem!important;color:var(--frost2)!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{font-size:.78rem!important;line-height:1.5}
.sb-head{padding:.4rem 0 .9rem}
.sb-brand,.mast-brand{display:flex;align-items:center;gap:.55rem}
.mark{width:14px;height:21px;flex-shrink:0}
.sb-brand .wm,.mast-brand .wm{font-stretch:125%;font-weight:700;letter-spacing:.02em;
  color:var(--frost);font-size:.98rem}
.sb-brand .ds,.mast-brand .ds{color:var(--frost3);font-size:.8rem}
.sb-title{font-stretch:118%;font-weight:600;font-size:1.4rem;color:var(--frost);
  margin:1rem 0 .15rem;letter-spacing:-.01em}
.sb-sub{color:var(--frost3);font-size:.8rem}
.step{display:flex;align-items:baseline;gap:.7rem;margin:1.4rem 0 .5rem;padding-top:1.05rem;
  border-top:1px solid var(--rule)}
.step b{color:var(--frost3);font-weight:500;font-size:.8rem;font-variant-numeric:tabular-nums;
  min-width:.9rem}
.step span{color:var(--frost);font-stretch:112%;font-weight:600;font-size:.98rem}
.sb-status{margin:.9rem 0 .2rem;display:grid;gap:.4rem}
.sb-status div{display:flex;align-items:center;gap:.55rem;color:var(--frost2);font-size:.8rem}
.sb-status i{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.drl-rep{margin-top:.9rem;padding:.85rem .9rem;background:var(--deck2);border-radius:10px}
.drl-rep p{color:var(--frost)!important;font-size:.84rem;margin:0 0 .6rem;line-height:1.5}
.drl-cap{color:var(--frost3);font-size:.75rem;margin-bottom:.3rem}
.drl-rep table{width:100%;border-collapse:collapse;font-size:.8rem;font-variant-numeric:tabular-nums}
.drl-rep td{padding:.28rem 0;color:var(--frost2);border-top:1px solid var(--rule)}
.drl-rep td+td{text-align:right}
.drl-rep tr.cur td{color:var(--frost);font-weight:600}

/* Buttons: ghost by default, solid frost for the one primary action */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
  font-family:var(--font)!important;font-weight:550!important;
  border-radius:999px!important;min-height:2.4rem;padding:.4rem 1.05rem!important;
  background:transparent!important;color:var(--frost)!important;
  border:1px solid var(--rule2)!important;box-shadow:none!important;
  transition:background .16s ease,border-color .16s ease,color .16s ease!important}
.stButton>button p,.stDownloadButton>button p,.stFormSubmitButton>button p{font-size:.86rem!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--deck2)!important;
  border-color:var(--frost3)!important}
.stButton>button[kind="primary"],button[data-testid^="stBaseButton-primary"],
.stFormSubmitButton>button{background:var(--frost)!important;color:var(--night)!important;
  border-color:var(--frost)!important;font-weight:600!important}
.stButton>button[kind="primary"] p,button[data-testid^="stBaseButton-primary"] p,
.stFormSubmitButton>button p{color:var(--night)!important}
.stButton>button[kind="primary"]:hover,button[data-testid^="stBaseButton-primary"]:hover,
.stFormSubmitButton>button:hover{background:#FFFFFF!important;border-color:#FFFFFF!important}

/* Inputs */
.stNumberInput input,.stTextInput input,.stTextArea textarea{background:var(--deck2)!important;
  color:var(--frost)!important;font-variant-numeric:tabular-nums}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div{
  background:var(--deck2)!important;border-color:var(--rule)!important;border-radius:8px!important}
.stNumberInput button{background:var(--deck2)!important;color:var(--frost2)!important;border:none!important}
.stNumberInput button:hover{color:var(--frost)!important}
[data-testid="stSliderThumbValue"],[data-testid="stThumbValue"]{color:var(--frost)!important;
  font-family:var(--font)!important;font-variant-numeric:tabular-nums;background:none!important}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"],
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:var(--frost3)!important}

/* Tabs: quiet text tabs with a frost underline */
.stTabs [data-baseweb="tab-list"]{gap:2.2rem!important;background:transparent!important;
  border-bottom:1px solid var(--rule)!important;padding:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;padding:.85rem 0!important;
  border-radius:0!important;color:var(--frost3)!important}
.stTabs [data-baseweb="tab"] p{font-size:.98rem!important;font-stretch:112%;font-weight:550!important}
.stTabs [data-baseweb="tab"]:hover{color:var(--frost2)!important}
.stTabs [aria-selected="true"]{color:var(--frost)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--frost)!important;height:2px!important}
.stTabs [data-baseweb="tab-border"]{display:none!important}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.4rem!important}

/* Alerts, expanders, captions, charts */
[data-testid="stAlert"],[data-testid="stAlertContainer"]{background:var(--deck)!important;
  border:1px solid var(--rule)!important;border-radius:10px!important;color:var(--frost2)!important}
[data-testid="stExpander"] details{border:1px solid var(--rule)!important;border-radius:10px!important;
  background:transparent!important}
[data-testid="stExpander"] summary{color:var(--frost2)!important}
[data-testid="stExpander"] summary:hover{color:var(--frost)!important}
[data-testid="stCaptionContainer"],.stCaption{color:var(--frost3)!important}
[data-testid="stPlotlyChart"]{background:transparent!important}
.modebar{background:transparent!important}
hr{border-color:var(--rule)!important}

/* Masthead + hero */
.mast{display:flex;justify-content:space-between;align-items:center;margin-bottom:2.6rem}
.mast-client{color:var(--frost3);font-size:.85rem}
.hero-k{display:flex;align-items:center;gap:.8rem;color:var(--frost3);font-size:.9rem}
.chip{display:inline-flex;align-items:center;gap:.4rem;padding:.16rem .62rem .18rem;
  border:1px solid currentColor;border-radius:999px;font-size:.78rem;font-weight:600}
.chip::before{content:"";width:.42rem;height:.42rem;border-radius:50%;background:currentColor}
.verdict{font-stretch:118%;font-weight:600;font-size:clamp(2.1rem,3.7vw,3.4rem);line-height:1.04;
  letter-spacing:-.018em;color:var(--frost);margin:.55rem 0 .8rem;max-width:20ch}
.verdict-sub{color:var(--frost2)!important;font-size:1.04rem;line-height:1.6;max-width:64ch;margin:0}
.verdict-sub span{color:var(--frost3)}
.live{color:var(--frost3)!important;font-size:.8rem;margin:.5rem 0 0}

/* Thermal ribbon, the signature element: surface temperature as colour */
.ribbon{margin:2.4rem 0 0}
.rb-cap{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;
  color:var(--frost3);font-size:.8rem;margin-bottom:.7rem}
.rb-scale{display:inline-flex;align-items:center;gap:.5rem;font-variant-numeric:tabular-nums}
.rb-scale i{display:inline-block;width:88px;height:6px;border-radius:3px}
.rb-heat{position:relative;height:8px;margin-bottom:5px}
.rb-heat i{position:absolute;bottom:0;height:4px;border-radius:2px;background:var(--heat);
  box-shadow:0 0 10px rgba(221,106,58,.65)}
.rb-band{height:30px;border-radius:7px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.22),inset 0 -10px 18px rgba(0,0,0,.16);
  animation:pour 1.1s cubic-bezier(.2,.7,.1,1) both}
@keyframes pour{from{clip-path:inset(0 100% 0 0 round 7px)}to{clip-path:inset(0 0 0 0 round 7px)}}
.rb-zones{position:relative;height:40px;margin-top:8px}
.rb-zones>div{position:absolute;top:0;height:100%;padding-left:7px;border-left:1px solid var(--rule2);
  overflow:hidden;white-space:nowrap}
.rb-zones span{display:block;color:var(--frost3);font-size:.74rem;overflow:hidden;text-overflow:ellipsis}
.rb-zones b{display:block;color:var(--frost2);font-size:.82rem;font-weight:550;
  font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis}
.rb-axis{display:flex;justify-content:space-between;color:var(--frost3);font-size:.74rem;
  margin-top:.3rem;font-variant-numeric:tabular-nums}

/* Read-outs: one instrument line, not cards */
.readouts{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));margin:2rem 0 2.6rem}
.ro{padding:.15rem 1.2rem .1rem}
.ro:first-child{padding-left:0}
.ro+.ro{border-left:1px solid var(--rule)}
.ro-l{color:var(--frost3);font-size:.8rem}
.ro-v{color:var(--frost);font-size:1.55rem;font-weight:600;font-stretch:108%;
  font-variant-numeric:tabular-nums;margin:.25rem 0 .15rem;line-height:1.1;white-space:nowrap}
.ro-v small{font-size:.9rem;font-weight:500;color:var(--frost2);margin-left:.2rem}
.ro-n{font-size:.78rem;color:var(--frost3)}
.sf{color:var(--frost2)!important;font-size:.98rem;line-height:1.6;max-width:72ch;margin:0 0 1.1rem}

/* Surface integrity note */
.note{display:flex;gap:.6rem 1.2rem;align-items:baseline;flex-wrap:wrap;padding:.9rem 0 .2rem;
  border-top:1px solid var(--rule);color:var(--frost2);font-size:.92rem}
.note b{color:var(--frost);font-weight:600}
.note .ok{color:var(--safe);font-weight:600}

/* Report */
.rp-rec{padding:0 0 .2rem 1.1rem;border-left:2px solid var(--risk,var(--frost3));margin-bottom:2.2rem}
.rp-rec p{color:var(--frost)!important;font-size:1.12rem;line-height:1.6;max-width:70ch;margin:.4rem 0 0}
.rp-k{color:var(--frost3);font-size:.82rem;margin-bottom:.6rem}
.rp-k span{color:var(--frost2);font-variant-numeric:tabular-nums;margin-left:.4rem}
.rp-gap{margin-top:1.8rem}
.rp-checks{list-style:none;margin:0;padding:0}
.rp-checks li{display:grid;grid-template-columns:1.3rem 1fr auto;gap:.5rem;align-items:baseline;
  padding:.62rem 0;border-top:1px solid var(--rule);color:var(--frost2);font-size:.92rem}
.rp-checks li i{font-style:normal;font-weight:700}
.rp-checks li.ok i{color:var(--safe)}
.rp-checks li.no i{color:var(--critical)}
.rp-checks li b{color:var(--frost);font-weight:550;font-variant-numeric:tabular-nums}
.rp-table{width:100%;border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
.rp-table th{text-align:left;color:var(--frost3);font-weight:500;padding:.4rem .6rem .5rem 0;
  border-bottom:1px solid var(--rule2)}
.rp-table td{padding:.55rem .6rem .55rem 0;border-bottom:1px solid var(--rule);color:var(--frost2)}
.rp-table td:first-child{color:var(--frost)}
.rp-table td i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:.55rem}
.rp-dl{display:grid;grid-template-columns:max-content 1fr;gap:.45rem 1.4rem;margin:0;font-size:.9rem}
.rp-dl dt{color:var(--frost3)}
.rp-dl dd{margin:0;color:var(--frost)}
.rp-muted{color:var(--frost3)!important}

/* Advisor */
.status{display:inline-flex;align-items:center;gap:.5rem;color:var(--frost2);font-size:.84rem;
  margin-bottom:.4rem}
.status i{width:7px;height:7px;border-radius:50%;background:var(--frost3)}
.status i.on{background:var(--safe)}
.h-sec{font-stretch:115%;font-weight:600;font-size:1.12rem;color:var(--frost);margin:2rem 0 .3rem}
.src{color:var(--frost3);font-size:.76rem;margin:.2rem 0 .9rem}
.analysis{max-width:78ch}
.an-h{color:var(--frost);font-weight:600;font-size:.95rem;margin:1.1rem 0 .3rem}
.an-b{color:var(--frost2);font-size:.95rem;line-height:1.7}
.thread{max-height:520px;overflow-y:auto;padding:.2rem .4rem .2rem 0;margin:.6rem 0 1rem}
.msg{margin:1.1rem 0}
.msg-u{display:flex;justify-content:flex-end}
.msg-u>div{max-width:72%;background:var(--deck2);color:var(--frost);padding:.7rem 1rem;
  border-radius:16px 16px 4px 16px;font-size:.93rem;line-height:1.6}
.msg-a{max-width:82%;padding-left:1rem;border-left:1px solid var(--rule2)}
.msg-a .who{display:flex;align-items:center;gap:.45rem;color:var(--frost3);font-size:.76rem;
  margin-bottom:.3rem}
.msg-a .who .mark{width:9px;height:14px}
.msg-a .body{color:var(--frost2);font-size:.94rem;line-height:1.7}
.msg-a .src{margin:.45rem 0 0}
.caret{display:inline-block;width:.45rem;height:1em;background:var(--frost2);vertical-align:-2px;
  margin-left:2px;animation:blink 1s steps(2) infinite}
.typing{display:inline-flex;gap:5px;padding:.4rem 0}
.typing i{width:6px;height:6px;border-radius:50%;background:var(--frost2);animation:typing 1.2s infinite}
.typing i:nth-child(2){animation-delay:.15s}
.typing i:nth-child(3){animation-delay:.3s}
.empty{color:var(--frost3);font-size:.9rem;padding:1.2rem 0;border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule);margin:.6rem 0 1rem}
.fu-k{color:var(--frost3);font-size:.8rem;margin:.2rem 0 .4rem}
.mem{color:var(--frost3);font-size:.78rem;padding-top:.55rem}

@keyframes blink{50%{opacity:0}}
@keyframes typing{0%,60%,100%{opacity:.25;transform:none}30%{opacity:1;transform:translateY(-3px)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulseGlow{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes typingBounce{0%,60%,100%{opacity:.4}30%{opacity:1}}

@media (max-width:900px){
  [data-testid="stMainBlockContainer"],.block-container{padding:1.2rem 1rem 4rem!important}
  .readouts{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:1.2rem}
  .ro{padding-left:0!important;border-left:none!important}
  .mast{margin-bottom:1.6rem}
  .rb-zones span{display:none}
  .msg-u>div,.msg-a{max-width:100%}
}
@media (prefers-reduced-motion:reduce){
  .rb-band,.caret,.typing i{animation:none!important}
}
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
        "reheat": (params["reheat"]["T"], params["reheat"]["duration"],
                   params["reheat"].get("mode","sequential"),
                   params["reheat"].get("pulses",0), params["reheat"].get("pulse_sec",0),
                   params["reheat"].get("pulse_window",0)),
        "h_cool": params["h_cool"], "h_reheat": params["h_reheat"],
        "melt": params.get("melt"), "late": params.get("late_cool_T"),
    }, sort_keys=True)
    _hash = hashlib.md5(_hs.encode()).hexdigest()[:8]
    if _hash != st.session_state.last_param_hash:
        with st.spinner(""):
            _ts = build_timeline(params["T_fill"],params["zones"],params["reheat"],
                                 params["h_cool"],params["h_reheat"],n_pts=120,
                                 melt=params.get("melt"),late_cool_T=params.get("late_cool_T"))
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

    # ─── HERO ─────────────────────────────────────────────────────────────────
    # The run recomputes automatically whenever the setup changes (hash above),
    # so the hero always shows the current recipe; no separate Compute step.
    if ats:
        st.markdown(_hero_html(DI_live, ats, params, heal_live), unsafe_allow_html=True)

    # ─── TABS ─────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        t("tab_belt"), t("tab_temp"), t("tab_crack"),
        t("tab_results"), t("tab_advisor"),
    ])
    _cfg3d = {"displayModeBar":True,"displaylogo":False,"scrollZoom":True,
              "modeBarButtonsToRemove":["pan3d","tableRotation"]}

    # ══ TAB 1: Production line ═══════════════════════════════════════════════
    with tab1:
        st.markdown(f"<p class='sf'>{t('sf_belt')}</p>", unsafe_allow_html=True)
        ts_b = ats if ats else build_timeline(params["T_fill"],params["zones"],
            params["reheat"],params["h_cool"],params["h_reheat"],n_pts=60,
            melt=params.get("melt"),late_cool_T=params.get("late_cool_T"))
        belt = fig_belt(ts_b,params["zones"],params["reheat"],
                        params["h_cool"],params["h_reheat"],n_frames=40)
        # Stable key + figure uirevision => camera rotation/zoom is preserved
        # across timeline drags AND Streamlit reruns (no more snap-to-default).
        st.plotly_chart(belt, width='stretch', key="belt3d", config=_cfg3d)
        export_figure_panel(belt, "uturn_belt", "exp_belt")

    # ══ TAB 2: Thermal history ═══════════════════════════════════════════════
    with tab2:
        st.markdown(f"<p class='sf'>{t('sf_temp')}</p>", unsafe_allow_html=True)
        if ats:
            ts_use=ats
        else:
            with st.spinner(""):
                ts_use=build_timeline(params["T_fill"],params["zones"],params["reheat"],
                                      params["h_cool"],params["h_reheat"],n_pts=120,
                                      melt=params.get("melt"),late_cool_T=params.get("late_cool_T"))
        charts=fig_charts(ts_use,params.get("_scenario",""))
        st.plotly_chart(charts,width='stretch',key="charts2d",config={"displaylogo":False})
        export_figure_panel(charts, "temperature_DI", "exp_temp")

    # ══ TAB 3: Surface integrity ═════════════════════════════════════════════
    with tab3:
        st.markdown(f"<p class='sf'>{t('sf_crack')}</p>", unsafe_allow_html=True)
        ts_cr = ats if ats else ts_use
        ta_   = np.array(ts_cr["times"]); DI_   = np.array(ts_cr["DI"])
        spans = _reheat_spans(ts_cr)
        has_reheat = bool(spans)
        # "Most damaged" = peak damage formed up to the end of the reheat;
        # "After reheat" = the healed residual at the end of the last hot-air
        # span. A well-healed (e.g. DRL-optimised) recipe renders crack-free.
        ie_reh = int(np.argmin(np.abs(ta_ - spans[-1][1]))) if has_reheat else len(DI_)-1
        ib     = int(np.argmax(DI_[:max(1, ie_reh+1)]))
        ia     = ie_reh if has_reheat else len(DI_)-1
        DIb    = float(DI_[ib]); DIa=float(DI_[ia])
        tb_    = float(ta_[ib]); ta2_=float(ta_[ia])
        hdrop  = DIb-DIa
        CRACK_TH = 0.25   # cracks show from Caution up, matching the risk bands

        def crack3d(DI_v, ttl):
            H_c=H_M*100; R_c=R_M*100
            n_th=72; n_zl=36
            th_s=np.linspace(0,2*np.pi,n_th); z_s=np.linspace(0,H_c,n_zl)
            TH_,ZS_=np.meshgrid(th_s,z_s); z_norm=ZS_/H_c
            has_crack = DI_v >= CRACK_TH
            rng_sf=np.random.default_rng(int(DI_v*1000)%9999)
            # Damage field φ on the side wall, concentrated near the top
            # fill-point and scaled by the damage index.
            n_hot=max(1,int(DI_v*6))
            hot_th=rng_sf.uniform(0,2*np.pi,n_hot); hot_z=rng_sf.uniform(0.78,1.0,n_hot)
            phi_sf=DI_v*np.clip(z_norm-0.45,0,1)*1.7
            for ha,hz in zip(hot_th,hot_z):
                d_th=np.abs(np.arctan2(np.sin(TH_-ha),np.cos(TH_-ha)))
                d_z=np.abs(z_norm-hz)
                phi_sf=phi_sf+DI_v*1.15*np.exp(-(d_th**2/0.32+d_z**2/0.055))
            phi_sf=np.clip(phi_sf,0,1)
            lit=dict(ambient=0.55,diffuse=0.8,specular=0.3,roughness=0.45,fresnel=0.2)
            f3=go.Figure()
            f3.add_trace(go.Surface(x=R_c*np.cos(TH_),y=R_c*np.sin(TH_),z=ZS_,
                surfacecolor=phi_sf,colorscale=DMG_SCALE,cmin=0,cmax=1,showscale=True,
                colorbar=dict(title=dict(text=t("damage"),font=dict(size=11,color=INK["frost3"])),
                              thickness=8,len=0.5,x=0.98,outlinewidth=0,
                              tickfont=dict(size=10,color=INK["frost3"])),
                lighting=lit,lightposition=dict(x=60,y=-120,z=400),opacity=0.95,
                hoverinfo="skip"))
            # Top surface (天面): damage gathers at the fill point in the centre
            rr=np.linspace(0,R_c,14); TT=np.linspace(0,2*np.pi,n_th)
            RR,TTg=np.meshgrid(rr,TT)
            cap=np.clip(float(phi_sf[-1].mean())*(0.55+0.45*(1-RR/R_c))
                        +DI_v*0.35*(1-RR/R_c),0,1)
            f3.add_trace(go.Surface(x=RR*np.cos(TTg),y=RR*np.sin(TTg),z=np.full_like(RR,H_c),
                surfacecolor=cap,colorscale=DMG_SCALE,cmin=0,cmax=1,showscale=False,
                lighting=lit,lightposition=dict(x=60,y=-120,z=400),opacity=0.98,
                hoverinfo="skip"))
            if has_crack:
                # Crack density & extent scale with how far above the threshold
                sev=float(np.clip((DI_v-CRACK_TH)/(1-CRACK_TH),0,1))
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
                cc=_crack_color(DI_v); hl=_lighten_color(cc,0.5)
                w=int(np.clip(5+sev*14,5,19))
                for col_,wd,op in ((_rgba(cc,0.35),w+6,1.0),(cc,w,0.98),(hl,max(1,w-6),0.8)):
                    f3.add_trace(go.Scatter3d(x=xm,y=ym,z=zm,mode="lines",opacity=op,
                        line=dict(color=col_,width=wd),showlegend=False,hoverinfo="skip"))
                f3.add_trace(go.Scatter3d(x=nxl,y=nyl,z=nzl,mode="markers",
                    marker=dict(size=5,color=INK["frost"],line=dict(color=cc,width=2)),
                    showlegend=False,hoverinfo="skip"))
            else:
                f3.add_trace(go.Scatter3d(x=[0],y=[0],z=[H_c*1.12],mode="markers+text",
                    marker=dict(size=8,color=RISK_COL["SAFE"]),
                    text=[t("crack_free")],textposition="top center",
                    textfont=dict(size=12,color=RISK_COL["SAFE"],family=FONT),
                    showlegend=False,hoverinfo="skip"))
            f3.update_layout(height=480,showlegend=False,uirevision="crack",
                title=dict(text=(f"{ttl}<br><span style='font-size:12px;color:{_rcol(DI_v)}'>"
                                 f"{_rname(DI_v)}, {t('di_short')} {DI_v:.3f}</span>"),
                           x=0,xanchor="left",y=0.97,font=dict(size=14,color=INK["frost"])),
                margin=dict(l=0,r=0,t=64,b=0),
                scene=dict(uirevision="crack",xaxis=dict(visible=False),
                           yaxis=dict(visible=False),zaxis=dict(visible=False),
                           aspectmode="data",bgcolor="rgba(0,0,0,0)",
                           camera=dict(eye=dict(x=1.45,y=-1.45,z=1.0),
                                       center=dict(x=0,y=0,z=-0.05))))
            return f3

        lbl_b=f"{t('before_reheat')}, {tb_:.1f} {t('min')}"
        lbl_a=f"{t('after_reheat') if has_reheat else t('end_cycle')}, {ta2_:.1f} {t('min')}"
        _fig_before = crack3d(DIb, lbl_b)
        _fig_after  = crack3d(DIa, lbl_a)
        cl,cr = st.columns(2, gap="large")
        cl.plotly_chart(_fig_before,width='stretch',key="crack_before",config=_cfg3d)
        cr.plotly_chart(_fig_after,width='stretch',key="crack_after",config=_cfg3d)

        if hdrop>0.005:
            eff=int(hdrop/max(DIb,0.01)*100)
            extra=(f"<span class='ok'>{t('crack_free')}</span>"
                   if DIa < CRACK_TH <= DIb else "")
            st.markdown(
                f"<div class='note'><b>{t('healed_by')} {hdrop:.3f}</b>"
                f"<span>{DIb:.3f} → {DIa:.3f}</span>"
                f"<span>{t('healing_eff')} {eff}%</span>"
                f"<span>{t('optimal_healing') if params['reheat']['T']>=62 else t('softening_only')}</span>"
                f"{extra}</div>", unsafe_allow_html=True)
        elif not has_reheat:
            st.info(t("no_reheat"))
        else:
            st.warning(t("min_healing"))
        export_figure_panel({t("before_reheat"): _fig_before,
                             (t("after_reheat") if has_reheat else t("end_cycle")): _fig_after},
                            "crack_propagation", "exp_crack")

    # ══ TAB 4: Report ════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"<p class='sf'>{t('sf_report')}</p>", unsafe_allow_html=True)
        if ats:
            show_results(ats,params)
        else:
            st.info(t("run_first"))

    # ══ TAB 5: AI Advisor ═══════════════════════════════════════════════════════
    with tab5:
        st.markdown(f"<p class='sf'>{t('sf_advisor')}</p>", unsafe_allow_html=True)

        # ── Load advisor (cached, but version-keyed so retraining doesn't
        #    need a page reload — bumping advisor_version creates a fresh
        #    cache entry while _load_advisor.clear() drops the old one) ──
        if "advisor_version" not in st.session_state:
            st.session_state.advisor_version = 0

        # Deploy-safety: @st.cache_resource persists the advisor INSTANCE across
        # reruns, and Streamlit re-execs this entry script fresh but does NOT
        # re-import already-loaded modules. So after a git deploy without a full
        # process restart, a stale advisor (old llm_advisor_v4 in sys.modules)
        # would linger — causing "chat_stream() got an unexpected kwarg" when
        # new main calls it with new params. Fix: key the cache on the advisor
        # file's mtime AND importlib.reload the module, so any deploy rebuilds a
        # fresh instance from current code automatically.
        try:
            import os as _os
            _adv_mtime = _os.path.getmtime(
                _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "llm_advisor_v4.py"))
        except Exception:
            _adv_mtime = 0.0

        @st.cache_resource(show_spinner=False)
        def _load_advisor(_version: int, _mtime: float):
            try:
                import importlib
                import llm_advisor_v4
                importlib.reload(llm_advisor_v4)   # pick up code from this deploy
                return llm_advisor_v4.PILLMAdvisor()
            except ImportError:
                return None

        advisor = _load_advisor(st.session_state.advisor_version, _adv_mtime)

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
                # Clean, client-facing status line — no technical jargon.
                _online = bool(_llm_up)
                st.markdown(f"<div class='status'><i class='{'on' if _online else ''}'></i>"
                            f"{t('adv_online') if _online else t('adv_ready')}</div>",
                            unsafe_allow_html=True)

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

                # Analysis of this run, grounded in the client data
                col_h1, col_h2 = st.columns([4, 1], vertical_alignment="bottom")
                col_h1.markdown(f"<div class='h-sec'>{t('detailed_analysis')}</div>",
                                unsafe_allow_html=True)
                analysis_key = f"llm_analysis_{id(ats)}_{DI_pk:.4f}"
                force_retry = col_h2.button(t("retry"), key="llm_retry_analysis",
                                            width='stretch')
                if force_retry and analysis_key in st.session_state:
                    del st.session_state[analysis_key]
                if analysis_key not in st.session_state:
                    with st.spinner(""):
                        st.session_state[analysis_key] = advisor.analyze(sim_ctx)
                        st.session_state[f"{analysis_key}_source"] = advisor.status()["last_source"]
                analysis_text = st.session_state[analysis_key]
                analysis_source = st.session_state.get(f"{analysis_key}_source", "unknown")
                st.markdown(f"<div class='src'>{t('source')}: {_src_label(analysis_source)}</div>",
                            unsafe_allow_html=True)
                # Section labels however the LLM marked them ("## ASSESSMENT",
                # "**ASSESSMENT:**", "ASSESSMENT:" …), shown in sentence case.
                sections = {"ASSESSMENT":"sec_assessment","PHYSICAL MEANING":"sec_physical",
                            "ROOT CAUSE":"sec_root","RECOMMENDATION":"sec_reco",
                            "COMPARISON":"sec_compare"}
                html_parts = ["<div class='analysis'>"]
                for raw in analysis_text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    clean   = re.sub(r"^[#*\s]+", "", line).rstrip(":*# ").strip()
                    clean_u = clean.upper()
                    sec, body = None, ""
                    for _s in sections:
                        if clean_u == _s or clean_u.startswith(_s + ":") or clean_u.startswith(_s + " "):
                            sec  = _s
                            body = clean[len(_s):].lstrip(":  ").strip()
                            break
                    if sec is not None:
                        html_parts.append(f"<div class='an-h'>{t(sections[sec])}</div>")
                        if body:
                            html_parts.append(f"<div class='an-b'>{_md_to_html(body)}</div>")
                    else:
                        html_parts.append(f"<div class='an-b'>{_md_to_html(line)}</div>")
                html_parts.append("</div>")
                st.markdown("".join(html_parts), unsafe_allow_html=True)

            else:
                st.info(t("run_first"))

            # ── Chat ────────────────────────────────────────────────────────
            st.markdown(f"<div class='h-sec'>{t('ask_advisor')}</div>"
                        f"<p class='sf'>{t('ask_hint')}</p>", unsafe_allow_html=True)

            def _bubble_html(role, content, source_label=None, cursor=False):
                """One message. User: right-aligned pill. Advisor: text on a
                hairline rule with its name and, once finished, its source."""
                if role == "user":
                    return f"<div class='msg msg-u'><div>{_md_to_html(content)}</div></div>"
                cur = "<span class='caret'></span>" if cursor else ""
                src = (f"<div class='src'>{t('source')}: {_src_label(source_label)}</div>"
                       if source_label else "")
                return (f"<div class='msg msg-a'><div class='who'>{MARK_SVG}{t('advisor')}</div>"
                        f"<div class='body'>{_md_to_html(content)}{cur}</div>{src}</div>")

            def _thinking_bubble(label):
                return (f"<div class='msg msg-a'><div class='who'>{MARK_SVG}{t('advisor')}</div>"
                        f"<div class='typing'><i></i><i></i><i></i></div>"
                        f"<div class='src'>{label}</div></div>")

            # Source of truth is the PER-SESSION list (not the shared advisor),
            # so each visitor only ever sees their own thread.
            history_pairs = []
            hist = st.session_state.chat_hist
            for i in range(0, len(hist) - (len(hist) % 2), 2):
                history_pairs.append((hist[i], hist[i+1] if i+1 < len(hist) else None))

            if history_pairs:
                thread = ["<div class='thread'>"]
                for user_msg, asst_msg in history_pairs:
                    thread.append(_bubble_html("user", user_msg["content"]))
                    if asst_msg:
                        thread.append(_bubble_html("assistant", asst_msg["content"]))
                thread.append("</div>")
                st.markdown("".join(thread), unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='empty'>{t('chat_empty')}</div>", unsafe_allow_html=True)

            # Suggested questions — clicking SENDS immediately (sets pending_q).
            # Before any chat: starter questions. After: contextual follow-ups.
            if history_pairs:
                _lu = history_pairs[-1][0]["content"]
                _la = history_pairs[-1][1]["content"] if history_pairs[-1][1] else ""
                _fk = f"fups_{len(hist)}_{st.session_state.get('lang','en')}"
                if _fk not in st.session_state:
                    with st.spinner(""):
                        st.session_state[_fk] = advisor.suggest_followups(
                            _lu, _la, lang=st.session_state.get("lang", "en"))
                suggestions = st.session_state[_fk]
                st.markdown(f"<div class='fu-k'>{t('followups')}</div>", unsafe_allow_html=True)
            elif st.session_state.get("lang") == "ja":
                suggestions = ["なぜ上面でヒケが発生するのですか？",
                               "DIを0.10未満にする熱風温度は？",
                               "ビオ数はヒケにどう影響しますか？",
                               "生産ラインのデータと比較してください"]
            else:
                suggestions = ["Why do sink marks form on the top surface?",
                               "What hot-air temperature keeps DI below 0.10?",
                               "How does the Biot number affect sink marks?",
                               "How does this compare with the production line?"]
            s_cols = st.columns(len(suggestions) if suggestions else 1)
            for i, (col, sq) in enumerate(zip(s_cols, suggestions)):
                if col.button(sq[:34]+"…" if len(sq)>34 else sq, help=sq,
                              key=f"suggest_{len(hist)}_{i}", width='stretch'):
                    st.session_state["pending_q"] = sq
                    st.rerun()

            # Bind the box to its OWN session-state key (not a transient value=)
            # so the text survives the form-submit rerun.
            with st.form("chat_composer", clear_on_submit=True, border=False):
                cc1, cc2 = st.columns([8, 1.3], vertical_alignment="bottom")
                user_q = cc1.text_input("Your question:", key="llm_chatbox",
                                        placeholder=t("chat_ph"), label_visibility="collapsed")
                sent = cc2.form_submit_button(t("send"), type="primary", width='stretch')

            lc_on   = st_info.get("langchain") or _groq
            mc1, mc2 = st.columns([3, 1], vertical_alignment="center")
            mc1.markdown(f"<div class='mem'>{t('mem_on') if lc_on else t('mem_basic')}</div>",
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
                if st_info["ollama_up"] and not st_info["model_warm"] and not _groq:
                    think_lbl = ("モデルを読み込み中（初回のみ30〜90秒）…" if _ja
                                 else "Loading the model, first question only (30–90 s)…")
                else:
                    think_lbl = "考えています…" if _ja else "Thinking…"

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
