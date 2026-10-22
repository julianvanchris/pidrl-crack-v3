/* PI-DRL site: the recipe, every view of it, and the bridges. The physics runs here (engine.js), so a change
   recomputes at once. Python (main_3d_v4.py) sends the strings, the presets and the advisor's state as render
   args, and answers the advisor requests this page sends back as the component value. The 3D line
   (components/line_studio) is a child frame that this page hosts the way Streamlit would. */
(function () {
  "use strict";
  const E = window.PIDRL, CH = window.CHARTS, RISK = CH.RISK, esc = CH.esc;
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const SC = document.getElementById("scroller");          // the page's scroll container (see style.css)
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const MODE_KEY = { none: "mode_none", sequential: "mode_seq", simultaneous: "mode_sim", pulsed: "mode_pul" };
  const TARGET = 2;                                        // the default starting point: "Target", stepped cooling
  const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked: keep going */ } } };

  // ---------------------------------------------------------------- bridge to Streamlit (component protocol v1)
  const send = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
  let A = null;                                            // the latest render args from Python
  window.addEventListener("message", e => {
    const d = e.data || {}, st = $("#stage");
    if (st && e.source && e.source === st.contentWindow) { onStage(d); return; }
    if (d.type === "streamlit:render") { const first = !A; A = d.args || {}; if (first) boot(); else onArgs(); }
  });
  send("streamlit:componentReady", { apiVersion: 1 });
  const post = v => send("streamlit:setComponentValue", { value: v, dataType: "json" });

  // ---------------------------------------------------------------- strings
  let LANG = "en";
  const t = k => (A.tr[LANG] && A.tr[LANG][k]) || A.tr.en[k] || k;
  function fmt(s, o) {                                     // Python's str.format for {k}, {k:.2f} and {k:+.3f}
    return String(s).replace(/\{(\w+)(?::(\+)?\.(\d+)f)?\}/g, (m, k, plus, p) => {
      if (!(k in o)) return m;
      if (p == null) return String(o[k]);
      const v = +o[k], s2 = E.rnd(v, +p).toFixed(+p);
      return plus && v >= 0 ? "+" + s2 : s2;
    });
  }
  const tf = (k, o) => fmt(t(k), o);
  const scName = sc => sc.name[LANG] || sc.name.en, scDesc = sc => sc.desc[LANG] || sc.desc.en;
  const rname = d => t("risk_" + CH.riskOf(d).toLowerCase());
  const num = (v, d) => (Math.abs(v - Math.round(v)) < 1e-9 && d <= 1 ? String(Math.round(v)) : v.toFixed(d));

  // ---------------------------------------------------------------- state
  const S = { r: null, scen: TARGET, dirty: false, ts: null, key: "", rev: 0, scope: "all", opt: null, best: {},
    caseOpt: {}, pend: null, anAsked: {}, chartDrawn: false, seen: {} };
  const clone = o => JSON.parse(JSON.stringify(o));
  function fromScenario(sc) {
    const reh = clone(sc.reheat);
    if (reh.mode === "pulsed") {
      reh.duration = 0;
      if (!reh.bursts || !reh.bursts.length) {
        const tc = sc.zones.reduce((s, z) => s + z.duration, 0);
        reh.bursts = E.evenBursts(reh.pulses || 5, reh.pulse_sec || 20, Math.min(reh.pulse_window == null ? 6 : reh.pulse_window, tc));
      }
    }
    return { T_fill: sc.T_fill, zones: clone(sc.zones).map((z, i) => ({ T: z.T, duration: z.duration, label: "Zone " + (i + 1) })),
      reheat: reh, h_cool: sc.h_cool, h_reheat: sc.h_reheat, melt: sc.melt, late: sc.late };
  }
  function valid(r) {
    try {
      return r && isFinite(r.T_fill) && Array.isArray(r.zones) && r.zones.length >= 1 && r.zones.length <= 12 &&
        r.zones.every(z => isFinite(z.T) && isFinite(z.duration)) && r.reheat && isFinite(r.reheat.T) &&
        isFinite(r.h_cool) && isFinite(r.h_reheat) && isFinite(r.melt) && isFinite(r.late);
    } catch (e) { return false; }
  }

  // ---------------------------------------------------------------- the model, and everything that reads it
  let queued = false, firstRender = true;
  function changed(opts = {}) {                            // a setting moved: recompute on the next frame
    if (opts.dirty !== false) S.dirty = true;
    if (opts.rows) rowsSig = "";
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; compute(); renderAll(); save(); });
  }
  function compute() {
    const r = S.r;
    r.zones.forEach((z, i) => { z.label = "Zone " + (i + 1); });
    S.ts = E.buildTimeline(r.T_fill, r.zones, r.reheat, r.h_cool, r.h_reheat, 120, r.melt, r.late);
    S.key = JSON.stringify([r.T_fill, r.zones.map(z => [z.T, z.duration]), r.reheat, r.h_cool, r.h_reheat, r.melt, r.late]);
    S.rev++;
  }
  function save() { store.set("pidrl.site", { lang: LANG, r: S.r, scen: S.scen, dirty: S.dirty, scope: S.scope }); }

  let PRE = null;                                          // the presets' timelines, computed once
  function presets() {
    if (!PRE) PRE = A.scenarios.map((sc, i) => {
      const ts = E.buildTimeline(sc.T_fill, sc.zones, sc.reheat, sc.h_cool, sc.h_reheat, 120, sc.melt, sc.late);
      let seed = 0; for (const ch of sc.id) seed += ch.codePointAt(0);
      return { id: sc.id, i, sc, ts, h_cool: sc.h_cool, peak: Math.max(...ts.DI), cycle: ts.t_total, seed: seed % 9973 };
    });
    return PRE.map(o => Object.assign({}, o, { name: scName(o.sc) }));
  }
  function options(withBest) {                             // presets, plus the optimiser's best for each timing
    const out = presets();
    if (withBest) for (const [m, b] of Object.entries(S.best)) {
      const ts = E.buildTimeline(b.T_fill, b.zones, b.reh, b.h_cool, b.h_reh, 120, b.melt, b.late);
      out.push({ id: "drl_" + m, ts, h_cool: b.h_cool, name: tf("opt_drl", { mode: t(MODE_KEY[m]) }), peak: Math.max(...ts.DI), cycle: ts.t_total, seed: 7 });
    }
    return out;
  }
  function why(o) {
    const ts = o.ts, kind = ts.t_reheat > 0 ? ts.reheat_mode : "none", bi = o.h_cool * E.C.R_M / E.C.K_TH, parts = [t("why_" + kind)];
    if (bi > 0.5) parts.push(tf("why_biot", { b: bi }));
    if (o.cycle > 30.5) parts.push(tf("why_slow", { t: o.cycle }));
    if (o.peak >= 0.25) parts.push(tf("why_cracks", { di: o.peak }));
    return parts.join(" ");
  }

  // ---------------------------------------------------------------- loading recipes
  function loadPreset(i) {
    S.r = fromScenario(A.scenarios[i]); S.scen = i; S.dirty = false; S.opt = null; S.best = {};
    changed({ dirty: false, rows: true }); flash($("#live .live-in"));
  }
  function applyRecipe(c) {                                // an optimiser result: cooling, air flow and its reheat
    S.r.zones = clone(c.zones); S.r.reheat = clone(c.reh); S.r.h_cool = c.h_cool; S.r.h_reheat = c.h_reh;
    const reh = S.r.reheat;                                // a new burst pattern comes as a count: lay the bursts out
    if (reh.mode === "pulsed" && !(reh.bursts && reh.bursts.length))
      reh.bursts = E.evenBursts(reh.pulses || 5, reh.pulse_sec || 20, Math.min(reh.pulse_window == null ? 6 : reh.pulse_window, tCool()));
  }
  function atPage(b, r) {                                  // the search scores on a coarser timeline; report the page's own
    const ts = E.buildTimeline(r.T_fill, b.zones, b.reh, b.h_cool, b.h_reh, 120, r.melt, r.late);
    b.di = Math.max(...ts.DI); b.t = ts.t_total; return b;
  }
  function loadOption(id) {
    const i = A.scenarios.findIndex(sc => sc.id === id);
    if (i >= 0) { loadPreset(i); return; }
    if (id.startsWith("drl_") && S.best[id.slice(4)]) { applyRecipe(S.best[id.slice(4)]); changed({ rows: true }); }
  }

  // ---------------------------------------------------------------- 02 the recipe: controls
  const SL = {
    T_fill: { l: "s_l_fill", min: 75, max: 88, step: 0.5, u: "°C", get: r => r.T_fill, set: (r, v) => { r.T_fill = v; } },
    reh_T: { l: "s_l_rehT", min: 40, max: 120, step: 1, u: "°C", get: r => r.reheat.T, set: (r, v) => { r.reheat.T = v; } },
    reh_D: { l: r => (r.reheat.mode === "simultaneous" ? "s_l_rehW" : "s_l_rehD"), h: r => (r.reheat.mode === "simultaneous" ? "reheat_window_help" : "reheat_help"),
      min: 0, max: 15, step: 0.5, u: "min", get: r => r.reheat.duration, set: (r, v) => { r.reheat.duration = v; } },
    pn: { l: "s_l_pn", h: "pulse_count_help", min: 1, max: 12, step: 1, u: "", get: r => r.reheat.pulses, set: (r, v) => pattern("pulses", v) },
    ps: { l: "s_l_ps", h: "pulse_dur_help", min: 5, max: 60, step: 5, u: "s", get: r => r.reheat.pulse_sec, set: (r, v) => pattern("pulse_sec", v) },
    pw: { l: "s_l_pw", h: "pulse_win_help", min: 2, max: 15, step: 0.5, u: "min", get: r => r.reheat.pulse_window, set: (r, v) => pattern("pulse_window", v) },
    melt: { l: "s_l_melt", h: "melting_point_help", min: 40, max: 100, step: 1, u: "°C", get: r => r.melt, set: (r, v) => { r.melt = v; } },
    late: { l: "s_l_late", h: "late_cool_help", min: 5, max: 30, step: 1, u: "°C", get: r => r.late, set: (r, v) => { r.late = v; } },
    h_cool: { l: "s_l_hc", min: 2, max: 20, step: 0.5, u: "W/m²K", get: r => r.h_cool, set: (r, v) => { r.h_cool = v; } },
    h_reh: { l: "s_l_hr", min: 5, max: 40, step: 0.5, u: "W/m²K", get: r => r.h_reheat, set: (r, v) => { r.h_reheat = v; } },
  };
  const unit = u => (u === "min" ? t("min") : u === "s" ? t("s_sec") : u);
  const tCool = () => S.r.zones.reduce((s, z) => s + z.duration, 0);
  function pattern(k, v) {                                 // a pattern slider moved: spread the bursts evenly again
    const reh = S.r.reheat; reh[k] = v;
    reh.bursts = E.evenBursts(reh.pulses, reh.pulse_sec, Math.min(reh.pulse_window, tCool())); delete reh.custom;
    rowsSig = "";
  }
  function buildSliders() {
    $$(".sl[data-key]").forEach(el => {
      const k = el.dataset.key, s = SL[k];
      el.innerHTML = `<div class="sl-top"><label for="sl_${k}"></label><output for="sl_${k}"></output></div>` +
        `<input type="range" id="sl_${k}" min="${s.min}" max="${s.max}" step="${s.step}"><div class="sl-ends"><span>${s.min}</span><span>${s.max}</span></div>`;
      const inp = $("input", el);
      inp.addEventListener("input", () => { s.set(S.r, +inp.value); syncSlider(el); changed(); });
    });
  }
  function syncSlider(el) {
    const k = el.dataset.key, s = SL[k], inp = $("input", el), v = +s.get(S.r);
    const l = typeof s.l === "function" ? s.l(S.r) : s.l, h = typeof s.h === "function" ? s.h(S.r) : s.h;
    $("label", el).textContent = t(l);
    if (h) el.title = t(h); else el.removeAttribute("title");
    if (document.activeElement !== inp) inp.value = v;
    inp.style.setProperty("--p", ((v - s.min) / (s.max - s.min) * 100).toFixed(1) + "%");
    $("output", el).innerHTML = esc(num(v, s.step < 1 ? 1 : 0)) + (s.u ? `<small>${esc(unit(s.u))}</small>` : "");
  }
  function buildModes() {
    const H = CH.heat, pic = m => {
      const cool = [0, 1, 2].map(k => `<rect x="${4 + k * 34}" y="24" width="32" height="22" rx="4" fill="${H(64 - k * 13)}"/>`).join("");
      if (m === "sequential") return cool + `<rect x="108" y="24" width="48" height="22" rx="4" fill="${H(92)}"/><path d="M114 14h36" stroke="${H(92)}" stroke-width="2" stroke-dasharray="3 3"/>`;
      if (m === "simultaneous") return cool + `<rect x="108" y="24" width="48" height="22" rx="4" fill="${H(30)}"/><rect x="4" y="14" width="100" height="5" rx="2.5" fill="${H(92)}"/>`;
      return cool + `<rect x="108" y="24" width="48" height="22" rx="4" fill="${H(25)}"/>` + [0, 1, 2, 3, 4].map(k => `<rect x="${6 + k * 14}" y="12" width="7" height="8" rx="2" fill="${H(100)}"/>`).join("");
    };
    $("#modes").innerHTML = ["sequential", "simultaneous", "pulsed"].map(m =>
      `<button class="mode" data-m="${m}" aria-pressed="false"><svg viewBox="0 0 160 54" aria-hidden="true">${pic(m)}</svg><b></b><span></span></button>`).join("");
    $("#modes").addEventListener("click", e => {
      const b = e.target.closest(".mode"); if (!b) return;
      const reh = S.r.reheat, m = b.dataset.m;
      if (reh.mode === m || (!reh.mode && m === "sequential")) return;
      reh.mode = m;
      if (m === "pulsed") {
        reh.pulses = reh.pulses || 5; reh.pulse_sec = reh.pulse_sec || 20; reh.pulse_window = reh.pulse_window || 6; reh.duration = 0;
        if (!reh.bursts || !reh.bursts.length) reh.bursts = E.evenBursts(reh.pulses, reh.pulse_sec, Math.min(reh.pulse_window, tCool()));
      } else if (!(reh.duration > 0)) reh.duration = 10;
      changed({ rows: true });
    });
    $("#rehParams").innerHTML = `<div class="sl" data-key="reh_D"></div><div class="pul"><div class="grid2c"><div class="sl" data-key="pn"></div>` +
      `<div class="sl" data-key="ps"></div></div><div class="sl" data-key="pw"></div><div class="rp-k" id="bK"></div><div class="zrows" id="brows"></div>` +
      `<div class="row-btns"><button class="btn ghost sm" id="bAdd"></button><button class="btn ghost sm" id="bEven"></button></div></div>`;
  }
  function stepper(f, v, u, i, a) {
    return `<div class="step" data-f="${f}" data-i="${i}"><button type="button" data-d="-1" aria-label="${esc(a)} −">−</button>` +
      `<input inputmode="decimal" value="${v}" aria-label="${esc(a)}"><small>${esc(u)}</small><button type="button" data-d="1" aria-label="${esc(a)} +">+</button></div>`;
  }
  let rowsSig = "";
  function renderRows() {
    const r = S.r, reh = r.reheat, pul = reh.mode === "pulsed";
    const sig = LANG + "|" + r.zones.length + "|" + (pul ? reh.bursts.length : "-");
    if (sig !== rowsSig) {                                 // rebuild only when rows come or go; values update in place
      rowsSig = sig;
      $("#zrows").innerHTML = r.zones.map((z, i) => `<div class="zr"><b><i></i>${esc(tf("zone_n", { n: i + 1 }))}</b>` +
        stepper("T", z.T, "°C", i, tf("zone_T", { n: i + 1 })) + stepper("duration", z.duration, t("min"), i, t("zone_min")) +
        `<button type="button" class="x" data-i="${i}" aria-label="${esc(tf("remove_zone", { n: i + 1 }))}">×</button></div>`).join("");
      $("#brows").innerHTML = pul ? reh.bursts.map((b, i) => `<div class="zr"><b><i style="background:${CH.heat(100)}"></i>${esc(tf("s_burst", { n: i + 1 }))}</b>` +
        stepper("bs", b[0], t("min"), i, tf("burst_at", { n: i + 1 })) + stepper("bd", b[1], t("s_sec"), i, t("burst_len")) +
        `<button type="button" class="x" data-b="${i}" aria-label="${esc(tf("remove_burst", { n: i + 1 }))}">×</button></div>`).join("") : "";
    }
    $$("#zrows .zr").forEach((row, i) => {
      const z = r.zones[i]; $("b i", row).style.background = CH.heat(z.T);
      $$(".step", row).forEach(stp => { const inp = $("input", stp); if (document.activeElement !== inp) inp.value = stp.dataset.f === "T" ? num(z.T, 1) : num(z.duration, 1); });
      $(".x", row).disabled = r.zones.length <= 1;
    });
    if (pul) $$("#brows .zr").forEach((row, i) => {
      const b = reh.bursts[i];
      $$(".step", row).forEach(stp => { const inp = $("input", stp); if (document.activeElement !== inp) inp.value = stp.dataset.f === "bs" ? (+b[0]).toFixed(2) : String(b[1]); });
      $(".x", row).disabled = reh.bursts.length <= 1;
    });
  }
  const LIM = { T: [10, null, 1], duration: [0.5, 35, 0.5], bs: [0, null, 0.1], bd: [5, 60, 5] };
  function stepTo(f, i, v) {
    const r = S.r, [lo, hi0, st] = LIM[f];
    const hi = f === "T" ? r.T_fill - 1 : f === "bs" ? tCool() : hi0;
    v = clamp(Math.round(v / st) * st, lo, hi);
    if (f === "T") r.zones[i].T = +v.toFixed(1);
    else if (f === "duration") r.zones[i].duration = +v.toFixed(1);
    else {
      const b = r.reheat.bursts[i];
      if (f === "bs") b[0] = E.rnd(v, 3); else b[1] = Math.trunc(v);
      r.reheat.custom = true; E.syncPattern(r.reheat);
    }
    changed();
  }
  function wireRows() {
    const onClick = e => {
      const btn = e.target.closest("button"); if (!btn) return;
      const r = S.r;
      if (btn.dataset.d) {
        const stp = btn.closest(".step"), f = stp.dataset.f, i = +stp.dataset.i;
        const cur = f === "T" ? r.zones[i].T : f === "duration" ? r.zones[i].duration : f === "bs" ? r.reheat.bursts[i][0] : r.reheat.bursts[i][1];
        stepTo(f, i, cur + (+btn.dataset.d) * LIM[f][2]);
      } else if (btn.classList.contains("x")) {
        if (btn.dataset.b != null) {
          if (r.reheat.bursts.length > 1) { r.reheat.bursts.splice(+btn.dataset.b, 1); r.reheat.custom = true; E.syncPattern(r.reheat); changed({ rows: true }); }
        } else if (r.zones.length > 1) { r.zones.splice(+btn.dataset.i, 1); changed({ rows: true }); }
      }
    };
    const onChange = e => {
      const inp = e.target.closest(".step input"); if (!inp) return;
      const stp = inp.closest(".step"), v = parseFloat(String(inp.value).replace(",", "."));
      if (isFinite(v)) stepTo(stp.dataset.f, +stp.dataset.i, v); else renderRows();
    };
    ["#zrows", "#brows"].forEach(s => { $(s).addEventListener("click", onClick); $(s).addEventListener("change", onChange);
      $(s).addEventListener("keydown", e => { if (e.key === "Enter" && e.target.matches(".step input")) e.target.blur(); }); });
    $("#zAdd").addEventListener("click", () => {
      const z = S.r.zones, last = z.length ? z[z.length - 1].T : S.r.T_fill;
      if (z.length >= 8) return;
      z.push({ T: Math.max(20, last - 8), duration: 5, label: "" }); changed({ rows: true });
    });
    $("#zReset").addEventListener("click", () => { S.r.zones = clone(A.scenarios[TARGET].zones); changed({ rows: true }); });
    $("#bAdd").addEventListener("click", () => {
      const b = S.r.reheat.bursts; if (b.length >= 12) return;
      const d = b[b.length - 1][1]; b.push([E.nextBurst(b, tCool(), d), d]); S.r.reheat.custom = true; E.syncPattern(S.r.reheat); changed({ rows: true });
    });
    $("#bEven").addEventListener("click", () => { const reh = S.r.reheat; reh.bursts = E.evenBursts(reh.pulses, reh.pulse_sec, Math.min(reh.pulse_window, tCool())); delete reh.custom; changed({ rows: true }); });
    $("#zbar").addEventListener("click", e => { const s = e.target.closest(".zb[data-i]"); if (s) { const row = $$("#zrows .zr")[+s.dataset.i]; if (row) $("input", row).focus(); } });
  }
  function renderZbar() {
    const ts = S.ts, T1 = ts.t_total, pc = v => (v / T1 * 100).toFixed(3) + "%", segs = [];
    S.r.zones.forEach((z, i) => {
      const c = CH.heat(z.T);
      segs.push(`<button type="button" class="zb${CH.lum(c) < 0.3 ? " dark" : ""}" data-i="${i}" style="flex:0 0 ${pc(z.duration)};background-color:${c}" title="${esc(tf("zone_n", { n: i + 1 }))}"><b>${num(z.T, 1)}°</b><span>${num(z.duration, 1)} ${esc(t("min"))}</span></button>`);
    });
    if (ts.reheat_mode === "sequential" && ts.t_reheat > 0) {
      const c = CH.heat(ts.reheat_T);
      segs.push(`<div class="zb reh${CH.lum(c) < 0.3 ? " dark" : ""}" style="flex:0 0 ${pc(ts.t_reheat)};background-color:${c}"><b>${num(ts.reheat_T, 0)}°</b><span>${esc(t("rb_reheat_seg"))}</span></div>`);
    }
    const c = CH.heat(ts.late_cool_T);
    segs.push(`<div class="zb fin${CH.lum(c) < 0.3 ? " dark" : ""}" style="flex:1 1 0;background-color:${c}"><b>${num(ts.late_cool_T, 0)}°</b><span>${esc(t("rb_final"))}</span></div>`);
    if (ts.reheat_mode !== "sequential") for (const [a, b] of E.reheatSpans(ts))
      segs.push(`<i class="zspan" style="left:${pc(a)};width:max(3px,${pc(b - a)})"></i>`);
    $("#zbar").innerHTML = segs.join("");
    const Tl = S.r.zones[S.r.zones.length - 1].T;
    $("#zNote").textContent = tf("cool_total", { t: tCool(), T: Tl, flag: Tl <= 42 ? "" : t("cool_flag") });
  }
  function renderControls() {
    const r = S.r, reh = r.reheat, mode = reh.mode || "sequential", pul = mode === "pulsed";
    // a: starting points, worst case to best option
    const opts = E.categorise(presets()), rank = { worst: 0, work: 1, good: 2, best: 3 };
    opts.sort((a, b) => rank[a.cat] - rank[b.cat] || b.peak - a.peak);
    $("#presets").innerHTML = opts.map(o => {
      const T = o.ts.t_total, idx = []; for (let k = 0; k < 28; k++) idx.push(Math.round(k * (o.ts.times.length - 1) / 27));
      const strip = idx.map(k => `${CH.heat(o.ts.T_surf[k])} ${(o.ts.times[k] / T * 100).toFixed(1)}%`).join(",");
      const col = { worst: RISK.CRITICAL, work: RISK.CAUTION, good: RISK.SAFE, best: RISK.SAFE }[o.cat];
      return `<button class="pre${o.i === S.scen ? " on" : ""}" data-i="${o.i}" aria-pressed="${o.i === S.scen}" title="${esc(scDesc(o.sc))}">` +
        `<i style="color:${col}">${esc(t("cat_" + o.cat))}</i><b>${esc(o.name)}</b><em style="color:${CH.rcol(o.peak)}">${o.peak.toFixed(2)}</em>` +
        `<s style="background:linear-gradient(90deg,${strip})"></s></button>`;
    }).join("");
    const sc = A.scenarios[S.scen];
    $("#preNote").innerHTML = esc(scDesc(sc)) + (S.dirty ? ` <span class="ed">${esc(t("s_edited"))}</span>` : "");
    // b to e: sliders, zones, reheat
    $$(".sl[data-key]").forEach(syncSlider);
    renderRows(); renderZbar();
    $$("#modes .mode").forEach(b => {
      const m = b.dataset.m, on = m === mode;
      b.classList.toggle("on", on); b.setAttribute("aria-pressed", on);
      $("b", b).textContent = t({ sequential: "reheat_seq", simultaneous: "reheat_sim", pulsed: "reheat_pulsed" }[m]);
      $("span", b).textContent = t({ sequential: "s_mseq_d", simultaneous: "s_msim_d", pulsed: "s_mpul_d" }[m]);
    });
    $(".sl[data-key=reh_D]").hidden = pul; $("#rehParams .pul").hidden = !pul;
    let note = "";
    if (pul) {
      $("#bK").textContent = tf("burst_list", { n: reh.bursts.length });
      $("#bAdd").textContent = t("add_burst"); $("#bAdd").disabled = reh.bursts.length >= 12; $("#bEven").textContent = t("even_pattern");
      const on = reh.bursts.reduce((s, b) => s + b[1], 0) / 60;
      note = tf("pulse_note", { n: reh.bursts.length, s: Math.trunc(reh.pulse_sec), on }) + (reh.custom ? " " + t("pattern_custom") : "");
    } else {
      const Tr = reh.duration > 0 ? hotEnd(S.ts) : null;     // the model's surface as the hot air ends
      if (Tr != null) note = tf("reach", { T: Tr, flag: Tr >= S.ts.mushy[0] ? t("reach_ok") : t("reach_soft") });
      if (mode === "simultaneous") note += (note ? " " : "") + t("reheat_sim_note");
    }
    $("#rehNote").textContent = note;
    $("#meltNote").textContent = tf("mushy_note", { lo: r.melt - 5, hi: r.melt + 5 });
    // f: the optimiser
    $("#scope").innerHTML = ["cooling", "all"].map(s => `<button type="button" data-s="${s}" class="${S.scope === s ? "on" : ""}" aria-pressed="${S.scope === s}">${esc(t("opt_scope_" + s))}</button>`).join("");
    const o = S.opt;
    if (!o) $("#optRes").innerHTML = "";
    else if (o.none) $("#optRes").innerHTML = `<p class="note">${esc(t("drl_na"))}</p>`;
    else $("#optRes").innerHTML = `<p class="opt-done">${esc(tf("drl_done", { mode: t(MODE_KEY[o.mode]), di: o.di, t: o.t }))}</p>` +
      `<table class="opt-tbl"><thead><tr><th>${esc(t("drl_compare"))}</th><th>${esc(t("s_col_di"))}</th><th>${esc(t("s_col_cycle"))}</th></tr></thead><tbody>` +
      Object.entries(o.best).sort((a, b) => a[1].di - b[1].di).map(([m, b]) => `<tr class="${m === o.mode ? "cur" : ""}"><td>${esc(t(MODE_KEY[m]))}</td>` +
        `<td class="n" style="color:${CH.rcol(b.di)}">${b.di.toFixed(3)}</td><td class="n">${b.t.toFixed(0)} ${esc(t("min"))}</td></tr>`).join("") + `</tbody></table>`;
  }
  function wireControls() {
    $("#presets").addEventListener("click", e => { const b = e.target.closest(".pre"); if (b) loadPreset(+b.dataset.i); });
    $("#scope").addEventListener("click", e => { const b = e.target.closest("button[data-s]"); if (b && b.dataset.s !== S.scope) { S.scope = b.dataset.s; renderControls(); renderCases(); save(); } });
    $("#optBtn").addEventListener("click", () => {
      const btn = $("#optBtn"); btn.classList.add("busy"); $("#optNote").textContent = t("drl_spin");
      setTimeout(() => {
        const r = S.r, best = E.optimise(r.T_fill, r.reheat, r.h_cool, r.h_reheat, r.melt, r.late, S.scope), kinds = Object.keys(best);
        kinds.forEach(k => atPage(best[k], r));
        if (kinds.length) {
          const pick = best[E.reheatKind(r.reheat)] || kinds.map(k => best[k]).reduce((a, b) => (b.score < a.score ? b : a));
          applyRecipe(pick);
          S.best = Object.fromEntries(kinds.map(k => [k, { zones: best[k].zones, reh: best[k].reh, h_cool: best[k].h_cool, h_reh: best[k].h_reh,
            T_fill: r.T_fill, melt: r.melt, late: r.late }]));
          S.opt = { mode: pick.mode, di: pick.di, t: pick.t, best: Object.fromEntries(kinds.map(k => [k, { di: best[k].di, t: best[k].t }])) };
        } else S.opt = { none: true };
        btn.classList.remove("busy"); $("#optNote").textContent = "";
        changed({ rows: true }); flash($("#live .live-in"));
      }, 40);
    });
  }

  // ---------------------------------------------------------------- what the top reheat does
  // After cooling, the hot air heals damage that has already formed: its effect is the drop across it. During
  // cooling (the top only, or bursts) it comes first and keeps the damage from forming, so its effect is the peak
  // of the same cooling without it, minus the peak with it.
  const early = ts => ts.t_reheat > 0 && ts.reheat_mode !== "sequential";
  const argmax = a => a.reduce((k, v, i) => (v > a[k] ? i : k), 0);
  const signed = v => (v >= 0 ? "+" : "−") + Math.abs(E.rnd(v, 3)).toFixed(3);
  function bare(ts, hCool) {                               // the same cooling, with no top reheat
    return E.buildTimeline(ts.T_fill, ts.zones, { T: ts.reheat_T, duration: 0, mode: "sequential" }, hCool, 12, 120, ts.melt, ts.late_cool_T);
  }
  function hotEnd(ts) {                                    // the surface as the hot air ends, from the model
    const sp = E.reheatSpans(ts);
    return sp.length ? ts.T_surf[E.nearest(ts.times, sp[sp.length - 1][1])] : null;
  }

  // ---------------------------------------------------------------- the live panel
  function readout(ts, hCool) {                            // _readout_cells of main_3d_v4.py, plus the reheat's effect
    const DI = Math.max(...ts.DI), tt = ts.t_total, Bi = hCool * E.C.R_M / E.C.K_TH, mode = ts.reheat_mode, Ta = ts.reheat_T;
    const spans = E.reheatSpans(ts), heal = E.heal(ts), caution = RISK.CAUTION;
    let rv, rn;
    if (mode === "pulsed" && spans.length) { rv = tf("reheat_val_pul", { n: Math.trunc(ts.n_pulse), s: Math.trunc(ts.pulse_sec) }); rn = tf("reheat_note_pul", { T: Ta }); }
    else if (spans.length) { const k = mode === "simultaneous" ? "sim" : "seq"; rv = tf("reheat_val_" + k, { T: Ta, d: ts.t_reheat }); rn = t("reheat_note_" + k); }
    else { rv = t("reheat_off"); rn = t("st_none"); }
    let fx = [t("ro_heal"), "–", t("st_none"), null, null];
    if (spans.length && early(ts)) { const pb = Math.max(...bare(ts, hCool).DI); fx = [t("ro_prevent"), signed(pb - DI), tf("st_without", { d: pb }), null, null]; }
    else if (spans.length) fx = [t("ro_heal"), signed(heal), heal > 0.02 ? t("st_effective") : t("st_minimal"), null, null];
    return [
      [t("ro_di"), DI.toFixed(3), t("st_limit"), CH.rcol(DI), null],
      [t("ro_cycle"), `${tt.toFixed(0)}<small>${esc(t("min"))}</small>`, tt <= 30 ? t("st_within") : tf("st_over", { d: tt - 30 }), null, tt <= 30 ? null : caution],
      [t("ro_exit"), `${ts.T_surf[ts.T_surf.length - 1].toFixed(0)}<small>°C</small>`, tf("st_exit", { t: tt }), null, null],
      [t("ro_biot"), Bi.toFixed(2), Bi <= 0.5 ? t("st_uniform") : t("st_uneven"), null, Bi <= 0.5 ? null : caution],
      fx,
      [t("ro_reheat"), esc(rv), rn, null, null]];
  }
  function heroData(ts, hCool) {
    const DI = Math.max(...ts.DI), risk = CH.riskOf(DI).toLowerCase(), vkey = risk === "safe" && DI >= 0.15 ? "safe_margin" : risk;
    return { k: t("outlook"), chip: rname(DI), chip_col: CH.rcol(DI), verdict: t("v_" + vkey), sub: tf("v_detail", { di: DI }),
      hint: t("v_" + vkey + "_hint"), cells: readout(ts, hCool) };
  }
  function renderLive() {
    const ts = S.ts, r = S.r, DI = Math.max(...ts.DI), col = CH.rcol(DI), h = heroData(ts, r.h_cool);
    $("#lvChip").textContent = h.chip; $("#lvChip").style.color = col;
    $("#lvDI").textContent = DI.toFixed(3); $("#lvDI").style.color = col;
    $("#lvVerdict").textContent = h.verdict;
    CH.spark($("#lvSpark"), ts);
    const cells = h.cells;
    $("#lvGrid").innerHTML = [1, 3, 4, 5].map(k => { const [l, v, n2, vc, nc] = cells[k];
      return `<div><span>${esc(l)}</span><b${vc ? ` style="color:${vc}"` : ""}>${v}</b><em${nc ? ` style="color:${nc}"` : ""}>${esc(n2)}</em></div>`; }).join("");
    const Tl = r.zones[r.zones.length - 1].T, add = r.reheat.mode === "simultaneous" || r.reheat.mode === "pulsed" ? 0 : r.reheat.duration;
    const tt = tCool() + add + clamp(30 - tCool() - add - 0.5, 2, 8), Bi = r.h_cool * E.C.R_M / E.C.K_TH, cr = -(r.T_fill - Tl) / Math.max(tCool(), 0.1);
    $("#lvChecks").innerHTML = [[tt <= 30, tf("sb_cycle", { t: tt })], [Bi <= 0.5, tf("sb_biot", { b: Bi })], [cr >= -5, tf("sb_rate", { r: cr })]]
      .map(([ok, s]) => `<div><i style="background:${ok ? RISK.SAFE : RISK.CAUTION}"></i>${esc(s)}</div>`).join("");
  }

  // ---------------------------------------------------------------- hero: the X-ray loop
  let XR = null, xrPlay = true, xrVisible = true, xrT0 = 0, xrRaf = 0;
  const XR_RUN = 14000, XR_HOLD = 1800;
  function xrLabels() { return { min: t("min"), top: t("s_xr_top"), core: t("s_xr_core"), front: t("s_xr_front"), hot: t("s_xr_hot"), di: t("s_xr_di") }; }
  function renderHero() {
    const ts = S.ts, DI = Math.max(...ts.DI), col = CH.rcol(DI), risk = CH.riskOf(DI), vkey = risk === "SAFE" && DI >= 0.15 ? "safe_margin" : risk.toLowerCase();
    $("#xrDI").textContent = DI.toFixed(3); $("#xrDI").style.color = col;
    $("#xrChip").textContent = rname(DI); $("#xrChip").style.color = col;
    $("#xrVerdict").textContent = t("v_" + vkey);
    $("#hmBand").textContent = `${num(S.r.melt - 5, 0)}–${num(S.r.melt + 5, 0)} °C`;
    if (XR) { XR.set(ts, S.r.h_cool, xrLabels()); if (!xrPlay || !xrVisible) xrFrame(performance.now()); }
  }
  function xrFrame(now) {
    if (!XR || !S.ts) return;
    const el = (now - xrT0) % (XR_RUN + XR_HOLD), tm = xrPlay ? Math.min(1, el / XR_RUN) * S.ts.t_total : S.ts.t_total * 0.42;
    XR.draw(tm, (now % 1100) / 1100);
    $("#xrCycle").textContent = `${tm.toFixed(1)} / ${S.ts.t_total.toFixed(0)} ${t("min")}`;
  }
  function xrLoop(now) { xrRaf = 0; if (xrPlay && xrVisible && !document.hidden) { xrFrame(now); xrRaf = requestAnimationFrame(xrLoop); } }
  function xrKick() { if (!xrRaf && xrPlay && xrVisible) xrRaf = requestAnimationFrame(xrLoop); }

  // ---------------------------------------------------------------- 01 the line: hosting the 3D stage
  let stageReady = false, stageStale = true, stageVisible = false, stageTimer = 0;
  function stageArgs() {
    const r = S.r, lab = {};
    for (const k in A.tr.en) if (k.startsWith("ls_")) lab[k.slice(3)] = t(k);
    Object.assign(lab, { min: t("min"), zone: t("zone_n"), reheat: t("rb_reheat_seg"), final: t("rb_final"), play: t("play"), pause: t("pause"), live: t("live_hint") });
    const opts = options(true), [worst, best] = E.worstBest(opts), alts = {};
    for (const [k, o] of [["worst", worst], ["best", best]]) alts[k] = Object.assign(view(o.ts, o.h_cool), { name: o.name, peak: o.peak, id: o.id });
    return Object.assign(view(S.ts, r.h_cool, (r.reheat.bursts || []).map(([s, d]) => [+s, Math.trunc(d)])), {
      geo: A.geo, heat: A.heat, lab, editable: S.ts.reheat_mode === "pulsed", alts, rev: String(S.rev),
      risk: [[0.8, RISK.CRITICAL, t("risk_critical")], [0.5, RISK.WARNING, t("risk_warning")], [0.25, RISK.CAUTION, t("risk_caution")], [0, RISK.SAFE, t("risk_safe")]] });
  }
  function view(ts, hCool, bursts) {                        // _view of main_3d_v4.py
    const mode = ts.reheat_mode, rd = (a, k) => a.map(x => +x.toFixed(k));
    if (bursts == null) bursts = ts.bursts.map(([s, e]) => [+s.toFixed(3), Math.round((e - s) * 60)]);
    return { times: rd(ts.times, 3), Ts: rd(ts.T_surf, 2), Tc: rd(ts.T_core, 2), DI: rd(ts.DI, 4), tc: ts.t_cool, tt: ts.t_total, mode,
      Tair: ts.reheat_T, Tfill: ts.T_fill, late: ts.late_cool_T, spans: E.reheatSpans(ts).map(([a, b]) => [+a.toFixed(3), +b.toFixed(3)]),
      bursts: mode === "pulsed" ? bursts : [], seq: mode === "sequential" && ts.t_reheat > 0 ? [ts.reheat_start, ts.reheat_end] : null,
      zones: ts.zones.map(z => ({ T: z.T, d: z.duration })), hero: heroData(ts, hCool) };
  }
  function pushStage(now) {
    stageStale = true;
    if (!stageReady || !stageVisible) return;
    clearTimeout(stageTimer);
    stageTimer = setTimeout(() => {
      const fr = $("#stage"); if (!fr.contentWindow) return;
      fr.contentWindow.postMessage({ type: "streamlit:render", args: stageArgs(), dataframes: [], disabled: false }, "*");
      stageStale = false;
    }, now ? 0 : 140);
  }
  function onStage(d) {
    if (d.type === "streamlit:componentReady") { stageReady = true; pushStage(true); }
    else if (d.type === "streamlit:setFrameHeight") $("#stage").style.height = Math.round(d.height) + "px";
    else if (d.type === "streamlit:setComponentValue") {
      const v = d.value || {};
      if (v.load) loadOption(v.load);
      else if (Array.isArray(v.bursts) && v.bursts.length) {
        const reh = S.r.reheat;
        reh.bursts = v.bursts.map(([s, x]) => [E.rnd(+s, 3), Math.trunc(+x)]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, 12);
        reh.mode = "pulsed"; reh.custom = true; reh.duration = 0; E.syncPattern(reh);
        changed({ rows: true });
      }
    }
  }

  // ---------------------------------------------------------------- 03 cases
  function renderCases() {
    const opts = E.categorise(presets()), [worst, best] = E.worstBest(opts), ts = S.ts;
    const mine = Math.max(...ts.DI), ok = mine < 0.25 && ts.t_total <= 30.5;
    $("#caseFigs").innerHTML = [[worst.peak, t("cat_worst")], [best.peak, t("cat_best")], [mine, tf("f_yours", { cat: t(ok ? "cat_good" : "cat_work") })]]
      .map(([v, l]) => `<div class="bn"><b style="color:${CH.rcol(v)}">${v.toFixed(3)}</b><span>${esc(l)}</span></div>`).join("");
    $("#caseScope").textContent = tf("cs_scope", { s: t("opt_scope_" + S.scope) });
    $("#caseAll").textContent = t("cs_opt_all");
    const col = { worst: RISK.CRITICAL, work: RISK.CAUTION, good: RISK.SAFE, best: RISK.SAFE };
    $("#caseCols").innerHTML = ["worst", "work", "good", "best"].map(cat => {
      const items = opts.filter(o => o.cat === cat).sort((a, b) => b.peak - a.peak);
      return `<div class="cs-col${S.seen.cases ? " in" : ""}" data-reveal><div class="cs-cat" style="color:${col[cat]}"><b>${esc(t("cat_" + cat))}</b><span>${esc(t("cat_" + cat + "_d"))}</span></div>` +
        (items.length ? items.map(o => caseCard(o)).join("") : `<p class="case-none">${esc(t("cs_none"))}</p>`) + `</div>`;
    }).join("");
  }
  function caseCard(o) {
    const r = S.caseOpt[o.id + "|" + S.scope], c = CH.rcol(o.peak), cur = o.i === S.scen && !S.dirty;
    let opt = "";
    if (r) opt = `<div class="opt"><span>${esc(t("cs_optimised"))}</span><b style="color:${c}">${o.peak.toFixed(3)}</b><span>→</span>` +
      `<b style="color:${CH.rcol(r.di)}">${r.di.toFixed(3)}</b><em>${esc(tf("cs_cycle", { t: r.t }))}, ${esc(t(MODE_KEY[r.mode]))}</em></div>` +
      (S.scope === "cooling" && r.di >= 0.25 ? `<p class="cs-note">${esc(t("cs_need_reheat"))}</p>` : "");
    return `<article class="case${cur ? " cur" : ""}"><div class="case-t">${CH.glyph(o.peak, o.seed)}<div><b>${esc(o.name)}</b>` +
      `<span><strong style="color:${c}">${o.peak.toFixed(3)}</strong>${esc(tf("cs_cycle", { t: o.cycle }))}${cur ? ` · ${esc(t("s_loaded"))}` : ""}</span></div></div>` +
      `<p>${esc(why(o))}</p>${opt}<div class="btns"><button class="btn ghost sm" data-load="${o.i}">${esc(t("cs_load"))}</button>` +
      (r ? `<button class="btn solid sm" data-lopt="${o.i}">${esc(t("cs_load_opt"))}</button>` : `<button class="btn ghost sm" data-opt="${o.i}">${esc(t("cs_opt"))}</button>`) + `</div></article>`;
  }
  function caseOpt(i) {
    const sc = A.scenarios[i], best = E.optimise(sc.T_fill, sc.reheat, sc.h_cool, sc.h_reheat, sc.melt, sc.late, S.scope), v = Object.values(best);
    S.caseOpt[sc.id + "|" + S.scope] = v.length ? atPage(v.reduce((a, b) => (b.score < a.score ? b : a)), sc) : null;
  }
  function wireCases() {
    $("#caseCols").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.load != null) { loadPreset(+b.dataset.load); go("recipe"); }
      else if (b.dataset.opt != null) { b.classList.add("busy"); b.textContent = t("drl_spin"); setTimeout(() => { caseOpt(+b.dataset.opt); renderCases(); }, 30); }
      else if (b.dataset.lopt != null) {
        const i = +b.dataset.lopt, r = S.caseOpt[A.scenarios[i].id + "|" + S.scope];
        loadPreset(i); if (r) { applyRecipe(r); changed({ rows: true }); } go("recipe");
      }
    });
    $("#caseAll").addEventListener("click", () => {
      const b = $("#caseAll"); b.classList.add("busy"); b.textContent = t("drl_spin");
      setTimeout(() => { A.scenarios.forEach((_, i) => caseOpt(i)); b.classList.remove("busy"); renderCases(); }, 40);
    });
  }

  // ---------------------------------------------------------------- 04 heat
  function renderHeat(animate) {
    const ts = S.ts, n = ts.times.length, [lo, hi] = ts.mushy;
    let mushy = 0, gap = 0, rate = 0;
    for (let i = 0; i < n; i++) {
      const dt = i ? ts.times[i] - ts.times[i - 1] : 0;
      if (ts.T_surf[i] >= lo && ts.T_surf[i] <= hi) mushy += dt;
      gap = Math.max(gap, Math.abs(ts.T_core[i] - ts.T_surf[i]));
      rate = Math.min(rate, ts.dTdt[i]);
    }
    const m = esc(t("min"));
    $("#heatFigs").innerHTML = [[`${mushy.toFixed(1)}<small>${m}</small>`, t("f_mushy")], [`${gap.toFixed(0)}<small>°C</small>`, t("f_gap")],
      [`${Math.abs(rate).toFixed(1)}<small>°C/${m}</small>`, t("f_rate")]].map(([v, l]) => `<div class="bn"><b>${v}</b><span>${esc(l)}</span></div>`).join("");
    const DI = Math.max(...ts.DI), I = CH.INK;
    const key = (c, st, w) => `<span><i style="border-top:${w}px ${st} ${c}"></i>`;
    $("#chLegend").innerHTML = `<b class="ch-t" style="color:${CH.rcol(DI)}">${esc(tf("ch_title", { di: DI, risk: rname(DI) }))}</b>` +
      [[I.ink, "solid", 2.4, "tr_surface"], [I.indigo, "dashed", 2, "tr_core"], [I.grey, "dotted", 2, "tr_air"], [I.umber, "solid", 2, "tr_dT"],
        [I.indigo, "dashed", 1.6, "tr_rate"], [I.ink, "solid", 2.4, "tr_di"]].map(([c, s, w, k]) => key(c, s, w) + esc(t(k)) + "</span>").join("") +
      `<span class="ch-hint">${esc(t("s_chart_hint"))}</span>`;
    CH.thermal($("#chart"), ts, { t, fmt, animate });
  }

  // ---------------------------------------------------------------- 05 surface
  function renderSurface() {
    const ts = S.ts, n = ts.DI.length, spans = E.reheatSpans(ts), has = spans.length > 0;
    const lab = { top: t("s_surf_top"), sec: t("s_surf_sec"), free: t("crack_free") };
    const card = (title, tm, di) => `<div class="sd"><div class="sd-h"><b>${esc(title)}</b><span>${esc(tf("f_at", { t: tm }))}</span></div>` +
      CH.surfaceArt(di, 4242, Object.assign({ label: title }, lab)) +
      `<div class="sd-f"><b style="color:${CH.rcol(di)}">${di.toFixed(3)}</b><i class="chip" style="color:${CH.rcol(di)}">${esc(rname(di))}</i></div></div>`;
    const sf = $("#surface .stand");
    sf.dataset.t = has && early(ts) ? "sf_crack_early" : "sf_crack"; sf.textContent = t(sf.dataset.t);
    if (has && early(ts)) {                                // hot air during cooling: the same cooling without it, then with it
      const b = bare(ts, S.r.h_cool), jb = argmax(b.DI), ja = argmax(ts.DI), Pb = b.DI[jb], Pa = ts.DI[ja], cut = Pb - Pa;
      $("#surf").innerHTML = card(t("s_without"), b.times[jb], Pb) + card(t("s_with"), ts.times[ja], Pa);
      $("#surfNote").innerHTML = cut > 0.005
        ? esc(tf("s_prevent_note", { a: Pa, b: Pb, p: Math.round(cut / Math.max(Pb, 0.01) * 100) })) +
          (Pa < 0.25 && Pb >= 0.25 ? ` <span class="ok">${esc(t("crack_free"))}</span>` : "")
        : esc(t("s_prevent_small"));
      return;
    }
    const ie = has ? E.nearest(ts.times, spans[spans.length - 1][1]) : n - 1;
    let ib = 0; for (let i = 0; i < Math.max(1, ie + 1); i++) if (ts.DI[i] > ts.DI[ib]) ib = i;
    const ia = has ? ie : n - 1, DIb = ts.DI[ib], DIa = ts.DI[ia], drop = DIb - DIa;
    $("#surf").innerHTML = card(t("before_reheat"), ts.times[ib], DIb) + card(has ? t("after_reheat") : t("end_cycle"), ts.times[ia], DIa);
    let note;
    if (drop > 0.005) {
      const eff = Math.trunc(drop / Math.max(DIb, 0.01) * 100);
      note = `<b>${esc(t("healing_eff"))} ${eff}%.</b> ${esc(hotEnd(ts) >= ts.mushy[0] ? t("optimal_healing") : t("softening_only"))}` +
        (DIa < 0.25 && DIb >= 0.25 ? ` <span class="ok">${esc(t("crack_free"))}</span>` : "");
    } else note = esc(has ? t("min_healing") : t("no_reheat"));
    $("#surfNote").innerHTML = note;
  }

  // ---------------------------------------------------------------- 06 report
  function renderReport() {
    const ts = S.ts, r = S.r, DI = Math.max(...ts.DI), Bi = r.h_cool * E.C.R_M / E.C.K_TH, tt = ts.t_total, nz = r.zones.length;
    const Tl = r.zones[nz - 1].T, spans = E.reheatSpans(ts), [sol, liq] = ts.mushy;
    let drop = 0;
    if (spans.length) { const is = E.nearest(ts.times, spans[0][0]), ie = E.nearest(ts.times, spans[spans.length - 1][1]); drop = ts.DI[is] - ts.DI[Math.min(ie, ts.DI.length - 1)]; }
    const rk = DI <= 0.10 ? "rec_excellent" : DI < 0.25 ? "rec_good" : DI <= 0.50 ? "rec_warning" : "rec_critical";
    $("#rec").style.borderLeftColor = CH.rcol(DI);
    $("#rec").innerHTML = `<div class="rp-k">${esc(t("rep_rec"))}</div><p>${esc(tf(rk, { di: DI, n: nz, t: tt }))}</p>`;
    const pre = spans.length > 0 && early(ts), Pb = pre ? Math.max(...bare(ts, r.h_cool).DI) : 0;
    const fx = pre ? Pb - DI : drop;                       // what the hot air does: damage kept off, or damage healed
    const checks = [[t("chk_time"), tt <= 30, `${tt.toFixed(0)} ${t("min")}`], [t("chk_biot"), Bi <= 0.5, Bi.toFixed(2)], [t("chk_last"), Tl <= 42, `${Tl.toFixed(0)} °C`],
      [t("chk_di"), DI < 0.25, DI.toFixed(3)], [t("chk_hike"), DI < 0.15, DI.toFixed(3)], [t("chk_heal"), !spans.length || fx > 0.01, spans.length ? signed(fx) : "–"]];
    $("#chkK").innerHTML = `${esc(t("rep_checks"))}<b>${checks.filter(c => c[1]).length} / ${checks.length}</b>`;
    $("#chk").innerHTML = checks.map(([l, ok, v]) => `<li class="${ok ? "ok" : "no"}"><i>${ok ? "✓" : "✕"}</i><span>${esc(l)}</span><b>${esc(v)}</b></li>`).join("");
    let acc = 0;
    $("#recipeTbl").innerHTML = `<table class="rp-table"><thead><tr><th>${esc(t("col_zone"))}</th><th>${esc(t("col_air"))}</th><th>${esc(t("col_min"))}</th><th>${esc(t("col_end"))}</th><th>${esc(t("col_phase"))}</th></tr></thead><tbody>` +
      r.zones.map((z, i) => { acc += z.duration; const ph = sol <= z.T && z.T <= liq ? t("ph_mushy") : z.T < sol ? t("ph_below") : t("ph_above");
        return `<tr><td><i style="background:${CH.heat(z.T)}"></i>${esc(tf("zone_n", { n: i + 1 }))}</td><td>${z.T.toFixed(0)}</td><td>${z.duration.toFixed(1)}</td><td>${acc.toFixed(1)} ${esc(t("min"))}</td><td>${esc(ph)}</td></tr>`; }).join("") + `</tbody></table>`;
    if (!spans.length) $("#rehDl").innerHTML = `<p class="note">${esc(t("rh_none"))}</p>`;
    else {
      const Tr = hotEnd(ts), state = Tr >= liq ? t("rh_full") : Tr >= sol ? t("rh_mushy") : t("rh_soft");
      const air = ts.reheat_mode === "pulsed" ? tf("belt_bursts", { n: ts.n_pulse, s: Math.trunc(ts.pulse_sec), T: r.reheat.T }) : `${r.reheat.T.toFixed(0)} °C, ${ts.t_reheat.toFixed(0)} ${t("min")}`;
      const rows = [[t("rh_mode"), t(MODE_KEY[ts.reheat_mode])], [t("rh_air"), air], [t("rh_reach"), `${Tr.toFixed(0)} °C, ${state}`]]
        .concat(pre ? [[t("rh_without"), Pb.toFixed(3)], [t("rh_prevent"), signed(fx)]] : [[t("rh_heal"), signed(fx)]]);
      $("#rehDl").innerHTML = `<dl class="rp-dl">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
    }
  }
  function download() {
    const ts = S.ts, r = S.r, DI = Math.max(...ts.DI);
    const out = { product: "LCWT401 deodorant stick", model: "PI-DRL v5", made: new Date().toISOString().slice(0, 10),
      starting_point: A.scenarios[S.scen].name.en + (S.dirty ? " (edited)" : ""),
      recipe: { T_fill: r.T_fill, zones: r.zones.map(z => ({ T: z.T, duration: z.duration })), reheat: r.reheat, h_cool: r.h_cool, h_reheat: r.h_reheat, melt: r.melt, late_cool_T: r.late },
      result: { peak_damage_index: +DI.toFixed(4), final_damage_index: +ts.DI[ts.DI.length - 1].toFixed(4), cycle_min: +ts.t_total.toFixed(2),
        risk: CH.riskOf(DI), biot: +(r.h_cool * E.C.R_M / E.C.K_TH).toFixed(3) } };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], { type: "application/json" }));
    a.download = "pidrl_recipe.json"; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  // ---------------------------------------------------------------- 07 advisor (the LLM runs in Python)
  function md(s) {                                         // small, safe Markdown: escape first, then mark up
    const inline = x => x.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+?)`/g, "<code>$1</code>")
      .replace(/(^|[^*\w])\*([^*\n]+?)\*(?![*\w])/g, "$1<em>$2</em>");
    const out = []; let list = null, para = [];
    const endPara = () => { if (para.length) { out.push("<p>" + para.join("<br>") + "</p>"); para = []; } };
    const endList = () => { if (list) { out.push(`<${list.tag}>${list.items.join("")}</${list.tag}>`); list = null; } };
    for (const raw of esc(s || "").split("\n")) {
      const l = raw.trim(), h = l.match(/^#{1,6}\s+(.*)/), b = l.match(/^[-•*]\s+(.*)/), nm = l.match(/^\d+[.)]\s+(.*)/);
      if (!l) { endPara(); endList(); }
      else if (h) { endPara(); endList(); out.push(`<h4>${inline(h[1].replace(/[:\s]+$/, ""))}</h4>`); }
      else if (b || nm) {
        endPara();
        const tag = b ? "ul" : "ol"; if (list && list.tag !== tag) endList();
        if (!list) list = { tag, items: [] };
        list.items.push(`<li>${inline((b || nm)[1])}</li>`);
      } else { endList(); para.push(inline(l)); }
    }
    endPara(); endList(); return out.join("");
  }
  function srcLabel(src) {
    const m = String(src || "").match(/^(groq|ollama):(.*)$/);
    return m ? `${m[1][0].toUpperCase() + m[1].slice(1)}, ${m[2]}` : t("rule_based");
  }
  const SECS = { "ASSESSMENT": "sec_assessment", "PHYSICAL MEANING": "sec_physical", "ROOT CAUSE": "sec_root", "RECOMMENDATION": "sec_reco", "COMPARISON": "sec_compare" };
  function analysisHtml(text) {
    const parts = [];
    for (const raw of String(text || "").split("\n")) {
      const line = raw.trim(); if (!line) continue;
      const clean = line.replace(/^[#*\s]+/, "").replace(/[:*#\s]+$/, "").trim(), up = clean.toUpperCase();
      let sec = null, len = 0;                              // "ASSESSMENT", "**Recommendations:** ...", "ROOT CAUSE: ..."
      for (const s of Object.keys(SECS)) { const m = up.match(new RegExp("^" + s + "S?(?=$|[:\\s*])")); if (m) { sec = s; len = m[0].length; break; } }
      if (sec) { parts.push(`<h4>${esc(t(SECS[sec]))}</h4>`); const body = clean.slice(len).replace(/^[:*\s]+/, ""); if (body) parts.push(md(body)); }
      else parts.push(md(line));
    }
    return parts.join("");
  }
  function payload() {
    const r = S.r;
    return { T_fill: r.T_fill, zones: r.zones.map(z => ({ T: z.T, duration: z.duration })), reheat: r.reheat, h_cool: r.h_cool, h_reheat: r.h_reheat,
      melt: r.melt, late: r.late, scenario: S.dirty ? "Custom" : A.scenarios[S.scen].id };
  }
  let slowTimer = 0, deadTimer = 0;
  function request(kind, extra) {
    if (S.pend) return false;
    const nonce = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    S.pend = Object.assign({ kind, nonce, slow: false }, extra || {});
    post(Object.assign({ kind, nonce, lang: LANG, recipe: payload(), key: S.key }, extra || {}));
    clearTimeout(slowTimer); clearTimeout(deadTimer);   // a local model can take a minute; say so, and never hang
    slowTimer = setTimeout(() => { if (S.pend && S.pend.nonce === nonce) { S.pend.slow = true; renderAdvisor(); } }, 8000);
    deadTimer = setTimeout(() => { if (S.pend && S.pend.nonce === nonce) { S.pend = null; A.err = "timeout"; renderAdvisor(); } }, 180000);
    renderAdvisor();
    return true;
  }
  const ask = q => { q = String(q || "").trim().slice(0, 1000); if (q) request("chat", { q }); };
  function renderAdvisor() {
    const adv = A.adv || {}, chat = A.chat || [], P = S.pend;
    $("#advStatus").innerHTML = adv.ok === false ? `<i></i>${esc(t("s_adv_off"))}` : `<i class="${adv.online ? "on" : ""}"></i>${esc(t(adv.online ? "adv_online" : "adv_ready"))}`;
    const who = `<div class="who">${esc(t("advisor"))}</div>`;
    let h = chat.map(m => (m.role === "user" ? `<div class="msg msg-u"><div>${md(m.content)}</div></div>`
      : `<div class="msg msg-a">${who}<div class="body">${md(m.content)}</div>${m.src ? `<div class="src">${esc(t("source"))}: ${esc(srcLabel(m.src))}</div>` : ""}</div>`)).join("");
    if (P && P.kind === "chat") h += `<div class="msg msg-u"><div>${md(P.q)}</div></div><div class="msg msg-a">${who}<div class="typing"><i></i><i></i><i></i></div><div class="src">${esc(t(P.slow ? "s_typing2" : "s_typing"))}</div></div>`;
    if (A.err && !P) h += `<p class="note">${esc(t("s_err"))}</p>`;
    $("#thread").innerHTML = h || `<div class="empty">${esc(t("chat_empty"))}</div>`;
    $("#thread").scrollTop = $("#thread").scrollHeight;
    const sugs = chat.length ? (A.fups || []) : [1, 2, 3, 4].map(k => t("s_q" + k));
    $("#sugs").innerHTML = (chat.length && sugs.length ? `<span class="fu-k">${esc(t("followups"))}</span>` : "") +
      sugs.map(q => `<button type="button" ${P ? "disabled" : ""}>${esc(q)}</button>`).join("");
    $("#ask").placeholder = t("chat_ph"); $("#ask").disabled = !!P; $("#composer button").disabled = !!P;
    $("#chatClear").disabled = !!P || !chat.length;
    const an = A.analysis;                                 // the analysis of this recipe
    if (an && an.key === S.key) $("#analysis").innerHTML = analysisHtml(an.text) + `<div class="src">${esc(t("source"))}: ${esc(srcLabel(an.src))}</div>`;
    else if (P && P.kind === "analyze") $("#analysis").innerHTML = `<div class="typing"><i></i><i></i><i></i></div>`;
    else $("#analysis").innerHTML = `<p class="an-empty">${esc(t(an ? "s_an_stale" : "s_an_empty"))}</p>`;
    $("#anBtn").disabled = !!P; $("#anBtn").textContent = t(an && an.key === S.key ? "retry" : "s_analyse");
  }
  function autoAnalyse() {                                 // once per recipe, when the advisor comes into view
    if (!S.seen.advisor || S.pend || (A.analysis && A.analysis.key === S.key) || S.anAsked[S.key] || (A.adv && A.adv.ok === false)) return;
    S.anAsked[S.key] = 1; request("analyze");
  }
  function wireAdvisor() {
    $("#composer").addEventListener("submit", e => { e.preventDefault(); const q = $("#ask").value; if (q.trim() && !S.pend) { $("#ask").value = ""; ask(q); } });
    $("#sugs").addEventListener("click", e => { const b = e.target.closest("button"); if (b) ask(b.textContent); });
    $("#chatClear").addEventListener("click", () => request("clear"));
    $("#anBtn").addEventListener("click", () => { S.anAsked[S.key] = 1; request("analyze"); });
  }
  function onArgs() {                                      // Python answered: the advisor's state
    if (S.pend && A.done === S.pend.nonce) { S.pend = null; clearTimeout(slowTimer); clearTimeout(deadTimer); }
    renderAdvisor(); autoAnalyse();
  }

  // ---------------------------------------------------------------- language
  function applyLang() {
    document.documentElement.lang = LANG;
    $$("[data-t]").forEach(el => { el.textContent = t(el.dataset.t); });
    $$("[data-th]").forEach(el => { el.innerHTML = t(el.dataset.th); });
    $$(".lang button").forEach(b => { const on = b.dataset.lang === LANG; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); });
    $("#xrPlay").textContent = xrPlay ? t("pause") : t("play");
    buildIndex(); rowsSig = "";
  }
  function setLang(l) { if (l === LANG) return; LANG = l; applyLang(); renderAll(); renderAdvisor(); save(); }

  // ---------------------------------------------------------------- rendering everything
  function renderAll() {
    renderControls(); renderLive(); renderHero(); renderCases(); renderSurface(); renderReport();
    if (S.seen.heat) { renderHeat(!S.chartDrawn); S.chartDrawn = true; } else S.heatStale = true;
    pushStage(false);
    if (A.analysis || S.pend) renderAdvisor();
    if (!firstRender) flash($("#lvDI"));
    firstRender = false;
  }
  function flash(el) { if (!el) return; el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }

  // ---------------------------------------------------------------- page: navigation, index, reveal, cursor
  function go(id) {
    const el = document.getElementById(id); document.body.classList.remove("menu-open");
    if (!el) return;
    // sections passed on the way (the chart, the 3D line) render as they come near and can push the target down:
    // if they did, go again once the scroll ends
    const y0 = el.offsetTop, again = () => { if (Math.abs(el.offsetTop - y0) > 4) el.scrollIntoView({ behavior: "smooth", block: "start" }); };
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    SC.addEventListener("scrollend", again, { once: true });
    setTimeout(() => SC.removeEventListener("scrollend", again), 3000);
  }
  function buildIndex() {
    $("#sideIndex").innerHTML = $$("main section[data-tk]").map(s => `<a href="#${s.id}" data-go="${s.id}"><span>${esc(t(s.dataset.tk))}</span><i></i></a>`).join("");
  }
  function wirePage() {
    document.addEventListener("click", e => {
      const a = e.target.closest("[data-go]"); if (!a) return;
      e.preventDefault(); go(a.dataset.go);
    });
    $$(".lang button").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
    $("#menuBtn").addEventListener("click", () => document.body.classList.toggle("menu-open"));
    $("#printBtn").addEventListener("click", () => window.print());
    $("#jsonBtn").addEventListener("click", download);
    $("#xrPlay").addEventListener("click", () => {
      xrPlay = !xrPlay; $("#xrPlay").textContent = xrPlay ? t("pause") : t("play"); $("#xrPlay").setAttribute("aria-pressed", String(!xrPlay));
      if (xrPlay) { xrT0 = performance.now(); xrKick(); } else xrFrame(performance.now());
    });

    // the top bar, progress, and the section under the bar (night sections turn the bar light). The page scrolls
    // in #scroller (see style.css); the wheel over the fixed bar and index scrolls it too.
    const bar = $("#topbar"), prog = $("#progress"), secs = $$("main section, footer.colo");
    [bar, $("#sideIndex")].forEach(el => el.addEventListener("wheel", e => { if (!document.body.classList.contains("menu-open")) SC.scrollBy({ top: e.deltaY, behavior: "auto" }); }, { passive: true }));
    const onScroll = () => {
      const y = SC.scrollTop, h = SC.scrollHeight - SC.clientHeight;
      bar.classList.toggle("solid", y > 30);
      prog.style.transform = `scaleX(${h > 0 ? (y / h).toFixed(4) : 0})`;
      let cur = secs[0];
      for (const s of secs) { const r = s.getBoundingClientRect(); if (r.top <= 64 && r.bottom > 64) { cur = s; break; } }
      document.body.classList.toggle("on-night", cur.classList.contains("night"));
      $$("#topnav a, #sideIndex a").forEach(a => a.classList.toggle("on", a.dataset.go === cur.id));
    };
    SC.addEventListener("scroll", onScroll, { passive: true }); onScroll();

    const reveal = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.classList.add("in"); reveal.unobserve(en.target); } }),
      { root: SC, rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    $$("[data-reveal]").forEach(el => reveal.observe(el));

    // sections that do work only once near the screen: the 3D stage, the chart, the advisor's first analysis
    const near = new IntersectionObserver(es => es.forEach(en => {
      const id = en.target.id;
      if (id === "line") {
        stageVisible = en.isIntersecting;
        if (en.isIntersecting && !$("#stage").getAttribute("src")) $("#stage").src = "../" + A.stage + "/index.html";
        if (en.isIntersecting && stageStale) pushStage(true);
      } else if (id === "hero") { xrVisible = en.isIntersecting; if (xrVisible) xrKick(); }
      else if (en.isIntersecting && !S.seen[id]) {
        S.seen[id] = 1;
        if (id === "heat") { renderHeat(true); S.chartDrawn = true; S.heatStale = false; }
        if (id === "cases") $$("#caseCols .cs-col").forEach(el => el.classList.add("in"));
        if (id === "advisor") autoAnalyse();
      } else if (en.isIntersecting && id === "heat" && S.heatStale) { renderHeat(false); S.heatStale = false; }
    }), { root: SC, rootMargin: "600px 0px 600px 0px", threshold: 0 });
    ["hero", "line", "cases", "heat", "advisor"].forEach(id => near.observe(document.getElementById(id)));
    document.addEventListener("visibilitychange", xrKick);

    let rz = 0;
    window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (S.seen.heat) renderHeat(false); }, 160); });

    // the cursor ring, and buttons that lean toward the pointer
    const cur = $("#cursor"); let mx = -100, my = -100, cx = -100, cy = -100, curRaf = 0;
    const tick = () => { cx += (mx - cx) * 0.22; cy += (my - cy) * 0.22; cur.style.transform = `translate(${cx.toFixed(1)}px,${cy.toFixed(1)}px)`;
      curRaf = Math.abs(mx - cx) + Math.abs(my - cy) > 0.3 ? requestAnimationFrame(tick) : 0; };
    document.addEventListener("pointermove", e => {
      if (e.pointerType !== "mouse") return;
      mx = e.clientX; my = e.clientY; cur.classList.add("on");
      cur.classList.toggle("big", !!e.target.closest("a, button, .pre, .mode, .zb, input[type=range], .chart"));
      if (!curRaf) curRaf = requestAnimationFrame(tick);
    }, { passive: true });
    document.documentElement.addEventListener("pointerleave", () => cur.classList.remove("on"));
    $$(".magnetic").forEach(b => {
      b.addEventListener("pointermove", e => { const r = b.getBoundingClientRect(); b.style.transform = `translate(${((e.clientX - r.left - r.width / 2) * 0.18).toFixed(1)}px,${((e.clientY - r.top - r.height / 2) * 0.28).toFixed(1)}px)`; });
      b.addEventListener("pointerleave", () => { b.style.transform = ""; });
    });
  }

  // ---------------------------------------------------------------- boot
  function boot() {
    CH.setStops(A.heat);
    const saved = store.get("pidrl.site") || {};
    LANG = saved.lang === "ja" || saved.lang === "en" ? saved.lang : (A.lang === "ja" ? "ja" : "en");
    S.scope = saved.scope === "cooling" ? "cooling" : "all";
    if (valid(saved.r) && saved.scen >= 0 && saved.scen < A.scenarios.length) { S.r = saved.r; S.scen = saved.scen; S.dirty = !!saved.dirty; }
    else S.r = fromScenario(A.scenarios[TARGET]);
    XR = CH.Xray($("#xraySvg"), $("#xrField"));
    buildModes(); buildSliders();                           // the reheat sliders live in what buildModes draws
    wireRows(); wireControls(); wireCases(); wireAdvisor(); applyLang();
    compute(); renderAll(); renderAdvisor(); wirePage();
    xrT0 = performance.now(); xrKick();
    SC.focus({ preventScroll: true });                     // so Page Down, Space and the arrow keys scroll the page
    const done = () => $("#loader").classList.add("done");
    Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 1800))]).then(() => setTimeout(done, 350));
  }
})();
