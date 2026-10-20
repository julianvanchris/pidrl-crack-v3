/* PI-DRL charts and drawings, in SVG with no library: the heat ramp, the thermal chart of section 04 (the same three
   panels as fig_charts in main_3d_v4.py), the sparkline of the live panel, the crack drawings of the cases and the
   surface section, and the X-ray of the hero. Everything reads a timeline from engine.js. */
(function (G) {
  "use strict";
  const INK = { ink: "#1F1D1A", ink2: "#4A4640", grey: "#8A847A", hair: "rgba(31,29,26,.12)", hair2: "rgba(31,29,26,.24)",
    indigo: "#2E4A6E", indigo2: "#7E98BC", umber: "#8C6A43", sand: "#E6D9BF", paper: "#F7F5F0", white: "#FBFAF7",
    hot: "#B4532A", hot2: "#D9875C", night: "#0F0E0C", nightText: "#EDE8DF", nightGrey: "#9C958A" };
  const RISK = { SAFE: "#4F8A3A", CAUTION: "#A8761A", WARNING: "#C2622B", CRITICAL: "#D04A35" };
  const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  let STOPS = [[15, "#2E4A6E"], [30, "#5D7FA6"], [45, "#A9BCCF"], [60, "#E6D9BF"], [72, "#D9A066"], [85, "#B4532A"],
    [100, "#7A2A16"]].map(([T, h]) => [T, hex2rgb(h)]);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------------------------------------------------------------- the heat ramp (fixed °C domain, as _heat)
  function setStops(list) { if (list && list.length) STOPS = list.map(([T, h]) => [+T, hex2rgb(h)]); }
  function heatRGB(T) {
    T = clamp(T, STOPS[0][0], STOPS[STOPS.length - 1][0]);
    for (let i = 1; i < STOPS.length; i++) {
      if (T <= STOPS[i][0]) {
        const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i], f = (T - t0) / (t1 - t0);
        return c0.map((c, k) => Math.round(c + (c1[k] - c) * f));
      }
    }
    return STOPS[STOPS.length - 1][1];
  }
  const heat = T => "#" + heatRGB(T).map(v => v.toString(16).padStart(2, "0")).join("");
  function rgba(hex, a) { const [r, g, b] = hex2rgb(hex); return `rgba(${r},${g},${b},${a})`; }
  function lum(hex) {                                      // relative luminance, to pick ink or paper text on a colour
    const c = hex2rgb(hex).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  const riskOf = d => (d >= 0.8 ? "CRITICAL" : d >= 0.5 ? "WARNING" : d >= 0.25 ? "CAUTION" : "SAFE");
  const rcol = d => RISK[riskOf(d)];
  function mulberry(seed) {                                // small seeded generator: the same case draws the same cracks
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const minus = s => String(s).replace(/^-/, "−");

  function ticks(lo, hi, n) {
    const raw = (hi - lo) / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return { list: out, dec: Math.max(0, Math.ceil(-Math.log10(step) - 1e-9)) };
  }
  const line = (xs, ys, X, Y) => xs.map((x, i) => (i ? "L" : "M") + X(x).toFixed(1) + " " + Y(ys[i]).toFixed(1)).join("");
  const reheatSpans = ts => (ts.reheat_mode === "pulsed" ? ts.bursts.map(b => b.slice())
    : ts.t_reheat > 0 && ts.reheat_end > ts.reheat_start ? [[ts.reheat_start, ts.reheat_end]] : []);
  const nearest = (a, v) => { let k = 0, d = Infinity; for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - v); if (e < d) { d = e; k = i; } } return k; };

  // ---------------------------------------------------------------- the thermal chart (section 04)
  // o: { t: label lookup, fmt: Python-style formatter, animate: draw the lines in }
  function thermal(host, ts, o) {
    const t = o.t, fmt = o.fmt;
    const W = Math.max(300, Math.round(host.clientWidth || 900)), narrow = W < 640;
    const sc = clamp(W / 1100, 0.66, 1);
    const ML = narrow ? 38 : 52, MR = narrow ? 8 : 14, MT = 30, GAP = narrow ? 44 : 50, MB = 40;
    const PH = [Math.round(300 * sc), Math.round(132 * sc), Math.round(196 * sc)];
    const top = [MT, MT + PH[0] + GAP, MT + PH[0] + GAP + PH[1] + GAP];
    const H = top[2] + PH[2] + MB, fs = narrow ? 10 : 11;
    const n = ts.times.length, T1 = ts.t_total;
    const X = v => ML + v / T1 * (W - ML - MR);
    const Te = ts.T_env.slice(); if (n > 1) Te[0] = Te[1];   // the model leaves the first air value unset
    const spans = reheatSpans(ts), Tair = ts.reheat_T, hotC = heat(Tair);
    const [msol, mliq] = ts.mushy;
    const lo1 = Math.min(...ts.T_surf, ...ts.T_core, ...Te) - 5, hi1 = Math.max(...ts.T_surf, ...ts.T_core, ...Te) + 5;
    const lo2 = Math.min(...ts.dT, ...ts.dTdt) - 3, hi2 = Math.max(...ts.dT, ...ts.dTdt) + 3;
    const Y = [v => top[0] + PH[0] * (1 - (v - lo1) / (hi1 - lo1)), v => top[1] + PH[1] * (1 - (v - lo2) / (hi2 - lo2)),
      v => top[2] + PH[2] * (1 - v / 1.05)];
    const yr = [[lo1, hi1], [lo2, hi2], [0, 1.05]];
    const p = [];
    const txt = (x, y, s, a = "") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" ${a}>${esc(s)}</text>`;

    // grid and axes
    for (let k = 0; k < 3; k++) {
      const tk = k === 2 ? { list: [0, 0.25, 0.5, 0.75, 1], dec: 2 } : ticks(yr[k][0], yr[k][1], k === 0 ? 5 : 3);
      for (const v of tk.list) {
        const y = Y[k](v);
        p.push(`<line x1="${ML}" x2="${W - MR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${INK.hair}"/>`);
        p.push(txt(ML - 8, y + 3.5, minus(v.toFixed(tk.dec)), `class="tk" text-anchor="end"`));
      }
    }
    const xstep = T1 <= 12 ? 2 : T1 <= 36 ? 5 : 10;
    for (let v = 0; v <= T1 + 1e-9; v += xstep) {
      const x = X(v);
      p.push(`<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${top[0]}" y2="${top[2] + PH[2]}" stroke="${INK.hair}" stroke-opacity=".55"/>`);
      p.push(txt(x, top[2] + PH[2] + 18, String(v), `class="tk" text-anchor="middle"`));
    }
    p.push(txt(W - MR, top[2] + PH[2] + 34, t("ax_time"), `class="tk" text-anchor="end"`));
    const heads = [[t("ch_temp"), t("ax_temp")], [t("ch_grad"), t("ax_grad")], [t("ch_di"), t("ax_di")]];
    heads.forEach(([h, u], k) => p.push(`<text x="${ML}" y="${top[k] - 12}" class="ph">${esc(h)}<tspan class="pu" dx="10">${esc(u)}</tspan></text>`));

    // hot air on the surface, across all three panels
    for (const [a, b] of spans) for (let k = 0; k < 3; k++)
      p.push(`<rect x="${X(a).toFixed(1)}" y="${top[k]}" width="${Math.max(1.2, X(b) - X(a)).toFixed(1)}" height="${PH[k]}" fill="${rgba(hotC, 0.16)}"/>`);
    // zone boundaries, and zone names where there is room
    let ta = 0;
    ts.zones.forEach((z, i) => {
      const x0 = ta; ta += z.duration;
      if (ta < T1 * 0.98) for (let k = 0; k < 3; k++)
        p.push(`<line x1="${X(ta).toFixed(1)}" x2="${X(ta).toFixed(1)}" y1="${top[k]}" y2="${top[k] + PH[k]}" stroke="${INK.hair2}" stroke-dasharray="2 3"/>`);
      if (z.duration >= 2.5 && X(ta) - X(x0) > 44) p.push(txt((X(x0) + X(ta)) / 2, top[0] + 13, fmt(t("zone_n"), { n: i + 1 }), `class="zl" text-anchor="middle"`));
    });

    // panel 1: mushy band, target and fill, then air, core and surface
    const yb = Y[0](mliq), yb2 = Y[0](msol);
    p.push(`<rect x="${ML}" y="${yb.toFixed(1)}" width="${W - ML - MR}" height="${(yb2 - yb).toFixed(1)}" fill="${rgba(INK.sand, 0.85)}"/>`);
    p.push(txt(W - MR - 4, yb - 5, fmt(t("ch_mushy"), { lo: msol, hi: mliq }), `class="an" fill="${INK.umber}" text-anchor="end"`));
    const yT = Y[0](37), yF = Y[0](ts.T_fill);
    p.push(`<line x1="${ML}" x2="${W - MR}" y1="${yT.toFixed(1)}" y2="${yT.toFixed(1)}" stroke="${rgba(INK.indigo, 0.7)}" stroke-dasharray="2 3"/>`);
    p.push(txt(W - MR - 4, yT + 13, fmt(t("ch_target"), { T: 37 }), `class="an" fill="${INK.indigo}" text-anchor="end"`));
    p.push(`<line x1="${ML}" x2="${W - MR}" y1="${yF.toFixed(1)}" y2="${yF.toFixed(1)}" stroke="${INK.hair2}" stroke-dasharray="2 3"/>`);
    p.push(txt(W - MR - 4, yF - 5, fmt(t("ch_fill"), { T: ts.T_fill }), `class="an" fill="${INK.grey}" text-anchor="end"`));
    if (spans.length) p.push(txt(X(spans[0][0]) + 4, top[0] + PH[0] - 7, fmt(t("ch_reheat"), { T: Tair }), `class="an" fill="${INK.hot}"`));
    p.push(`<path class="fade" d="${line(ts.times, Te, X, Y[0])}" fill="none" stroke="${INK.grey}" stroke-width="1.3" stroke-dasharray="1.5 3.5" stroke-linecap="round"/>`);
    p.push(`<path class="fade" d="${line(ts.times, ts.T_core, X, Y[0])}" fill="none" stroke="${INK.indigo}" stroke-width="2" stroke-dasharray="7 5"/>`);
    p.push(`<path class="line" d="${line(ts.times, ts.T_surf, X, Y[0])}" fill="none" stroke="${INK.ink}" stroke-width="2.4" stroke-linejoin="round"/>`);

    // panel 2: core minus surface, and the cooling rate
    p.push(`<line x1="${ML}" x2="${W - MR}" y1="${Y[1](0).toFixed(1)}" y2="${Y[1](0).toFixed(1)}" stroke="${INK.hair2}"/>`);
    p.push(`<path class="fade" d="${line(ts.times, ts.dTdt, X, Y[1])}" fill="none" stroke="${INK.indigo}" stroke-width="1.6" stroke-dasharray="9 5"/>`);
    p.push(`<path class="line" d="${line(ts.times, ts.dT, X, Y[1])}" fill="none" stroke="${INK.umber}" stroke-width="2" stroke-linejoin="round"/>`);

    // panel 3: risk bands, thresholds, the damage index
    for (const [a, b, key] of [[0, 0.25, "SAFE"], [0.25, 0.5, "CAUTION"], [0.5, 0.8, "WARNING"], [0.8, 1.05, "CRITICAL"]])
      p.push(`<rect x="${ML}" y="${Y[2](b).toFixed(1)}" width="${W - ML - MR}" height="${(Y[2](a) - Y[2](b)).toFixed(1)}" fill="${rgba(RISK[key], 0.07)}"/>`);
    for (const [v, key] of [[0.25, "CAUTION"], [0.5, "WARNING"], [0.8, "CRITICAL"]]) {
      const y = Y[2](v);
      p.push(`<line x1="${ML}" x2="${W - MR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${rgba(RISK[key], 0.6)}" stroke-dasharray="5 4"/>`);
      p.push(txt(W - MR - 4, y - 5, t("risk_" + key.toLowerCase()), `class="an" fill="${RISK[key]}" text-anchor="end"`));
    }
    const area = line(ts.times, ts.DI, X, Y[2]) + `L${X(T1).toFixed(1)} ${Y[2](0).toFixed(1)}L${X(0).toFixed(1)} ${Y[2](0).toFixed(1)}Z`;
    p.push(`<path class="fade" d="${area}" fill="${rgba(INK.ink, 0.06)}"/>`);
    p.push(`<path class="line" d="${line(ts.times, ts.DI, X, Y[2])}" fill="none" stroke="${INK.ink}" stroke-width="2.4" stroke-linejoin="round"/>`);
    if (spans.length) {
      const is = nearest(ts.times, spans[0][0]), ie = nearest(ts.times, spans[spans.length - 1][1]);
      const drop = ts.DI[is] - ts.DI[Math.min(ie, n - 1)];
      if (drop > 0.005) {
        const s = fmt(t("ch_heal"), { d: drop }), x = clamp(X((spans[0][0] + spans[spans.length - 1][1]) / 2), ML + 70, W - MR - 70);
        const y = Math.max(top[2] + 16, Y[2](ts.DI[Math.min(ie, n - 1)] + 0.07));
        const w = s.length * 6.3 + 18;
        p.push(`<rect x="${(x - w / 2).toFixed(1)}" y="${(y - 15).toFixed(1)}" width="${w.toFixed(1)}" height="22" rx="6" fill="${INK.white}" stroke="${rgba(hotC, 0.6)}"/>`);
        p.push(txt(x, y + 0.5, s, `class="an" fill="${INK.hot}" text-anchor="middle"`));
      }
    }
    // hover read-out
    p.push(`<g class="xh" visibility="hidden"><line class="xl" y1="${top[0]}" y2="${top[2] + PH[2]}" stroke-width="1.5"/>` +
      `<circle class="d1" r="5" fill="${INK.ink}" stroke="${INK.paper}" stroke-width="1.5"/><circle class="d2" r="5" fill="${INK.indigo}" stroke="${INK.paper}" stroke-width="1.5"/>` +
      `<circle class="d3" r="4.5" fill="${INK.umber}" stroke="${INK.paper}" stroke-width="1.5"/><circle class="d4" r="5.5" stroke="${INK.paper}" stroke-width="1.5"/></g>`);
    p.push(`<rect class="hit" x="${ML}" y="${top[0]}" width="${W - ML - MR}" height="${top[2] + PH[2] - top[0]}" fill="transparent"/>`);

    host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(t("th_temp"))}" style="font-size:${fs}px">` +
      `<style>.tk{fill:${INK.grey};font:500 ${fs - 0.5}px var(--mono)}.ph{fill:${INK.ink};font:600 ${fs + 2}px var(--sans)}.pu{fill:${INK.grey};font:500 ${fs - 0.5}px var(--mono)}` +
      `.zl{fill:${INK.grey};font:500 ${fs - 1}px var(--sans)}.an{font:500 ${fs - 0.5}px var(--sans)}</style>${p.join("")}</svg><div class="ch-tip" aria-hidden="true"></div>`;
    const svg = host.firstChild, tip = host.lastChild, xh = svg.querySelector(".xh");
    svg.querySelectorAll("path.line").forEach(el => el.style.setProperty("--len", Math.ceil(el.getTotalLength() + 2)));
    if (o.animate) svg.classList.add("draw");

    // hover, touch and keyboard: a crosshair through the three panels and a read-out card
    let idx = -1;
    const inSpan = tm => spans.some(([a, b]) => a <= tm && tm < b);
    function show(i) {
      idx = clamp(i, 0, n - 1);
      const tm = ts.times[idx], x = X(tm), hot = inSpan(tm), di = ts.DI[idx];
      xh.setAttribute("visibility", "visible");
      const xl = xh.querySelector(".xl"); xl.setAttribute("x1", x); xl.setAttribute("x2", x);
      xl.setAttribute("stroke", hot ? hotC : rgba(INK.ink, 0.55));
      const put = (c, y) => { const d = xh.querySelector(c); d.setAttribute("cx", x); d.setAttribute("cy", y); };
      put(".d1", Y[0](ts.T_surf[idx])); put(".d2", Y[0](ts.T_core[idx])); put(".d3", Y[1](ts.dT[idx])); put(".d4", Y[2](di));
      xh.querySelector(".d4").setAttribute("fill", rcol(di));
      const row = (k, v) => `<div><span>${esc(k)}</span><span>${esc(v)}</span></div>`;
      tip.innerHTML = `<b>${tm.toFixed(1)} ${esc(t("min"))}</b>` + row(t("tr_surface"), ts.T_surf[idx].toFixed(1) + " °C") +
        row(t("tr_core"), ts.T_core[idx].toFixed(1) + " °C") + row(t("tr_air"), Te[idx].toFixed(0) + " °C") +
        row(t("tr_dT"), minus(ts.dT[idx].toFixed(1)) + " °C") + row(t("tr_di"), di.toFixed(3));
      const r = svg.getBoundingClientRect(), k = r.width / W, px = x * k, tw = tip.offsetWidth || 200;
      tip.style.transform = `translate(${Math.round(px + 18 + tw > r.width ? px - tw - 18 : px + 18)}px,${Math.round((top[0] + 10) * k)}px)`;
      tip.classList.add("on");
    }
    function hide() { idx = -1; xh.setAttribute("visibility", "hidden"); tip.classList.remove("on"); }
    const hit = svg.querySelector(".hit");
    const at = e => { const r = svg.getBoundingClientRect(); return nearest(ts.times, ((e.clientX - r.left) / r.width * W - ML) / (W - ML - MR) * T1); };
    hit.addEventListener("pointermove", e => show(at(e)));
    hit.addEventListener("pointerdown", e => show(at(e)));
    hit.addEventListener("pointerleave", e => { if (e.pointerType === "mouse") hide(); });
    host.tabIndex = 0;
    host.onkeydown = e => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const step = Math.max(1, Math.round(n / 60)) * (e.shiftKey ? 5 : 1);
        show((idx < 0 ? 0 : idx) + (e.key === "ArrowRight" ? step : -step)); e.preventDefault();
      } else if (e.key === "Escape") hide();
    };
    host.onblur = hide;
  }

  // ---------------------------------------------------------------- the sparkline (live panel)
  function spark(svg, ts) {
    const W = 300, H = 70, T1 = ts.t_total, pk = Math.max(...ts.DI);
    const top = Math.max(0.32, pk * 1.15), X = v => v / T1 * W, Y = v => H - 3 - v / top * (H - 8), col = rcol(pk);
    const d = line(ts.times, ts.DI, X, Y), spans = reheatSpans(ts);
    svg.innerHTML = spans.map(([a, b]) => `<rect x="${X(a).toFixed(1)}" y="0" width="${Math.max(1.5, X(b) - X(a)).toFixed(1)}" height="${H}" fill="${rgba(INK.hot2, 0.14)}"/>`).join("") +
      `<line x1="0" x2="${W}" y1="${Y(0.25).toFixed(1)}" y2="${Y(0.25).toFixed(1)}" stroke="${RISK.CAUTION}" stroke-opacity=".7" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>` +
      `<path d="${d}L${W} ${H}L0 ${H}Z" fill="${rgba(col, 0.13)}"/>` +
      `<path d="${d}" fill="none" stroke="${col}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>` +
      `<line x1="0" x2="${W}" y1="${H - 0.5}" y2="${H - 0.5}" stroke="${INK.hair2}" vector-effect="non-scaling-stroke"/>`;
  }

  // ---------------------------------------------------------------- cracks (the 3D line's rule: none below 0.25,
  // complete by 0.6)
  function cracks(di, seed, c, R, detail) {
    const rng = mulberry(seed), u = (a, b) => a + (b - a) * rng(), out = [];
    const f = di >= 0.25 ? 0.18 + 0.82 * Math.min(1, (di - 0.25) / 0.35) : 0;
    if (f > 0) {
      const arms = 3 + Math.round(f * 2);
      for (let k = 0; k < arms; k++) {
        let ang = 2 * Math.PI * k / arms + u(-0.45, 0.45), x = 0, y = 0;
        const pts = [[c, c]], steps = Math.floor(2 + f * 6) * detail;
        for (let s = 0; s < steps; s++) {
          ang += u(-0.45, 0.45) / Math.sqrt(detail); const st = R * u(0.12, 0.19) / detail;
          x += Math.cos(ang) * st; y += Math.sin(ang) * st;
          if (Math.hypot(x, y) > R * 0.96) break;
          pts.push([c + x, c + y]);
        }
        out.push(pts);
        if (detail > 1 && pts.length > 4) {                // a side branch on the bigger drawings
          const j = 2 + Math.floor(rng() * (pts.length - 3)), [bx, by] = pts[j];
          let a2 = ang + (rng() < 0.5 ? -1 : 1) * u(0.7, 1.2), x2 = bx - c, y2 = by - c; const br = [[bx, by]];
          for (let s = 0; s < 3 + Math.floor(f * 4); s++) {
            a2 += u(-0.4, 0.4); x2 += Math.cos(a2) * R * 0.07; y2 += Math.sin(a2) * R * 0.07;
            if (Math.hypot(x2, y2) > R * 0.95) break;
            br.push([c + x2, c + y2]);
          }
          if (br.length > 1) out.push(br);
        }
      }
    }
    return { f, paths: out.map(pts => "M" + pts.map(([x, y]) => x.toFixed(1) + " " + y.toFixed(1)).join(" L")) };
  }
  function glyph(di, seed) {                               // case cards: a top view of the stick, 46 px
    const { f, paths } = cracks(di, seed, 23, 19, 1), col = rcol(di);
    const ring = f === 0 ? RISK.SAFE : "rgba(31,29,26,.22)";
    return `<svg viewBox="0 0 46 46" aria-hidden="true"><circle cx="23" cy="23" r="19" fill="#EFEBE3" stroke="${ring}" stroke-width="1.3"/>` +
      paths.map(d => `<path d="${d}" fill="none" stroke="#1F1D1A" stroke-width="3.6" stroke-linecap="round" stroke-opacity=".5"/><path d="${d}" fill="none" stroke="${col}" stroke-width="1.6" stroke-linecap="round"/>`).join("") +
      (f === 0 ? `<path d="M16.5 23.5l4.5 4.5 8.5-9" fill="none" stroke="${RISK.SAFE}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>` : "") + `</svg>`;
  }
  // Section 05: the top of the stick seen from above (cracks by the 3D line's rule), and a schematic section through
  // it with the top drawn down by the damage index.
  function surfaceArt(di, seed, o) {
    const { f, paths } = cracks(di, seed, 150, 118, 2), col = rcol(di), id = "s" + seed + "_" + Math.round(di * 1000);
    const glowA = Math.min(0.55, di * 0.75), depth = 26 * Math.min(1, di / 0.6);
    const cx = 415, w = 170, y0 = 74, yb = 266, x0 = cx - w / 2, x1 = cx + w / 2;
    const sink = `M${x0} ${y0} Q${cx} ${(y0 + 2 * depth).toFixed(1)} ${x1} ${y0}`;
    let side = "";
    if (f > 0) {                                           // cracks running down from the sink, in section
      const rng = mulberry(seed + 7), ln = 30 + 70 * f;
      for (let k = 0; k < 1 + Math.round(f * 2); k++) {
        let x = cx + (k - 1) * 16 + (rng() - 0.5) * 8, y = y0 + depth; const pts = [[x, y]];
        for (let s = 0; s < 4; s++) { x += (rng() - 0.5) * 12; y += ln / 4; pts.push([x, y]); }
        side += `<path d="M${pts.map(([a, b]) => a.toFixed(1) + " " + b.toFixed(1)).join(" L")}" fill="none" stroke="#1F1D1A" stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>`;
      }
    }
    return `<svg viewBox="0 0 520 300" role="img" aria-label="${esc(o.label)}"><defs>` +
      `<radialGradient id="w${id}" cx=".42" cy=".38" r=".7"><stop offset="0" stop-color="#F1ECE2"/><stop offset=".75" stop-color="#E2DACB"/><stop offset="1" stop-color="#CFC5B3"/></radialGradient>` +
      `<radialGradient id="g${id}"><stop offset="0" stop-color="${col}" stop-opacity="${glowA.toFixed(2)}"/><stop offset=".75" stop-color="${col}" stop-opacity="0"/></radialGradient>` +
      `<linearGradient id="v${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ECE6DA"/><stop offset="1" stop-color="#BDB3A2"/></linearGradient>` +
      `<clipPath id="c${id}"><path d="${sink} L${x1} ${yb} L${x0} ${yb} Z"/></clipPath>` +
      `<marker id="a${id}" viewBox="0 0 8 8" refX="4" refY="7" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L4 8L8 0" fill="${col}"/></marker></defs>` +
      `<circle cx="150" cy="150" r="128" fill="none" stroke="rgba(237,232,223,.16)"/>` +
      `<circle cx="150" cy="150" r="118" fill="url(#w${id})"/><circle cx="150" cy="150" r="118" fill="url(#g${id})"/>` +
      paths.map(d => `<path class="ck" d="${d}" fill="none" stroke="#1F1D1A" stroke-width="5" stroke-linecap="round" stroke-opacity=".55"/>` +
        `<path class="ck" d="${d}" fill="none" stroke="${col}" stroke-width="2.2" stroke-linecap="round"/>`).join("") +
      (f === 0 ? `<text x="150" y="160" text-anchor="middle" style="font:italic 400 30px var(--serif);fill:${RISK.SAFE}">${esc(o.free)}</text>` : "") +
      `<g clip-path="url(#c${id})"><rect x="${x0}" y="${y0 - 2}" width="${w}" height="${yb - y0 + 2}" fill="url(#v${id})"/>${side}</g>` +
      `<path d="${sink}" fill="none" stroke="${INK.nightText}" stroke-width="1.6"/>` +
      `<path d="M${x0} ${y0} L${x0} ${yb} M${x1} ${y0} L${x1} ${yb}" stroke="rgba(237,232,223,.35)" stroke-width="1.2"/>` +
      `<path d="M${x0 - 10} ${y0 - 10} L${x0 - 10} ${yb + 6} M${x1 + 10} ${y0 - 10} L${x1 + 10} ${yb + 6}" stroke="rgba(237,232,223,.18)" stroke-width="1.2"/>` +
      (depth > 3 ? `<path d="M${cx} ${y0 - 24} L${cx} ${(y0 + depth - 5).toFixed(1)}" stroke="${col}" stroke-width="1.3" marker-end="url(#a${id})"/>` : "") +
      `<text x="150" y="296" text-anchor="middle" class="sv-l">${esc(o.top)}</text><text x="${cx}" y="296" text-anchor="middle" class="sv-l">${esc(o.sec)}</text></svg>`;
  }

  // ---------------------------------------------------------------- the hero X-ray: a section through the stick
  // Top surface and core come from the model. The side wall follows the cooling air alone (the model's surface
  // node without the hot air), and the field between them is drawn from the distance to each surface: a picture
  // of the model, not a solved field.
  function sideWall(ts, hCool) {
    const C = G.PIDRL.C, n = ts.times.length, out = new Array(n).fill(ts.T_fill), zb = []; let ta = 0;
    for (const z of ts.zones) { zb.push([ta, ta + z.duration, z.T]); ta += z.duration; }
    const tauC = Math.max(2, C.RHO * C.CP * C.R_M ** 2 / (C.K_TH * (hCool * C.R_M / C.K_TH + 0.1) * 10)) / 60;
    const env = tm => { for (const [a, b, T] of zb) if (a <= tm && tm <= b) return T; return ts.late_cool_T; };
    for (let i = 1; i < n; i++) {
      const tm = ts.times[i], dt = tm - ts.times[i - 1];
      const seqHot = ts.reheat_mode === "sequential" && ts.reheat_start <= tm && tm < ts.reheat_end;
      const e = seqHot ? ts.reheat_T : tm <= ts.t_cool ? env(tm) : ts.late_cool_T;
      const tau = seqHot || tm <= ts.t_cool ? tauC : tauC * 1.5;
      out[i] = e + (out[i - 1] - e) * Math.exp(-dt / Math.max(tau, 0.01));
    }
    return out;
  }
  function Xray(svg, canvas) {
    const NX = canvas.width, NY = canvas.height, ctx = canvas.getContext("2d"), img = ctx.createImageData(NX, NY);
    const BX = 150, BY = 96, BW = 220, BH = 352, cx = BX + BW / 2, BX0 = 40, BX1 = 480;
    const field = new Float32Array(NX * NY);
    let ts = null, wall = null, spans = [], L = {};
    const mono = `font:500 11px var(--mono);letter-spacing:.1em;fill:${INK.nightGrey}`, big = `font:400 36px var(--serif);fill:${INK.nightText}`;
    svg.innerHTML = `<defs><linearGradient id="xrAir" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${INK.hot2}" stop-opacity="0"/><stop offset="1" stop-color="${INK.hot2}" stop-opacity=".95"/></linearGradient></defs>` +
      `<path d="M${BX - 12} ${BY - 8} L${BX - 12} ${BY + BH + 12} L${BX + BW + 12} ${BY + BH + 12} L${BX + BW + 12} ${BY - 8}" fill="none" stroke="rgba(237,232,223,.22)" stroke-width="1.4"/>` +
      `<path class="xr-mask" fill="${INK.night}"/><path class="xr-front" fill="none" stroke="rgba(237,232,223,.8)" stroke-width="1.3" stroke-linecap="round"/>` +
      `<path class="xr-top" fill="none" stroke="${INK.nightText}" stroke-width="2"/><g class="xr-cracks"></g>` +
      `<g class="xr-air"><path d="M${cx - 46} 22 L${cx + 46} 22 L${cx + 28} 48 L${cx - 28} 48 Z" fill="#2A2621" stroke="rgba(237,232,223,.3)"/>` +
      [-34, -12, 12, 34].map(dx => `<path class="xr-jet" d="M${cx + dx * 0.6} 52 Q${cx + dx * 0.85} 70 ${cx + dx} ${BY - 6}" fill="none" stroke="url(#xrAir)" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="6 7"/>`).join("") + `</g>` +
      `<g class="xr-cool">${[0.22, 0.48, 0.74].map(f => { const y = BY + BH * f; return `<path d="M${BX - 58} ${y} q8 -6 16 0 t16 0" fill="none" stroke="${INK.indigo2}" stroke-width="1.6" stroke-linecap="round"/><path d="M${BX + BW + 26} ${y} q8 -6 16 0 t16 0" fill="none" stroke="${INK.indigo2}" stroke-width="1.6" stroke-linecap="round"/>`; }).join("")}</g>` +
      `<line class="xr-l1" stroke="rgba(237,232,223,.45)"/><line class="xr-l2" stroke="rgba(237,232,223,.45)"/>` +
      `<circle class="xr-d1" r="4.5" fill="${INK.nightText}"/><circle class="xr-d2" r="4.5" fill="${INK.nightText}" stroke="${INK.night}" stroke-width="1.5"/>` +
      `<text class="xr-k1" x="400" style="${mono}"></text><text class="xr-v1" x="398" style="${big}"></text>` +
      `<text class="xr-k2" x="400" style="${mono}"></text><text class="xr-v2" x="398" style="${big}"></text>` +
      `<text class="xr-air-t" x="${cx + 58}" y="40" style="font:500 11px var(--mono);fill:${INK.hot2}"></text>` +
      `<text class="xr-di" x="${BX - 12}" y="${BY - 22}" style="font:500 11px var(--mono);letter-spacing:.04em"></text>` +
      `<text class="xr-fr" x="${BX - 22}" text-anchor="end" style="font:500 10.5px var(--sans);fill:${INK.nightGrey}"></text>` +
      `<g class="xr-bar"></g><line class="xr-ph" y1="498" y2="528" stroke="${INK.nightText}" stroke-width="1.5"/><circle class="xr-pd" cy="515" r="4" fill="${INK.nightText}"/>` +
      `<text x="${BX0}" y="548" style="${mono}">0</text><text class="xr-t1" x="${BX1}" y="548" text-anchor="end" style="${mono}"></text>`;
    const q = s => svg.querySelector(s);
    const E = {}; ["mask", "front", "top", "cracks", "air", "cool", "l1", "l2", "d1", "d2", "k1", "v1", "k2", "v2", "air-t", "di", "fr", "bar", "ph", "pd", "t1"]
      .forEach(k => { E[k] = q(".xr-" + k); });
    function set(t, hCool, labels) {
      ts = t; wall = sideWall(t, hCool); spans = reheatSpans(t); L = labels;
      const T1 = t.t_total, X = v => BX0 + v / T1 * (BX1 - BX0), segs = [];
      let ta = 0;
      for (const z of t.zones) { segs.push(`<rect x="${X(ta).toFixed(1)}" y="512" width="${Math.max(1, X(ta + z.duration) - X(ta) - 1.5).toFixed(1)}" height="6" rx="1.5" fill="${heat(z.T)}"/>`); ta += z.duration; }
      if (t.reheat_mode === "sequential" && t.t_reheat > 0) segs.push(`<rect x="${X(t.reheat_start).toFixed(1)}" y="512" width="${Math.max(1, X(t.reheat_end) - X(t.reheat_start) - 1.5).toFixed(1)}" height="6" rx="1.5" fill="${heat(t.reheat_T)}"/>`);
      const fin0 = t.reheat_mode === "sequential" ? t.t_cool + t.t_reheat : t.t_cool;
      segs.push(`<rect x="${X(fin0).toFixed(1)}" y="512" width="${Math.max(1, X(T1) - X(fin0)).toFixed(1)}" height="6" rx="1.5" fill="${heat(t.late_cool_T)}"/>`);
      if (t.reheat_mode !== "sequential") for (const [a, b] of spans) segs.push(`<rect x="${X(a).toFixed(1)}" y="502" width="${Math.max(1.5, X(b) - X(a)).toFixed(1)}" height="5" rx="1" fill="${INK.hot2}"/>`);
      E.bar.innerHTML = segs.join("");
      E.t1.textContent = T1.toFixed(0) + " " + L.min;
      E.k1.textContent = L.top.toUpperCase(); E.k2.textContent = L.core.toUpperCase();
    }
    // marching squares on the sampled field: the solidification front, at the melting point
    function front(level) {
      const sx = BW / NX, sy = BH / NY, X = i => BX + (i + 0.5) * sx, Y = j => BY + (j + 0.5) * sy, out = [];
      const v = (i, j) => field[j * NX + i];
      let ymin = Infinity;
      for (let j = 0; j < NY - 1; j++) for (let i = 0; i < NX - 1; i++) {
        const a = v(i, j), b = v(i + 1, j), c = v(i + 1, j + 1), d = v(i, j + 1);
        const k = (a > level) | ((b > level) << 1) | ((c > level) << 2) | ((d > level) << 3);
        if (k === 0 || k === 15) continue;
        const lp = (p, q2) => (level - p) / (q2 - p);
        const T = () => [X(i) + lp(a, b) * sx, Y(j)], R = () => [X(i + 1), Y(j) + lp(b, c) * sy];
        const B = () => [X(i) + lp(d, c) * sx, Y(j + 1)], Lf = () => [X(i), Y(j) + lp(a, d) * sy];
        const e = { 1: [Lf, T], 14: [Lf, T], 2: [T, R], 13: [T, R], 3: [Lf, R], 12: [Lf, R], 4: [R, B], 11: [R, B], 6: [T, B], 9: [T, B],
          7: [Lf, B], 8: [Lf, B], 5: [Lf, T, R, B], 10: [T, R, B, Lf] }[k].map(fn => fn());
        for (let m = 0; m < e.length; m += 2) {
          out.push(`M${e[m][0].toFixed(1)} ${e[m][1].toFixed(1)}L${e[m + 1][0].toFixed(1)} ${e[m + 1][1].toFixed(1)}`);
          ymin = Math.min(ymin, e[m][1], e[m + 1][1]);
        }
      }
      return [out.join(""), ymin];
    }
    function draw(tm, phase) {                             // tm: minutes into the cycle; phase 0..1 for the air
      if (!ts) return null;
      const n = ts.times.length; let i = 0;
      while (i < n - 1 && ts.times[i + 1] <= tm) i++;
      const j = Math.min(n - 1, i + 1), f = j > i ? clamp((tm - ts.times[i]) / (ts.times[j] - ts.times[i]), 0, 1) : 0;
      const mix = a => a[i] + (a[j] - a[i]) * f;
      const Ts = mix(ts.T_surf), Tc = mix(ts.T_core), Tw = mix(wall), di = mix(ts.DI);
      const hot = spans.some(([a, b]) => a <= tm && tm < b);
      const dl = 0.17 * BW / 2, lo = Math.min(Ts, Tc, Tw) - 1, hi = Math.max(Ts, Tc, Tw) + 1;
      for (let jj = 0; jj < NY; jj++) {
        const yTop = (jj + 0.5) / NY * BH, eT = Math.exp(-yTop / dl), eB = 0.65 * Math.exp(-(BH - yTop) / dl);
        for (let ii = 0; ii < NX; ii++) {
          const xs = (ii + 0.5) / NX * BW, eW = Math.min(1, Math.exp(-Math.min(xs, BW - xs) / dl) + eB);
          const T = clamp(Tc + (Ts - Tc) * eT + (Tw - Tc) * eW * (1 - eT), lo, hi);
          field[jj * NX + ii] = T;
          const [r, g, b] = heatRGB(T), k = (jj * NX + ii) * 4;
          img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      const [fd, fy] = front(ts.melt);
      E.front.setAttribute("d", fd);
      E.fr.textContent = fd ? L.front : ""; E.fr.setAttribute("y", Math.min(BY + BH - 10, Math.max(BY + 14, fy + 4)).toFixed(1));
      const depth = 26 * Math.min(1, di / 0.6), qy = (BY + 2 * depth).toFixed(1);
      E.top.setAttribute("d", `M${BX} ${BY} Q${cx} ${qy} ${BX + BW} ${BY}`);
      E.mask.setAttribute("d", `M${BX - 1} ${BY - 1} L${BX + BW + 1} ${BY - 1} L${BX + BW + 1} ${BY} Q${cx} ${qy} ${BX - 1} ${BY} Z`);
      if (di >= 0.25) {
        const k = Math.min(1, (di - 0.25) / 0.35), y = BY + depth, ln = 18 + 50 * k;
        E.cracks.innerHTML = [[-14, -0.35], [0, 0.05], [15, 0.4]].slice(0, 1 + Math.round(k * 2)).map(([dx, a]) =>
          `<path d="M${cx + dx} ${y.toFixed(1)} l${(Math.sin(a) * ln * 0.5).toFixed(1)} ${(ln * 0.5).toFixed(1)} l${(-Math.sin(a) * ln * 0.25).toFixed(1)} ${(ln * 0.5).toFixed(1)}" fill="none" stroke="${INK.night}" stroke-width="2.4" stroke-linecap="round"/>`).join("");
      } else E.cracks.innerHTML = "";
      E.air.style.opacity = hot ? 1 : 0.1;
      E.air.querySelectorAll(".xr-jet").forEach(p => p.setAttribute("stroke-dashoffset", (-phase * 52).toFixed(1)));
      E.cool.style.opacity = tm <= ts.t_cool ? 0.85 : 0.3;
      E.cool.setAttribute("transform", `translate(${(Math.sin(phase * Math.PI * 2) * 2).toFixed(2)} 0)`);
      const y1 = BY + 10 + depth * 0.9, y2 = BY + BH / 2;
      E.d1.setAttribute("cx", cx); E.d1.setAttribute("cy", y1); E.d2.setAttribute("cx", cx); E.d2.setAttribute("cy", y2);
      [["l1", y1, y1 - 34], ["l2", y2, y2]].forEach(([k, ya, yb]) => { E[k].setAttribute("x1", cx + 6); E[k].setAttribute("y1", ya); E[k].setAttribute("x2", 392); E[k].setAttribute("y2", yb); });
      E.k1.setAttribute("y", y1 - 62); E.v1.setAttribute("y", y1 - 30); E.v1.textContent = Ts.toFixed(0) + "°C";
      E.k2.setAttribute("y", y2 - 28); E.v2.setAttribute("y", y2 + 6); E.v2.textContent = Tc.toFixed(0) + "°C";
      E["air-t"].textContent = hot ? L.hot.replace("{T}", ts.reheat_T.toFixed(0)) : "";
      E.di.textContent = L.di + "  " + di.toFixed(3);
      E.di.style.fill = di >= 0.25 ? rcol(di) : INK.nightGrey;
      const x = BX0 + tm / ts.t_total * (BX1 - BX0);
      E.ph.setAttribute("x1", x); E.ph.setAttribute("x2", x); E.pd.setAttribute("cx", x);
      return { tm, di, hot };
    }
    return { set, draw };
  }

  G.CHARTS = { INK, RISK, heat, heatRGB, rgba, lum, rcol, riskOf, setStops, thermal, spark, glyph, surfaceArt, Xray, esc, minus };
})(window);
