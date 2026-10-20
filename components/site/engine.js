/* PI-DRL engine: the physics of main_3d_v4.py ported line for line, so the site recomputes the moment a
   setting changes, with no round trip to the server. Python stays the reference: tools/check_engine.py runs
   both on every preset and on the optimiser and fails if they disagree.
   build_timeline: a surface node and a core node relaxing toward their air (cooling zones, hot air, late
   air), the liquid fraction across the mushy band (melting point +/- 5 C), and the damage index ODE
   dDI/dt = k_form (1 - DI) - k_heal DI, integrated exactly per step. */
(function (G) {
  "use strict";
  const C = { T_SOL: 62, T_LIQ: 72, T_MID: 67, K_TH: 0.25, R_M: 0.0125, H_M: 0.04, RHO: 990, CP: 2000,
    T_FILL_DEF: 80, T_TARGET: 37, T_ROOM: 23, CRACK_TH: 0.25, CYCLE: 30, CYCLE_OK: 30.5 };
  const clip = (v, a, b) => Math.min(b, Math.max(a, v));
  // Python's round(x, k): the exact binary value rounded to k decimals, an exact tie to the even digit
  // (69.25 -> 69.2, 4.25 -> 4.2). toFixed rounds the exact value too but sends ties up, so ties are found
  // exactly, from the double's mantissa and exponent, and sent to the even neighbour.
  const _dv = new DataView(new ArrayBuffer(8));
  function rnd(x, k) {
    if (!isFinite(x) || x === 0) return x;
    if (x < 0) return -rnd(-x, k);
    _dv.setFloat64(0, x);
    const hi = _dv.getUint32(0), lo = _dv.getUint32(4), ex = (hi >>> 20) & 0x7ff;
    let M = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo), E = ex - 1075;
    if (ex) M |= 1n << 52n; else E = -1074;
    const P = 10n ** BigInt(k);
    if (E < 0) {                                           // x * 10^k = M * P / 2^-E: a tie when that is an odd half
      const den = 1n << BigInt(-E), num = 2n * M * P;
      if (num % den === 0n) {
        const q = num / den;
        if (q % 2n === 1n) { const lowK = (q - 1n) / 2n; return Number(lowK % 2n === 0n ? lowK : lowK + 1n) / Math.pow(10, k); }
      }
    }
    return Number(x.toFixed(k));
  }
  const sum = a => a.reduce((s, x) => s + x, 0);

  function lf(T, tSol = C.T_SOL, tLiq = C.T_LIQ, tMid = C.T_MID) {      // liquid fraction across the mushy band
    if (T >= tLiq) return 1;
    if (T <= tSol) return 0;
    return 0.5 * (1 + Math.sin(Math.PI * (T - tMid) / (tLiq - tSol)));
  }
  const meltBand = melt => (melt == null ? [C.T_SOL, C.T_LIQ, C.T_MID] : [+melt - 5, +melt + 5, +melt]);

  // ---------------------------------------------------------------- hot-air bursts (pulsed top reheat)
  function evenBursts(n, sec, window) {
    n = Math.trunc(n); const slot = window / Math.max(n, 1);
    return Array.from({ length: n }, (_, k) => [rnd(k * slot, 3), Math.trunc(sec)]);
  }
  function burstSpans(reh, tCool) {
    let b = reh.bursts;
    if (!b || !b.length) {
      const n = Math.trunc(reh.pulses || 0), sec = +(reh.pulse_sec || 0);
      if (n <= 0 || sec <= 0) return [];
      b = evenBursts(n, sec, Math.min(+(reh.pulse_window == null ? 6 : reh.pulse_window), tCool));
    }
    const out = [];
    for (const [s0, d0] of b) {
      const s = +s0, d = +d0, e = Math.min(s + d / 60, tCool);
      if (d > 0 && s >= 0 && s < tCool && e > s) out.push([s, e]);
    }
    return out.sort((a, c) => a[0] - c[0] || a[1] - c[1]);
  }
  function median(a) { const s = a.slice().sort((x, y) => x - y), m = s.length; return m % 2 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2; }
  function nextBurst(bursts, tCool, sec = 20) {           // start of one more burst (the add button)
    const st = bursts.map(b => +b[0]).sort((a, c) => a - c);
    const step = st.length > 1 ? median(st.slice(1).map((x, i) => x - st[i])) : 1.2;
    let s = st.length ? st[st.length - 1] + step : 0;
    if (s + sec / 60 > tCool) {
      const sorted = bursts.slice().sort((a, c) => a[0] - c[0] || a[1] - c[1]);
      const edges = [0, ...sorted.flatMap(b => [+b[0], +b[0] + b[1] / 60]), tCool];
      let best = null;
      for (let i = 0; i < edges.length - 1; i += 2) {
        const g = [edges[i + 1] - edges[i], edges[i]];
        if (!best || g[0] > best[0] || (g[0] === best[0] && g[1] > best[1])) best = g;
      }
      s = best[1] + best[0] / 2 - sec / 120;
    }
    return rnd(clip(s, 0, Math.max(tCool - sec / 60, 0)), 2);
  }
  function syncPattern(reh) {                              // after a hand edit: the closest even pattern
    const b = reh.bursts || [];
    if (!b.length) return;
    reh.pulses = b.length;
    reh.pulse_sec = Math.trunc(clip(Math.round(sum(b.map(x => x[1])) / b.length / 5) * 5, 5, 60));
    reh.pulse_window = clip(Math.round(Math.max(...b.map(x => x[0] + x[1] / 60)) * 2) / 2, 2, 15);
  }

  // ---------------------------------------------------------------- the timeline
  function buildTimeline(Tfill, zones, reh, hCool, hReh, nPts = 120, melt = null, lateCool = null) {
    Tfill = +Tfill; hCool = +hCool; hReh = +hReh;
    const mode = String(reh.mode || "sequential").toLowerCase();
    const tCool = sum(zones.map(z => +z.duration));
    const tReh = +(reh.duration || 0);
    const spans = mode === "pulsed" ? burstSpans(reh, tCool) : [];
    const pulsed = spans.length > 0, nPulse = spans.length;
    const pulseS = pulsed ? rnd(sum(spans.map(([s, e]) => (e - s) * 60)) / spans.length, 2) : 0;
    const simul = mode === "simultaneous" && tReh > 0;
    const surfReheat = simul || pulsed;
    const [tSol, tLiq, tMid] = meltBand(melt);
    const lateT = lateCool == null ? C.T_ROOM : +lateCool;

    const zb = []; let ta = 0;
    for (const z of zones) { zb.push([ta, ta + +z.duration, +z.T]); ta += +z.duration; }
    const pulseWin = pulsed ? Math.max(...spans.map(s => s[1])) : 0;
    let rehS, rehE, tFin, tTot;
    if (surfReheat) {
      rehS = 0; rehE = pulsed ? pulseWin : Math.min(tReh, tCool);
      tFin = clip(30 - tCool - 0.5, 2, 8); tTot = tCool + tFin;
    } else {
      rehS = ta; rehE = ta + tReh;
      tFin = clip(30 - tCool - tReh - 0.5, 2, 8); tTot = tCool + tReh + tFin;
    }
    const surfHot = tm => (simul ? rehS <= tm && tm < rehE : spans.some(([s, e]) => s <= tm && tm < e));
    if (pulsed) {
      const dMin = Math.min(...spans.map(([s, e]) => e - s));
      nPts = Math.max(nPts, Math.trunc(tTot / Math.max(dMin / 2, 0.02)) + 1, 260);
    }
    const times = Array.from({ length: nPts }, (_, i) => (nPts > 1 ? tTot * i / (nPts - 1) : 0));
    const tauC = Math.max(2, C.RHO * C.CP * C.R_M ** 2 / (C.K_TH * (hCool * C.R_M / C.K_TH + 0.1) * 10)) / 60;
    const tauR = Math.max(1, C.RHO * C.CP * C.R_M ** 2 / (C.K_TH * (hReh * C.R_M / C.K_TH + 0.1) * 10)) / 60;
    const tauSkin = 0.30;                                  // the thin top skin re-melts fast under hot air
    const zoneEnv = tm => { for (const [t0, t1, Tz] of zb) if (t0 <= tm && tm <= t1) return Tz; return C.T_ROOM; };

    const Ts = new Array(nPts).fill(0), Tc = new Array(nPts).fill(0), Te = new Array(nPts).fill(0), H = new Array(nPts).fill(0);
    Ts[0] = Tc[0] = Tfill;
    for (let i = 1; i < nPts; i++) {
      const tm = times[i], dt = times[i] - times[i - 1];
      let envCore, tauCore, Hcore, envSurf, tauSurf, Hsurf;
      if (surfReheat) {                                    // the core follows the cooling schedule, never reheated
        if (tm <= tCool) { envCore = zoneEnv(tm); tauCore = tauC; Hcore = hCool; }
        else { envCore = lateT; tauCore = tauC * 1.5; Hcore = hCool; }
        if (surfHot(tm)) { envSurf = +reh.T; tauSurf = pulsed ? Math.min(tauR, tauSkin) : tauR; Hsurf = hReh; }
        else { envSurf = envCore; tauSurf = tauCore; Hsurf = Hcore; }
      } else {
        let env, tau;
        if (tm < rehS) { env = zoneEnv(tm); tau = tauC; Hsurf = hCool; }
        else if (tm < rehE) { env = +reh.T; tau = tauR; Hsurf = hReh; }
        else { env = lateT; tau = tauC * 1.5; Hsurf = hCool; }
        envSurf = envCore = env; tauSurf = tauCore = tau; Hcore = Hsurf;
      }
      H[i] = Hsurf; Te[i] = envSurf;
      Ts[i] = envSurf + (Ts[i - 1] - envSurf) * Math.exp(-dt / Math.max(tauSurf, 0.01));
      Ts[i] = clip(Ts[i], Math.min(C.T_ROOM, lateT) - 2, Tfill + 2);
      const lag = clip(1 / (1 + Hcore * C.R_M / C.K_TH * 0.5), 0.3, 0.9);
      Tc[i] = envCore + (Tc[i - 1] - envCore) * Math.exp(-dt / Math.max(tauCore * (1 + lag), 0.01));
    }

    const DIa = new Array(nPts).fill(0); let DI = 0;
    for (let i = 1; i < nPts; i++) {
      const dt = times[i] - times[i - 1], T = Ts[i], Tc_ = Tc[i], Bi = H[i] * C.R_M / C.K_TH;
      const fl = lf(T, tSol, tLiq, tMid), tm = times[i];
      const mw = 4 * fl * (1 - fl);                        // a bell that peaks at fl = 0.5
      const BiR = clip((Bi - 0.15) / 0.85, 0, 1);
      const cs = clip(Math.max(0, Tfill - Te[i]) / 60, 0, 1);
      const inv = clip((T - Tc_) / 20, 0, 1);              // surface warmer than the core: the sink cannot form
      const kForm = fl > 0.01 ? (0.60 * BiR * cs * (1 + 2 * mw) + 0.15 * BiR * cs) * (1 - 0.9 * inv) : 0;
      const hd = Math.max(0, T - 55) / 10;
      const inReh = surfReheat ? surfHot(tm) : rehS <= tm && tm < rehE && tReh > 0;
      const kHeal = clip(hd * (inReh ? 3.5 : 0.2) + 1.5 * inv, 0, 2.5);
      const kT = kForm + kHeal;
      if (kT > 1e-9) { const eq = kForm / kT; DI = eq + (DI - eq) * Math.exp(-kT * dt); }
      DI = clip(DI, 0, 1); DIa[i] = DI;
    }
    const dTdt = Ts.map((_, i) => {                        // numpy.gradient, first order at the ends
      if (nPts < 2) return 0;
      if (i === 0) return (Ts[1] - Ts[0]) / (times[1] - times[0]);
      if (i === nPts - 1) return (Ts[i] - Ts[i - 1]) / (times[i] - times[i - 1]);
      return (Ts[i + 1] - Ts[i - 1]) / (times[i + 1] - times[i - 1]);
    });
    const tReheatOut = pulsed ? sum(spans.map(([s, e]) => e - s)) : simul ? rehE - rehS : tReh;
    return {
      times, T_surf: Ts, T_core: Tc, T_env: Te, dT: Tc.map((c, i) => c - Ts[i]), dTdt, DI: DIa, T_fill: Tfill,
      t_cool: tCool, t_reheat: tReheatOut, t_final: tFin, t_total: tTot, reheat_start: rehS, reheat_end: rehE,
      reheat_T: +reh.T, reheat_mode: pulsed ? "pulsed" : simul ? "simultaneous" : "sequential",
      n_pulse: pulsed ? nPulse : 0, pulse_sec: pulsed ? pulseS : 0, pulse_window: pulsed ? pulseWin : 0,
      bursts: spans.map(([s, e]) => [rnd(s, 4), rnd(e, 4)]), melt: melt == null ? tMid : +melt, late_cool_T: lateT,
      mushy: [tSol, tLiq], zones: zones.map(z => ({ T: +z.T, duration: +z.duration, label: z.label })),
    };
  }

  // ---------------------------------------------------------------- the optimiser (grid search)
  function reheatKind(reh) {
    if (reh.mode === "pulsed") return "pulsed";
    if (+(reh.duration || 0) <= 0) return "none";
    return reh.mode || "sequential";
  }
  function zonePlan(Tfill, nz, dist, budget) {
    const temps = dist === "top_heavy"
      ? Array.from({ length: nz }, (_, k) => Tfill - (Tfill - C.T_TARGET) * Math.pow((k + 1) / nz, 0.55))
      : Array.from({ length: nz }, (_, k) => Tfill - (k + 1) * (Tfill - C.T_TARGET) / nz);
    let durs;
    if (dist === "mushy_dwell") { const raw = temps.map(T => (T >= 55 && T <= 75 ? 6 : 3)); const r = sum(raw); durs = raw.map(d => d * budget / r); }
    else durs = temps.map(() => budget / nz);
    return temps.map((T, k) => ({ T: rnd(T, 1), duration: rnd(durs[k], 1), label: "Zone " + (k + 1) }));
  }
  const copyReh = r => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, k === "bursts" && v ? v.map(b => b.slice()) : v]));
  function optimise(Tfill, reh0, hCool0, hReh0, melt, late, scope = "all") {
    reh0 = copyReh(reh0);
    const cands = [[reheatKind(reh0), reh0, [+hReh0]]];
    if (scope === "all") {
      for (const T of [65, 72, 78, 85, 92, 100, 110]) for (const m of ["sequential", "simultaneous"])
        cands.push([m, { T, duration: 10, mode: m }, [12, 18, 25]]);
      for (const T of [95, 105]) for (const np of [3, 5, 7])
        cands.push(["pulsed", { T, duration: 0, mode: "pulsed", pulses: np, pulse_sec: 20, pulse_window: 5 }, [18, 25]]);
    }
    const best = {};
    for (const [kind, reh, hRehs] of cands) {
      const overlap = kind !== "sequential";
      const budget = overlap ? 27 : 30 - +(reh.duration || 0) - 3;
      const lean = kind === "pulsed";
      for (const hR of hRehs) for (const hC of [+hCool0, Math.max(2, +hCool0 * 0.6)])
        for (const nz of lean ? [3, 4] : [3, 4, 5]) for (const dist of lean ? ["mushy_dwell", "top_heavy"] : ["mushy_dwell", "linear", "top_heavy"]) {
          const zones = zonePlan(Tfill, nz, dist, budget);
          const tt = buildTimeline(Tfill, zones, reh, hC, hR, 80, melt, late);
          if (tt.t_total > 30.5) continue;
          const pk = Math.max(...tt.DI), score = 0.6 * pk + 0.4 * tt.DI[tt.DI.length - 1];
          if (score < ((best[kind] && best[kind].score) ?? Infinity))
            best[kind] = { score, di: pk, t: tt.t_total, mode: kind, zones, reh: copyReh(reh), h_cool: hC, h_reh: hR };
        }
    }
    return best;
  }

  // ---------------------------------------------------------------- cases and read-outs
  const rlbl = d => (d >= 0.8 ? "CRITICAL" : d >= 0.5 ? "WARNING" : d >= 0.25 ? "CAUTION" : "SAFE");
  function reheatSpans(ts) {
    if (ts.reheat_mode === "pulsed") return ts.bursts.map(b => b.slice());
    if (ts.t_reheat > 0 && ts.reheat_end > ts.reheat_start) return [[ts.reheat_start, ts.reheat_end]];
    return [];
  }
  const nearest = (a, v) => { let k = 0, d = Infinity; a.forEach((x, i) => { const e = Math.abs(x - v); if (e < d) { d = e; k = i; } }); return k; };
  function heal(ts) {                                      // damage healed by the hot air: its start minus its end
    if (ts.t_reheat <= 0) return 0;
    const i0 = nearest(ts.times, ts.reheat_start), i1 = nearest(ts.times, ts.reheat_end);
    return ts.DI[i0] - ts.DI[Math.min(i1, ts.DI.length - 1)];
  }
  function worstBest(opts) {
    const worst = opts.reduce((a, o) => (o.peak > a.peak ? o : a));
    const ok = opts.filter(o => o.cycle <= C.CYCLE_OK);
    const pool = ok.length ? ok : opts;
    const best = pool.reduce((a, o) => (o.peak < a.peak || (o.peak === a.peak && o.cycle < a.cycle) ? o : a));
    return [worst, best];
  }
  function categorise(opts) {
    const [worst, best] = worstBest(opts);
    for (const o of opts) o.cat = o === worst ? "worst" : o === best ? "best" : o.peak < 0.25 && o.cycle <= C.CYCLE_OK ? "good" : "work";
    return opts;
  }

  const API = { C, clip, rnd, lf, meltBand, evenBursts, burstSpans, nextBurst, syncPattern, buildTimeline, reheatKind,
    zonePlan, optimise, rlbl, reheatSpans, heal, nearest, worstBest, categorise };
  G.PIDRL = API;
  if (typeof module !== "undefined" && module.exports) module.exports = API;
})(typeof window !== "undefined" ? window : globalThis);
