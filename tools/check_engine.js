// Runs the site's engine (components/site/engine.js) on the cases in a JSON file and prints its results as JSON.
// Called by tools/check_engine.py, which compares them with the Python of main_3d_v4.py.
const fs = require("fs"), path = require("path");
const E = require(path.join(__dirname, "..", "components", "site", "engine.js"));
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = { timelines: [], optimiser: [] };
for (const c of cases.timelines) {
  const a = c.args;
  const ts = E.buildTimeline(a.T_fill, a.zones, a.reheat, a.h_cool, a.h_reheat, 120, a.melt, a.late);
  out.timelines.push({ name: c.name, js: { times: ts.times, T_surf: ts.T_surf, T_core: ts.T_core, DI: ts.DI, t_total: ts.t_total,
    t_reheat: ts.t_reheat, reheat_start: ts.reheat_start, reheat_end: ts.reheat_end, reheat_mode: ts.reheat_mode, n_pulse: ts.n_pulse,
    pulse_sec: ts.pulse_sec } });
}
for (const c of cases.optimiser) {
  const a = c.args;
  const best = E.optimise(a.T_fill, a.reheat, a.h_cool, a.h_reheat, a.melt, a.late, c.scope);
  out.optimiser.push({ name: c.name, scope: c.scope, js: Object.fromEntries(Object.entries(best).map(([k, v]) => [k, { di: v.di, t: v.t, zones: v.zones }])) });
}
process.stdout.write(JSON.stringify(out));
