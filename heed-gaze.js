/*!
 * heed-gaze 0.5.0 — the gaze model behind heed-face
 * Face landmarks -> 22 features -> weighted ridge regression -> a point on the screen (fractions
 * of the viewport), plus an offset learned from recent checks and clicks, and the saved form of
 * the samples. No DOM and no camera, so it also runs in Node (see test/).
 *
 *   import { features, createGaze } from './heed-gaze.js';
 *   const gaze = createGaze();
 *   gaze.samples.grid.push({ x, y, f: features(landmarks, matrix, aspect).vec });
 *   gaze.refit(innerWidth, innerHeight);
 *   gaze.predict(f);             // { x, y }
 */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const median = a => { const s = a.slice().sort((p, q) => p - q); return s.length ? s[Math.floor(s.length / 2)] : 0; };

// saved samples: STORE_VERSION changes whenever features() changes; NF is the length of its vec
export const STORE_VERSION = 1, NF = 22, KINDS = ['grid', 'pursuit', 'clicks', 'checks'];
const WEIGHTS = { grid: 3, pursuit: 1, clicks: 2, checks: 3 };

// ---------- features
function eyeFeat(lm, inner, outer, top, bottom, iris, ax) {
  const ux = (lm[outer].x - lm[inner].x) * ax, uy = lm[outer].y - lm[inner].y, len = Math.hypot(ux, uy) || 1e-6;
  const nx = -uy / len, ny = ux / len, rx = (lm[iris].x - lm[inner].x) * ax, ry = lm[iris].y - lm[inner].y;
  return { h: (rx * ux + ry * uy) / (len * len), v: (rx * nx + ry * ny) / len, open: (lm[bottom].y - lm[top].y) / len, len };
}
export function features(lm, matrix, ax) {
  const m = matrix.data; // 4x4 column-major
  const pitch = Math.atan2(m[6], m[10]), yaw = Math.asin(clamp(-m[2], -1, 1)), roll = Math.atan2(m[1], m[0]);
  const tx = m[12] / 10, ty = m[13] / 10, tz = m[14] / 10;
  let r = { h: 0, v: 0, open: 0, len: 0 }, l = r;
  if (lm.length >= 478) { r = eyeFeat(lm, 133, 33, 159, 145, 468, ax); l = eyeFeat(lm, 362, 263, 386, 374, 473, ax); }
  const hm = (r.h - l.h) / 2, vm = (r.v - l.v) / 2, open = (r.open + l.open) / 2;
  const nose = lm[1], iod = Math.hypot((lm[33].x - lm[263].x) * ax, lm[33].y - lm[263].y);
  const base = [yaw, pitch, roll, tx, ty, tz, r.h, r.v, l.h, l.v, open, nose.x, nose.y, iod];
  const vec = base.concat([yaw * yaw, pitch * pitch, hm * hm, vm * vm, yaw * pitch, hm * yaw, vm * pitch, hm * vm]);
  return { yaw, pitch, roll, vec };
}

// ---------- weighted ridge regression with lambda chosen by 5-fold cross-validation
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-9;
    for (let r = 0; r < n; r++) { if (r === c) continue; const f = M[r][c] / d; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-9));
}
export function fitModel(samples, W, H) {
  const k = samples[0].f.length, sw = samples.reduce((a, s) => a + s.w, 0);
  const mean = Array(k).fill(0), sd = Array(k).fill(0);
  samples.forEach(s => s.f.forEach((x, i) => { mean[i] += s.w * x / sw; }));
  samples.forEach(s => s.f.forEach((x, i) => { sd[i] += s.w * (x - mean[i]) ** 2 / sw; }));
  // a feature that only varies by rounding (saved samples keep 6 digits) is constant: z = 0
  for (let i = 0; i < k; i++) { sd[i] = Math.sqrt(sd[i]); if (!(sd[i] > 1e-5 * (Math.abs(mean[i]) + 1e-3))) sd[i] = Infinity; }
  const z = f => [1].concat(f.map((x, i) => (x - mean[i]) / sd[i]));
  const rows = samples.map(s => ({ x: z(s.f), tx: s.x, ty: s.y, w: s.w }));
  const train = (set, lambda) => {
    const p = k + 1, A = Array.from({ length: p }, () => Array(p).fill(0)), bx = Array(p).fill(0), by = Array(p).fill(0);
    set.forEach(r => { for (let i = 0; i < p; i++) { bx[i] += r.w * r.x[i] * r.tx; by[i] += r.w * r.x[i] * r.ty; for (let j = 0; j < p; j++) A[i][j] += r.w * r.x[i] * r.x[j]; } });
    for (let i = 1; i < p; i++) A[i][i] += lambda * sw / set.length;
    return [solve(A, bx), solve(A, by)];
  };
  const dot = (w, x) => w.reduce((a, wi, i) => a + wi * x[i], 0);
  const err = (wts, set) => set.reduce((a, r) => a + Math.hypot((dot(wts[0], r.x) - r.tx) * W, (dot(wts[1], r.x) - r.ty) * H), 0) / set.length;
  let best = { lambda: 1, cv: Infinity };
  if (rows.length >= 15) {
    [0.01, 0.1, 0.3, 1, 3, 10, 30].forEach(lambda => {
      let e = 0;
      for (let fold = 0; fold < 5; fold++) {
        const tr = rows.filter((_, i) => i % 5 !== fold), te = rows.filter((_, i) => i % 5 === fold);
        if (te.length && tr.length > k) e += err(train(tr, lambda), te) / 5;
      }
      if (e < best.cv) best = { lambda, cv: e };
    });
  } else best = { lambda: 3, cv: null };
  const wts = train(rows, best.lambda);
  return { lambda: best.lambda, cvErrorPx: best.cv === null ? null : Math.round(best.cv),
           predict: f => { const x = z(f); return { x: dot(wts[0], x), y: dot(wts[1], x) }; } };
}

// ---------- samples, model and offset
// The offset is a shift between where the model puts the gaze and where the eyes are, e.g. after
// sitting differently than at calibration: the median miss over this visit's last 8 check points
// and clicks, applied once there are 3, so one careless click cannot move it far.
// samples: { grid, pursuit, clicks, checks }, each a list of { x, y, f } with x, y in 0..1.
export function createGaze({ maxClicks = 200, maxChecks = 60 } = {}) {
  const samples = { grid: [], pursuit: [], clicks: [], checks: [] };
  const feedback = [];                   // this visit's latest check points and clicks, for the offset
  let model = null, offset = { x: 0, y: 0 };
  const fresh = list => list.filter(s => !s.restored);
  const mid = list => ({ x: median(list.map(d => d.x)), y: median(list.map(d => d.y)) });
  const addFeedback = list => { feedback.push(...list); feedback.splice(0, Math.max(0, feedback.length - 8)); };
  function updateOffset() {
    offset = !model || feedback.length < 3 ? { x: 0, y: 0 } : mid(feedback.map(s => { const p = model.predict(s.f); return { x: s.x - p.x, y: s.y - p.y }; }));
  }

  const gaze = {
    samples,
    get model() { return model; },
    get offset() { return offset; },
    all() { return KINDS.flatMap(k => samples[k].map(s => Object.assign({ w: WEIGHTS[k] }, s))); },
    // fits all samples (W, H only scale the reported errors); false while there are fewer than 8
    refit(W, H) {
      const all = gaze.all();
      if (all.length < 8) return false;
      model = fitModel(all, W, H);
      updateOffset();
      return true;
    },
    predict(f) { const p = model.predict(f); return { x: p.x + offset.x, y: p.y + offset.y }; },
    // a click is a sample and evidence for the offset; the offset follows on the next refit
    addClick(s) { samples.clicks.push(s); if (samples.clicks.length > maxClicks) samples.clicks.shift(); addFeedback([s]); },
    // check points become samples and replace the offset evidence
    addChecks(points) { feedback.splice(0); addFeedback(points); samples.checks = samples.checks.concat(points).slice(-maxChecks); },
    // A calibration starts afresh: the new grid and pursuit (null keeps this visit's pursuit), only this
    // visit's clicks, no checks and no offset evidence. Returns a function that undoes it.
    restart(grid, pursuit) {
      const prev = Object.assign({}, samples), prevFeedback = feedback.splice(0);
      Object.assign(samples, { grid, pursuit: pursuit || fresh(samples.pursuit), clicks: fresh(samples.clicks), checks: [] });
      return () => { Object.assign(samples, prev); addFeedback(prevFeedback); };
    },
    // Check points: before is how far the shown gaze point missed; after is what is left once an offset
    // is removed, each point corrected with the offset learned from the other points (fresh data).
    // offsetPx is where the gaze point sat relative to the targets.
    measure(points, W, H) {
      const px = d => Math.hypot(d.x * W, d.y * H), mean = a => a.reduce((s, v) => s + v, 0) / a.length, diag = Math.hypot(W, H);
      const seen = points.map(p => { const g = gaze.predict(p.f); return { x: g.x - p.x, y: g.y - p.y }; }), shift = mid(seen);
      const miss = points.map(p => { const g = model.predict(p.f); return { x: p.x - g.x, y: p.y - g.y }; });
      const held = miss.map((m, i) => { const c = mid(miss.filter((_, j) => j !== i)); return px({ x: m.x - c.x, y: m.y - c.y }); });
      const beforePx = Math.round(mean(seen.map(px))), afterPx = Math.round(mean(held));
      return { beforePx, afterPx, beforePct: Math.round(beforePx / diag * 100), afterPct: Math.round(afterPx / diag * 100),
               offsetPx: { x: Math.round(shift.x * W), y: Math.round(shift.y * H) } };
    },
    // saved form: each sample as [x, y, ...f] with 6 significant digits
    toJSON() {
      const d = { v: STORE_VERSION }, pack = s => [s.x, s.y].concat(s.f).map(v => +v.toPrecision(6));
      KINDS.forEach(k => { d[k] = samples[k].map(pack); });
      return d;
    },
    // loads a saved form, marking its samples as restored; false for another format version
    load(d) {
      if (!d || d.v !== STORE_VERSION) return false;
      KINDS.forEach(k => { samples[k] = (d[k] || []).filter(r => r.length === NF + 2).map(r => ({ x: r[0], y: r[1], f: r.slice(2), restored: true })); });
      return true;
    },
    reset() { model = null; offset = { x: 0, y: 0 }; feedback.splice(0); KINDS.forEach(k => { samples[k] = []; }); }
  };
  return gaze;
}
