// Unit tests for the gaze model. Run: node --test test/*.test.mjs
// A synthetic face stands in for MediaPipe: its iris landmarks follow a point of regard (gx, gy),
// with a little landmark jitter. bias shifts the iris as a changed sitting position would (a pure
// offset); gain -1 mirrors the mapping (a distortion that no offset can fix).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { features, fitModel, createGaze, STORE_VERSION, NF } from '../heed-gaze.js';

const W = 1280, H = 800, DIAG = Math.hypot(W, H);
const BASE = {
  133: [0.45, 0.45], 33: [0.38, 0.45], 159: [0.415, 0.44], 145: [0.415, 0.46], 468: [0.415, 0.45],
  362: [0.55, 0.45], 263: [0.62, 0.45], 386: [0.585, 0.44], 374: [0.585, 0.46], 473: [0.585, 0.45], 1: [0.5, 0.55]
};
let seed = 1;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -50, 1];

function landmarks(gx, gy, { bias = { x: 0, y: 0 }, gain = 1, jitter = 0.0002 } = {}) {
  const k = 0.012 * gain, dx = -k * (gx - 0.5 + bias.x) + rnd() * jitter, dy = k * 0.6 * (gy - 0.5 + bias.y) + rnd() * jitter;
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (const i in BASE) lm[i] = { x: BASE[i][0] + rnd() * jitter, y: BASE[i][1] + rnd() * jitter, z: 0 };
  lm[468] = { x: BASE[468][0] + dx, y: BASE[468][1] + dy, z: 0 };
  lm[473] = { x: BASE[473][0] + dx, y: BASE[473][1] + dy, z: 0 };
  return lm;
}
const face = (gx, gy, opts) => features(landmarks(gx, gy, opts), { data: IDENTITY.map((v, i) => i >= 12 && i <= 14 ? v + rnd() * 0.5 : v) }, 4 / 3).vec;
const sample = (x, y, opts) => ({ x, y, f: face(x, y, opts) });
const errPx = (p, s) => Math.hypot((p.x - s.x) * W, (p.y - s.y) * H);
const meanErrPx = (gaze, points) => points.reduce((a, s) => a + errPx(gaze.predict(s.f), s), 0) / points.length;
const randomPoints = (n, opts) => Array.from({ length: n }, () => sample(0.1 + 0.8 * (rnd() + 0.5), 0.1 + 0.8 * (rnd() + 0.5), opts));

function calibrated(opts) {
  const gaze = createGaze(opts);
  const g4 = [0.06, 0.36, 0.64, 0.94];
  g4.forEach(y => g4.forEach(x => gaze.samples.grid.push(sample(x, y))));
  gaze.samples.grid.push(sample(0.5, 0.5));
  for (let t = 0; t < 150; t++) gaze.samples.pursuit.push(sample(0.5 + 0.43 * Math.sin(t / 10), 0.5 + 0.42 * Math.sin(t / 6.6 + Math.PI / 2)));
  assert.ok(gaze.refit(W, H));
  return gaze;
}
const CHECK = [[0.5, 0.5], [0.2, 0.25], [0.8, 0.25], [0.2, 0.75], [0.8, 0.75]];
const SHIFT = { bias: { x: 0.12, y: -0.15 } };   // reads as gaze further right and higher

test('features: identity head pose, 22 values, and a head turn reads as yaw', () => {
  const f = features(landmarks(0.5, 0.5, { jitter: 0 }), { data: IDENTITY }, 4 / 3);
  assert.equal(f.vec.length, NF);
  assert.ok([f.yaw, f.pitch, f.roll].every(v => v === 0));
  const a = 0.2, turned = IDENTITY.slice(); turned[0] = Math.cos(a); turned[2] = -Math.sin(a); turned[8] = Math.sin(a); turned[10] = Math.cos(a);
  assert.ok(Math.abs(features(landmarks(0.5, 0.5), { data: turned }, 4 / 3).yaw - a) < 1e-9);
});

test('a calibration predicts unseen points within 3% of the screen diagonal', () => {
  const gaze = calibrated();
  const err = meanErrPx(gaze, randomPoints(30));
  assert.ok(err < 0.03 * DIAG, err + ' px');
});

test('refit needs 8 samples', () => {
  const gaze = createGaze();
  for (let i = 0; i < 7; i++) gaze.samples.grid.push(sample(0.1 * i, 0.5));
  assert.equal(gaze.refit(W, H), false);
  gaze.samples.grid.push(sample(0.9, 0.5));
  assert.equal(gaze.refit(W, H), true);
});

test('a feature that differs only by rounding cannot become a predictor', () => {
  // Two groups whose labels differ by 0.1; feature 10 is the same value, exact in one group and
  // rounded to 6 digits in the other, as with saved and live samples. It must not explain the gap.
  const exact = 0.2857142857142857, rounded = +exact.toPrecision(6), samples = [];
  for (let i = 0; i < 60; i++) {
    const gx = (i % 10) / 10, gy = Math.floor(i / 10) / 6, f = Array.from({ length: NF }, () => rnd() * 0.01);
    f[6] = gx; f[7] = gy; f[10] = i % 2 ? exact : rounded;
    samples.push({ x: gx + (i % 2 ? 0.1 : 0), y: gy, f, w: 1 });
  }
  const model = fitModel(samples, W, H), f = samples[0].f.slice();
  const a = model.predict(Object.assign(f.slice(), { 10: exact })), b = model.predict(Object.assign(f.slice(), { 10: rounded }));
  assert.ok(Math.abs(a.x - b.x) < 1e-9 && Math.abs(a.y - b.y) < 1e-9, JSON.stringify([a, b]));
});

test('the offset starts with the third click and follows a shift', () => {
  const gaze = calibrated();
  const clicks = [[0.3, 0.3], [0.7, 0.4], [0.5, 0.7], [0.2, 0.6]].map(([x, y]) => sample(x, y, SHIFT));
  gaze.addClick(clicks[0]); gaze.addClick(clicks[1]); gaze.refit(W, H);
  assert.deepEqual(gaze.offset, { x: 0, y: 0 }, 'two clicks: no offset');
  gaze.addClick(clicks[2]); gaze.refit(W, H);
  assert.ok(gaze.offset.x < 0 && gaze.offset.y > 0, 'three clicks: shifts left and down against the bias ' + JSON.stringify(gaze.offset));
});

test('a check measures a pure shift, names its direction, and corrects it', () => {
  const gaze = calibrated();
  const pts = CHECK.map(([x, y]) => sample(x, y, SHIFT));
  const m = gaze.measure(pts, W, H);
  assert.ok(m.offsetPx.x > 100 && m.offsetPx.y < -80, 'gaze sat right and above: ' + JSON.stringify(m.offsetPx));
  assert.ok(m.beforePx > 150 && m.afterPx < m.beforePx / 4, m.beforePx + ' → ' + m.afterPx + ' px');
  const before = meanErrPx(gaze, randomPoints(20, SHIFT));
  gaze.addChecks(pts); gaze.refit(W, H);
  const after = meanErrPx(gaze, randomPoints(20, SHIFT));
  assert.ok(after < 0.03 * DIAG && after < before / 4, 'unseen points: ' + Math.round(before) + ' → ' + Math.round(after) + ' px');
});

test('a check of a distortion is rated beyond what an offset fixes', () => {
  const gaze = calibrated();
  const m = gaze.measure(CHECK.map(([x, y]) => sample(x, y, { gain: -1 })), W, H);
  assert.ok(m.afterPct > 15, m.afterPct + '%');
});

test('saved form: round trip, version and row checks', () => {
  const gaze = calibrated();
  gaze.addClick(sample(0.4, 0.4)); gaze.addChecks(CHECK.map(([x, y]) => sample(x, y))); gaze.refit(W, H);
  const d = JSON.parse(JSON.stringify(gaze.toJSON()));
  assert.equal(d.v, STORE_VERSION);
  assert.ok(d.grid.flat().every(v => v === +v.toPrecision(6)), 'six significant digits');

  const again = createGaze();
  assert.ok(again.load(d) && again.refit(W, H));
  assert.deepEqual(Object.fromEntries(Object.entries(again.samples).map(([k, v]) => [k, v.length])), { grid: 17, pursuit: 150, clicks: 1, checks: 5 });
  assert.ok(again.samples.grid.every(s => s.restored));
  const probe = randomPoints(10);
  // the model, not the offset: the offset restarts with each visit
  assert.ok(probe.every(s => errPx(again.model.predict(s.f), gaze.model.predict(s.f)) < 1), 'restored model predicts like the original');

  assert.equal(createGaze().load(Object.assign({}, d, { v: STORE_VERSION + 1 })), false);
  const short = createGaze(); short.load(Object.assign({}, d, { grid: [d.grid[0].slice(0, 5)].concat(d.grid.slice(1)) }));
  assert.equal(short.samples.grid.length, 16, 'a row of the wrong length is dropped');
});

test('a new calibration keeps only this visit\'s clicks, and undo restores everything', () => {
  const gaze = createGaze();
  const old = calibrated(); old.addClick(sample(0.1, 0.1)); old.addChecks(CHECK.map(([x, y]) => sample(x, y)));
  gaze.load(JSON.parse(JSON.stringify(old.toJSON())));
  const mine = sample(0.6, 0.6); gaze.addClick(mine);
  const counts = () => Object.fromEntries(Object.entries(gaze.samples).map(([k, v]) => [k, v.length]));
  const before = counts(), grid = CHECK.map(([x, y]) => sample(x, y));
  const undo = gaze.restart(grid, null);
  assert.deepEqual(counts(), { grid: 5, pursuit: 0, clicks: 1, checks: 0 });
  assert.equal(gaze.samples.clicks[0], mine);
  undo();
  assert.deepEqual(counts(), before);
});

test('clicks and checks are capped', () => {
  const gaze = createGaze({ maxClicks: 3, maxChecks: 4 });
  for (let i = 0; i < 5; i++) gaze.addClick(sample(i / 5, 0.5));
  gaze.addChecks(CHECK.map(([x, y]) => sample(x, y)));
  assert.equal(gaze.samples.clicks.length, 3);
  assert.equal(gaze.samples.checks.length, 4);
  assert.equal(gaze.samples.clicks[0].x, 0.4);
});
