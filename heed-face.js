/*!
 * heed-face 0.5.0 — face-mesh adapter for heed
 * Front camera -> MediaPipe Face Landmarker (on-device, vendored) ->
 *   head pose, expressions (relative to the person's own neutral face), blinks, presence,
 *   and a gaze point from a calibrated model that keeps learning from clicks and taps.
 * Gaze dwell on [data-heed] regions is recorded as kind 'gaze'. No frames are stored or sent.
 *
 *   import { attachFace } from './heed-face.js';
 *   const face = attachFace(h);
 *   await face.start();          // camera permission
 *   await face.calibrate();      // ~50 s: 17-point grid, smooth pursuit, validation
 *   // afterwards every mouse click / tap adds a sample and refits the model
 */
import { FilesetResolver, FaceLandmarker } from './vendor/vision_bundle.mjs';

const VENDOR = new URL('./vendor/', import.meta.url).href;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const DEG = 180 / Math.PI;
const median = a => { const s = a.slice().sort((p, q) => p - q); return s.length ? s[Math.floor(s.length / 2)] : 0; };
const wait = ms => new Promise(r => setTimeout(r, ms));

// ---------- features
function eyeFeat(lm, inner, outer, top, bottom, iris, ax) {
  const ux = (lm[outer].x - lm[inner].x) * ax, uy = lm[outer].y - lm[inner].y, len = Math.hypot(ux, uy) || 1e-6;
  const nx = -uy / len, ny = ux / len, rx = (lm[iris].x - lm[inner].x) * ax, ry = lm[iris].y - lm[inner].y;
  return { h: (rx * ux + ry * uy) / (len * len), v: (rx * nx + ry * ny) / len, open: (lm[bottom].y - lm[top].y) / len, len };
}
function features(lm, matrix, ax) {
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
function fitModel(samples, W, H) {
  const k = samples[0].f.length, sw = samples.reduce((a, s) => a + s.w, 0);
  const mean = Array(k).fill(0), sd = Array(k).fill(0);
  samples.forEach(s => s.f.forEach((x, i) => { mean[i] += s.w * x / sw; }));
  samples.forEach(s => s.f.forEach((x, i) => { sd[i] += s.w * (x - mean[i]) ** 2 / sw; }));
  for (let i = 0; i < k; i++) sd[i] = Math.sqrt(sd[i]) || 1;
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

// ---------- expressions (facial action proxies, not felt emotion)
const EXPRESSIONS = [
  { label: 'smiling', meaning: 'positive', valence: 'positive', score: b => 0.75 * av(b, 'mouthSmileLeft', 'mouthSmileRight') + 0.25 * av(b, 'cheekSquintLeft', 'cheekSquintRight') },
  { label: 'brows raised', meaning: 'surprise or interest', valence: 'neutral', score: b => 0.5 * ((b.browInnerUp || 0) + av(b, 'browOuterUpLeft', 'browOuterUpRight')) / 2 * 2 * 0.5 + 0.3 * av(b, 'eyeWideLeft', 'eyeWideRight') + 0.2 * (b.jawOpen || 0) },
  { label: 'frowning', meaning: 'concentration or confusion', valence: 'negative', score: b => 0.75 * av(b, 'browDownLeft', 'browDownRight') + 0.25 * av(b, 'eyeSquintLeft', 'eyeSquintRight') },
  { label: 'mouth corners down', meaning: 'doubt or sadness', valence: 'negative', score: b => 0.7 * av(b, 'mouthFrownLeft', 'mouthFrownRight') + 0.3 * av(b, 'mouthPressLeft', 'mouthPressRight') },
  { label: 'nose wrinkled', meaning: 'dislike or disgust', valence: 'negative', score: b => 0.65 * av(b, 'noseSneerLeft', 'noseSneerRight') + 0.35 * av(b, 'mouthUpperUpLeft', 'mouthUpperUpRight') }
];
function av(b, x, y) { return ((b[x] || 0) + (b[y] || 0)) / 2; }

// ---------- adapter
export function attachFace(h, opts = {}) {
  const fps = opts.fps || 15, sampleEvery = opts.sampleMs || 500, maxClicks = opts.maxClicks || 200;
  let ctx = null;
  h.use({ name: 'face', setup(c) {
    ctx = c;
    // pausing saves what is in progress, so nothing open is lost
    const off = c.bus.on('pausing', () => { closeGaze('pause'); endEpisode(); });
    return () => { off(); stop(); };
  } });

  const listeners = new Set();
  const state = {
    status: 'idle', error: null, present: false, fps: 0,
    pose: { yaw: 0, pitch: 0, roll: 0 },
    expr: { smile: 0, browDown: 0, browUp: 0, jawOpen: 0, squint: 0, blink: 0 },
    expression: { label: 'neutral', meaning: '', valence: 'neutral', scores: {} },
    blinks: 0, blinkRate: 0, gaze: null, calibrated: false, landmarks: null,
    model: { source: 'none', samples: 0, clicks: 0, cvErrorPx: null, validationErrorPx: null }
  };
  ctx.faceState = state;

  let video, stream, landmarker, running = false, lastDetect = 0, lastSample = 0, startedAt = 0;
  let model = null, smooth = null, latest = null, sink = null, refitTimer = 0;
  let eyesClosed = false, blinkBase = null, closedAt = 0, blinkTimes = [], awaySince = 0, frames = 0, fpsT0 = 0;
  let gazeRegion = null, gazeSince = 0;
  const ring = [];                       // last ~1.5 s of frames for click lookup
  const samples = { grid: [], pursuit: [], clicks: [] };
  let exprBase = null, exprBaseBuf = [], exprSmooth = {}, episode = null, candidate = null;

  const emit = () => listeners.forEach(fn => { try { fn(state); } catch (e) { console.error('[heed-face]', e); } });

  async function load() {
    const fileset = await FilesetResolver.forVisionTasks(VENDOR + 'wasm');
    const base = { modelAssetPath: VENDOR + 'face_landmarker.task' };
    // A regular canvas for the GL context: Safari before 17 has no WebGL on OffscreenCanvas.
    const options = d => ({ canvas: document.createElement('canvas'), baseOptions: Object.assign({}, base, { delegate: d }), runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    try { return await FaceLandmarker.createFromOptions(fileset, options('GPU')); }
    catch (e) { return await FaceLandmarker.createFromOptions(fileset, options('CPU')); }
  }

  async function start() {
    if (running) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      state.status = 'error'; state.error = 'This browser has no camera API. Open the page over HTTPS in Safari or Chrome.'; emit(); return;
    }
    state.status = 'loading'; state.error = null; emit();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      video = document.createElement('video');
      video.muted = true; video.playsInline = true; video.setAttribute('playsinline', '');
      video.srcObject = stream;
      await video.play();
      landmarker = landmarker || await load();
    } catch (e) {
      stop();
      state.status = 'error';
      state.error = e && e.name === 'NotAllowedError' ? 'Camera permission was declined. Allow it in the browser settings and try again.' : String(e && e.message || e);
      emit(); return;
    }
    running = true; state.status = 'running'; fpsT0 = performance.now(); frames = 0; startedAt = Date.now();
    ctx.record({ kind: 'face-start', calibrated: state.calibrated });
    loop(); emit();
  }

  function stop() {
    if (running && ctx) { closeGaze('stop'); endEpisode(); ctx.record({ kind: 'face-stop' }); }
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; video = null;
    if (state.status === 'running' || state.status === 'loading') state.status = 'idle';
    state.present = false; state.gaze = null; emit();
  }

  function loop() {
    if (!running) return;
    const next = () => running && (video && video.requestVideoFrameCallback ? video.requestVideoFrameCallback(loop) : requestAnimationFrame(loop));
    const t = performance.now();
    if (t - lastDetect < 1000 / fps || !video || video.readyState < 2) { next(); return; }
    lastDetect = t;
    let res;
    try { res = landmarker.detectForVideo(video, t); } catch (e) { next(); return; }
    frames++;
    if (t - fpsT0 > 1000) { state.fps = Math.round(frames * 1000 / (t - fpsT0)); frames = 0; fpsT0 = t; }
    process(res);
    next();
  }

  function process(res) {
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    const wall = Date.now();
    if (!lm) {
      if (state.present) { awaySince = wall; closeGaze('away'); endEpisode(); }
      state.present = false; state.landmarks = null; state.gaze = null;
      if (wall - lastSample > sampleEvery) { lastSample = wall; ctx.record({ kind: 'face', present: false }); }
      emit(); return;
    }
    if (!state.present && awaySince) {
      const ms = wall - awaySince;
      if (ms > 1500) {
        const ev = ctx.record({ kind: 'away', ms });
        ctx.finding({ level: 'notice', type: 'away', trigger: ev.id, text: 'Looked away from the screen for ' + (ms / 1000).toFixed(1) + ' s' });
      }
      awaySince = 0;
    }
    state.present = true; state.landmarks = lm;

    const ax = video && video.videoHeight ? video.videoWidth / video.videoHeight : 4 / 3;
    const f = features(lm, res.facialTransformationMatrixes[0], ax);
    state.pose = { yaw: f.yaw * DEG, pitch: f.pitch * DEG, roll: f.roll * DEG };

    const bs = {};
    ((res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || []).forEach(c => { bs[c.categoryName] = c.score; });
    state.expr = { smile: av(bs, 'mouthSmileLeft', 'mouthSmileRight'), browDown: av(bs, 'browDownLeft', 'browDownRight'),
                   browUp: bs.browInnerUp || 0, jawOpen: bs.jawOpen || 0, squint: av(bs, 'eyeSquintLeft', 'eyeSquintRight') };

    // blinks relative to this person's resting eye openness
    const blink = av(bs, 'eyeBlinkLeft', 'eyeBlinkRight');
    blinkBase = blinkBase === null ? blink : (eyesClosed ? blinkBase : blinkBase * 0.97 + blink * 0.03);
    const closeAt = Math.min(0.92, blinkBase + 0.3), openAt = Math.min(0.85, blinkBase + 0.15);
    if (!eyesClosed && blink > closeAt) { eyesClosed = true; closedAt = wall; }
    else if (eyesClosed && (blink < openAt || wall - closedAt > 1200)) {
      eyesClosed = false;
      if (wall - closedAt <= 1200) { state.blinks++; blinkTimes.push(wall); } else blinkBase = blink;
    }
    state.expr.blink = blink;
    blinkTimes = blinkTimes.filter(x => wall - x < 60000);
    state.blinkRate = Math.round(blinkTimes.length * 60000 / Math.max(10000, Math.min(60000, wall - startedAt)));

    // frame ring for click samples; calibration sink
    if (!eyesClosed) { ring.push({ t: wall, f: f.vec }); while (ring.length && wall - ring[0].t > 1500) ring.shift(); }
    latest = eyesClosed ? null : f.vec;
    if (sink && !eyesClosed) sink(f.vec, wall, bs);

    updateExpression(bs, wall);

    if (model && !eyesClosed) {
      const g = model.predict(f.vec);
      smooth = smooth ? { x: smooth.x * 0.65 + g.x * 0.35, y: smooth.y * 0.65 + g.y * 0.35 } : g;
      const W = innerWidth, H = innerHeight;
      state.gaze = { x: clamp(smooth.x, -0.2, 1.2) * W, y: clamp(smooth.y, -0.2, 1.2) * H, onScreen: smooth.x >= 0 && smooth.x <= 1 && smooth.y >= 0 && smooth.y <= 1 };
      trackGaze(state.gaze);
      if (ctx.point && state.gaze.onScreen) ctx.point(state.gaze.x, state.gaze.y, 'gaze');
    }

    if (wall - lastSample > sampleEvery) {
      lastSample = wall;
      ctx.record({ kind: 'face', present: true, yaw: +state.pose.yaw.toFixed(1), pitch: +state.pose.pitch.toFixed(1), roll: +state.pose.roll.toFixed(1),
        smile: +state.expr.smile.toFixed(2), browDown: +state.expr.browDown.toFixed(2), browUp: +state.expr.browUp.toFixed(2),
        jawOpen: +state.expr.jawOpen.toFixed(2), blinkRate: state.blinkRate, expression: state.expression.label,
        gx: state.gaze ? Math.round(state.gaze.x) : null, gy: state.gaze ? Math.round(state.gaze.y) : null });
    }
    emit();
  }

  // ---------- expressions: scores relative to the person's own neutral face
  function updateExpression(bs, wall) {
    const raw = {}; EXPRESSIONS.forEach(x => { raw[x.label] = x.score(bs); });
    if (!exprBase) {
      exprBaseBuf.push(raw);
      if (exprBaseBuf.length >= 20) setBaseline(exprBaseBuf);
      state.expression = { label: 'neutral', meaning: 'learning your neutral face', valence: 'neutral', scores: {} };
      return;
    }
    let best = null;
    EXPRESSIONS.forEach(x => {
      const d = Math.max(0, raw[x.label] - exprBase[x.label]);
      exprSmooth[x.label] = (exprSmooth[x.label] || 0) * 0.6 + d * 0.4;
      if (!best || exprSmooth[x.label] > exprSmooth[best.label]) best = x;
    });
    const label = best && exprSmooth[best.label] > 0.15 ? best.label : 'neutral';
    const meta = EXPRESSIONS.find(x => x.label === label) || { meaning: '', valence: 'neutral' };
    state.expression = { label, meaning: meta.meaning, valence: meta.valence, scores: Object.assign({}, exprSmooth) };
    // episodes: a label must hold 500 ms to start; non-neutral episodes of 700 ms or more are recorded
    if (!episode || episode.label !== label) {
      if (!candidate || candidate.label !== label) candidate = { label, t0: wall };
      else if (wall - candidate.t0 >= 500) { endEpisode(); episode = { label, meta, t0: candidate.t0, peak: 0, region: focusRegion() }; candidate = null; }
    } else candidate = null;
    if (episode) episode.peak = Math.max(episode.peak, exprSmooth[label] || 0);
  }
  function setBaseline(buf) {
    exprBase = {}; EXPRESSIONS.forEach(x => { exprBase[x.label] = median(buf.map(r => r[x.label])); });
  }
  function focusRegion() { return ctx.focus && Date.now() - ctx.focus.t < 3000 ? ctx.focus.region : (gazeRegion ? gazeRegion.getAttribute('data-heed') : null); }
  function endEpisode() {
    if (!episode) return;
    const ms = Date.now() - episode.t0, e = episode; episode = null;
    if (e.label === 'neutral' || ms < 700) return;
    const ev = ctx.record({ kind: 'emotion', label: e.label, meaning: e.meta.meaning, valence: e.meta.valence, ms, peak: +e.peak.toFixed(2), region: e.region });
    ctx.finding({ level: 'info', type: 'expression', trigger: ev.id, region: e.region,
      text: e.label.charAt(0).toUpperCase() + e.label.slice(1) + ' (' + e.meta.meaning + ') for ' + (ms / 1000).toFixed(1) + ' s' + (e.region ? ' on ' + ctx.label(e.region) : '') });
  }

  // ---------- gaze dwell
  function closeGaze(reason) {
    if (!gazeRegion) return;
    const ms = Date.now() - gazeSince;
    if (ms >= 150) ctx.record(Object.assign({ kind: 'gaze', ms, reason }, ctx.regionInfo(gazeRegion)));
    gazeRegion = null;
  }
  function trackGaze(g) {
    const el = g.onScreen ? document.elementFromPoint(g.x, g.y) : null;
    const r = el ? ctx.regionOf(el) : null;
    if (r === gazeRegion) { if (r && Date.now() - gazeSince > 1000) { closeGaze('checkpoint'); gazeRegion = r; gazeSince = Date.now(); } return; }
    closeGaze('move');
    if (r) { gazeRegion = r; gazeSince = Date.now(); }
  }

  // ---------- model fitting
  function allSamples() {
    return samples.grid.map(s => Object.assign({ w: 3 }, s))
      .concat(samples.pursuit.map(s => Object.assign({ w: 1 }, s)))
      .concat(samples.clicks.map(s => Object.assign({ w: 2 }, s)));
  }
  function refit(source) {
    const all = allSamples();
    if (all.length < 8) return false;
    model = fitModel(all, innerWidth, innerHeight);
    state.calibrated = true; smooth = null;
    state.model = Object.assign({}, state.model, { source: source || state.model.source, samples: all.length, clicks: samples.clicks.length, cvErrorPx: model.cvErrorPx });
    emit();
    return true;
  }

  // Every mouse click or tap is a labelled sample: people look where they click.
  function onPointerDown(e) {
    if (!running || !ring.length || (e.target && e.target.closest && e.target.closest('.heed-calib'))) return;
    const t = Date.now(), win = ring.filter(r => t - r.t <= 250 && t - r.t >= 0);
    if (win.length < 1) return;
    const f = win[0].f.map((_, i) => median(win.map(r => r.f[i])));
    samples.clicks.push({ x: e.clientX / innerWidth, y: e.clientY / innerHeight, f });
    if (samples.clicks.length > maxClicks) samples.clicks.shift();
    ctx.record({ kind: 'calibration-click', x: Math.round(e.clientX), y: Math.round(e.clientY), clicks: samples.clicks.length,
                 predictedErrorPx: model ? Math.round(Math.hypot((model.predict(f).x - e.clientX / innerWidth) * innerWidth, (model.predict(f).y - e.clientY / innerHeight) * innerHeight)) : null });
    clearTimeout(refitTimer);
    refitTimer = setTimeout(() => refit(samples.grid.length ? 'calibration + clicks' : 'clicks only'), 250);
  }
  document.addEventListener('pointerdown', onPointerDown, true);

  // ---------- calibration: 17-point grid, smooth pursuit, validation
  async function calibrate(copts = {}) {
    if (!running) await start();
    if (!running) return false;
    const quick = !!copts.quick;
    const layer = document.createElement('div');
    layer.setAttribute('data-heed-ignore', '');
    layer.className = 'heed-calib';
    layer.innerHTML = '<div class="heed-calib-card"><b>Gaze calibration</b><p>About 50 seconds. Hold the device steady at a comfortable distance and keep your head still. Follow the dot with your eyes.</p><button type="button" class="go">Start</button></div>' +
      '<i></i><span class="heed-calib-status"></span><button type="button" class="cancel">Cancel</button>';
    document.body.appendChild(layer);
    ctx.calibrating = true;
    const dot = layer.querySelector('i'), status = layer.querySelector('.heed-calib-status'), card = layer.querySelector('.heed-calib-card');
    let cancelled = false;
    layer.querySelector('.cancel').onclick = () => { cancelled = true; };
    await new Promise(res => { layer.querySelector('.go').onclick = res; layer.querySelector('.cancel').addEventListener('click', res); });
    card.remove();
    const place = (x, y) => { dot.style.left = (x * 100) + '%'; dot.style.top = (y * 100) + '%'; };
    const baseBuf = [];
    const collect = async (x, y, ms) => {
      const got = [];
      sink = (f, t, bs) => { got.push(f); const raw = {}; EXPRESSIONS.forEach(e => { raw[e.label] = e.score(bs); }); baseBuf.push(raw); };
      const t0 = Date.now();
      while (!cancelled && Date.now() - t0 < 4000 && !(Date.now() - t0 >= ms && got.length >= 5)) await wait(40);
      sink = null;
      return got.length >= 2 ? got[0].map((_, i) => median(got.map(g => g[i]))) : null;
    };

    // 1. grid (randomised order)
    const g4 = [0.06, 0.36, 0.64, 0.94];
    let grid = quick ? [0.1, 0.5, 0.9].flatMap(y => [0.1, 0.5, 0.9].map(x => [x, y])) : g4.flatMap(y => g4.map(x => [x, y])).concat([[0.5, 0.5]]);
    grid = grid.map(p => [Math.random(), p]).sort((a, b) => a[0] - b[0]).map(p => p[1]);
    samples.grid = [];
    for (let i = 0; i < grid.length && !cancelled; i++) {
      const [x, y] = grid[i];
      status.textContent = 'Point ' + (i + 1) + ' of ' + grid.length;
      place(x, y); dot.className = ''; await wait(700); dot.className = 'hold';
      const f = await collect(x, y, 900);
      if (f) samples.grid.push({ x, y, f });
    }
    if (baseBuf.length >= 20) setBaseline(baseBuf);

    // 2. smooth pursuit: the dot glides along a Lissajous path; targets lag-compensated by 150 ms
    if (!quick && !cancelled) {
      samples.pursuit = [];
      status.textContent = 'Follow the moving dot';
      dot.className = 'pursuit';
      const T = 14000, t0 = Date.now(), pos = t => { const s = t / 1000; return [0.5 + 0.43 * Math.sin(2 * Math.PI * s / 7), 0.5 + 0.42 * Math.sin(2 * Math.PI * s / 4.6 + Math.PI / 2)]; };
      place(...pos(0)); await wait(900);
      const t1 = Date.now();
      sink = (f, t) => { const [x, y] = pos(Math.max(0, t - t1 - 150)); if (t - t1 > 400) samples.pursuit.push({ x, y, f }); };
      while (!cancelled && Date.now() - t1 < T) { place(...pos(Date.now() - t1)); await wait(16); }
      sink = null;
    }

    // 3. fit, then validate on fresh points
    if (cancelled || !refit(quick ? 'quick calibration' : 'calibration')) {
      layer.remove(); ctx.calibrating = false;
      ctx.record({ kind: 'calibration', ok: false, points: samples.grid.length });
      return false;
    }
    let validationErrorPx = null;
    if (!quick && !cancelled) {
      dot.className = '';
      const val = [[0.25, 0.2], [0.75, 0.22], [0.5, 0.5], [0.22, 0.78], [0.78, 0.8]], errs = [];
      for (let i = 0; i < val.length && !cancelled; i++) {
        status.textContent = 'Checking accuracy ' + (i + 1) + ' of ' + val.length;
        const [x, y] = val[i]; place(x, y); dot.className = ''; await wait(700); dot.className = 'hold';
        const f = await collect(x, y, 800);
        if (f) { const p = model.predict(f); errs.push(Math.hypot((p.x - x) * innerWidth, (p.y - y) * innerHeight)); samples.grid.push({ x, y, f }); }
      }
      if (errs.length) validationErrorPx = Math.round(errs.reduce((a, b) => a + b, 0) / errs.length);
      refit('calibration');
    }
    state.model.validationErrorPx = validationErrorPx;
    const diag = Math.hypot(innerWidth, innerHeight);
    const pct = validationErrorPx ? Math.round(validationErrorPx / diag * 100) : null;
    const quality = pct === null ? '' : pct <= 8 ? 'good' : pct <= 15 ? 'fair' : 'poor';
    ctx.record({ kind: 'calibration', ok: true, points: samples.grid.length, samples: allSamples().length, pursuit: samples.pursuit.length,
                 meanErrorPx: model.cvErrorPx, validationErrorPx, validationPctOfScreen: pct, lambda: model.lambda });
    if (validationErrorPx !== null && !cancelled) {
      dot.remove(); status.remove();
      const done = document.createElement('div'); done.className = 'heed-calib-card';
      done.innerHTML = '<b>Calibration ' + quality + '</b><p>Average error on fresh points: ' + validationErrorPx + ' px, about ' + pct + '% of the screen diagonal. Each click or tap from now on refines it.</p><button type="button" class="go">Done</button>';
      layer.appendChild(done);
      await new Promise(res => { done.querySelector('.go').onclick = res; layer.querySelector('.cancel').addEventListener('click', res); setTimeout(res, 8000); });
    }
    layer.remove(); ctx.calibrating = false;
    emit();
    return true;
  }

  const style = document.createElement('style');
  style.textContent =
    '.heed-calib{position:fixed;inset:0;z-index:2147483600;background:var(--heed-card,#fff);color:var(--heed-card-fg,#09090b);font:15px/1.45 system-ui,-apple-system,sans-serif}' +
    '.heed-calib-card{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:min(360px,calc(100vw - 48px));display:grid;gap:10px;padding:20px;border:1px solid var(--heed-line,#e4e4e7);border-radius:12px;background:var(--heed-card,#fff)}' +
    '.heed-calib-card p{margin:0;color:var(--heed-muted,#71717a)}.heed-calib-card .go{justify-self:start;background:var(--heed-card-fg,#09090b);color:var(--heed-card,#fff);border-color:var(--heed-card-fg,#09090b)}' +
    '.heed-calib-status{position:absolute;left:50%;bottom:max(24px,env(safe-area-inset-bottom,0px));transform:translateX(-50%);font-size:13px;color:var(--heed-muted,#71717a)}' +
    '.heed-calib i{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:var(--heed-tint,#2563eb);transition:left .45s ease,top .45s ease,transform .2s}' +
    '.heed-calib i.hold{transform:scale(.5)}.heed-calib i.pursuit{transition:none;transform:scale(.6)}' +
    '.heed-calib button{font:inherit;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--heed-line,#e4e4e7);background:var(--heed-btn,#fafafa);color:inherit;cursor:pointer}' +
    '.heed-calib .cancel{position:absolute;right:max(16px,env(safe-area-inset-right,0px));top:max(16px,env(safe-area-inset-top,0px))}' +
    '@media (prefers-reduced-motion: reduce){.heed-calib i{transition:none}}';
  document.head.appendChild(style);

  return {
    state, start, stop, calibrate,
    get video() { return video; },
    onFrame(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    debug() { return { latest, eyesClosed, running, samples: { grid: samples.grid.length, pursuit: samples.pursuit.length, clicks: samples.clicks.length } }; },
    resetCalibration() { model = null; samples.grid = []; samples.pursuit = []; samples.clicks = []; state.calibrated = false; state.gaze = null; emit(); }
  };
}
