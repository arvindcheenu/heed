/*!
 * heed-face 0.1.0 — face-mesh adapter for heed
 * Front camera -> MediaPipe Face Landmarker (on-device, vendored) ->
 *   head pose, expression (blendshapes), blinks, presence, and a calibrated gaze point.
 * Gaze dwell on [data-heed] regions is recorded as kind 'gaze' (counts as coverage
 * when the spec's signals include 'gaze'). No frames leave the device or are stored:
 * only derived numbers are recorded.
 *
 * Usage:
 *   import { attachFace } from './heed-face.js';
 *   const face = attachFace(h, { fps: 15 });
 *   await face.start();        // asks for camera permission
 *   await face.calibrate();    // 9-point, ~15 s; enables the gaze point
 *   face.onFrame(state => …);  // pose, expression, gaze, blinkRate, present
 */
import { FilesetResolver, FaceLandmarker } from './vendor/vision_bundle.mjs';

const VENDOR = new URL('./vendor/', import.meta.url).href;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const DEG = 180 / Math.PI;

// ---------- feature extraction
function eyeRatio(lm, inner, outer, top, bottom, iris) {
  const ax = lm[outer].x - lm[inner].x, ay = lm[outer].y - lm[inner].y;
  const len2 = ax * ax + ay * ay || 1e-6;
  const along = ((lm[iris].x - lm[inner].x) * ax + (lm[iris].y - lm[inner].y) * ay) / len2; // 0 inner .. 1 outer
  const h = (lm[bottom].y - lm[top].y) || 1e-6;
  const down = (lm[iris].y - lm[top].y) / h; // 0 top lid .. 1 bottom lid
  return [along, down];
}

function features(lm, matrix) {
  const m = matrix.data; // 4x4, column-major
  const pitch = Math.atan2(m[6], m[10]);
  const yaw = Math.asin(clamp(-m[2], -1, 1));
  const roll = Math.atan2(m[1], m[0]);
  let h = 0, v = 0;
  if (lm.length >= 478) {
    const r = eyeRatio(lm, 133, 33, 159, 145, 468);  // subject's right eye
    const l = eyeRatio(lm, 362, 263, 386, 374, 473); // subject's left eye
    h = (r[0] - l[0]) / 2;
    v = (r[1] + l[1]) / 2;
  }
  return { yaw, pitch, roll, h, v, vec: [yaw, pitch, h, v] };
}

// ---------- ridge regression for the gaze mapping
function solve(A, b) { // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-9;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-9));
}

function fitGaze(samples, lambda = 0.5) {
  const k = samples[0].f.length;
  const mean = Array(k).fill(0), sd = Array(k).fill(0);
  samples.forEach(s => s.f.forEach((x, i) => { mean[i] += x / samples.length; }));
  samples.forEach(s => s.f.forEach((x, i) => { sd[i] += (x - mean[i]) ** 2 / samples.length; }));
  for (let i = 0; i < k; i++) sd[i] = Math.sqrt(sd[i]) || 1;
  const X = samples.map(s => [1].concat(s.f.map((x, i) => (x - mean[i]) / sd[i])));
  const fit = target => {
    const p = k + 1, A = Array.from({ length: p }, () => Array(p).fill(0)), b = Array(p).fill(0);
    X.forEach((row, n) => {
      for (let i = 0; i < p; i++) { b[i] += row[i] * target[n]; for (let j = 0; j < p; j++) A[i][j] += row[i] * row[j]; }
    });
    for (let i = 1; i < p; i++) A[i][i] += lambda;
    return solve(A, b);
  };
  const wx = fit(samples.map(s => s.x)), wy = fit(samples.map(s => s.y));
  return f => {
    const z = [1].concat(f.map((x, i) => (x - mean[i]) / sd[i]));
    let x = 0, y = 0;
    for (let i = 0; i < z.length; i++) { x += wx[i] * z[i]; y += wy[i] * z[i]; }
    return { x, y }; // viewport fractions
  };
}

const median = a => { const s = a.slice().sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };

// ---------- adapter
export function attachFace(h, opts = {}) {
  const fps = opts.fps || 15, sampleEvery = opts.sampleMs || 500;
  let ctx = null;
  h.use({ name: 'face', setup(c) { ctx = c; return () => stop(); } });

  const listeners = new Set();
  const state = {
    status: 'idle', error: null, present: false, fps: 0,
    pose: { yaw: 0, pitch: 0, roll: 0 },
    expr: { smile: 0, browDown: 0, browUp: 0, jawOpen: 0, squint: 0 },
    blinks: 0, blinkRate: 0, gaze: null, calibrated: false, landmarks: null
  };

  let video, stream, landmarker, running = false, lastDetect = 0, lastSample = 0;
  let mapGaze = null, smooth = null, latestFeat = null, calibBucket = null;
  let eyesClosed = false, blinkBase = null, closedAt = 0, blinkTimes = [], awaySince = 0, frames = 0, fpsT0 = 0;
  let gazeRegion = null, gazeSince = 0;

  function emit() { listeners.forEach(fn => { try { fn(state); } catch (e) { console.error('[heed-face]', e); } }); }

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
    running = true; state.status = 'running'; fpsT0 = performance.now(); frames = 0;
    ctx.record({ kind: 'face-start', calibrated: state.calibrated });
    loop();
    emit();
  }

  function stop() {
    if (running && ctx) { closeGaze('stop'); ctx.record({ kind: 'face-stop' }); }
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
    process(res, t);
    next();
  }

  function process(res, t) {
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    const wall = Date.now();
    if (!lm) {
      if (state.present) { awaySince = wall; closeGaze('away'); }
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

    const f = features(lm, res.facialTransformationMatrixes[0]);
    latestFeat = f.vec;
    if (calibBucket && !eyesClosed) calibBucket.push(f.vec.slice());
    state.pose = { yaw: f.yaw * DEG, pitch: f.pitch * DEG, roll: f.roll * DEG };

    const bs = {};
    ((res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || []).forEach(c => { bs[c.categoryName] = c.score; });
    const avg = (a, b) => ((bs[a] || 0) + (bs[b] || 0)) / 2;
    state.expr = { smile: avg('mouthSmileLeft', 'mouthSmileRight'), browDown: avg('browDownLeft', 'browDownRight'),
                   browUp: bs.browInnerUp || 0, jawOpen: bs.jawOpen || 0, squint: avg('eyeSquintLeft', 'eyeSquintRight') };

    // Blinks relative to this person's resting eye openness: narrow or downcast eyes
    // (common when looking down at a phone) raise the score without being a blink.
    const blink = avg('eyeBlinkLeft', 'eyeBlinkRight');
    blinkBase = blinkBase === null ? blink : (eyesClosed ? blinkBase : blinkBase * 0.97 + blink * 0.03);
    const closeAt = Math.min(0.92, blinkBase + 0.3), openAt = Math.min(0.85, blinkBase + 0.15);
    if (!eyesClosed && blink > closeAt) { eyesClosed = true; closedAt = wall; }
    else if (eyesClosed && (blink < openAt || wall - closedAt > 1200)) {
      eyesClosed = false;
      if (wall - closedAt <= 1200) { state.blinks++; blinkTimes.push(wall); } else blinkBase = blink; // a long "closure" was a new baseline
    }
    state.expr.blink = blink; state.expr.blinkBase = blinkBase;
    blinkTimes = blinkTimes.filter(x => wall - x < 60000);
    const span = Math.min(60000, wall - (ctx.self.started || wall)) || 1;
    state.blinkRate = Math.round(blinkTimes.length * 60000 / Math.max(span, 10000));

    if (mapGaze && !eyesClosed) {
      const g = mapGaze(f.vec);
      smooth = smooth ? { x: smooth.x * 0.7 + g.x * 0.3, y: smooth.y * 0.7 + g.y * 0.3 } : g;
      const W = innerWidth, H = innerHeight;
      state.gaze = { x: clamp(smooth.x, -0.2, 1.2) * W, y: clamp(smooth.y, -0.2, 1.2) * H, onScreen: smooth.x >= 0 && smooth.x <= 1 && smooth.y >= 0 && smooth.y <= 1 };
      trackGaze(state.gaze);
      if (ctx.point && state.gaze.onScreen) ctx.point(state.gaze.x, state.gaze.y, 'gaze');
    }

    if (wall - lastSample > sampleEvery) {
      lastSample = wall;
      ctx.record({ kind: 'face', present: true, yaw: +state.pose.yaw.toFixed(1), pitch: +state.pose.pitch.toFixed(1), roll: +state.pose.roll.toFixed(1),
        smile: +state.expr.smile.toFixed(2), browDown: +state.expr.browDown.toFixed(2), browUp: +state.expr.browUp.toFixed(2),
        jawOpen: +state.expr.jawOpen.toFixed(2), blinkRate: state.blinkRate,
        gx: state.gaze ? Math.round(state.gaze.x) : null, gy: state.gaze ? Math.round(state.gaze.y) : null });
    }
    emit();
  }

  function closeGaze(reason) {
    if (!gazeRegion) return;
    const ms = Date.now() - gazeSince;
    if (ms >= 150) ctx.record(Object.assign({ kind: 'gaze', ms, reason }, ctx.regionInfo(gazeRegion)));
    gazeRegion = null;
  }
  function trackGaze(g) {
    const el = g.onScreen ? document.elementFromPoint(g.x, g.y) : null;
    const r = el ? ctx.regionOf(el) : null;
    if (r === gazeRegion) {
      if (r && Date.now() - gazeSince > 1000) { closeGaze('checkpoint'); gazeRegion = r; gazeSince = Date.now(); }
      return;
    }
    closeGaze('move');
    if (r) { gazeRegion = r; gazeSince = Date.now(); }
  }

  // 9-point calibration: look at each dot; ~1.5 s per dot.
  async function calibrate(copts = {}) {
    if (!running) await start();
    if (!running) return false;
    const pts = copts.points || [0.1, 0.5, 0.9].flatMap(y => [0.1, 0.5, 0.9].map(x => [x, y]));
    const layer = document.createElement('div');
    layer.setAttribute('data-heed-ignore', '');
    layer.className = 'heed-calib';
    layer.innerHTML = '<p>Keep your head still and follow the dot with your eyes.</p><i></i><button type="button">Cancel</button>';
    document.body.appendChild(layer);
    const dot = layer.querySelector('i');
    let cancelled = false;
    layer.querySelector('button').onclick = () => { cancelled = true; };
    const samples = [];
    const wait = ms => new Promise(r => setTimeout(r, ms));
    for (const [x, y] of pts) {
      if (cancelled) break;
      dot.style.left = (x * 100) + '%'; dot.style.top = (y * 100) + '%';
      dot.classList.remove('hold'); await wait(650); dot.classList.add('hold');
      // collect detector frames, not wall-clock polls: slow devices still get enough samples
      calibBucket = [];
      const t0 = Date.now();
      while (!cancelled && Date.now() - t0 < 4000 && !(Date.now() - t0 >= 800 && calibBucket.length >= 5)) await wait(50);
      const got = calibBucket; calibBucket = null;
      if (got.length >= 2) samples.push({ x, y, f: got[0].map((_, i) => median(got.map(g => g[i]))) });
    }
    layer.remove();
    if (cancelled || samples.length < 6) {
      ctx.record({ kind: 'calibration', ok: false, points: samples.length });
      return false;
    }
    mapGaze = fitGaze(samples);
    const err = samples.map(s => { const p = mapGaze(s.f); return Math.hypot((p.x - s.x) * innerWidth, (p.y - s.y) * innerHeight); });
    state.calibrated = true; smooth = null;
    ctx.record({ kind: 'calibration', ok: true, points: samples.length, meanErrorPx: Math.round(err.reduce((a, b) => a + b, 0) / err.length) });
    emit();
    return true;
  }

  // Calibration layer styles (tokens fall back to neutral values).
  const style = document.createElement('style');
  style.textContent =
    '.heed-calib{position:fixed;inset:0;z-index:2147483600;background:var(--heed-card,#fff);color:var(--heed-card-fg,#09090b);font:15px/1.4 system-ui,-apple-system,sans-serif}' +
    '.heed-calib p{position:absolute;left:50%;top:50%;transform:translate(-50%,calc(-50% + 48px));margin:0;width:min(320px,80vw);text-align:center;color:var(--heed-muted,#71717a)}' +
    '.heed-calib i{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:var(--heed-tint,#2563eb);transition:left .45s ease,top .45s ease,transform .2s}' +
    '.heed-calib i.hold{transform:scale(.55)}' +
    '.heed-calib button{position:absolute;right:max(16px,env(safe-area-inset-right,0px));top:max(16px,env(safe-area-inset-top,0px));font:inherit;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--heed-line,#e4e4e7);background:var(--heed-btn,#fafafa);color:inherit}' +
    '@media (prefers-reduced-motion: reduce){.heed-calib i{transition:none}}';
  document.head.appendChild(style);

  return {
    state, start, stop, calibrate,
    get video() { return video; },
    onFrame(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    debug() { return { latestFeat, eyesClosed, running, lastDetect }; },
    resetCalibration() { mapGaze = null; state.calibrated = false; state.gaze = null; emit(); }
  };
}
