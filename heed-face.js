/*!
 * heed-face 0.5.0 — face-mesh adapter for heed
 * Front camera -> MediaPipe Face Landmarker (on-device, vendored) ->
 *   head pose, expressions (relative to the person's own neutral face), blinks, presence,
 *   and a gaze point from a calibrated model that keeps learning from clicks and taps.
 * Gaze dwell on [data-heed] regions is recorded as kind 'gaze'. No frames are stored or sent.
 * The calibration samples (derived numbers) are kept in localStorage for the next visit.
 *
 *   import { attachFace } from './heed-face.js';
 *   const face = attachFace(h);  // { persist: false } keeps the calibration for this page only
 *   await face.start();          // camera permission; restores a saved calibration
 *   await face.calibrate();      // ~50 s: 17-point grid, smooth pursuit, validation
 *   await face.check();          // ~8 s: 5 points measure the offset and correct it
 *   // afterwards every mouse click / tap adds a sample and refits the model
 * The gaze model itself (features, regression, offset, saved form) is in heed-gaze.js.
 */
import { FilesetResolver, FaceLandmarker } from './vendor/vision_bundle.mjs';
import { features, createGaze, median, STORE_VERSION } from './heed-gaze.js';

const VENDOR = new URL('./vendor/', import.meta.url).href;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const DEG = 180 / Math.PI;
const wait = ms => new Promise(r => setTimeout(r, ms));
const shuffle = a => a.map(p => [Math.random(), p]).sort((p, q) => p[0] - q[0]).map(p => p[1]);

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
  const storeKey = opts.persist === false ? null : typeof opts.persist === 'string' ? opts.persist : 'heed-face:calibration';
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
    model: emptyModel(),
    saved: null                          // { at, applied, reason } while a calibration is stored on this device
  };
  ctx.faceState = state;
  function emptyModel() { return { source: 'none', samples: 0, clicks: 0, cvErrorPx: null, validationErrorPx: null, correctionPx: null, check: null, restoredFrom: null }; }

  let video, stream, landmarker, running = false, lastDetect = 0, lastSample = 0, startedAt = 0;
  let smooth = null, latest = null, sink = null, refitTimer = 0;
  let eyesClosed = false, blinkBase = null, closedAt = 0, blinkTimes = [], awaySince = 0, frames = 0, fpsT0 = 0;
  let gazeRegion = null, gazeSince = 0;
  const ring = [];                       // last ~1.5 s of frames for click lookup
  const gaze = createGaze({ maxClicks });
  const peek = readSaved();
  if (peek) state.saved = { at: peek.at, applied: false, reason: null };
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
    if (!gaze.model) restore();
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

    if (gaze.model && !eyesClosed) {
      const g = gaze.predict(f.vec);
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

  // ---------- model fitting (the model lives in heed-gaze.js; this keeps the state in step)
  function refit(source) {
    if (!gaze.refit(innerWidth, innerHeight)) return false;
    const m = gaze.model, o = gaze.offset;
    state.calibrated = true; smooth = null;
    state.model = Object.assign({}, state.model, { source: source || state.model.source, samples: gaze.all().length, clicks: gaze.samples.clicks.length, cvErrorPx: m.cvErrorPx,
      correctionPx: o.x || o.y ? { x: Math.round(o.x * innerWidth), y: Math.round(o.y * innerHeight) } : null });
    emit();
    return true;
  }

  // ---------- persistence: samples, not frames; the next visit refits from them
  function readSaved() {
    if (!storeKey) return null;
    try { const d = JSON.parse(localStorage.getItem(storeKey)); return d && d.v === STORE_VERSION ? d : null; } catch (e) { return null; }
  }
  function camera() {
    const track = stream && stream.getVideoTracks()[0];
    return { label: track ? track.label : '', aspect: video && video.videoHeight ? +(video.videoWidth / video.videoHeight).toFixed(3) : null };
  }
  function save() {
    if (!storeKey || !gaze.model || !running) return false;
    const d = Object.assign(gaze.toJSON(), { at: Date.now(), w: innerWidth, h: innerHeight, validationErrorPx: state.model.validationErrorPx }, camera());
    try { localStorage.setItem(storeKey, JSON.stringify(d)); state.saved = { at: d.at, applied: true, reason: null }; return true; } catch (e) { return false; }
  }
  // A saved calibration only fits the camera and screen orientation it was made with.
  function restore() {
    const d = readSaved(), cam = camera();
    if (!d) return;
    const reason = d.label !== cam.label || Math.abs(d.aspect - cam.aspect) > 0.05 ? 'camera' : (d.w > d.h) !== (innerWidth > innerHeight) ? 'orientation' : null;
    state.saved = { at: d.at, applied: false, reason };
    if (reason) return;
    gaze.load(d);
    if (!refit('saved calibration')) { gaze.reset(); state.saved.reason = 'format'; return; }
    state.saved.applied = true;
    Object.assign(state.model, { validationErrorPx: d.validationErrorPx, restoredFrom: d.at });
    ctx.record({ kind: 'calibration', ok: true, restored: true, savedAt: d.at, points: gaze.samples.grid.length, samples: gaze.all().length,
                 pursuit: gaze.samples.pursuit.length, meanErrorPx: gaze.model.cvErrorPx, validationErrorPx: d.validationErrorPx, lambda: gaze.model.lambda });
  }

  // Every mouse click or tap is a labelled sample: people look where they click.
  function onPointerDown(e) {
    if (!running || !ring.length || (e.target && e.target.closest && e.target.closest('.heed-calib'))) return;
    const t = Date.now(), win = ring.filter(r => t - r.t <= 250 && t - r.t >= 0);
    if (win.length < 1) return;
    const f = win[0].f.map((_, i) => median(win.map(r => r.f[i]))), s = { x: e.clientX / innerWidth, y: e.clientY / innerHeight, f };
    gaze.addClick(s);
    const g = gaze.model && gaze.predict(f);
    ctx.record({ kind: 'calibration-click', x: Math.round(e.clientX), y: Math.round(e.clientY), clicks: gaze.samples.clicks.length,
                 predictedErrorPx: g ? Math.round(Math.hypot((g.x - s.x) * innerWidth, (g.y - s.y) * innerHeight)) : null });
    clearTimeout(refitTimer);
    refitTimer = setTimeout(() => { if (refit(gaze.samples.grid.length ? 'calibration + clicks' : 'clicks only')) save(); }, 250);
  }
  document.addEventListener('pointerdown', onPointerDown, true);

  // ---------- full-screen dot layer shared by calibration and the gaze check
  function overlay(title, text) {
    const layer = document.createElement('div');
    layer.setAttribute('data-heed-ignore', '');
    layer.className = 'heed-calib';
    layer.innerHTML = '<div class="heed-calib-card"><b>' + title + '</b><p>' + text + '</p><button type="button" class="go">Start</button></div>' +
      '<i></i><span class="heed-calib-status"></span><button type="button" class="cancel">Cancel</button>';
    document.body.appendChild(layer);
    ctx.calibrating = true;
    const cancel = layer.querySelector('.cancel');
    const o = { dot: layer.querySelector('i'), status: layer.querySelector('.heed-calib-status'), cancelled: false };
    cancel.onclick = () => { o.cancelled = true; };
    o.begin = () => new Promise(res => { layer.querySelector('.go').onclick = res; cancel.addEventListener('click', res); })
      .then(() => layer.querySelector('.heed-calib-card').remove());
    o.place = (x, y) => { o.dot.style.left = (x * 100) + '%'; o.dot.style.top = (y * 100) + '%'; };
    // median features over at least ms and 5 frames (4 s at most); onFrame also sees each frame
    o.collect = async (ms, onFrame) => {
      const got = [];
      sink = (f, t, bs) => { got.push(f); if (onFrame) onFrame(bs); };
      const t0 = Date.now();
      while (!o.cancelled && Date.now() - t0 < 4000 && !(Date.now() - t0 >= ms && got.length >= 5)) await wait(40);
      sink = null;
      return got.length >= 2 ? got[0].map((_, i) => median(got.map(g => g[i]))) : null;
    };
    // a fixation target: 700 ms for the eyes to land, then collect
    o.fixate = async (x, y, ms, onFrame) => { o.place(x, y); o.dot.className = ''; await wait(700); o.dot.className = 'hold'; return o.collect(ms, onFrame); };
    // result card; resolves with the data-act of the button pressed, or 'done' on cancel or after autoMs
    o.finish = (html, buttons, autoMs) => {
      o.dot.remove(); o.status.remove();
      const card = document.createElement('div'); card.className = 'heed-calib-card';
      card.innerHTML = html + '<div class="heed-calib-actions">' + buttons + '</div>';
      layer.appendChild(card);
      return new Promise(res => {
        card.querySelectorAll('button').forEach(b => { b.onclick = () => res(b.dataset.act); });
        cancel.addEventListener('click', () => res('done'));
        if (autoMs) setTimeout(() => res('done'), autoMs);
      });
    };
    o.close = () => { layer.remove(); ctx.calibrating = false; };
    return o;
  }
  const grade = pct => pct <= 8 ? 'good' : pct <= 15 ? 'fair' : 'poor';
  const saved = ok => !storeKey ? '' : ok ? ' It is saved on this device for your next visit.' : ' It could not be saved on this device (storage full or blocked), so it lasts until you leave the page.';

  // ---------- calibration: 17-point grid, smooth pursuit, validation
  async function calibrate(copts = {}) {
    if (!running) await start();
    if (!running) return false;
    const quick = !!copts.quick;
    const o = overlay('Gaze calibration', 'About 50 seconds. Hold the device steady at a comfortable distance and keep your head still. Follow the dot with your eyes.');
    await o.begin();
    const baseBuf = [], grid = [], pursuit = [];
    const toBase = bs => { const raw = {}; EXPRESSIONS.forEach(e => { raw[e.label] = e.score(bs); }); baseBuf.push(raw); };

    // 1. grid (randomised order)
    const g4 = [0.06, 0.36, 0.64, 0.94];
    const targets = shuffle(quick ? [0.1, 0.5, 0.9].flatMap(y => [0.1, 0.5, 0.9].map(x => [x, y])) : g4.flatMap(y => g4.map(x => [x, y])).concat([[0.5, 0.5]]));
    for (let i = 0; i < targets.length && !o.cancelled; i++) {
      const [x, y] = targets[i];
      o.status.textContent = 'Point ' + (i + 1) + ' of ' + targets.length;
      const f = await o.fixate(x, y, 900, toBase);
      if (f) grid.push({ x, y, f });
    }
    if (baseBuf.length >= 20) setBaseline(baseBuf);

    // 2. smooth pursuit: the dot glides along a Lissajous path; targets lag-compensated by 150 ms
    if (!quick && !o.cancelled) {
      o.status.textContent = 'Follow the moving dot';
      o.dot.className = 'pursuit';
      const T = 14000, pos = t => { const s = t / 1000; return [0.5 + 0.43 * Math.sin(2 * Math.PI * s / 7), 0.5 + 0.42 * Math.sin(2 * Math.PI * s / 4.6 + Math.PI / 2)]; };
      o.place(...pos(0)); await wait(900);
      const t1 = Date.now();
      sink = (f, t) => { const [x, y] = pos(Math.max(0, t - t1 - 150)); if (t - t1 > 400) pursuit.push({ x, y, f }); };
      while (!o.cancelled && Date.now() - t1 < T) { o.place(...pos(Date.now() - t1)); await wait(16); }
      sink = null;
    }

    // 3. fit, then validate on fresh points. A calibration starts afresh: of the older samples it keeps
    // only this visit's clicks, and the offset restarts. Cancelling leaves the previous model as it was.
    const undo = o.cancelled ? null : gaze.restart(grid, quick ? null : pursuit);
    if (!undo || !refit(quick ? 'quick calibration' : 'calibration')) {
      if (undo) undo();
      o.close();
      ctx.record({ kind: 'calibration', ok: false, points: grid.length });
      return false;
    }
    let validationErrorPx = null;
    if (!quick && !o.cancelled) {
      const val = [[0.25, 0.2], [0.75, 0.22], [0.5, 0.5], [0.22, 0.78], [0.78, 0.8]], errs = [];
      for (let i = 0; i < val.length && !o.cancelled; i++) {
        o.status.textContent = 'Checking accuracy ' + (i + 1) + ' of ' + val.length;
        const [x, y] = val[i];
        const f = await o.fixate(x, y, 800);
        if (f) { const p = gaze.model.predict(f); errs.push(Math.hypot((p.x - x) * innerWidth, (p.y - y) * innerHeight)); gaze.samples.grid.push({ x, y, f }); }
      }
      if (errs.length) validationErrorPx = Math.round(errs.reduce((a, b) => a + b, 0) / errs.length);
      refit('calibration');
    }
    Object.assign(state.model, { validationErrorPx, check: null, restoredFrom: null });
    const stored = save();
    const diag = Math.hypot(innerWidth, innerHeight);
    const pct = validationErrorPx ? Math.round(validationErrorPx / diag * 100) : null;
    ctx.record({ kind: 'calibration', ok: true, points: gaze.samples.grid.length, samples: gaze.all().length, pursuit: gaze.samples.pursuit.length,
                 meanErrorPx: gaze.model.cvErrorPx, validationErrorPx, validationPctOfScreen: pct, lambda: gaze.model.lambda });
    if (validationErrorPx !== null && !o.cancelled) {
      await o.finish('<b>Calibration ' + grade(pct) + '</b><p>Average error on fresh points: ' + validationErrorPx + ' px, about ' + pct + '% of the screen diagonal. Each click or tap from now on refines it.' + saved(stored) + '</p>',
        '<button type="button" class="go" data-act="done">Done</button>', 8000);
    }
    o.close();
    emit();
    return true;
  }

  // ---------- gaze check: 5 fixation targets measure how far and in which direction the gaze point
  // misses, and set the offset from them. If a shift does not explain the miss, it offers a new calibration.
  async function check() {
    if (!running) await start();
    if (!running) return null;
    if (!gaze.model) return { recalibrated: await calibrate() };
    const o = overlay('Gaze check', 'About 8 seconds. Look at each dot until it shrinks. heed measures how far off the gaze point is and corrects the offset.');
    await o.begin();
    const targets = shuffle([[0.5, 0.5], [0.2, 0.25], [0.8, 0.25], [0.2, 0.75], [0.8, 0.75]]), pts = [];
    for (let i = 0; i < targets.length && !o.cancelled; i++) {
      const [x, y] = targets[i];
      o.status.textContent = 'Point ' + (i + 1) + ' of ' + targets.length;
      const f = await o.fixate(x, y, 800);
      if (f) pts.push({ x, y, f });
    }
    if (o.cancelled || pts.length < 3) {
      if (!o.cancelled) await o.finish('<b>Gaze check incomplete</b><p>Your eyes were not visible on enough points. Try again in even light.</p>', '<button type="button" class="go" data-act="done">Done</button>', 8000);
      o.close(); ctx.record({ kind: 'calibration-check', ok: false, points: pts.length });
      return null;
    }
    const result = Object.assign({ ok: true, points: pts.length }, gaze.measure(pts, innerWidth, innerHeight)), { beforePx, afterPx } = result;
    result.quality = grade(result.afterPct);
    // the check points become the offset's evidence and calibration samples
    gaze.addChecks(pts);
    refit();
    state.model.check = Object.assign({ at: Date.now() }, result);
    save();
    ctx.record(Object.assign({ kind: 'calibration-check' }, result));
    const dx = result.offsetPx.x, dy = result.offsetPx.y, parts = [];
    if (Math.abs(dx) >= 10) parts.push(Math.abs(dx) + ' px ' + (dx < 0 ? 'left' : 'right'));
    if (Math.abs(dy) >= 10) parts.push(Math.abs(dy) + ' px ' + (dy < 0 ? 'above' : 'below'));
    const where = parts.length ? 'The gaze point sat mostly ' + parts.join(' and ') + (Math.abs(dy) >= 10 ? ' the dot. ' : ' of the dot. ') : 'There was no consistent shift. ';
    const poor = result.quality === 'poor';
    const act = await o.finish('<b>Gaze check: ' + result.quality + '</b><p>Before: ' + beforePx + ' px off on average (' + result.beforePct + '% of the screen diagonal). ' + where +
        'With the offset corrected: about ' + afterPx + ' px (' + result.afterPct + '%). ' + (poor ? 'A shift does not explain the error, so a new calibration should help.' : 'Clicks and taps keep it on target.') + '</p>',
      poor ? '<button type="button" class="go" data-act="redo">Recalibrate</button><button type="button" data-act="done">Not now</button>' : '<button type="button" class="go" data-act="done">Done</button>',
      poor ? 0 : 8000);
    o.close();
    emit();
    if (act === 'redo') result.recalibrated = await calibrate();
    return result;
  }

  const style = document.createElement('style');
  style.textContent =
    '.heed-calib{position:fixed;inset:0;z-index:2147483600;background:var(--heed-card,#fff);color:var(--heed-card-fg,#09090b);font:15px/1.45 system-ui,-apple-system,sans-serif}' +
    '.heed-calib-card{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:min(360px,calc(100vw - 48px));display:grid;gap:10px;padding:20px;border:1px solid var(--heed-line,#e4e4e7);border-radius:12px;background:var(--heed-card,#fff)}' +
    '.heed-calib-card p{margin:0;color:var(--heed-muted,#71717a)}.heed-calib-card .go{justify-self:start;background:var(--heed-card-fg,#09090b);color:var(--heed-card,#fff);border-color:var(--heed-card-fg,#09090b)}' +
    '.heed-calib-actions{display:flex;flex-wrap:wrap;gap:8px}' +
    '.heed-calib-status{position:absolute;left:50%;bottom:max(24px,env(safe-area-inset-bottom,0px));transform:translateX(-50%);font-size:13px;color:var(--heed-muted,#71717a)}' +
    '.heed-calib i{position:absolute;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:var(--heed-tint,#2563eb);transition:left .45s ease,top .45s ease,transform .2s}' +
    '.heed-calib i.hold{transform:scale(.5)}.heed-calib i.pursuit{transition:none;transform:scale(.6)}' +
    '.heed-calib button{font:inherit;font-size:14px;padding:6px 12px;border-radius:8px;border:1px solid var(--heed-line,#e4e4e7);background:var(--heed-btn,#fafafa);color:inherit;cursor:pointer}' +
    '.heed-calib .cancel{position:absolute;right:max(16px,env(safe-area-inset-right,0px));top:max(16px,env(safe-area-inset-top,0px))}' +
    '@media (prefers-reduced-motion: reduce){.heed-calib i{transition:none}}';
  document.head.appendChild(style);

  return {
    state, start, stop, calibrate, check,
    get video() { return video; },
    onFrame(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    debug() { const s = gaze.samples; return { latest, eyesClosed, running, offset: gaze.offset, samples: { grid: s.grid.length, pursuit: s.pursuit.length, clicks: s.clicks.length, checks: s.checks.length } }; },
    // forgets the model, its samples and the copy saved on this device
    resetCalibration() {
      clearTimeout(refitTimer); closeGaze('reset');
      gaze.reset();
      state.calibrated = false; state.gaze = null; state.model = emptyModel(); state.saved = null;
      if (storeKey) try { localStorage.removeItem(storeKey); } catch (e) {}
      emit();
    }
  };
}
