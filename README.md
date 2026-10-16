# heed 0.4 — face, trails, transitions and behaviour

Static site. Open `index.html` over HTTPS (phones only allow the camera on secure pages, or on `localhost`).

## Put it on your phone (GitHub Pages, about 3 minutes)

1. Create a repo, e.g. `arvindcheenu/heed-demo`, and push this folder's contents to `main`.
2. Repo → Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/`.
3. Open `https://arvindcheenu.github.io/heed-demo/` on the phone, tap **Start camera**, allow the camera, then **Calibrate gaze**.

Any static HTTPS host works the same way. Locally on the laptop: `python3 -m http.server` and open `http://localhost:8000`.

## What the camera adds

`heed-face.js` runs MediaPipe Face Landmarker (478 landmarks, 52 blendshapes, head transform) entirely on the device. Model and WASM are vendored in `vendor/`, so the page makes no third-party requests. No video frames are stored or sent; heed records only derived numbers.

| Event | Fields | Rate |
| --- | --- | --- |
| `face` | `present, yaw, pitch, roll` (degrees), `smile, browDown, browUp, jawOpen` (0–1), `blinkRate` (/min), `gx, gy` (px, after calibration) | 2 Hz |
| `gaze` | `region, ms` — gaze dwell on a `data-heed` region; counts as coverage when `signals` includes `gaze` | on region change |
| `away` | `ms` — face out of view > 1.5 s; also a `notice` finding | per episode |
| `calibration` | `ok, points, meanErrorPx` | per run |
| `face-start` / `face-stop` | | |

Gaze is a 9-point ridge regression on head yaw/pitch plus iris position within each eye. On a phone's front camera expect region-level accuracy (roughly a quarter of the screen), better horizontally than vertically, and only while the head stays near its calibration pose. Recalibrate when posture changes.

**Show attention** toggles only the visible layer (region tint, gaze dot, the floating face readout). Capture continues while it is off. The browser still asks for camera permission and shows its camera indicator; a page cannot suppress either.

## The playground

Two layouts share one heed instance: **Chat** (a question, an answer with three cited claims and its sources, and a composer that adds turns) and **Dashboard** (filters, KPI tiles, an AI insight with two cited claims, four charts). One claim in each is false against its own evidence (Factory B in chat; "alerts doubled" on the dashboard), so the unverified check and sampling have something real to catch.

Toolbar toggles map to `render`: **Coverage** (`overlay`), **Trail** (`trail`), **Transitions** (`links`), **Notify** (`toasts`).

## Dwell and transitions

`h.query(spec).transitions({ regions })` returns `{ kind, dwell, visits, counts }`: ms and visit count per region, and `"from>to"` move counts. Visits come from gaze when the spec counts gaze and gaze exists, otherwise from cursor dwell; consecutive pieces of one region within 1.5 s merge into one visit, and each person's sequence is counted separately.

- **Transitions over the page** (`render.links`): arrows between the regions' live positions, thicker for more frequent moves, with the count on each arrow.
- **Matrices** (`heed.matrix(h, el)`): dwell per region (seconds, share, visits), the from → to heatmap, the three most frequent moves, and the regions in the current layout that were never attended.

Both follow Who and When, so Peers or Last 30 s change them live.

## Trails and behaviour

**Trail** (`render.trail`) draws the last 15 s of cursor, touch and gaze as a fading path, with fixations as circles sized by duration and numbered in order: a scanpath over the live page. Gaze is pink and cursor blue.

**Notify** (`render.toasts`) pops a short card for each finding at or above `notify` (default `notice`).

**Event trails** (`heed.timeline(h, canvas)`) shows the last 60 s as swimlanes: cursor, gaze, in view, fixations, typing, scroll, actions, presence and findings.

Behaviours detected by the `behaviors` plugin:

| Finding | Level | When |
| --- | --- | --- |
| `revisit` | info | Back on a region after more than 4 s elsewhere |
| `skim` | notice | A region was in view under 0.7 s and never looked at or hovered |
| `quick` | notice | A claim was accepted with under 1.5 s spent on the claim itself |
| `hesitation` | info / notice | Hovered an action for over 1.2 s, then clicked or moved away |
| `idle` | info | No input for 10 s |
| `left` | notice | The page was hidden for over 2 s |
| `fast-scroll` | debug | Scrolling faster than 2500 px/s |
| `frown`, `smile`, `surprise` | info | Sustained expression (camera on), tied to the region in focus |
| `away` | notice | Face out of view over 1.5 s (camera on) |
| `unverified` | warn / notice / info | Claim accepted without, with a glance at, or after checking its cited evidence |

Fixations are recorded as `fixation` events (`src, x, y, ms, region`), using a dispersion threshold of 30 px for cursor and 90 px for gaze. `h.export().trail` holds the raw points.

## Using it in your own page

```html
<script src="heed.js"></script>
<script type="module">
  import { attachFace } from './heed-face.js';
  const h = heed.create({ spec: { purpose: 'verify' } });
  const face = attachFace(h, { fps: 15 });
  startButton.onclick = async () => { await face.start(); await face.calibrate(); h.set({ ...h.spec, signals: ['gaze', 'dwell'] }); };
  face.onFrame(s => console.log(s.pose, s.expr, s.gaze));
</script>
```

## Tested

Headless Chrome on a 390 × 844 phone viewport, with a face video as the camera: camera start and model load from `vendor/`, face found (478 landmarks), head pose and expression, 9-point calibration, gaze driving coverage, gaze trail and gaze dwell, the unverified check, the sampling question, composer typing effort and new turns, the dashboard with gaze matrices, swimlanes, JSON export, and no page errors or failed requests (17 checks). WebKit (Safari's engine) ran the same model on the test face: 478 landmarks, 52 blendshapes, head transform. Desktop Chrome ran the cursor-only path with no errors.

Not tested: a real phone, real eye movement (the test face does not look around, so gaze accuracy is unmeasured), and Firefox.

## Third-party

`vendor/` contains `@mediapipe/tasks-vision` 0.10.14 and the `face_landmarker.task` model (float16, v1) from Google, Apache License 2.0.
