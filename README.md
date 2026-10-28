# heed 0.5

Attention-aware toolkit for any web page, with a playground at **https://heed.arvindchee.nu**. It captures where people attend (cursor, scroll, typing, actions, and with the camera on, face, expression and gaze), records it as events, shows it back on the page, and reduces a session to a factual report that an on-device LLM can narrate.

Everything runs in the browser. Video frames never leave the device or get stored; heed keeps only derived numbers. The gaze calibration (those numbers, not images) is kept in the browser's local storage so the next visit can use it.

## Run it locally

No build and no install; the vendored files are in the repo. Serve the folder over HTTP, because ES modules and the camera do not work from `file://`:

```sh
python3 -m http.server 8000    # or: npx serve
```

Open http://localhost:8000. Browsers allow the camera on `localhost`. From a phone on the same network the camera needs HTTPS, so use the hosted playground or an HTTPS tunnel. The on-device summary downloads its model weights from Hugging Face the first time.

## Files

| File | What it is |
| --- | --- |
| `heed.js` | Core: kernel, spec, capture plugins, behaviours, renderers, matrices, timeline, report. No dependencies. |
| `heed-face.js` | Camera plugin: face mesh, head pose, expressions, blinks, calibrated gaze that learns from clicks, saved calibration, gaze check. |
| `heed-gaze.js` | The gaze model `heed-face.js` imports: features, ridge regression, offset correction, saved form. No DOM or camera, so it runs in Node. |
| `heed-llm.js` | Optional summary written by a small model running in the browser (WebLLM, WebGPU). |
| `index.html` | Playground with a chat layout and a dashboard layout. |
| `vendor/` | MediaPipe Face Landmarker 0.10.14 and model, WebLLM 0.2.85. Apache 2.0. |

## Using the playground

1. **Recording** (red dot, top left) pauses and resumes capture. While paused nothing is stored; open gaze dwell and expression episodes are saved at the moment you pause.
2. **Start camera**, then **Calibrate gaze** (about 50 s on a phone). Hold the device steady. The calibration is saved, so on your next visit gaze works as soon as the camera starts.
3. If the gaze dot sits off to one side of where you look, use **Check gaze** (about 8 s) to measure and correct the offset.
4. Read, check and accept or flag claims in **Chat** or **Dashboard**.
5. Every **click or tap** after the camera starts adds a calibration sample and refits the gaze model. Without the full calibration, clicks alone start gaze after eight samples.
6. **Build report**, then **Summarize on this device** for a narrative written from the report.

**Forget gaze calibration** (in the heed panel) deletes the saved calibration from this browser.

Toggles: **Coverage** (tint by attention), **Trail** (path and numbered fixations), **Transitions** (arrows between regions, loops for refixations), **Notify** (toasts for behaviours).

## Gaze calibration

Three phases, then continuous learning:

1. **Grid**: 17 points (4×4 plus centre) in random order. Each point waits 700 ms for the eyes to land, then takes the median of at least 5 frames. Grid samples weigh 3.
2. **Smooth pursuit**: 14 s of a dot gliding along a Lissajous path; each frame is labelled with where the dot was 150 ms earlier, to allow for eye lag. Weight 1.
3. **Validation**: 5 fresh points, never used before the error is measured. The result card reports the average error in pixels and as a share of the screen diagonal (good ≤ 8%, fair ≤ 15%).
4. **Clicks and taps**: each one pairs the median face features from the 250 ms before it with the click position. Weight 2, last 200 kept.

The model is a weighted ridge regression on 22 features (head yaw, pitch, roll and position; iris position in each eye along and across the eye axis; eye openness; face position and size; and second-order terms), with the regularisation picked by 5-fold cross-validation. Expect region-level accuracy on a phone's front camera, best when the head stays near its calibration pose.

### Saved between visits

After a calibration, and after each click or check that changes the model, the labelled samples (feature numbers and target positions) are written to `localStorage` under `heed-face:calibration`. The next time the camera starts, heed refits the model from them, unless the camera or the screen orientation has changed. In that case it says so and leaves the saved copy for when they match again. A new calibration starts afresh: of the older samples it keeps only the clicks from the current visit. Cancelling a calibration leaves the previous one untouched.

### Gaze check and offset correction

Sitting a little differently from the calibration often shifts the whole gaze estimate in one direction. **Check gaze** shows 5 dots (centre and 4 quadrants), measures where heed placed your gaze for each, and reports:

- **Before**: the average error, and in which direction the gaze point sat (e.g. "149 px right and 115 px above the dot").
- **After**: the error once the offset is corrected. Each point is measured with the offset learned from the other four, so the figure is not flattered by its own data.

The offset is the median miss over the visit's 8 most recent check points and clicks. It applies once there are 3, so a single careless click cannot move it far. After a check, the next clicks keep updating it. The check points also become calibration samples (weight 3, last 60 kept). If the error stays above 15% of the screen diagonal after the correction, a shift does not explain the error, and the card offers **Recalibrate**.

## Expressions

MediaPipe gives 52 blendshape scores per frame. heed combines them into five expressions and measures each against **your own neutral face** (learned from your first frames, then from the calibration grid):

| Expression | From | Possible meaning shown |
| --- | --- | --- |
| smiling | mouth smile, cheek raise | positive |
| brows raised | inner and outer brow raise, eyes wide, jaw | surprise or interest |
| frowning | brows down, eye squint | concentration or confusion |
| mouth corners down | mouth frown, lips pressed | doubt or sadness |
| nose wrinkled | nose sneer, upper lip raise | dislike or disgust |

These are facial movements. The meaning column is one reading, and the report and summary present it that way. An expression must hold 500 ms to start an episode; episodes of 700 ms or more are recorded as `emotion` events with the region in focus, e.g. "Frowning (concentration or confusion) for 2.4 s on Source 2: share".

## Idle and away

- **Idle**: no input for `spec.idleMs` (default 10 s). With the camera on, heed tells apart idle while still looking (and at what) from idle with the face out of view. Calibration never counts as idle.
- **Away**: face out of view for over 1.5 s.
- **Left the page**: tab hidden for over 2 s.

## Dwell and transitions

`h.query(spec).transitions({ regions })` returns dwell and visits per region and counts of moves between consecutive **fixations**. A refixation inside the same region counts as a self-transition, so the diagonal of the matrix is filled. The source is gaze when gaze exists, otherwise the cursor. On the page, self-transitions appear as small loops with their count.

## Report and summary

`heed.report(h)` returns counted facts: time recorded and paused, dwell and share per region, regions never attended, top moves and refixations, each claim with what happened to its evidence (checked, glanced, not looked at) and whether it contradicts its evidence (`data-heed-truth="false"` on the claim), idle and away time, expressions and where they happened, typing effort, calibration accuracy (and whether the calibration came from an earlier visit, plus the last gaze check), and sampling answers.

`heed-llm.js` passes only that JSON to a small instruction model in the browser, with instructions to use only those facts and to describe expressions as movements. Models: Qwen2.5 0.5B (default on phones) or 1.5B (default on desktop), in 16-bit form where the GPU supports it, else 32-bit. Weights download once from Hugging Face and stay cached. Needs a browser with WebGPU; without it the page says so and the factual report stands on its own.

## The spec

| Key | Values | Meaning |
| --- | --- | --- |
| `who` | `self`, `peers`, `all` | Whose attention a view shows |
| `when` | `live`, `{ last: ms }`, `{ from, to }` | Which interval |
| `level` | `trace` … `error` | Rendering threshold; producers always run |
| `purpose` | `reflect`, `verify`, `sample`, `coordinate` | Macro that sets the knobs below |
| `signals` | e.g. `['gaze','dwell']` | Event kinds that count as coverage |
| `threshold` | ms, default 800 | Attention needed for evidence to count as checked |
| `idleMs` | ms, default 10000 | Idle threshold |
| `render` | `{ overlay, trail, links, toasts }` | What to draw |
| `notify` | level, default `notice` | Toast threshold |
| `checks` | `['unverified']` | Checks to run |
| `ask` | `false` or `{ on, question, options }` | Attention-triggered experience sampling |

## API

```html
<script src="heed.js"></script>
<script type="module">
  import { attachFace } from './heed-face.js';
  import { summarize } from './heed-llm.js';
  const h = heed.create({ spec: { purpose: 'verify' } });
  const face = attachFace(h);             // { persist: false } keeps the calibration for this page only
  await face.start(); await face.calibrate();   // start() restores a saved calibration
  const check = await face.check();       // { beforePx, afterPx, offsetPx, quality, ... }
  face.resetCalibration();                // forgets the model and the saved copy
  h.pause(); h.resume();
  const facts = heed.report(h);
  const text = await summarize(facts, { size: 'small', onText: t => console.log(t) });
</script>
```

Mark regions with `data-heed="<id>"` (optional `data-heed-label`, `data-heed-cites`, `data-heed-truth`), actions with `data-heed-action`, and UI to ignore with `data-heed-ignore`.

Events: `dwell`, `view`, `gaze`, `fixation`, `typing`, `scroll`, `action`, `face`, `emotion`, `away`, `hidden`, `idle`, `calibration` (with `restored: true` when loaded from an earlier visit), `calibration-click`, `calibration-check`, `recording`, `prompt`, `response`.

## Tested (0.5)

Headless Chrome on a 390×844 phone viewport with a face video as the camera: camera and model load from `vendor/`, face found (478 landmarks), neutral baseline, full calibration (22 points, pursuit, validation, result card), taps refining the model, pause storing nothing while the camera runs, gaze-based claim check, idle while still looking, a smile detected after a switch from a neutral to a smiling face and recorded with its region, the report using camera gaze, and no page errors or failed requests. Desktop Chrome: pause and resume, self-transitions in data and matrix, report, idle. WebKit (Safari's engine): face model (478 landmarks, 52 blendshapes, head transform) and page interactions. On-device summary: on a software (CPU-emulated) WebGPU adapter the model downloaded and the engine started, but generation was too slow to finish there (an 8-token reply ran past 25 minutes), so the written summary has not been seen yet; it needs a real GPU.

Saved calibration and gaze check: headless Chrome at 1280×800 with a synthetic face model in place of MediaPipe, whose iris landmarks follow a simulated point of regard. The test covered a full calibration (6 px validation error on that synthetic face), the saved copy surviving a reload, the restored model working without recalibrating, a cancelled recalibration keeping the old model and saved copy, and a simulated change in sitting position (error 9 → 193 px) that the check named by direction and corrected (to 16 px). It also covered clicks following a further drift (six clicks roughly halved the error, e.g. 143 → 70 px), a distortion that no offset explains (rated poor, with Recalibrate offered), a different camera not restoring, Forget clearing storage, `persist: false`, the report fields, and no page errors.

Not tested: a real phone or real eye movement (including for the saved calibration and the gaze check), the summary text on a real GPU (the test face is a photo, so gaze accuracy is unmeasured here), and Firefox.
