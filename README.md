# heed 0.5

Attention-aware toolkit for any web page, with a playground at **https://heed.arvindchee.nu**. It captures where people attend (cursor, scroll, typing, actions, and with the camera on, face, expression and gaze), records it as events, shows it back on the page, and reduces a session to a factual report that an on-device LLM can narrate.

Everything runs in the browser. Video frames never leave the device or get stored; heed keeps only derived numbers.

## Files

| File | What it is |
| --- | --- |
| `heed.js` | Core: kernel, spec, capture plugins, behaviours, renderers, matrices, timeline, report. No dependencies. |
| `heed-face.js` | Camera plugin: face mesh, head pose, expressions, blinks, calibrated gaze that learns from clicks. |
| `heed-llm.js` | Optional summary written by a small model running in the browser (WebLLM, WebGPU). |
| `index.html` | Playground with a chat layout and a dashboard layout. |
| `vendor/` | MediaPipe Face Landmarker 0.10.14 and model, WebLLM 0.2.85. Apache 2.0. |

## Using the playground

1. **Recording** (red dot, top left) pauses and resumes capture. While paused nothing is stored; open gaze dwell and expression episodes are saved at the moment you pause.
2. **Start camera**, then **Calibrate gaze** (about 50 s on a phone). Hold the device steady.
3. Read, check and accept or flag claims in **Chat** or **Dashboard**.
4. Every **click or tap** after the camera starts adds a calibration sample and refits the gaze model. Without the full calibration, clicks alone start gaze after eight samples.
5. **Build report**, then **Summarize on this device** for a narrative written from the report.

Toggles: **Coverage** (tint by attention), **Trail** (path and numbered fixations), **Transitions** (arrows between regions, loops for refixations), **Notify** (toasts for behaviours).

## Gaze calibration

Three phases, then continuous learning:

1. **Grid**: 17 points (4×4 plus centre) in random order. Each point waits 700 ms for the eyes to land, then takes the median of at least 5 frames. Grid samples weigh 3.
2. **Smooth pursuit**: 14 s of a dot gliding along a Lissajous path; each frame is labelled with where the dot was 150 ms earlier, to allow for eye lag. Weight 1.
3. **Validation**: 5 fresh points, never used before the error is measured. The result card reports the average error in pixels and as a share of the screen diagonal (good ≤ 8%, fair ≤ 15%).
4. **Clicks and taps**: each one pairs the median face features from the 250 ms before it with the click position. Weight 2, last 200 kept.

The model is a weighted ridge regression on 22 features (head yaw, pitch, roll and position; iris position in each eye along and across the eye axis; eye openness; face position and size; and second-order terms), with the regularisation picked by 5-fold cross-validation. Expect region-level accuracy on a phone's front camera, best when the head stays near its calibration pose.

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

`heed.report(h)` returns counted facts: time recorded and paused, dwell and share per region, regions never attended, top moves and refixations, each claim with what happened to its evidence (checked, glanced, not looked at) and whether it contradicts its evidence (`data-heed-truth="false"` on the claim), idle and away time, expressions and where they happened, typing effort, calibration accuracy, and sampling answers.

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
  const face = attachFace(h);
  await face.start(); await face.calibrate();
  h.pause(); h.resume();
  const facts = heed.report(h);
  const text = await summarize(facts, { size: 'small', onText: t => console.log(t) });
</script>
```

Mark regions with `data-heed="<id>"` (optional `data-heed-label`, `data-heed-cites`, `data-heed-truth`), actions with `data-heed-action`, and UI to ignore with `data-heed-ignore`.

Events: `dwell`, `view`, `gaze`, `fixation`, `typing`, `scroll`, `action`, `face`, `emotion`, `away`, `hidden`, `idle`, `calibration`, `calibration-click`, `recording`, `prompt`, `response`.

## Tested (0.5)

Headless Chrome on a 390×844 phone viewport with a face video as the camera: camera and model load from `vendor/`, face found (478 landmarks), neutral baseline, full calibration (22 points, pursuit, validation, result card), taps refining the model, pause storing nothing while the camera runs, gaze-based claim check, idle while still looking, a smile detected after a switch from a neutral to a smiling face and recorded with its region, the report using camera gaze, and no page errors or failed requests. Desktop Chrome: pause and resume, self-transitions in data and matrix, report, idle. WebKit (Safari's engine): face model (478 landmarks, 52 blendshapes, head transform) and page interactions. On-device summary: WebLLM with Qwen2.5 0.5B on a software WebGPU adapter.

Not tested: a real phone or real eye movement (the test face is a photo, so gaze accuracy is unmeasured here), and Firefox.
