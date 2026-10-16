/*!
 * heed-llm 0.5.0 — on-device session summary (meta layer)
 * Runs a small instruction model in the browser with WebLLM (WebGPU). The model sees only
 * heed.report(h): counted facts, never raw video or keystrokes. Weights download once from
 * Hugging Face and are cached by the browser.
 */
const MODELS = {
  small:  { f16: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC', f32: 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 0.5B' },
  medium: { f16: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC', f32: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC', name: 'Qwen2.5 1.5B' }
};

const SYSTEM = `You write a short behavioural summary of one person's session in an attention-tracking tool.
Use only the JSON facts you are given. Never invent numbers, regions or reasons. Name regions exactly as written.
Facial expressions are muscle movements: describe them as "frowned", "smiled", "raised brows" and give the listed possible meaning as one possibility, never as a fact about feelings.
If a claim "contradicts its evidence" and was accepted, say so plainly. If evidence was "not looked at", say so.
Write plain prose in four short paragraphs with these bold lead-ins: **Focus.** **Checking.** **Expressions and pauses.** **Worth a look.**
The last paragraph gives one or two concrete things to revisit. Keep it under 220 words.`;

let engine = null, engineId = null, loading = null;

export async function gpuInfo() {
  if (!('gpu' in navigator)) return { ok: false, reason: 'This browser has no WebGPU, so the on-device model cannot run here. The facts above are complete without it.' };
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) return { ok: false, reason: 'WebGPU is present but no GPU adapter is available.' };
    return { ok: true, f16: a.features.has('shader-f16') };
  } catch (e) { return { ok: false, reason: String(e && e.message || e) }; }
}

export function modelFor(size, f16) { const m = MODELS[size] || MODELS.small; return { id: f16 ? m.f16 : m.f32, name: m.name + (f16 ? '' : ' (f32)') }; }

export async function summarize(report, { size = 'small', onStatus = () => {}, onText = () => {}, signal } = {}) {
  const gpu = await gpuInfo();
  if (!gpu.ok) throw new Error(gpu.reason);
  const { id, name } = modelFor(size, gpu.f16);
  if (engineId !== id) {
    if (!loading) {
      loading = (async () => {
        onStatus('Loading ' + name + ' (first time downloads the weights)…', 0);
        const webllm = await import('./vendor/web-llm.js');
        if (engine) { try { await engine.unload(); } catch (e) {} }
        engine = await webllm.CreateMLCEngine(id, { initProgressCallback: p => onStatus(p.text, p.progress) });
        engineId = id;
      })().finally(() => { loading = null; });
    }
    await loading;
  }
  onStatus('Writing the summary with ' + name + '…', 1);
  const facts = JSON.stringify(report, null, 1);
  const chunks = await engine.chat.completions.create({
    messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: 'Session facts:\n' + facts }],
    stream: true, temperature: 0.2, max_tokens: 420
  });
  let text = '';
  for await (const c of chunks) {
    if (signal && signal.aborted) { try { await engine.interruptGenerate(); } catch (e) {} break; }
    text += (c.choices[0] && c.choices[0].delta && c.choices[0].delta.content) || '';
    onText(text);
  }
  onStatus('Summary written on this device by ' + name + '.', 1);
  return text;
}
