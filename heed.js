/*!
 * heed 0.4.0 — attention-aware toolkit (MVP)
 * Capture -> Record -> Revisualize for any DOM, driven by one declarative spec:
 *   { who, when, level, purpose }  (+ render / checks / ask knobs)
 * Browser-only signals: pointer dwell, viewport visibility, keystroke dynamics, actions.
 * No dependencies. Works as a <script> global (window.heed) or CommonJS module.
 */
(function (global, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else global.heed = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------- constants
  var LEVELS = ['trace', 'debug', 'info', 'notice', 'warn', 'error'];
  var rank = function (l) { var i = LEVELS.indexOf(l); return i < 0 ? 2 : i; };
  var now = function () { return Date.now(); };
  var uid = function () { return Math.random().toString(36).slice(2, 10); };
  var MIN_DWELL = 120; // ms; shorter passes are treated as transit, not attention

  // Purposes are macros: they only set knobs. A spec is complete without one.
  var PURPOSES = {
    reflect:    { who: 'self',  level: 'info',   render: { overlay: true },  checks: [],             ask: false },
    coordinate: { who: 'peers', level: 'info',   render: { overlay: true },  checks: [],             ask: false },
    verify:     { who: 'self',  level: 'notice', render: { overlay: true },  checks: ['unverified'], ask: false },
    sample:     { who: 'self',  level: 'warn',   render: { overlay: false, trail: false, toasts: false }, checks: ['unverified'],
                  ask: { on: 'warn', question: 'Did you check the evidence for this claim?',
                         options: ['Yes', 'No', "Didn't need to"], cooldownMs: 0, max: 20 } }
  };

  var DEFAULTS = {
    who: 'self',            // self | peers | all
    when: 'live',           // live | { last: ms } | { from: t, to: t }
    level: 'info',          // rendering threshold; producers always run
    purpose: null,
    signals: null,          // which event kinds count toward coverage; null = auto
    threshold: 800,         // ms of attention for a region to count as checked
    render: { overlay: true, trail: true, links: false, toasts: true },
    notify: 'notice',       // toast threshold, separate from the rendering level
    trailMs: 15000,         // how much trail to draw
    checks: [],
    ask: false
  };

  function merge(a, b) {
    var out = {}, k;
    for (k in a) out[k] = a[k];
    for (k in b) {
      if (b[k] === undefined) continue;
      out[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object' && !Array.isArray(a[k]))
        ? merge(a[k], b[k]) : b[k];
    }
    return out;
  }

  function normalize(spec) {
    spec = spec || {};
    var p = spec.purpose && PURPOSES[spec.purpose] ? PURPOSES[spec.purpose] : {};
    var s = merge(merge(DEFAULTS, p), spec);
    if (!s.signals) {
      var fine = typeof matchMedia === 'function' && matchMedia('(pointer: fine)').matches;
      s.signals = fine ? ['dwell'] : ['view', 'dwell'];
    }
    if (LEVELS.indexOf(s.level) < 0) s.level = 'info';
    return s;
  }

  // ---------------------------------------------------------------- event bus
  function createBus() {
    var map = {};
    return {
      on: function (type, fn) {
        (map[type] = map[type] || []).push(fn);
        return function () { map[type] = (map[type] || []).filter(function (f) { return f !== fn; }); };
      },
      emit: function (type, payload) {
        [type, '*'].forEach(function (k) {
          (map[k] || []).slice().forEach(function (fn) {
            try { fn(payload, type); } catch (e) { console.error('[heed]', e); }
          });
        });
      }
    };
  }

  // ---------------------------------------------------------------- kernel
  function create(opts) {
    opts = opts || {};
    var root = opts.root || document;
    var me = { id: opts.participant || ('p-' + uid()), session: opts.session || ('s-' + uid()), started: now() };
    var bus = createBus();
    var events = [], findings = [], cleanups = [];
    var spec = normalize(opts.spec);

    function within(t, when) {
      if (!when || when === 'live') return true;
      if (when.last) return t >= now() - when.last;
      return t >= (when.from || 0) && t <= (when.to || Infinity);
    }
    function whoMatch(ev, who) {
      if (who === 'all') return true;
      if (who === 'peers') return ev.who !== me.id;
      return ev.who === me.id;
    }

    function regionOf(el) {
      while (el && el !== root && el.nodeType === 1) {
        if (el.hasAttribute('data-heed-ignore')) return null;
        if (el.hasAttribute('data-heed')) return el;
        el = el.parentNode;
      }
      return null;
    }
    function regionInfo(el) {
      return {
        region: el.getAttribute('data-heed'),
        rkind: el.getAttribute('data-heed-kind') || 'region',
        cites: (el.getAttribute('data-heed-cites') || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean)
      };
    }
    function regions() {
      var scope = root.querySelectorAll ? root : document;
      return Array.prototype.slice.call(scope.querySelectorAll('[data-heed]'));
    }

    function record(e) {
      var ev = merge({ id: uid(), t: now(), who: me.id, session: me.session }, e);
      events.push(ev);
      bus.emit('event', ev);
      bus.emit('event:' + ev.kind, ev);
      return ev;
    }
    function ingest(ev) { // events from peers: stored and rendered, never re-checked
      if (events.some(function (x) { return x.id === ev.id; })) return;
      ev.remote = true;
      events.push(ev);
      bus.emit('event', ev);
    }
    function finding(f) {
      var x = merge({ id: uid(), t: now(), who: me.id, level: 'info' }, f);
      findings.push(x);
      bus.emit('finding', x);
      if (rank(x.level) >= rank(spec.level)) bus.emit('surface', x);
      return x;
    }

    function coverage(region, q) {
      var s = q ? normalize(merge(spec, q)) : spec, ms = 0;
      for (var i = 0; i < events.length; i++) {
        var e = events[i];
        if (e.region !== region || !e.ms || s.signals.indexOf(e.kind) < 0) continue;
        if (!whoMatch(e, s.who) || !within(e.t, s.when)) continue;
        ms += e.ms;
      }
      return ms;
    }

    function view(q) {
      var s = q ? normalize(q) : spec;
      return {
        spec: s,
        coverage: function (r) { return coverage(r, s); },
        regions: function () {
          var list = regions().map(function (el) {
            var info = regionInfo(el);
            return { id: info.region, kind: info.rkind, ms: coverage(info.region, s) };
          });
          var max = Math.max.apply(null, list.map(function (r) { return r.ms; }).concat([1]));
          list.forEach(function (r) { r.share = r.ms / max; });
          return list;
        },
        events: function () { return events.filter(function (e) { return whoMatch(e, s.who) && within(e.t, s.when); }); },
        findings: function () { return findings.filter(function (f) { return rank(f.level) >= rank(s.level) && within(f.t, s.when); }); },
        // Visits in order, per person: consecutive pieces of one region merge into one visit.
        // Source: gaze when the spec counts gaze and gaze exists, else cursor dwell.
        sequence: function (src) {
          var kind = src || (s.signals.indexOf('gaze') >= 0 && events.some(function (e) { return e.kind === 'gaze'; }) ? 'gaze' : 'dwell');
          var byWho = {};
          events.forEach(function (e) {
            if (e.kind !== kind || !e.region || !e.ms || !whoMatch(e, s.who) || !within(e.t, s.when)) return;
            (byWho[e.who] = byWho[e.who] || []).push({ region: e.region, a: e.t - e.ms, b: e.t, ms: e.ms });
          });
          var seqs = Object.keys(byWho).map(function (w) {
            var out = [];
            byWho[w].sort(function (p, q) { return p.a - q.a; }).forEach(function (v) {
              var l = out[out.length - 1];
              if (l && l.region === v.region && v.a - l.b < 1500) { l.b = v.b; l.ms += v.ms; } else out.push(merge({}, v));
            });
            return out;
          });
          return { kind: kind, seqs: seqs };
        },
        // Dwell (ms), visits and region-to-region transition counts, optionally limited to some regions.
        transitions: function (o) {
          o = o || {};
          var sq = this.sequence(o.src), keep = o.regions || null, counts = {}, dwell = {}, visits = {};
          sq.seqs.forEach(function (seq) {
            var prev = null;
            seq.forEach(function (v) {
              if (keep && keep.indexOf(v.region) < 0) return;
              dwell[v.region] = (dwell[v.region] || 0) + v.ms;
              visits[v.region] = (visits[v.region] || 0) + 1;
              if (prev && prev !== v.region) { var k = prev + '>' + v.region; counts[k] = (counts[k] || 0) + 1; }
              prev = v.region;
            });
          });
          return { kind: sq.kind, counts: counts, dwell: dwell, visits: visits };
        }
      };
    }

    var instance = {
      self: me,
      get spec() { return spec; },
      set: function (next) { spec = normalize(next); bus.emit('spec', spec); return instance; },
      query: view,
      coverage: coverage,
      on: bus.on,
      use: function (plugin) {
        var c = plugin.setup(ctx);
        if (typeof c === 'function') cleanups.push(c);
        return instance;
      },
      export: function () {
        return { heed: '0.4.0', participant: me.id, session: me.session, started: me.started,
                 exported: now(), spec: spec, events: events.slice(), findings: findings.slice(), trail: (ctx.trail || []).slice() };
      },
      trail: function () { return (ctx.trail || []).slice(); },
      fixations: function () { return (ctx.fixations || []).slice(); },
      flush: function () { bus.emit('flush'); },
      destroy: function () { bus.emit('flush'); cleanups.forEach(function (fn) { try { fn(); } catch (e) {} }); cleanups = []; }
    };

    var ctx = {
      root: root, self: me, bus: bus, events: events, findings: findings,
      get spec() { return spec; },
      record: record, ingest: ingest, finding: finding,
      regionOf: regionOf, regionInfo: regionInfo, regions: regions,
      coverage: coverage, view: view, rank: rank, heed: instance
    };

    var list = opts.plugins || defaultPlugins();
    list.forEach(function (p) { instance.use(p); });
    return instance;
  }

  // ---------------------------------------------------------------- plugins
  // A plugin is { name, setup(ctx) -> cleanup? }. Everything below is replaceable.

  // Pointer dwell: time the cursor rests inside a region.
  var pointer = {
    name: 'pointer',
    setup: function (ctx) {
      var cur = null, since = 0, doc = ctx.root.ownerDocument || document;
      function close(reason) {
        if (!cur) return;
        var ms = now() - since;
        if (ms >= MIN_DWELL) ctx.record(merge({ kind: 'dwell', ms: ms, reason: reason }, ctx.regionInfo(cur)));
        cur = null;
      }
      function move(e) {
        if (e.pointerType && e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
        var r = ctx.regionOf(e.target);
        if (r === cur) return;
        close('move');
        if (r) { cur = r; since = now(); }
      }
      function leave() { close('leave'); }
      function vis() { if (doc.hidden) close('hidden'); }
      doc.addEventListener('pointermove', move, { passive: true });
      doc.addEventListener('pointerleave', leave);
      doc.addEventListener('visibilitychange', vis);
      var tick = setInterval(function () { // checkpoint long dwells so live views stay current
        if (cur && now() - since > 1000) { var el = cur; close('checkpoint'); cur = el; since = now(); }
      }, 1000);
      var off = ctx.bus.on('flush', function () { close('flush'); });
      return function () {
        doc.removeEventListener('pointermove', move); doc.removeEventListener('pointerleave', leave);
        doc.removeEventListener('visibilitychange', vis); clearInterval(tick); off();
      };
    }
  };

  // Viewport: time a region is at least 60% visible. Touch devices rely on this.
  var viewport = {
    name: 'viewport',
    setup: function (ctx) {
      if (typeof IntersectionObserver !== 'function') return;
      var seen = new Map();
      function close(el) {
        var t = seen.get(el); if (!t) return;
        seen.delete(el);
        var ms = now() - t;
        if (ms >= MIN_DWELL) ctx.record(merge({ kind: 'view', ms: ms }, ctx.regionInfo(el)));
      }
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.intersectionRatio >= 0.6) { if (!seen.has(en.target)) seen.set(en.target, now()); }
          else close(en.target);
        });
      }, { threshold: [0, 0.6, 1] });
      ctx.regions().forEach(function (el) { io.observe(el); });
      var tick = setInterval(function () { seen.forEach(function (t, el) { if (now() - t > 2000) { close(el); seen.set(el, now()); } }); }, 2000);
      var off = ctx.bus.on('flush', function () { Array.from(seen.keys()).forEach(close); });
      return function () { io.disconnect(); clearInterval(tick); off(); };
    }
  };

  // Keystroke dynamics on text inputs inside regions (after Schutz et al. 2026):
  // keys, backspaces, mean inter-key interval, pauses > 2 s. Emitted per composition.
  var keys = {
    name: 'keys',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document, comp = null;
      function start(target) {
        var r = ctx.regionOf(target);
        comp = { el: target, region: r ? ctx.regionInfo(r).region : null, t0: now(), last: 0, keys: 0, back: 0, pauses: 0, gaps: [] };
      }
      function down(e) {
        var t = e.target;
        if (!t || !(t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && /text|search|email|url/.test(t.type)) || t.isContentEditable)) return;
        if (!comp || comp.el !== t) start(t);
        var n = now();
        if (comp.last) { var g = n - comp.last; comp.gaps.push(g); if (g > 2000) comp.pauses++; }
        comp.last = n; comp.keys++;
        if (e.key === 'Backspace' || e.key === 'Delete') comp.back++;
      }
      function end(reason) {
        if (!comp || comp.keys === 0) { comp = null; return; }
        var g = comp.gaps, mean = g.length ? g.reduce(function (a, b) { return a + b; }, 0) / g.length : 0;
        var ev = ctx.record({ kind: 'typing', region: comp.region, ms: now() - comp.t0, keys: comp.keys,
          backspaces: comp.back, pauses: comp.pauses, meanIKI: Math.round(mean),
          chars: (comp.el.value || comp.el.textContent || '').length, reason: reason });
        ctx.finding({ level: 'debug', type: 'effort', trigger: ev.id, region: comp.region,
          text: comp.keys + ' keys, ' + comp.back + ' corrections, ' + comp.pauses + ' pauses, ' + Math.round(mean) + ' ms between keys' });
        comp = null;
      }
      function blur(e) { if (comp && e.target === comp.el) end('blur'); }
      doc.addEventListener('keydown', down, true);
      doc.addEventListener('focusout', blur, true);
      var off = ctx.bus.on('flush', function () { end('flush'); });
      var off2 = ctx.bus.on('compose:end', function () { end('submit'); });
      return function () { doc.removeEventListener('keydown', down, true); doc.removeEventListener('focusout', blur, true); off(); off2(); };
    }
  };

  // Actions: clicks on [data-heed-action] inside a region (accept, flag, open...).
  var actions = {
    name: 'actions',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      function click(e) {
        var a = e.target.closest && e.target.closest('[data-heed-action]');
        if (!a || a.closest('[data-heed-ignore]')) return;
        var r = ctx.regionOf(a);
        ctx.record(merge({ kind: 'action', action: a.getAttribute('data-heed-action'), el: a },
          r ? ctx.regionInfo(r) : { region: null, cites: [] }));
      }
      doc.addEventListener('click', click, true);
      return function () { doc.removeEventListener('click', click, true); };
    }
  };

  // Check: a claim accepted without attention to the evidence it cites.
  // Produces warn (never looked), notice (glanced), info (checked).
  var unverified = {
    name: 'unverified',
    setup: function (ctx) {
      return ctx.bus.on('event:action', function (ev) {
        if (ev.remote || ev.action !== 'accept' || !ctx.spec.checks || ctx.spec.checks.indexOf('unverified') < 0) return;
        ctx.heed.flush();
        var cites = ev.cites || [];
        if (!cites.length) return;
        var s = { who: 'self', when: 'live' }; // judged on your own attention, whole session
        var cov = cites.map(function (c) { return { region: c, ms: ctx.coverage(c, s) }; });
        var worst = Math.min.apply(null, cov.map(function (c) { return c.ms; }));
        var level = worst === 0 ? 'warn' : worst < ctx.spec.threshold ? 'notice' : 'info';
        var label = function (c) { return c.region + ' (' + (c.ms / 1000).toFixed(1) + ' s)'; };
        ctx.finding({ level: level, type: 'unverified', trigger: ev.id, region: ev.region, cites: cov, el: ev.el,
          text: level === 'warn' ? 'Accepted "' + ev.region + '" without attending to ' + cov.filter(function (c) { return !c.ms; }).map(label).join(', ')
              : level === 'notice' ? 'Accepted "' + ev.region + '" after a glance at ' + cov.map(label).join(', ')
              : 'Accepted "' + ev.region + '" after checking ' + cov.map(label).join(', ') });
      });
    }
  };

  // Renderer: tint each region by coverage under the current spec.
  // At level info or lower, every region shows its share; at notice or above,
  // only regions involved in a surfaced finding are marked.
  var overlay = {
    name: 'overlay',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      var style = doc.createElement('style');
      style.textContent =
        '[data-heed]{position:relative}' +
        '[data-heed][data-heed-on]::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;' +
        'background:var(--heed-tint,#2563eb);opacity:calc(var(--heed-share,0)*.16);transition:opacity .3s}' +
        '[data-heed][data-heed-on]::after{content:attr(data-heed-badge);position:absolute;top:-9px;right:10px;z-index:1;pointer-events:none;' +
        'font:500 11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;padding:1px 6px;border-radius:999px;' +
        'background:var(--heed-badge-bg,#eff6ff);color:var(--heed-badge-fg,#1d4ed8);border:1px solid var(--heed-badge-line,#bfdbfe)}' +
        '[data-heed][data-heed-mark="warn"]{outline:2px solid var(--heed-warn,#d97706);outline-offset:2px}' +
        '[data-heed][data-heed-mark="notice"]{outline:2px dashed var(--heed-notice,#ca8a04);outline-offset:2px}';
      doc.head.appendChild(style);
      var marks = {};
      function paint() {
        var s = ctx.spec, on = s.render && s.render.overlay, low = rank(s.level) <= rank('info');
        var v = ctx.view(s), list = v.regions(), byId = {};
        list.forEach(function (r) { byId[r.id] = r; });
        ctx.regions().forEach(function (el) {
          var id = el.getAttribute('data-heed'), r = byId[id];
          if (on && low && r) {
            el.setAttribute('data-heed-on', '');
            el.style.setProperty('--heed-share', r.share.toFixed(3));
            el.setAttribute('data-heed-badge', (r.ms / 1000).toFixed(1) + ' s');
          } else {
            el.removeAttribute('data-heed-on'); el.style.removeProperty('--heed-share'); el.removeAttribute('data-heed-badge');
          }
          var m = on ? marks[id] : null;
          if (m && rank(m) >= rank(s.level)) el.setAttribute('data-heed-mark', m); else el.removeAttribute('data-heed-mark');
        });
      }
      var offF = ctx.bus.on('finding', function (f) {
        if (f.type !== 'unverified') return;
        (f.cites || []).forEach(function (c) {
          var lvl = c.ms === 0 ? 'warn' : c.ms < ctx.spec.threshold ? 'notice' : null;
          if (lvl && (!marks[c.region] || rank(lvl) > rank(marks[c.region]))) marks[c.region] = lvl;
          if (!lvl) delete marks[c.region];
        });
        paint();
      });
      var offS = ctx.bus.on('spec', paint);
      var tick = setInterval(paint, 500);
      paint();
      return function () { clearInterval(tick); offF(); offS(); style.remove(); };
    }
  };

  // Ask: attention-triggered experience sampling. When a finding at the spec's
  // ask level surfaces, ask one question and store the answer with the trigger.
  var ask = {
    name: 'ask',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document, asked = 0, lastAt = 0, open = null;
      var style = doc.createElement('style');
      style.textContent =
        '.heed-ask{position:fixed;z-index:2147483000;right:max(16px,env(safe-area-inset-right,0px));bottom:max(16px,env(safe-area-inset-bottom,0px));' +
        'width:min(340px,calc(100vw - 32px));box-sizing:border-box;padding:16px;border-radius:12px;' +
        'background:var(--heed-card,#fff);color:var(--heed-card-fg,#09090b);border:1px solid var(--heed-line,#e4e4e7);' +
        'box-shadow:0 10px 30px -10px rgba(0,0,0,.25);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;display:grid;gap:10px}' +
        '.heed-ask small{font-size:12px;color:var(--heed-muted,#71717a)}' +
        '.heed-ask p{margin:0;font-weight:600}' +
        '.heed-ask div{display:flex;flex-wrap:wrap;gap:8px}' +
        '.heed-ask button{font:inherit;font-size:13px;padding:6px 12px;border-radius:8px;cursor:pointer;' +
        'border:1px solid var(--heed-line,#e4e4e7);background:var(--heed-btn,#fafafa);color:inherit}' +
        '.heed-ask button:hover{background:var(--heed-btn-hover,#f4f4f5)}' +
        '.heed-ask button:focus-visible{outline:2px solid var(--heed-tint,#2563eb);outline-offset:2px}';
      doc.head.appendChild(style);
      function show(f) {
        var a = ctx.spec.ask;
        if (!a || open) return;
        if (a.max && asked >= a.max) return;
        if (a.cooldownMs && now() - lastAt < a.cooldownMs) return;
        if (rank(f.level) < rank(a.on || 'warn')) return;
        asked++; lastAt = now();
        var card = doc.createElement('div'), t0 = now();
        card.className = 'heed-ask'; card.setAttribute('data-heed-ignore', ''); card.setAttribute('role', 'dialog');
        card.setAttribute('aria-label', 'Quick question');
        var small = doc.createElement('small'); small.textContent = 'Quick question · ' + (f.region || '');
        var p = doc.createElement('p'); p.textContent = a.question;
        var row = doc.createElement('div');
        (a.options || ['Yes', 'No']).forEach(function (o) {
          var b = doc.createElement('button'); b.type = 'button'; b.textContent = o;
          b.addEventListener('click', function () {
            ctx.record({ kind: 'response', region: f.region, finding: f.id, trigger: f.trigger,
                         question: a.question, answer: o, latencyMs: now() - t0 });
            card.remove(); open = null;
          });
          row.appendChild(b);
        });
        var skip = doc.createElement('button'); skip.type = 'button'; skip.textContent = 'Skip';
        skip.style.marginLeft = 'auto';
        skip.addEventListener('click', function () {
          ctx.record({ kind: 'response', region: f.region, finding: f.id, trigger: f.trigger, question: a.question, answer: null, latencyMs: now() - t0 });
          card.remove(); open = null;
        });
        row.appendChild(skip);
        card.appendChild(small); card.appendChild(p); card.appendChild(row);
        doc.body.appendChild(card); open = card;
        ctx.record({ kind: 'prompt', region: f.region, finding: f.id, question: a.question });
        var first = row.querySelector('button'); if (first) first.focus({ preventScroll: true });
      }
      var off = ctx.bus.on('surface', show);
      return function () { off(); style.remove(); if (open) open.remove(); };
    }
  };

  // Transport: share events with peers (other tabs/windows on the same origin).
  var broadcast = {
    name: 'broadcast',
    setup: function (ctx) {
      if (typeof BroadcastChannel !== 'function') return;
      var ch;
      try { ch = new BroadcastChannel('heed'); } catch (e) { return; }
      var send = function (ev) {
        if (ev.remote) return;
        var copy = {}; for (var k in ev) if (k !== 'el') copy[k] = ev[k];
        try { ch.postMessage(copy); } catch (e) {}
      };
      ch.onmessage = function (m) { if (m.data && m.data.who !== ctx.self.id) ctx.ingest(m.data); };
      var off = ctx.bus.on('event', send);
      return function () { off(); ch.close(); };
    }
  };

  // Store: keep the session in localStorage when available (best effort).
  var store = {
    name: 'store',
    setup: function (ctx) {
      var key = 'heed:' + ctx.self.session;
      var save = function () {
        try { localStorage.setItem(key, JSON.stringify(ctx.heed.export(), function (k, v) { return k === 'el' ? undefined : v; })); } catch (e) {}
      };
      var tick = setInterval(save, 3000);
      return function () { clearInterval(tick); save(); };
    }
  };


  // Points: raw positions (cursor, touch, gaze) in page coordinates, kept in memory
  // for trails and fixations. Other plugins add points with ctx.point(clientX, clientY, src).
  var points = {
    name: 'points',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document, last = {};
      ctx.trail = [];
      ctx.point = function (cx, cy, src) {
        var t = now();
        if (last[src] && t - last[src] < 30) return;
        last[src] = t;
        var p = { x: Math.round(cx + (window.scrollX || 0)), y: Math.round(cy + (window.scrollY || 0)), t: t, src: src };
        ctx.trail.push(p);
        if (ctx.trail.length > 6000) ctx.trail.splice(0, 2000);
        ctx.bus.emit('point', p);
      };
      function move(e) { ctx.point(e.clientX, e.clientY, e.pointerType === 'touch' ? 'touch' : 'pointer'); }
      doc.addEventListener('pointermove', move, { passive: true });
      doc.addEventListener('pointerdown', move, { passive: true });
      return function () { doc.removeEventListener('pointermove', move); doc.removeEventListener('pointerdown', move); };
    }
  };

  // Behaviours: turn raw streams into named moments, each a finding.
  //   fixation (event), revisit, skim, quick accept, hesitation, idle, left page,
  //   fast scroll, and expression-at-region (frown, smile, raised brows) when a face adapter runs.
  var behaviors = {
    name: 'behaviors',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      var say = function (level, type, text, extra) { return ctx.finding(merge({ level: level, type: type, text: text }, extra || {})); };
      var visits = {}, skimmed = {}, lastInput = now(), idleOn = false, hiddenAt = 0, hover = null;
      var lastFace = null, exprAt = {}, lastScroll = { y: window.scrollY || 0, t: now() }, fastAt = 0, scrollAt = 0;
      ctx.fixations = []; ctx.focus = null;

      // fixations: dispersion threshold per source
      var R = { pointer: 30, touch: 40, gaze: 90 }, MINF = { pointer: 250, touch: 200, gaze: 200 }, cl = {};
      function endCluster(src) {
        var c = cl[src]; if (!c) return; delete cl[src];
        var ms = c.t1 - c.t0; if (ms < MINF[src]) return;
        var el = doc.elementFromPoint(c.x - (window.scrollX || 0), c.y - (window.scrollY || 0));
        var r = el ? ctx.regionOf(el) : null, region = r ? ctx.regionInfo(r).region : null;
        var f = { x: Math.round(c.x), y: Math.round(c.y), ms: ms, src: src, region: region, t0: c.t0, t: c.t1 };
        ctx.fixations.push(f); if (ctx.fixations.length > 1500) ctx.fixations.splice(0, 500);
        ctx.bus.emit('fixation', f);
        ctx.record({ kind: 'fixation', src: src, x: f.x, y: f.y, ms: ms, region: region });
        if (region) ctx.focus = { region: region, t: now(), src: src };
      }
      ctx.openFixations = function () { return Object.keys(cl).map(function (k) { return merge({ src: k }, cl[k]); }); };
      var offP = ctx.bus.on('point', function (p) {
        var c = cl[p.src];
        if (c && Math.hypot(p.x - c.x, p.y - c.y) <= R[p.src]) { c.n++; c.x += (p.x - c.x) / c.n; c.y += (p.y - c.y) / c.n; c.t1 = p.t; c.seen = p.t; }
        else { endCluster(p.src); cl[p.src] = { x: p.x, y: p.y, n: 1, t0: p.t, t1: p.t, seen: p.t }; }
      });

      var offE = ctx.bus.on('event', function (ev) {
        if (ev.remote) return;
        if ((ev.kind === 'dwell' || ev.kind === 'gaze') && ev.region) {
          var v = visits[ev.region] || (visits[ev.region] = { n: 0, end: 0 }), start = ev.t - ev.ms;
          if (start - v.end > 4000) {
            v.n++;
            if (v.n >= 2) say('info', 'revisit', 'Returned to ' + ev.region + ' (visit ' + v.n + ')', { region: ev.region, trigger: ev.id });
          }
          v.end = ev.t; ctx.focus = { region: ev.region, t: ev.t, src: ev.kind };
        }
        if (ev.kind === 'view' && ev.region && ev.ms < 700 && ev.reason !== 'checkpoint') {
          var looked = ctx.coverage(ev.region, { who: 'self', when: 'live', signals: ['dwell', 'gaze'] });
          if (!looked && (!skimmed[ev.region] || ev.t - skimmed[ev.region] > 15000)) {
            skimmed[ev.region] = ev.t;
            say('notice', 'skim', 'Scrolled past ' + ev.region + ' in ' + (ev.ms / 1000).toFixed(1) + ' s', { region: ev.region, trigger: ev.id });
          }
        }
        if (ev.kind === 'action' && ev.action === 'accept' && ev.region) {
          ctx.heed.flush(); // close open dwell so time on the claim is counted
          var read = ctx.coverage(ev.region, { who: 'self', when: 'live' });
          if (read < 1500) say('notice', 'quick', 'Accepted ' + ev.region + ' after ' + (read / 1000).toFixed(1) + ' s on the claim itself', { region: ev.region, trigger: ev.id });
        }
        if (ev.kind === 'face' && ev.present) {
          var at = ctx.focus && now() - ctx.focus.t < 3000 ? ' while on ' + ctx.focus.region : '';
          var region = at ? ctx.focus.region : null;
          [['browDown', 0.45, 'frown', 'Frowned'], ['smile', 0.55, 'smile', 'Smiled'], ['browUp', 0.6, 'surprise', 'Raised brows']].forEach(function (x) {
            if (ev[x[0]] > x[1] && lastFace && lastFace[x[0]] > x[1] && (!exprAt[x[2]] || now() - exprAt[x[2]] > 8000)) {
              exprAt[x[2]] = now();
              say('info', x[2], x[3] + at, { region: region, trigger: ev.id });
            }
          });
          lastFace = ev;
        }
      });

      // hesitation over action controls (mouse)
      function endHover(how) {
        if (!hover) return;
        var ms = now() - hover.t, el = hover.el; hover = null;
        if (ms < 1200) return;
        var r = ctx.regionOf(el), reg = r ? ctx.regionInfo(r).region : 'page', label = el.getAttribute('data-heed-action');
        say(how === 'click' ? 'info' : 'notice', 'hesitation', how === 'click'
          ? 'Hovered ' + label + ' on ' + reg + ' for ' + (ms / 1000).toFixed(1) + ' s before clicking'
          : 'Hovered ' + label + ' on ' + reg + ' for ' + (ms / 1000).toFixed(1) + ' s, then moved away', { region: reg });
      }
      function over(e) {
        if (e.pointerType && e.pointerType !== 'mouse') return;
        var a = e.target.closest && e.target.closest('[data-heed-action]');
        if (!a || (hover && hover.el === a)) return;
        endHover('moved'); hover = { el: a, t: now() };
      }
      function out(e) { if (hover && !hover.el.contains(e.relatedTarget)) endHover('moved'); }
      function click(e) { if (hover && hover.el.contains(e.target)) endHover('click'); }

      // idle, leaving the page, scrolling
      function act() {
        if (idleOn) { ctx.record({ kind: 'idle', ms: now() - lastInput }); idleOn = false; }
        lastInput = now();
      }
      function scroll() {
        act();
        var t = now(), y = window.scrollY || 0;
        if (t - scrollAt < 250) return;
        var vy = (y - lastScroll.y) / Math.max(1, (t - lastScroll.t)) * 1000;
        scrollAt = t; lastScroll = { y: y, t: t };
        ctx.record({ kind: 'scroll', y: Math.round(y), vy: Math.round(vy) });
        if (Math.abs(vy) > 2500 && t - fastAt > 3000) { fastAt = t; say('debug', 'fast-scroll', 'Fast scroll, ' + Math.round(Math.abs(vy)) + ' px/s'); }
      }
      function vis() {
        if (doc.hidden) { hiddenAt = now(); return; }
        if (!hiddenAt) return;
        var ms = now() - hiddenAt; hiddenAt = 0; lastInput = now();
        var ev = ctx.record({ kind: 'hidden', ms: ms });
        if (ms > 2000) say('notice', 'left', 'Left the page for ' + (ms / 1000).toFixed(1) + ' s', { trigger: ev.id });
      }
      var tick = setInterval(function () {
        var t = now();
        Object.keys(cl).forEach(function (k) {
          var c = cl[k];
          if (k === 'pointer' && t - c.seen < 8000) c.t1 = t;         // a resting cursor keeps its fixation
          if (k !== 'pointer' && t - c.seen > 400) endCluster(k);       // touch and gaze end when the stream pauses
          if (t - c.t0 > 8000) { endCluster(k); }
        });
        if (!doc.hidden && !idleOn && t - lastInput > 10000) { idleOn = true; say('info', 'idle', 'No input for 10 s'); }
      }, 250);

      doc.addEventListener('pointerover', over, true);
      doc.addEventListener('pointerout', out, true);
      doc.addEventListener('click', click, true);
      ['pointermove', 'keydown', 'touchstart', 'wheel'].forEach(function (n) { doc.addEventListener(n, act, { passive: true, capture: true }); });
      window.addEventListener('scroll', scroll, { passive: true });
      doc.addEventListener('visibilitychange', vis);
      var offF = ctx.bus.on('flush', function () { Object.keys(cl).forEach(endCluster); });
      return function () {
        offP(); offE(); offF(); clearInterval(tick);
        doc.removeEventListener('pointerover', over, true); doc.removeEventListener('pointerout', out, true); doc.removeEventListener('click', click, true);
        ['pointermove', 'keydown', 'touchstart', 'wheel'].forEach(function (n) { doc.removeEventListener(n, act, true); });
        window.removeEventListener('scroll', scroll); doc.removeEventListener('visibilitychange', vis);
      };
    }
  };

  // Trail renderer: a fading path per source plus a numbered scanpath of fixations,
  // drawn on a fixed canvas over the page (spec.render.trail, last spec.trailMs).
  var trail = {
    name: 'trail',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      var cv = doc.createElement('canvas');
      cv.setAttribute('data-heed-ignore', ''); cv.setAttribute('aria-hidden', 'true');
      cv.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:2147482000';
      doc.body.appendChild(cv);
      var g = cv.getContext('2d'), raf = 0, colors = {}, colorsAt = 0;
      function readColors() {
        var cs = getComputedStyle(doc.documentElement);
        var v = function (n, f) { return (cs.getPropertyValue(n) || '').trim() || f; };
        colors = { pointer: v('--heed-tint', '#2563eb'), touch: v('--heed-tint', '#2563eb'), gaze: v('--heed-gaze', '#db2777'),
                   ink: v('--heed-card', '#ffffff'), text: v('--heed-card-fg', '#09090b') };
      }
      function draw() {
        raf = requestAnimationFrame(draw);
        var on = ctx.spec.render && ctx.spec.render.trail && !doc.hidden;
        cv.style.display = on ? 'block' : 'none';
        if (!on) return;
        var dpr = window.devicePixelRatio || 1, W = innerWidth, H = innerHeight;
        if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
        if (now() - colorsAt > 1000) { readColors(); colorsAt = now(); }
        g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
        var t = now(), win = ctx.spec.trailMs || 15000, sx = window.scrollX || 0, sy = window.scrollY || 0;
        var pts = ctx.trail || [], bySrc = {};
        for (var i = pts.length - 1; i >= 0 && t - pts[i].t < win; i--) (bySrc[pts[i].src] = bySrc[pts[i].src] || []).unshift(pts[i]);
        g.lineCap = 'round'; g.lineJoin = 'round';
        Object.keys(bySrc).forEach(function (src) {
          var a = bySrc[src]; g.strokeStyle = colors[src] || colors.pointer; g.lineWidth = src === 'gaze' ? 2 : 1.5;
          for (var j = 1; j < a.length; j++) {
            if (a[j].t - a[j - 1].t > 600) continue;
            g.globalAlpha = 0.55 * (1 - (t - a[j].t) / win);
            g.beginPath(); g.moveTo(a[j - 1].x - sx, a[j - 1].y - sy); g.lineTo(a[j].x - sx, a[j].y - sy); g.stroke();
          }
        });
        // scanpath
        var fx = (ctx.fixations || []).filter(function (f) { return t - f.t < win; });
        var open = ctx.openFixations ? ctx.openFixations().filter(function (c) { return c.t1 - c.t0 > 150; }) : [];
        ['pointer', 'touch', 'gaze'].forEach(function (src) {
          var list = fx.filter(function (f) { return f.src === src; });
          open.filter(function (c) { return c.src === src; }).forEach(function (c) { list.push({ x: c.x, y: c.y, ms: c.t1 - c.t0, t: c.t1, open: true }); });
          if (!list.length) return;
          var col = colors[src] || colors.pointer;
          g.strokeStyle = col; g.lineWidth = 1;
          for (var k = 1; k < list.length; k++) {
            g.globalAlpha = 0.35 * (1 - (t - list[k].t) / win);
            g.beginPath(); g.moveTo(list[k - 1].x - sx, list[k - 1].y - sy); g.lineTo(list[k].x - sx, list[k].y - sy); g.stroke();
          }
          list.forEach(function (f, n) {
            var r = Math.min(34, 6 + Math.sqrt(f.ms) / 2.2), age = 1 - (t - f.t) / win;
            g.globalAlpha = Math.max(0.15, age) * 0.22; g.fillStyle = col;
            g.beginPath(); g.arc(f.x - sx, f.y - sy, r, 0, Math.PI * 2); g.fill();
            g.globalAlpha = Math.max(0.2, age) * (f.open ? 1 : 0.8); g.lineWidth = f.open ? 2 : 1.25;
            g.beginPath(); g.arc(f.x - sx, f.y - sy, r, 0, Math.PI * 2); g.stroke();
            if (list.length - n <= 12 && r >= 9) {
              g.globalAlpha = Math.max(0.35, age); g.fillStyle = col;
              g.font = '600 10px ui-monospace,SFMono-Regular,Menlo,monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
              g.fillText(String(n + 1), f.x - sx, f.y - sy);
            }
          });
        });
        g.globalAlpha = 1;
      }
      draw();
      return function () { cancelAnimationFrame(raf); cv.remove(); };
    }
  };

  // Toasts: notify behaviour as it happens (spec.render.toasts, at or above spec.notify).
  var toasts = {
    name: 'toasts',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      var box = doc.createElement('div');
      box.setAttribute('data-heed-ignore', ''); box.setAttribute('aria-live', 'polite'); box.className = 'heed-toasts';
      var style = doc.createElement('style');
      style.textContent =
        '.heed-toasts{position:fixed;z-index:2147482500;top:max(12px,env(safe-area-inset-top,0px));right:max(12px,env(safe-area-inset-right,0px));' +
        'display:flex;flex-direction:column;gap:6px;width:min(320px,calc(100vw - 24px));pointer-events:none}' +
        '.heed-toast{display:flex;gap:8px;align-items:flex-start;padding:8px 10px;border-radius:10px;background:var(--heed-card,#fff);color:var(--heed-card-fg,#09090b);' +
        'border:1px solid var(--heed-line,#e4e4e7);box-shadow:0 8px 20px -12px rgba(0,0,0,.35);font:12.5px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;' +
        'animation:heed-in .18s ease-out}' +
        '.heed-toast b{flex:none;font:600 10.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;padding:0 6px;border-radius:999px;border:1px solid currentColor}' +
        '.heed-toast[data-level="warn"] b,.heed-toast[data-level="error"] b{color:var(--heed-warn,#d97706)}' +
        '.heed-toast[data-level="notice"] b{color:var(--heed-notice,#ca8a04);border-style:dashed}' +
        '.heed-toast[data-level="info"] b,.heed-toast[data-level="debug"] b{color:var(--heed-tint,#2563eb)}' +
        '@keyframes heed-in{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}' +
        '@media (prefers-reduced-motion: reduce){.heed-toast{animation:none}}';
      doc.head.appendChild(style); doc.body.appendChild(box);
      var recent = {};
      var off = ctx.bus.on('finding', function (f) {
        var s = ctx.spec;
        if (!s.render || !s.render.toasts || rank(f.level) < rank(s.notify || 'notice')) return;
        if (recent[f.text] && now() - recent[f.text] < 3000) return;
        recent[f.text] = now();
        var el = doc.createElement('div'); el.className = 'heed-toast'; el.setAttribute('data-level', f.level);
        var b = doc.createElement('b'); b.textContent = f.level;
        var span = doc.createElement('span'); span.textContent = f.text || f.type;
        el.appendChild(b); el.appendChild(span); box.insertBefore(el, box.firstChild);
        while (box.children.length > 4) box.removeChild(box.lastChild);
        setTimeout(function () { el.remove(); }, 4500);
      });
      return function () { off(); box.remove(); style.remove(); };
    }
  };

  // Timeline: swimlanes of the last N seconds of events and findings, drawn into a canvas.
  function timeline(h, canvas, opts) {
    opts = opts || {};
    var win = opts.windowMs || 60000, LBL = 78, LANE = 22, TOP = 4, AXIS = 18;
    var lanes = opts.lanes || [
      { name: 'Cursor', kinds: ['dwell'] }, { name: 'Gaze', kinds: ['gaze'] }, { name: 'In view', kinds: ['view'], noLabels: true },
      { name: 'Fixations', kinds: ['fixation'] }, { name: 'Typing', kinds: ['typing'] }, { name: 'Scroll', kinds: ['scroll'] },
      { name: 'Actions', kinds: ['action', 'prompt', 'response', 'calibration'] }, { name: 'Presence', kinds: ['face', 'away', 'hidden', 'idle'] },
      { name: 'Findings', findings: true }
    ];
    var g = canvas.getContext('2d'), timer;
    canvas.style.height = (TOP + lanes.length * LANE + AXIS) + 'px';
    function col(n, f) { return (getComputedStyle(canvas).getPropertyValue(n) || '').trim() || f; }
    function draw() {
      var dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 600, H = TOP + lanes.length * LANE + AXIS;
      if (canvas.width !== Math.round(W * dpr)) canvas.width = Math.round(W * dpr);
      if (canvas.height !== Math.round(H * dpr)) canvas.height = Math.round(H * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
      var C = { fg: col('--heed-card-fg', '#09090b'), muted: col('--heed-muted', '#71717a'), line: col('--heed-line', '#e4e4e7'),
                tint: col('--heed-tint', '#2563eb'), warn: col('--heed-warn', '#d97706'), gaze: col('--heed-gaze', '#db2777') };
      var t1 = Date.now(), t0 = t1 - win, X = function (t) { return LBL + (t - t0) / win * (W - LBL - 4); };
      var evs = h.query({ who: 'self', when: { last: win + 5000 }, level: 'trace' }).events();
      var fs = h.query({ who: 'self', when: { last: win }, level: 'trace' }).findings();
      g.font = '11px system-ui,-apple-system,"Segoe UI",sans-serif'; g.textBaseline = 'middle';
      lanes.forEach(function (ln, i) {
        var y = TOP + i * LANE, mid = y + LANE / 2;
        g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText(ln.name, 0, mid);
        g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(LBL, y + LANE - 0.5); g.lineTo(W, y + LANE - 0.5); g.stroke();
        if (ln.findings) {
          fs.forEach(function (f) {
            var x = X(f.t); if (x < LBL) return;
            var hot = rank(f.level) >= rank('notice');
            g.fillStyle = hot ? C.warn : rank(f.level) >= rank('info') ? C.tint : C.muted;
            g.beginPath(); g.arc(x, mid, hot ? 4.5 : 3, 0, Math.PI * 2); g.fill();
          });
          return;
        }
        var blocks = [];
        evs.forEach(function (e) {
          if (ln.kinds.indexOf(e.kind) < 0) return;
          var bad = e.kind === 'away' || e.kind === 'hidden' || e.kind === 'idle' || (e.kind === 'face' && !e.present);
          var color = bad ? C.warn : e.kind === 'gaze' || (e.kind === 'fixation' && e.src === 'gaze') ? C.gaze : C.tint;
          if (e.kind === 'face') { blocks.push({ a: e.t - 500, b: e.t, key: bad ? 'away' : 'here', color: color, alpha: bad ? 0.9 : 0.3, thin: true }); return; }
          if (e.ms) { blocks.push({ a: e.t - e.ms, b: e.t, key: (e.region || '') + '|' + color, region: e.region, color: color, alpha: bad ? 0.8 : 0.5, label: !ln.noLabels }); return; }
          var x = X(e.t); if (x < LBL) return;
          g.fillStyle = color; g.fillRect(x - 1, y + 4, 2, LANE - 9);
          if (e.kind === 'action' && e.action) { g.fillStyle = C.fg; g.textAlign = 'left'; g.font = '10px ui-monospace,SFMono-Regular,Menlo,monospace'; g.fillText(e.action, x + 4, mid - 0.5); g.font = '11px system-ui,-apple-system,"Segoe UI",sans-serif'; }
        });
        // merge consecutive pieces of the same region (checkpoints split long dwells)
        blocks.sort(function (p, q) { return p.a - q.a; });
        var merged = [];
        blocks.forEach(function (bk) {
          var last = merged[merged.length - 1];
          if (last && last.key === bk.key && bk.a - last.b < 450) last.b = Math.max(last.b, bk.b); else merged.push(bk);
        });
        merged.forEach(function (bk) {
          var a = Math.max(LBL, X(bk.a)), b = X(bk.b); if (b < LBL) return;
          g.globalAlpha = ln.noLabels ? 0.22 : bk.alpha; g.fillStyle = bk.color;
          if (bk.thin) g.fillRect(a, mid - 2, Math.max(1, b - a), 4);
          else g.fillRect(a, y + 4, Math.max(2, b - a - 1), LANE - 9);
          g.globalAlpha = 1;
          if (bk.label && bk.region && b - a > 50) {
            g.save(); g.beginPath(); g.rect(a, y, b - a - 2, LANE); g.clip();
            g.fillStyle = C.fg; g.textAlign = 'left'; g.font = '10px ui-monospace,SFMono-Regular,Menlo,monospace'; g.fillText(bk.region, a + 4, mid - 0.5); g.restore();
            g.font = '11px system-ui,-apple-system,"Segoe UI",sans-serif';
          }
        });
      });
      var ay = TOP + lanes.length * LANE + 10; g.fillStyle = C.muted; g.textAlign = 'center';
      for (var s = 0; s <= win / 1000; s += 10) { var x = X(t1 - s * 1000); g.fillText(s === 0 ? 'now' : '-' + s + ' s', Math.min(W - 14, Math.max(LBL + 10, x)), ay); }
    }
    draw(); timer = setInterval(draw, opts.refreshMs || 250);
    return { stop: function () { clearInterval(timer); }, draw: draw };
  }


  // Links renderer: transitions drawn over the page as arrows between the regions'
  // live positions; width and opacity scale with how often attention moved A -> B.
  var links = {
    name: 'links',
    setup: function (ctx) {
      var doc = ctx.root.ownerDocument || document;
      var cv = doc.createElement('canvas');
      cv.setAttribute('data-heed-ignore', ''); cv.setAttribute('aria-hidden', 'true');
      cv.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none;z-index:2147481900';
      doc.body.appendChild(cv);
      var g = cv.getContext('2d'), raf = 0, last = 0;
      function css(n, f) { return (getComputedStyle(doc.documentElement).getPropertyValue(n) || '').trim() || f; }
      function draw() {
        raf = requestAnimationFrame(draw);
        var on = ctx.spec.render && ctx.spec.render.links && !doc.hidden;
        cv.style.display = on ? 'block' : 'none';
        if (!on || now() - last < 90) return;
        last = now();
        var dpr = window.devicePixelRatio || 1, W = innerWidth, H = innerHeight;
        if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
        g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
        var pos = {}, ids = [];
        ctx.regions().forEach(function (el) {
          if (!el.getClientRects().length) return;
          var r = el.getBoundingClientRect(), id = el.getAttribute('data-heed');
          pos[id] = { x: r.left + r.width / 2, y: r.top + Math.min(r.height / 2, 28), r: r }; ids.push(id);
        });
        var tr = ctx.view(ctx.spec).transitions({ regions: ids });
        var keys = Object.keys(tr.counts); if (!keys.length) return;
        var max = Math.max.apply(null, keys.map(function (k) { return tr.counts[k]; }));
        var col = tr.kind === 'gaze' ? css('--heed-gaze', '#db2777') : css('--heed-tint', '#2563eb');
        var ink = css('--heed-card', '#ffffff'), txt = css('--heed-card-fg', '#09090b');
        keys.forEach(function (k) {
          var ab = k.split('>'), A = pos[ab[0]], B = pos[ab[1]]; if (!A || !B) return;
          var c = tr.counts[k], f = c / max, dx = B.x - A.x, dy = B.y - A.y, d = Math.hypot(dx, dy) || 1;
          var nx = -dy / d, ny = dx / d, bend = Math.min(80, d * 0.18);
          var cx = (A.x + B.x) / 2 + nx * bend, cy = (A.y + B.y) / 2 + ny * bend;
          var tx = B.x - cx, ty = B.y - cy, tl = Math.hypot(tx, ty) || 1, ex = B.x - tx / tl * 10, ey = B.y - ty / tl * 10;
          g.globalAlpha = 0.25 + 0.6 * f; g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1 + 4 * Math.sqrt(f); g.lineCap = 'round';
          g.beginPath(); g.moveTo(A.x, A.y); g.quadraticCurveTo(cx, cy, ex, ey); g.stroke();
          var ang = Math.atan2(ty, tx), hl = 7 + 4 * f;
          g.beginPath(); g.moveTo(B.x, B.y); g.lineTo(B.x - hl * Math.cos(ang - 0.45), B.y - hl * Math.sin(ang - 0.45)); g.lineTo(B.x - hl * Math.cos(ang + 0.45), B.y - hl * Math.sin(ang + 0.45)); g.closePath(); g.fill();
          var mx = 0.25 * A.x + 0.5 * cx + 0.25 * B.x, my = 0.25 * A.y + 0.5 * cy + 0.25 * B.y, label = String(c);
          g.globalAlpha = 1; g.font = '600 10px ui-monospace,SFMono-Regular,Menlo,monospace'; g.textAlign = 'center'; g.textBaseline = 'middle';
          var w = g.measureText(label).width + 8;
          g.fillStyle = ink; g.strokeStyle = col; g.lineWidth = 1;
          g.beginPath(); if (g.roundRect) g.roundRect(mx - w / 2, my - 8, w, 16, 8); else g.rect(mx - w / 2, my - 8, w, 16); g.fill(); g.stroke();
          g.fillStyle = txt; g.fillText(label, mx, my + 0.5);
        });
        Object.keys(tr.dwell).forEach(function (id) {
          var P = pos[id]; if (!P) return;
          g.globalAlpha = 0.9; g.fillStyle = col; g.beginPath(); g.arc(P.x, P.y, 3.5, 0, Math.PI * 2); g.fill();
        });
        g.globalAlpha = 1;
      }
      draw();
      return function () { cancelAnimationFrame(raf); cv.remove(); };
    }
  };

  // Matrix view: per-region dwell and a from -> to transition heatmap, rendered as HTML
  // into an element. Limited to regions currently laid out on the page.
  function matrix(h, el, opts) {
    opts = opts || {};
    var doc = el.ownerDocument || document;
    if (!doc.getElementById('heed-mx-style')) {
      var st = doc.createElement('style'); st.id = 'heed-mx-style';
      st.textContent =
        '.heed-mx{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:20px;font:12px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--heed-card-fg,#09090b)}' +
        '@media (max-width:760px){.heed-mx{grid-template-columns:minmax(0,1fr)}}' +
        '.heed-mx h3{margin:0 0 8px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;color:var(--heed-muted,#71717a)}' +
        '.heed-mx table{border-collapse:collapse;font-variant-numeric:tabular-nums}' +
        '.heed-mx .dw td{padding:3px 6px 3px 0;white-space:nowrap}.heed-mx .dw td.bar{width:100%}' +
        '.heed-mx .dw i{display:block;height:8px;border-radius:999px;background:var(--heed-tint,#2563eb);min-width:2px}' +
        '.heed-mx .num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right;color:var(--heed-muted,#71717a)}' +
        '.heed-mx .tm{overflow-x:auto}.heed-mx .tm td,.heed-mx .tm th{width:30px;height:26px;text-align:center;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--heed-line,#e4e4e7)}' +
        '.heed-mx .tm th{font-weight:500;color:var(--heed-muted,#71717a);background:transparent}.heed-mx .tm th.row{text-align:left;width:auto;padding:0 8px;white-space:nowrap;font-family:system-ui,-apple-system,sans-serif}' +
        '.heed-mx .tm td.diag{background:var(--heed-btn,#fafafa);color:var(--heed-line,#d4d4d8)}' +
        '.heed-mx .key{margin:8px 0 0;color:var(--heed-muted,#71717a)}.heed-mx .key b{font-weight:600;color:var(--heed-card-fg,#09090b)}.heed-mx .top{margin:0 0 10px}.heed-mx .top b{font-weight:600}' +
        '.heed-mx .empty{color:var(--heed-muted,#71717a);margin:0}';
      doc.head.appendChild(st);
    }
    var esc = function (x) { return String(x).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
    function draw() {
      var regs = Array.prototype.slice.call(doc.querySelectorAll('[data-heed]')).filter(function (r) { return r.getClientRects().length && !r.closest('[data-heed-ignore]'); });
      var ids = regs.map(function (r) { return r.getAttribute('data-heed'); });
      var label = {}; regs.forEach(function (r) { label[r.getAttribute('data-heed')] = r.getAttribute('data-heed-label') || r.getAttribute('data-heed'); });
      var v = h.query(h.spec), tr = v.transitions({ regions: ids });
      var src = tr.kind === 'gaze' ? 'gaze' : 'cursor dwell';
      var seen = ids.filter(function (id) { return tr.dwell[id]; });
      if (!seen.length) { el.innerHTML = '<p class="heed-mx empty">No ' + src + ' on these regions yet. Move through the layout above' + (tr.kind === 'gaze' ? '' : ' with the cursor, or start the camera and calibrate') + '.</p>'; return; }
      var tint = tr.kind === 'gaze' ? 'var(--heed-gaze,#db2777)' : 'var(--heed-tint,#2563eb)';
      var order = seen.slice().sort(function (a, b) { return (tr.dwell[b] || 0) - (tr.dwell[a] || 0); });
      var never = ids.filter(function (id) { return !tr.dwell[id]; });
      var maxD = Math.max.apply(null, order.map(function (id) { return tr.dwell[id] || 0; }).concat([1]));
      var total = order.reduce(function (a, id) { return a + (tr.dwell[id] || 0); }, 0) || 1;
      var dw = '<table class="dw"><tbody>' + order.map(function (id) {
        var ms = tr.dwell[id] || 0;
        return '<tr><td>' + esc(label[id]) + '</td><td class="bar"><i style="width:' + (ms / maxD * 100).toFixed(1) + '%;background:' + tint + '"></i></td>' +
          '<td class="num">' + (ms / 1000).toFixed(1) + ' s</td><td class="num">' + Math.round(ms / total * 100) + '%</td><td class="num">' + (tr.visits[id] || 0) + '×</td></tr>';
      }).join('') + '</tbody></table>' +
        (never.length ? '<p class="key"><b>Never attended (' + never.length + '):</b> ' + never.map(function (id) { return esc(label[id]); }).join(', ') + '</p>' : '<p class="key">Every region in this layout was attended.</p>');
      var keys = Object.keys(tr.counts), maxC = Math.max.apply(null, keys.map(function (k) { return tr.counts[k]; }).concat([1]));
      var top = keys.sort(function (a, b) { return tr.counts[b] - tr.counts[a]; }).slice(0, 3).map(function (k) {
        var ab = k.split('>'); return esc(label[ab[0]] || ab[0]) + ' → ' + esc(label[ab[1]] || ab[1]) + ' <b>' + tr.counts[k] + '</b>';
      });
      var n = order.length, head = '<tr><th class="row">from ↓ to →</th>' + order.map(function (_, i) { return '<th>' + (i + 1) + '</th>'; }).join('') + '</tr>';
      var rows = order.map(function (a, i) {
        return '<tr><th class="row">' + (i + 1) + ' ' + esc(label[a]) + '</th>' + order.map(function (b) {
          if (a === b) return '<td class="diag">·</td>';
          var c = tr.counts[a + '>' + b] || 0, pct = Math.round(15 + 70 * c / maxC);
          return c ? '<td style="background:color-mix(in srgb,' + tint + ' ' + pct + '%,transparent);color:' + (pct > 55 ? 'var(--heed-card,#fff)' : 'inherit') + '">' + c + '</td>' : '<td></td>';
        }).join('') + '</tr>';
      }).join('');
      el.innerHTML = '<div class="heed-mx"><div><h3>Dwell · ' + src + '</h3>' + dw + '</div><div><h3>Transitions · ' + keys.length + ' kinds, ' +
        keys.reduce(function (a, k) { return a + tr.counts[k]; }, 0) + ' moves</h3>' + (top.length ? '<p class="top">Most frequent: ' + top.join(' · ') + '</p>' : '') +
        '<div class="tm"><table><tbody>' + head + rows + '</tbody></table></div><p class="key">Row is where attention left, column is where it went next.</p></div></div>';
      if (n === 0) el.innerHTML = '';
    }
    draw();
    var timer = setInterval(draw, opts.refreshMs || 700);
    return { stop: function () { clearInterval(timer); }, draw: draw };
  }

  function defaultPlugins() {
    return [points, pointer, viewport, keys, actions, behaviors, unverified, overlay, links, trail, toasts, ask, broadcast, store];
  }

  // JSON-safe export (drops DOM references).
  function serialize(data) { return JSON.stringify(data, function (k, v) { return k === 'el' ? undefined : v; }, 2); }

  return {
    version: '0.4.0',
    create: create,
    normalize: normalize,
    serialize: serialize,
    LEVELS: LEVELS,
    purposes: PURPOSES,
    timeline: timeline,
    matrix: matrix,
    plugins: { links: links, points: points, behaviors: behaviors, trail: trail, toasts: toasts, pointer: pointer, viewport: viewport, keys: keys, actions: actions,
               unverified: unverified, overlay: overlay, ask: ask, broadcast: broadcast, store: store }
  };
});
