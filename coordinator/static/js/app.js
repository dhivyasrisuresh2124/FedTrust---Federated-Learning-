/* Shared helpers for every page. No framework: plain DOM + Chart.js. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts = {}) {
  const init = { headers: { 'Content-Type': 'application/json' }, ...opts };
  if (opts.body && typeof opts.body !== 'string') init.body = JSON.stringify(opts.body);
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function toast(msg, ms = 3500) {
  const t = document.createElement('div');
  t.className = 'toast'; t.setAttribute('role', 'status'); t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

/* ---- formatting ---- */
const fmt = {
  pct: (x, d = 1) => (x == null || Number.isNaN(x) ? '–' : (x * 100).toFixed(d) + '%'),
  num: (x, d = 3) => (x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(d)),
  int: (x) => (x == null ? '–' : Number(x).toLocaleString()),
  bytes: (n) => {
    if (n == null) return '–';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
  },
  dur: (s) => {
    if (s == null) return '–';
    if (s < 60) return s.toFixed(0) + ' s';
    return Math.floor(s / 60) + ' min ' + String(Math.round(s % 60)).padStart(2, '0') + ' s';
  },
  eps: (e) => (e == null ? 'no DP' : e >= 100 ? e.toFixed(0) : e.toFixed(1)),
  ago: (ts) => {
    if (!ts) return '–';
    const s = Date.now() / 1000 - ts;
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ago';
    return new Date(ts * 1000).toLocaleDateString();
  },
};

const MODE_LABEL = {
  trust_adaptive: 'Trust-weighted + adaptive (proposed)',
  trust_fixed: 'Trust-weighted + fixed λ',
  uniform_fd: 'Uniform averaging (FedMD-style)',
  local_only: 'Local only',
};
const MODE_SHORT = { trust_adaptive: 'Proposed', trust_fixed: 'Trust + fixed λ', uniform_fd: 'Uniform FD', local_only: 'Local only' };

const badge = (status) => `<span class="badge ${esc(status)}">${esc(status)}</span>`;

/* ---- colours (read from CSS so dark mode just works) ---- */
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = (i) => css(`--s${(i % 8) + 1}`);
const MODE_COLOR = () => ({ trust_adaptive: css('--teal'), trust_fixed: css('--s6'), uniform_fd: css('--s2'), local_only: css('--faint') });

/* ---- charts ---- */
const charts = new Map();
function drawChart(canvasId, cfg) {
  const el = document.getElementById(canvasId);
  if (!el) return null;
  const old = charts.get(canvasId);
  if (old) old.destroy();
  Chart.defaults.font.family = css('--sans');
  Chart.defaults.font.size = 13;
  Chart.defaults.color = css('--muted');
  Chart.defaults.borderColor = css('--line');
  const ch = new Chart(el, cfg);
  charts.set(canvasId, ch);
  return ch;
}

function lineChart(canvasId, { labels, datasets, yTitle, xTitle, yMin, yMax, pct = false, step = false, stacked = false, legend = false }) {
  const ds = datasets.map((d) => ({
    borderWidth: 2, pointRadius: labels.length > 30 ? 0 : 3, pointHoverRadius: 5, tension: 0, fill: stacked,
    stepped: step, spanGaps: true, ...d,
  }));
  return drawChart(canvasId, {
    type: 'line',
    data: { labels, datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: legend, position: 'bottom', labels: { boxWidth: 12, boxHeight: 12, usePointStyle: false } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct ? fmt.pct(c.parsed.y) : fmt.num(c.parsed.y, 3)}` } },
      },
      scales: {
        x: { title: { display: !!xTitle, text: xTitle }, grid: { display: false } },
        y: { title: { display: !!yTitle, text: yTitle }, min: yMin, max: yMax, stacked,
          ticks: { callback: (v) => (pct ? Math.round(v * 100) + '%' : v) } },
      },
    },
  });
}

function barChart(canvasId, { labels, datasets, yTitle, pct = true, yMin, yMax, legend = true, errorBars }) {
  return drawChart(canvasId, {
    type: 'bar',
    data: { labels, datasets: datasets.map((d) => ({ borderRadius: 3, ...d })) },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
      plugins: {
        legend: { display: legend, position: 'bottom', labels: { boxWidth: 12, boxHeight: 12 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct ? fmt.pct(c.parsed.y) : fmt.num(c.parsed.y, 3)}` } },
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: !!yTitle, text: yTitle }, min: yMin, max: yMax, ticks: { callback: (v) => (pct ? Math.round(v * 100) + '%' : v) } },
      },
    },
  });
}

/* ---- polling that pauses when the tab is hidden ---- */
function poll(fn, ms, { immediate = true } = {}) {
  let timer = null; let stopped = false;
  const tick = async () => {
    if (stopped) return;
    if (!document.hidden) { try { await fn(); } catch (e) { console.warn(e); } }
    if (!stopped) timer = setTimeout(tick, ms);
  };
  if (immediate) tick(); else timer = setTimeout(tick, ms);
  return () => { stopped = true; clearTimeout(timer); };
}

/* ---- header: what the scheduler is doing right now ---- */
async function refreshRunner() {
  const dot = $('#runner-dot'); const txt = $('#runner-text');
  if (!dot) return;
  try {
    const m = await api('/api/meta');
    if (m.running) {
      dot.className = 'dot busy';
      txt.textContent = `Running · ${m.stage || 'starting'}` + (m.queued ? ` · ${m.queued} queued` : '');
    } else {
      dot.className = 'dot ok';
      txt.textContent = m.queued ? `${m.queued} queued` : 'Idle';
    }
  } catch (_) { dot.className = 'dot bad'; txt.textContent = 'Coordinator unreachable'; }
}
document.addEventListener('DOMContentLoaded', () => poll(refreshRunner, 3000));

/* ---- confirm helper for destructive actions ---- */
const confirmAction = (msg) => window.confirm(msg);
