'use strict';
let META = null, NODES = [];
const NOISE_DEFAULT_LAST = 0.35;

const PRESETS = [
  { title: 'Proposed method', sub: 'Trust-weighted teacher, adaptive distillation, DP on', cfg: { mode: 'trust_adaptive' } },
  { title: 'Uniform-average baseline', sub: 'Every campus counts equally (FedMD-style)', cfg: { mode: 'uniform_fd' } },
  { title: 'No collaboration', sub: 'Each campus trains alone', cfg: { mode: 'local_only' } },
  { title: 'Strong privacy', sub: 'ε = 5 per release', cfg: { mode: 'trust_adaptive', dp: { enabled: true, epsilon: 5 } } },
  { title: 'Highly non-IID', sub: 'α = 0.1: campuses see few classes', cfg: { mode: 'trust_adaptive', alpha: 0.1 } },
];

const FIELDS = ['n_rounds', 'warmup_epochs', 'local_epochs', 'distill_epochs', 'batch_size', 'lr', 'anchor_per_round', 'seed'];
const alphaFromSlider = (v) => Math.pow(10, parseFloat(v));
const fmtAlpha = (a) => (a < 1 ? a.toFixed(2) : a < 10 ? a.toFixed(1) : Math.round(a));

function fill(cfg) {
  FIELDS.forEach((k) => { if (cfg[k] != null) $('#' + k).value = cfg[k]; });
  $('#mode').value = cfg.mode; $('#dataset').value = cfg.dataset;
  $('#alpha').value = Math.log10(cfg.alpha); $('#alpha-v').textContent = fmtAlpha(cfg.alpha);
  $('#dp-enabled').checked = cfg.dp.enabled; $('#dp-epsilon').value = cfg.dp.epsilon; $('#dp-delta').value = cfg.dp.delta;
  ['temperature', 'lambda_max', 'lambda_floor', 'lambda_fixed'].forEach((k) => ($('#d-' + k).value = cfg.distill[k]));
  ['tau', 'tau_sample', 'ema'].forEach((k) => ($('#t-' + k).value = cfg.trust[k]));
  $('#name').value = cfg.name === 'Untitled experiment' ? '' : cfg.name;
  $('#dp-box').style.opacity = cfg.dp.enabled ? 1 : .45;
  $('#mode-hint').textContent = META.modes[cfg.mode];
}

function renderCampuses() {
  const el = $('#campuses');
  if (!NODES.length) { el.innerHTML = `<div class="empty"><h3>No campuses registered</h3><p>Add at least two on the Campuses page.</p><a class="btn primary" href="/nodes">Go to Campuses</a></div>`; return; }
  const online = NODES.filter((n) => n.online);
  el.innerHTML = NODES.map((n, i) => {
    const oi = online.indexOf(n);
    const noise = oi >= 0 && oi === online.length - 1 && online.length >= 3 ? NOISE_DEFAULT_LAST : 0;
    return `<div class="campus-card${n.online ? '' : ' off'}" data-id="${esc(n.id)}">
      <header><label class="switch"><input type="checkbox" class="c-on" ${n.online ? 'checked' : 'disabled'}> <span>${esc(n.name)}</span></label>
      <span class="badge ${n.online ? 'online' : 'offline'}">${n.online ? 'online' : 'offline'}</span></header>
      <div class="row">
        <div><label class="f">Model architecture</label><select class="c-model">${META.models.map((m) => `<option ${m === META.models[(oi < 0 ? i : oi) % 4] ? 'selected' : ''}>${m}</option>`).join('')}</select></div>
        <div><label class="f">Label noise</label><input type="number" class="c-noise" min="0" max="0.9" step="0.05" value="${noise}"><div class="hint">Share of wrong labels: lowers data quality.</div></div>
        <div><label class="f">Data fraction</label><input type="number" class="c-frac" min="0.1" max="1" step="0.1" value="1"><div class="hint">Smaller means a smaller campus.</div></div>
      </div></div>`;
  }).join('');
  $$('.c-on').forEach((c) => c.addEventListener('change', countSel)); countSel();
}
const countSel = () => ($('#n-sel').textContent = `${$$('.c-on:checked').length} selected`);

function collect() {
  const num = (id) => parseFloat($('#' + id).value);
  const cfg = {
    name: $('#name').value.trim() || `${MODE_SHORT[$('#mode').value]} · ${new Date().toLocaleTimeString()}`,
    mode: $('#mode').value, dataset: $('#dataset').value, alpha: alphaFromSlider($('#alpha').value),
    dp: { enabled: $('#dp-enabled').checked, epsilon: num('dp-epsilon'), delta: num('dp-delta'), sigma: null },
    distill: { temperature: num('d-temperature'), lambda_max: num('d-lambda_max'), lambda_floor: num('d-lambda_floor'), lambda_fixed: num('d-lambda_fixed') },
    trust: { tau: num('t-tau'), tau_sample: num('t-tau_sample'), ema: num('t-ema') },
    campuses: $$('.campus-card').filter((c) => $('.c-on', c).checked).map((c) => ({
      node_id: c.dataset.id, model: $('.c-model', c).value, label_noise: num2($('.c-noise', c)), data_fraction: num2($('.c-frac', c)) })),
  };
  FIELDS.forEach((k) => (cfg[k] = k === 'lr' ? num(k) : parseInt($('#' + k).value, 10)));
  return cfg;
}
const num2 = (el) => parseFloat(el.value);

async function init() {
  [META, NODES] = await Promise.all([api('/api/meta'), api('/api/nodes')]);
  $('#mode').innerHTML = Object.entries(META.modes).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
  $('#dataset').innerHTML = META.datasets.map((d) => `<option>${d}</option>`).join('');
  fill(META.defaults); renderCampuses();
  $('#presets').innerHTML = PRESETS.map((p, i) => `<button type="button" class="preset" data-i="${i}"><b>${esc(p.title)}</b><span>${esc(p.sub)}</span></button>`).join('');
  $$('.preset').forEach((b) => b.addEventListener('click', () => {
    const p = PRESETS[b.dataset.i].cfg; const base = JSON.parse(JSON.stringify(META.defaults));
    base.mode = p.mode; if (p.alpha) base.alpha = p.alpha; if (p.dp) Object.assign(base.dp, p.dp);
    fill(base); toast(`Loaded preset: ${PRESETS[b.dataset.i].title}`);
  }));
  $('#alpha').addEventListener('input', () => ($('#alpha-v').textContent = fmtAlpha(alphaFromSlider($('#alpha').value))));
  $('#mode').addEventListener('change', () => ($('#mode-hint').textContent = META.modes[$('#mode').value]));
  $('#dp-enabled').addEventListener('change', () => ($('#dp-box').style.opacity = $('#dp-enabled').checked ? 1 : .45));
  $('#go').addEventListener('click', async () => {
    $('#err').innerHTML = '';
    try {
      const r = await api('/api/experiments', { method: 'POST', body: collect() });
      location.href = `/experiments/${r.id}`;
    } catch (e) { $('#err').innerHTML = `<div class="callout err" style="margin-bottom:12px" role="alert">${esc(e.message)}</div>`; }
  });
}
init();
