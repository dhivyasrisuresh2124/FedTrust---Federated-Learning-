'use strict';
let META, ONLINE = [];
const KINDS = {
  modes: { hint: 'Runs local-only, uniform averaging, trust + fixed λ, and the proposed method.', vals: null, modes: ['local_only', 'uniform_fd', 'trust_fixed', 'trust_adaptive'] },
  privacy: { hint: 'Runs the proposed method at several privacy levels. Smaller ε means more noise.', label: 'ε values', def: '1 5 10 20 50 off', key: 'dp.epsilon', modes: ['trust_adaptive'] },
  noniid: { hint: 'Compares local-only, uniform averaging and the proposed method as data becomes more skewed.', label: 'Dirichlet α values', def: '0.1 0.3 1 10', key: 'alpha', modes: ['local_only', 'uniform_fd', 'trust_adaptive'] },
};
const parseVals = () => $('#vals').value.trim().split(/\s+/).filter(Boolean).map((v) => (v === 'off' ? 'off' : parseFloat(v))).filter((v) => v === 'off' || !Number.isNaN(v));

function updateForm() {
  const k = KINDS[$('#kind').value];
  $('#kind-hint').textContent = k.hint;
  $('#vals-box').style.display = k.key ? '' : 'none';
  if (k.key) { $('#vals-l').textContent = k.label; $('#vals').value = k.def; }
  const n = (k.key ? parseVals().length : 1) * k.modes.length * (parseInt($('#seeds').value, 10) || 1);
  $('#runs').textContent = `This queues ${n} runs. Campuses are shared, so they run one after another.`;
}

async function loadList() {
  const st = await api('/api/studies'); const el = $('#list');
  if (!st.length) { el.innerHTML = '<div class="empty"><h3>No studies yet</h3><p>Start one on the right to compare methods across seeds.</p></div>'; return; }
  el.innerHTML = `<table><thead><tr><th>Name</th><th>Kind</th><th>Status</th><th class="r">Progress</th><th>Created</th></tr></thead><tbody>` +
    st.map((s) => `<tr class="click" onclick="location.href='/studies/${esc(s.id)}'"><td><a href="/studies/${esc(s.id)}">${esc(s.name)}</a></td><td>${esc(s.kind)}</td><td>${badge(s.status)}</td><td class="r">${s.done}/${s.runs}</td><td>${fmt.ago(s.created_at)}</td></tr>`).join('') + '</tbody></table>';
}

async function start() {
  $('#err').innerHTML = '';
  const kind = $('#kind').value, k = KINDS[kind];
  if (ONLINE.length < 2) { $('#err').innerHTML = '<div class="callout err" style="margin-bottom:12px">Two online campuses are needed.</div>'; return; }
  const seeds = Array.from({ length: parseInt($('#seeds').value, 10) || 1 }, (_, i) => 42 + i);
  const spec = {
    name: $('#sname').value.trim() || `${kind} study`, kind, seeds, modes: k.modes,
    base: { n_rounds: parseInt($('#rounds').value, 10) || 8,
      campuses: ONLINE.map((n, i) => ({ node_id: n.id, model: META.models[i % 4], label_noise: i === ONLINE.length - 1 && ONLINE.length >= 3 ? 0.35 : 0 })) },
  };
  if (k.key) spec.vary = { [k.key]: parseVals() };
  try { const r = await api('/api/studies', { method: 'POST', body: spec }); location.href = `/studies/${r.id}`; }
  catch (e) { $('#err').innerHTML = `<div class="callout err" style="margin-bottom:12px" role="alert">${esc(e.message)}</div>`; }
}

(async function () {
  META = await api('/api/meta'); ONLINE = (await api('/api/nodes')).filter((n) => n.online);
  ['kind', 'vals', 'seeds'].forEach((id) => $('#' + id).addEventListener('input', updateForm));
  $('#kind').addEventListener('change', updateForm); updateForm();
  $('#go').addEventListener('click', start);
  poll(loadList, 4000);
})();
