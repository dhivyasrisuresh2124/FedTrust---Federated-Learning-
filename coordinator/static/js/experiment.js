'use strict';
let stopPoll = null;

function kpi(v, l, d = '') { return `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div>${d ? `<div class="d">${d}</div>` : ''}</div>`; }

function render(e) {
  const cfg = e.config, rounds = e.rounds, sum = e.summary;
  document.title = `${e.name} · FedTrust`;
  $('#title').textContent = e.name;
  $('#subtitle').innerHTML = `${esc(MODE_LABEL[cfg.mode])} · ${cfg.campuses.length} campuses · α = ${cfg.alpha} · seed ${cfg.seed} · ${cfg.dp.enabled ? 'ε = ' + (cfg.dp.sigma != null ? 'σ ' + cfg.dp.sigma : cfg.dp.epsilon) + ' per release' : 'no DP'}`;
  const active = e.status === 'running' || e.status === 'queued';
  $('#actions').innerHTML = `<a class="btn" href="/api/experiments/${EID}/export.csv">Export CSV</a>` +
    (active ? `<button class="btn danger" id="cancel">Cancel run</button>` : `<button class="btn danger" id="del">Delete</button>`);
  if ($('#cancel')) $('#cancel').onclick = async () => { await api(`/api/experiments/${EID}/cancel`, { method: 'POST' }); toast('Cancelling after the current step'); };
  if ($('#del')) $('#del').onclick = async () => { if (confirmAction('Delete this experiment and its results?')) { await api(`/api/experiments/${EID}`, { method: 'DELETE' }); location.href = '/experiments'; } };

  const done = Math.max(0, rounds.length - 1);
  let st = `<div style="margin-bottom:20px"><div style="display:flex;justify-content:space-between;margin-bottom:6px"><span>${badge(e.status)} <span class="muted">${esc(e.stage || '')}</span></span><span class="muted num">round ${done} of ${e.total_rounds}</span></div>
    <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="${e.total_rounds}" aria-valuenow="${done}"><i style="width:${(done / e.total_rounds) * 100}%"></i></div></div>`;
  if (e.status === 'failed') st += `<div class="callout err" role="alert" style="margin-bottom:20px"><b>Run failed.</b> ${esc(e.error)}</div>`;
  const errs = rounds.flatMap((r) => (r.errors || []).map((m) => `Round ${r.round}: ${m}`));
  if (errs.length) st += `<div class="callout warn" style="margin-bottom:20px"><b>${errs.length} campus error(s)</b> (the round continued without the affected campus)<br>${errs.slice(0, 4).map(esc).join('<br>')}</div>`;
  $('#state').innerHTML = st;

  if (!rounds.length) { $('#kpis').innerHTML = ''; return; }
  const last = rounds[rounds.length - 1], first = rounds[0];
  const gain = last.mean_acc - first.mean_acc;
  const t = last.teacher;
  const tot = rounds.slice(1).reduce((a, r) => a + r.comm.bytes, 0);
  const fed = (last.comm.fedavg_equiv_bytes || 0) * (rounds.length - 1);
  $('#kpis').innerHTML =
    kpi(fmt.pct(last.mean_acc), 'Mean accuracy', `<span class="${gain >= 0 ? 'pos' : 'neg'}">${gain >= 0 ? '+' : ''}${(gain * 100).toFixed(1)} pts since warm-up</span>`) +
    kpi(fmt.pct(last.min_acc), 'Weakest campus', `spread ±${fmt.pct(last.std_acc)}`) +
    kpi(t ? fmt.pct(t.acc) : '–', 'Teacher accuracy', t ? `uniform average: ${fmt.pct(t.uniform_acc)}` : 'n/a for this method') +
    kpi(last.privacy.sigma ? 'σ ' + last.privacy.sigma.toFixed(2) : 'none', 'Noise added', last.privacy.epsilon_total != null ? `ε spent ≈ ${fmt.eps(last.privacy.epsilon_total)} (${esc(last.privacy.method)} composition)` : 'privacy off');
  $('#kpis').innerHTML += kpi(fmt.bytes(tot), 'Data sent in total', fed ? `weight sharing would send ~${fmt.bytes(fed)}` : '') +
    kpi(sum ? 'round ' + sum.convergence_round : '–', 'Reached 95% of final gain');

  const labels = rounds.map((r) => r.round);
  const names = last.campuses.map((c) => c.name);
  const per = (key) => names.map((n, i) => ({ label: n, data: rounds.map((r) => r.campuses[i][key]), borderColor: seriesColor(i), backgroundColor: seriesColor(i) }));
  lineChart('c-acc', { labels, xTitle: 'Round', pct: true, yMin: undefined, legend: true,
    datasets: [...per('acc'), { label: 'Mean', data: rounds.map((r) => r.mean_acc), borderColor: css('--ink'), borderDash: [6, 4], borderWidth: 2.5, pointRadius: 0 }] });
  const collab = cfg.mode !== 'local_only';
  const rr = rounds.slice(1);
  if (collab) {
    lineChart('c-trust', { labels: rr.map((r) => r.round), xTitle: 'Round', pct: true, yMin: 0, legend: true, datasets: per('trust').map((d) => ({ ...d, data: d.data.slice(1) })) });
    lineChart('c-teacher', { labels: rr.map((r) => r.round), xTitle: 'Round', pct: true, legend: true, datasets: [
      { label: 'Trust-weighted teacher', data: rr.map((r) => r.teacher && r.teacher.acc), borderColor: css('--teal'), backgroundColor: css('--teal') },
      { label: 'Uniform average', data: rr.map((r) => r.teacher && r.teacher.uniform_acc), borderColor: css('--s2'), backgroundColor: css('--s2'), borderDash: [5, 4] }] });
    lineChart('c-lambda', { labels: rr.map((r) => r.round), xTitle: 'Round', yMin: 0, legend: true, datasets: per('lambda').map((d) => ({ ...d, data: d.data.slice(1) })) });
    lineChart('c-entropy', { labels: rr.map((r) => r.round), xTitle: 'Round', yMin: 0, yMax: 1, legend: true, datasets: per('entropy').map((d) => ({ ...d, data: d.data.slice(1) })) });
  } else {
    ['c-trust', 'c-teacher', 'c-lambda', 'c-entropy'].forEach((id) => { const p = $('#' + id).closest('.panel-b'); p.innerHTML = '<div class="empty"><p>Not used by the local-only baseline: campuses share nothing.</p></div>'; });
  }
  barChart('c-comm', { labels: rr.map((r) => r.round), pct: false, legend: false, yTitle: 'Bytes',
    datasets: [{ label: 'Bytes exchanged', data: rr.map((r) => r.comm.bytes), backgroundColor: css('--blue') }] });
  $('#comm-hint').textContent = 'Only predictions on the anchor subset are exchanged, so cost stays flat and independent of model size.';

  $('#final').innerHTML = `<table><thead><tr><th>Campus</th><th>Model</th><th class="r">Parameters</th><th class="r">Private samples</th><th class="r">Accuracy</th><th class="r">Avg trust</th><th class="r">Avg λ</th><th class="r">Released-prediction accuracy</th></tr></thead><tbody>` +
    last.campuses.map((c, i) => { const m = (k) => { const v = rounds.map((r) => r.campuses[i][k]).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
      return `<tr><td><span class="swatch" style="background:${seriesColor(i)}"></span>${esc(c.name)}</td><td>${esc(c.model)}</td><td class="r">${fmt.int(c.n_params)}</td><td class="r">${c.n_private}</td>
      <td class="r"><b>${fmt.pct(c.acc)}</b></td><td class="r">${collab ? fmt.pct(m('trust')) : '–'}</td><td class="r">${collab ? fmt.num(m('lambda'), 2) : '–'}</td><td class="r">${collab ? fmt.pct(m('released_acc')) : '–'}</td></tr>`; }).join('') + '</tbody></table>' +
    (sum && sum.trust_vs_released_acc_corr != null ? `<p class="hint" style="margin-top:10px">Correlation between a campus's trust and the accuracy of what it released: <b>${sum.trust_vs_released_acc_corr.toFixed(2)}</b> (1 means trust tracks true quality).</p>` : '');

  const pcs = last.campuses.map((c) => c.per_class);
  if (pcs[0]) {
    const nC = pcs[0].length;
    $('#perclass').innerHTML = `<table><thead><tr><th>Campus</th>${Array.from({ length: nC }, (_, i) => `<th class="r">${i}</th>`).join('')}</tr></thead><tbody>` +
      last.campuses.map((c) => `<tr><td>${esc(c.name)}</td>${c.per_class.map((v) => `<td class="r" style="background:color-mix(in srgb, var(--teal) ${v == null ? 0 : Math.round(v * 55)}%, transparent)">${v == null ? '–' : Math.round(v * 100)}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
  } else $('#perclass').innerHTML = '<p class="muted">Available when the run completes.</p>';
  $('#cfg').textContent = JSON.stringify(cfg, null, 2);
}

async function load() {
  const e = await api(`/api/experiments/${EID}`);
  render(e);
  if (!['running', 'queued'].includes(e.status) && stopPoll) { stopPoll(); stopPoll = null; }
}
stopPoll = poll(load, 2000);
