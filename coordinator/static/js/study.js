'use strict';
let stop = null;
function render(s) {
  document.title = `${s.name} · FedTrust`;
  $('#title').textContent = s.name;
  const exps = s.experiments, done = exps.filter((e) => ['completed', 'failed', 'cancelled'].includes(e.status)).length;
  const active = s.status === 'running' || s.status === 'queued';
  $('#sub').textContent = `${done} of ${exps.length} runs finished`;
  $('#actions').innerHTML = `<a class="btn" href="/api/studies/${SID}/export.csv">Export CSV</a>` + (active ? '<button class="btn danger" id="cancel">Cancel study</button>' : '<button class="btn danger" id="del">Delete</button>');
  if ($('#cancel')) $('#cancel').onclick = async () => { await api(`/api/studies/${SID}/cancel`, { method: 'POST' }); toast('Cancelling'); };
  if ($('#del')) $('#del').onclick = async () => { if (confirmAction('Delete this study and all its runs?')) { await api(`/api/studies/${SID}`, { method: 'DELETE' }); location.href = '/studies'; } };
  const failed = exps.filter((e) => e.status === 'failed');
  $('#state').innerHTML = `<div style="margin-bottom:20px"><div class="progress"><i style="width:${(done / exps.length) * 100}%"></i></div></div>` +
    (failed.length ? `<div class="callout err" style="margin-bottom:20px"><b>${failed.length} run(s) failed.</b> ${esc(failed[0].error || '')}</div>` : '');

  const rows = s.aggregate.rows; if (!rows.length) { $('#tbl').innerHTML = '<p class="muted">Results appear as runs complete.</p>'; }
  else {
    const varied = s.config.vary && Object.keys(s.config.vary)[0];
    const xs = [...new Set(rows.map((r) => r.vary_value))], modes = [...new Set(rows.map((r) => r.mode))];
    const col = MODE_COLOR();
    if (!varied) {
      $('#chart-h').textContent = 'Final accuracy by method';
      barChart('c-main', { labels: modes.map((m) => MODE_SHORT[m]), pct: true, legend: false, yMin: 0,
        datasets: [{ label: 'Mean accuracy', data: modes.map((m) => rows.find((r) => r.mode === m).acc_mean), backgroundColor: modes.map((m) => col[m]) },
                   { label: 'Weakest campus', data: modes.map((m) => rows.find((r) => r.mode === m).min_acc_mean), backgroundColor: modes.map((m) => col[m] + '66') }] });
      drawLegendless();
    } else {
      $('#chart-h').textContent = varied === 'dp.epsilon' ? 'Accuracy vs privacy level (ε per release)' : 'Accuracy vs non-IID severity (Dirichlet α)';
      lineChart('c-main', { labels: xs.map((x) => (x === 'off' ? 'no DP' : x)), pct: true, legend: true, xTitle: varied === 'dp.epsilon' ? 'ε (smaller = more private)' : 'α (smaller = more skewed)',
        datasets: modes.map((m) => ({ label: MODE_SHORT[m], borderColor: col[m], backgroundColor: col[m], data: xs.map((x) => { const r = rows.find((q) => q.mode === m && q.vary_value === x); return r ? r.acc_mean : null; }) })) });
    }
    $('#tbl').innerHTML = `<table><thead><tr><th>${varied ? esc(varied) : 'Setting'}</th><th>Method</th><th class="r">Runs</th><th class="r">Accuracy (mean ± sd)</th><th class="r">Weakest campus</th><th class="r">Rounds to 95%</th><th class="r">Teacher acc.</th><th class="r">Data sent</th><th class="r">ε spent</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${esc(r.vary_value === 'null' || r.vary_value === 'None' ? '–' : r.vary_value)}</td><td><span class="swatch" style="background:${col[r.mode]}"></span>${esc(MODE_SHORT[r.mode])}</td><td class="r">${r.runs}</td><td class="r"><b>${fmt.pct(r.acc_mean)}</b> ± ${(r.acc_std * 100).toFixed(1)}</td><td class="r">${fmt.pct(r.min_acc_mean)}</td><td class="r">${r.convergence_round.toFixed(1)}</td><td class="r">${r.teacher_acc != null ? fmt.pct(r.teacher_acc) : '–'}</td><td class="r">${fmt.bytes(r.comm_bytes)}</td><td class="r">${fmt.eps(r.epsilon_total)}</td></tr>`).join('') + '</tbody></table>';
  }
  $('#runs').innerHTML = `<table><thead><tr><th>Run</th><th>Status</th><th class="r">Accuracy</th></tr></thead><tbody>` + exps.map((e) => `<tr class="click" onclick="location.href='/experiments/${esc(e.id)}'"><td><a href="/experiments/${esc(e.id)}">${esc(e.name)}</a></td><td>${badge(e.status)}</td><td class="r">${e.summary ? fmt.pct(e.summary.final_mean_acc) : '–'}</td></tr>`).join('') + '</tbody></table>';
  return active;
}
function drawLegendless() {}
async function load() { const s = await api(`/api/studies/${SID}`); if (!render(s) && stop) { stop(); stop = null; } }
stop = poll(load, 3000);
