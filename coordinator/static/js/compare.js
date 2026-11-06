'use strict';
(async function () {
  const ids = new URLSearchParams(location.search).get('ids') || '';
  const list = await api('/api/compare?ids=' + encodeURIComponent(ids));
  if (list.length < 1) { $('#tbl').innerHTML = '<div class="empty"><h3>Nothing to compare</h3><p>Select two or more completed experiments on the Experiments page.</p></div>'; return; }
  $('#sub').textContent = `${list.length} experiments`;
  const n = Math.max(...list.map((e) => e.rounds.length));
  const labels = Array.from({ length: n }, (_, i) => i);
  const ds = (key) => list.map((e, i) => ({ label: e.name, data: e.rounds.map((r) => r[key]), borderColor: seriesColor(i), backgroundColor: seriesColor(i) }));
  lineChart('c-mean', { labels, xTitle: 'Round', pct: true, legend: true, datasets: ds('mean_acc') });
  lineChart('c-min', { labels, xTitle: 'Round', pct: true, legend: true, datasets: ds('min_acc') });
  $('#tbl').innerHTML = `<table><thead><tr><th>Experiment</th><th>Method</th><th class="r">α</th><th class="r">ε/release</th><th class="r">Final accuracy</th><th class="r">Weakest</th><th class="r">Rounds to 95%</th><th class="r">Data sent</th></tr></thead><tbody>` +
    list.map((e, i) => { const s = e.summary || {}, c = e.config; return `<tr><td><span class="swatch" style="background:${seriesColor(i)}"></span><a href="/experiments/${esc(e.id)}">${esc(e.name)}</a></td><td>${esc(MODE_SHORT[c.mode])}</td><td class="r">${c.alpha}</td><td class="r">${c.dp.enabled ? fmt.eps(c.dp.epsilon) : 'no DP'}</td><td class="r"><b>${fmt.pct(s.final_mean_acc)}</b></td><td class="r">${fmt.pct(s.final_min_acc)}</td><td class="r">${s.convergence_round ?? '–'}</td><td class="r">${fmt.bytes(s.total_comm_bytes)}</td></tr>`; }).join('') + '</tbody></table>';
})();
