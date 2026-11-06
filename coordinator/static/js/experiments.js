'use strict';
const picked = new Set();
async function load() {
  const exps = await api('/api/experiments?limit=300');
  const el = $('#list');
  if (!exps.length) { el.innerHTML = `<div class="empty"><h3>No experiments yet</h3><p>Queue your first run to see trust scores and accuracy by round.</p><a class="btn primary" href="/experiments/new">New experiment</a></div>`; return; }
  el.innerHTML = `<table><thead><tr><th><span class="sr-only">Select</span></th><th>Name</th><th>Method</th><th>Status</th><th class="r">Accuracy</th><th class="r">Weakest</th><th class="r">ε spent</th><th class="r">Sent</th><th>Created</th></tr></thead><tbody>` +
    exps.map((e) => { const s = e.summary; return `<tr><td><input type="checkbox" data-id="${esc(e.id)}" aria-label="Select ${esc(e.name)}" ${picked.has(e.id) ? 'checked' : ''} ${e.status === 'completed' ? '' : 'disabled'}></td>
    <td><a href="/experiments/${esc(e.id)}">${esc(e.name)}</a></td><td>${esc(MODE_SHORT[e.tags.mode] || (s && MODE_SHORT[s.mode]) || '')}</td><td>${badge(e.status)}${e.status === 'running' ? ` <span class="muted num">${e.progress}/${e.total_rounds}</span>` : ''}</td>
    <td class="r">${s ? fmt.pct(s.final_mean_acc) : '–'}</td><td class="r">${s ? fmt.pct(s.final_min_acc) : '–'}</td><td class="r">${s ? fmt.eps(s.epsilon_total) : '–'}</td><td class="r">${s ? fmt.bytes(s.total_comm_bytes) : '–'}</td><td>${fmt.ago(e.created_at)}</td></tr>`; }).join('') + '</tbody></table>';
  $$('input[data-id]').forEach((c) => c.addEventListener('change', () => { c.checked ? picked.add(c.dataset.id) : picked.delete(c.dataset.id); $('#cmp').disabled = picked.size < 2; }));
  $('#cmp').disabled = picked.size < 2;
}
$('#cmp').addEventListener('click', () => (location.href = '/compare?ids=' + [...picked].join(',')));
poll(load, 4000);
