'use strict';

function histogram(counts) {
  if (!counts) return '';
  const m = Math.max(...counts, 1);
  return `<div class="mini-bars" role="img" aria-label="Samples per class">` +
    counts.map((c, i) => `<i title="class ${i}: ${c}" style="height:${Math.max(3, (c / m) * 100)}%"></i>`).join('') + '</div>';
}

async function loadNodes() {
  const nodes = await api('/api/nodes');
  $('#count').textContent = nodes.length ? `${nodes.filter((n) => n.online).length} online` : '';
  const el = $('#list');
  if (!nodes.length) {
    el.innerHTML = `<div class="empty"><h3>No campuses yet</h3><p>Start campus services with <code>python scripts/start_cluster.py</code> (they register automatically) or add one by address.</p></div>`;
    return;
  }
  el.innerHTML = nodes.map((n) => {
    const i = n.info;
    return `<article class="campus-card${n.online ? '' : ' off'}">
      <header><div><b class="serif" style="font-size:18px">${esc(n.name)}</b><div class="muted num">${esc(n.url)}</div></div>
        <div class="actions">${n.online ? `<span class="badge online">${esc(i.state)}</span>` : `<span class="badge offline">${esc(n.error || 'offline')}</span>`}
        <button class="btn small danger" data-del="${esc(n.id)}" aria-label="Remove ${esc(n.name)}">Remove</button></div></header>
      ${i ? `<div class="row" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr))">
        <div><div class="muted">Model</div><b>${esc(i.model || 'not assigned')}</b></div>
        <div><div class="muted">Parameters</div><b class="num">${fmt.int(i.n_params || null)}</b></div>
        <div><div class="muted">Private samples</div><b class="num">${i.n_private ?? '–'}</b></div>
        <div><div class="muted">Epochs trained</div><b class="num">${i.epochs_done}</b></div>
        <div><div class="muted">Releases</div><b class="num">${i.releases}</b></div>
        <div><div class="muted">Class mix</div>${histogram(i.class_counts) || '<span class="faint">–</span>'}</div></div>` : ''}
    </article>`;
  }).join('');
  $$('[data-del]').forEach((b) => b.addEventListener('click', async () => {
    if (!confirmAction('Remove this campus from the coordinator? The node itself keeps running.')) return;
    await api(`/api/nodes/${b.dataset.del}`, { method: 'DELETE' }); loadNodes();
  }));
}

$('#n-add').addEventListener('click', async () => {
  const url = $('#n-url').value.trim(); const name = $('#n-name').value.trim();
  if (!url) { toast('Enter the campus address first'); return; }
  try {
    await api('/api/nodes', { method: 'POST', body: { url, name } });
    $('#n-url').value = ''; $('#n-name').value = ''; toast('Campus added'); loadNodes();
  } catch (e) { toast(e.message); }
});
poll(loadNodes, 3000);
