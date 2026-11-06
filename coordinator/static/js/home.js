'use strict';

function federationSVG(nodes, trustByNode, accByNode, running) {
  const W = 640, H = 430, cx = W / 2, cy = H / 2, rx = 235, ry = 150;
  const n = nodes.length;
  const pos = nodes.map((_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(n, 1);
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)];
  });
  let s = `<svg class="fed" viewBox="0 0 ${W} ${H}" role="img" aria-label="Federation of ${n} campuses around the coordinator">`;
  nodes.forEach((nd, i) => {
    const [x, y] = pos[i];
    const t = trustByNode[nd.id];
    const rel = t != null ? t * n : 1;
    const w = Math.max(1.5, Math.min(13, 1.5 + 4 * rel));
    s += `<line class="edge${running && nd.online ? ' flow' : ''}" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke-width="${w.toFixed(1)}" ${nd.online ? '' : 'style="stroke:var(--line-strong);opacity:.6"'}/>`;
  });
  s += `<circle class="core" cx="${cx}" cy="${cy}" r="48"/><text x="${cx}" y="${cy - 2}" text-anchor="middle" font-size="14" font-weight="600" style="fill:var(--blue-ink)">Coordinator</text>` +
       `<text x="${cx}" y="${cy + 15}" text-anchor="middle" font-size="12" style="fill:var(--blue-ink);opacity:.85">trust → teacher</text>`;
  nodes.forEach((nd, i) => {
    const [x, y] = pos[i];
    const color = nd.online ? 'var(--teal)' : 'var(--red)';
    const t = trustByNode[nd.id];
    const acc = accByNode[nd.id];
    const below = y > cy + 5;
    const ty = below ? y + 46 : y - 44;
    s += `<g><circle class="ring" cx="${x}" cy="${y}" r="26" stroke="${color}"/>` +
         `<text x="${x}" y="${y + 5}" text-anchor="middle" font-size="15" font-weight="700" class="serif">${t != null ? (t * 100).toFixed(0) + '%' : (nd.online ? '·' : '×')}</text>` +
         `<text x="${x}" y="${ty}" text-anchor="middle" font-size="14" font-weight="600">${esc(nd.name)}</text>` +
         `<text x="${x}" y="${ty + 16}" text-anchor="middle" font-size="12.5" style="fill:var(--muted)">${esc(nd.info && nd.info.model ? nd.info.model : (nd.online ? 'idle' : 'offline'))}${acc != null ? ' · ' + (acc * 100).toFixed(1) + '% acc' : ''}</text></g>`;
  });
  return s + '</svg>';
}

async function loadHome() {
  const [nodes, exps] = await Promise.all([api('/api/nodes'), api('/api/experiments?limit=8')]);
  const wrap = $('#fed-wrap');
  const online = nodes.filter((n) => n.online).length;
  $('#fed-sub').textContent = nodes.length ? `${online} of ${nodes.length} campuses online` : 'No campuses yet';

  const running = exps.some((e) => e.status === 'running');
  const done = exps.find((e) => e.status === 'completed');
  let trust = {}, acc = {};
  if (done) {
    const full = await api(`/api/experiments/${done.id}`);
    const last = full.rounds[full.rounds.length - 1];
    if (last) last.campuses.forEach((c) => { acc[c.node_id] = c.acc; if (c.trust != null) trust[c.node_id] = c.trust; });
    const s = done.summary || {};
    $('#latest').innerHTML = `
      <div class="serif" style="font-size:34px;font-weight:600">${fmt.pct(s.final_mean_acc)}</div>
      <div class="muted">mean test accuracy across campuses · ${esc(MODE_SHORT[s.mode] || s.mode)}</div>
      <p style="margin-top:12px" class="muted">Gain over warm-up <b class="${(s.gain_over_warmup || 0) >= 0 ? 'pos' : 'neg'}">${s.gain_over_warmup != null ? (s.gain_over_warmup >= 0 ? '+' : '') + (s.gain_over_warmup * 100).toFixed(1) + ' pts' : '–'}</b>
      · privacy budget ε = <b>${fmt.eps(s.epsilon_total)}</b></p>
      <a href="/experiments/${esc(done.id)}">Open “${esc(done.name)}”</a>`;
  }
  if (!nodes.length) {
    wrap.innerHTML = `<div class="empty"><h3>No campuses registered</h3><p>Start the campus services, then register them so the coordinator can reach them.</p><a class="btn primary" href="/nodes">Add campuses</a></div>`;
  } else {
    wrap.innerHTML = federationSVG(nodes, trust, acc, running);
  }

  const rec = $('#recent');
  if (!exps.length) {
    rec.innerHTML = `<div class="empty"><h3>No experiments yet</h3><p>Run the proposed method on your campuses to see trust scores and accuracy by round.</p><a class="btn primary" href="/experiments/new">New experiment</a></div>`;
  } else {
    rec.innerHTML = `<table><thead><tr><th>Name</th><th>Method</th><th>Status</th><th class="r">Accuracy</th><th class="r">ε spent</th><th>When</th></tr></thead><tbody>` +
      exps.map((e) => `<tr class="click" onclick="location.href='/experiments/${esc(e.id)}'"><td><a href="/experiments/${esc(e.id)}">${esc(e.name)}</a></td>
      <td>${esc(MODE_SHORT[(e.summary && e.summary.mode) || e.tags.mode] || e.tags.mode || '')}</td><td>${badge(e.status)}</td>
      <td class="r">${e.summary ? fmt.pct(e.summary.final_mean_acc) : '–'}</td><td class="r">${e.summary ? fmt.eps(e.summary.epsilon_total) : '–'}</td><td>${fmt.ago(e.created_at)}</td></tr>`).join('') +
      '</tbody></table>';
  }
}
poll(loadHome, 4000);
