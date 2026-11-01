// Interactive replay of program tracing: each step adds one execution trace
// to the trie and drains its probability mass. Data comes from
// window.TRACE_DATA, which the notebook writes to trace-data.js.
(function () {
  const root = document.getElementById('trace-replay');
  if (!root || !window.d3 || !window.TRACE_DATA) return;

  const css = `
.trace-replay { margin: 1.5em 0; font-size: 0.85em; }
.trace-replay .tr-controls { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; margin-bottom: 8px; }
.trace-replay label { display: inline-flex; gap: 6px; align-items: center; }
.trace-replay select, .trace-replay button { font: inherit; padding: 3px 10px; border: 1px solid #ccc; border-radius: 4px; background: #fff; color: #333; cursor: pointer; }
.trace-replay button:hover, .trace-replay select:hover { border-color: #888; }
.trace-replay .tr-buttons { display: inline-flex; gap: 6px; }
.trace-replay .tr-view { width: 100%; height: 440px; border: 1px solid #e3ddd3; border-radius: 4px; background: #fffdf9; touch-action: none; }
.trace-replay .tr-status { margin-top: 6px; color: #555; font-variant-numeric: tabular-nums; }
.trace-replay .tr-hint { color: #888; }
.trace-replay .link { fill: none; stroke: #c9c2b6; stroke-width: 1.2; }
.trace-replay .link.current { stroke: #d9822b; stroke-width: 2; }
.trace-replay .edge-label { font-family: Menlo, Monaco, monospace; font-size: 10px; fill: #666; }
.trace-replay .node circle { stroke-width: 1.5; }
.trace-replay .node.leaf text { font-family: Menlo, Monaco, monospace; font-size: 10px; fill: #444; }
.trace-replay .node.current.leaf text { fill: #b5651d; font-weight: bold; }
.trace-replay .legend text { font-size: 10px; fill: #666; }
`;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  const PASS_COLORS = { 'pass 1': '#3b6ea8', 'pass 2': '#8a4fa3' };
  const LEAF_COLOR = '#555';
  const CURRENT = '#d9822b';

  root.classList.add('trace-replay');
  root.innerHTML = `
    <div class="tr-controls">
      <label>Program <select class="tr-program">
        <option value="awrs">AWRS on the test case</option>
        <option value="sis">SIS with AWRS</option>
      </select></label>
      <label>Order <select class="tr-order">
        <option value="sampling">sampling</option>
        <option value="greedy">greedy</option>
      </select></label>
      <span class="tr-buttons">
        <button class="tr-step">Next trace</button>
        <button class="tr-play">Play</button>
        <button class="tr-all">All</button>
        <button class="tr-reset">Reset</button>
      </span>
    </div>
    <svg class="tr-view"></svg>
    <div class="tr-status"></div>
    <div class="tr-hint">Drag to pan; pinch or ctrl+scroll to zoom; click a node to fold it.</div>`;

  const svg = d3.select(root).select('svg');
  const canvas = svg.append('g');
  const linkLayer = canvas.append('g');
  const labelLayer = canvas.append('g');
  const nodeLayer = canvas.append('g');
  const legend = svg.append('g').attr('class', 'legend').attr('transform', 'translate(10,14)');
  legend.append('rect').attr('x', -8).attr('y', -10).attr('width', 236).attr('height', 20)
    .attr('rx', 3).attr('fill', '#fffdf9').attr('opacity', 0.9);
  [['pass 1', PASS_COLORS['pass 1']], ['pass 2', PASS_COLORS['pass 2']], ['finished trace', LEAF_COLOR]]
    .forEach(([name, color], i) => {
      const g = legend.append('g').attr('transform', `translate(${i * 72},0)`);
      g.append('circle').attr('r', 4).attr('fill', color);
      g.append('text').attr('x', 8).attr('dy', '0.32em').text(name);
    });

  let userZoomed = false;
  const zoom = d3.zoom()
    .scaleExtent([0.05, 4])
    .filter(e => (e.type !== 'wheel' || e.ctrlKey || e.metaKey) && !e.button)
    .on('zoom', e => {
      canvas.attr('transform', e.transform);
      if (e.sourceEvent) userZoomed = true;
    });
  svg.call(zoom).on('dblclick.zoom', null);

  const els = {
    program: root.querySelector('.tr-program'),
    order: root.querySelector('.tr-order'),
    step: root.querySelector('.tr-step'),
    play: root.querySelector('.tr-play'),
    all: root.querySelector('.tr-all'),
    reset: root.querySelector('.tr-reset'),
    status: root.querySelector('.tr-status'),
  };

  let data, order, shown, explored, folded, timer = null;

  function load() {
    data = TRACE_DATA[els.program.value];
    order = data.orders[els.order.value];
    // Each node's children, for walking up and down the trie.
    data.nodes.forEach((n, i) => { n.id = i; });
    reset();
  }

  function reset() {
    stop();
    shown = 0;
    explored = new Float64Array(data.nodes.length);  // explored mass below each node
    folded = new Set();
    userZoomed = false;
    render(true);
  }

  function pathTo(id) {
    const ids = [];
    for (let i = id; i !== null; i = data.nodes[i].p) ids.push(i);
    return ids.reverse();
  }

  function step() {
    if (shown >= order.length) { stop(); return; }
    const leaf = order[shown++];
    const m = data.nodes[leaf].m;
    pathTo(leaf).forEach(i => { explored[i] += m; });
    render(false);
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    els.play.textContent = 'Play';
  }

  function play() {
    if (timer) { stop(); return; }
    if (shown >= order.length) reset();
    els.play.textContent = 'Pause';
    timer = setInterval(step, 150);
  }

  function showAll() {
    stop();
    while (shown < order.length) {
      const leaf = order[shown++];
      const m = data.nodes[leaf].m;
      pathTo(leaf).forEach(i => { explored[i] += m; });
    }
    render(false);
  }

  function render(instant) {
    // The revealed part of the trie: every node on a trace explored so far.
    const visible = new Set([0]);
    for (let k = 0; k < shown; k++) pathTo(order[k]).forEach(i => visible.add(i));
    const current = new Set(shown ? pathTo(order[shown - 1]) : []);

    const children = new Map();
    visible.forEach(i => {
      const p = data.nodes[i].p;
      if (p === null) return;
      if (!children.has(p)) children.set(p, []);
      children.get(p).push(i);
    });
    const hier = d3.hierarchy(0, i => (folded.has(i) ? null : children.get(i)));
    d3.tree().nodeSize([16, 104])(hier);
    const nodes = hier.descendants();
    const links = hier.links();
    const t = svg.transition().duration(instant ? 0 : 250);

    const linkPath = d => `M${d.source.y},${d.source.x}C${(d.source.y + d.target.y) / 2},${d.source.x} ${(d.source.y + d.target.y) / 2},${d.target.x} ${d.target.y},${d.target.x}`;
    linkLayer.selectAll('path').data(links, d => d.target.data)
      .join(enter => enter.append('path').attr('class', 'link').attr('d', linkPath))
      .classed('current', d => current.has(d.target.data))
      .transition(t).attr('d', linkPath);

    labelLayer.selectAll('text').data(links, d => d.target.data)
      .join(enter => enter.append('text').attr('class', 'edge-label').attr('text-anchor', 'end'))
      .text(d => data.nodes[d.target.data].e)
      .transition(t)
      .attr('x', d => d.target.y - 8)
      .attr('y', d => d.target.x - 4);

    const node = nodeLayer.selectAll('g.node').data(nodes, d => d.data)
      .join(enter => {
        const g = enter.append('g').attr('class', 'node')
          .attr('transform', d => `translate(${d.y},${d.x})`)
          .style('cursor', 'pointer')
          .on('click', (e, d) => {
            if (folded.has(d.data)) folded.delete(d.data); else folded.add(d.data);
            render(false);
          });
        g.append('circle').attr('r', 4.5);
        g.append('text').attr('x', 8).attr('dy', '0.32em');
        g.append('title');
        return g;
      });
    node.classed('leaf', d => data.nodes[d.data].o !== null)
      .classed('current', d => current.has(d.data))
      .transition(t).attr('transform', d => `translate(${d.y},${d.x})`);
    node.select('circle')
      .attr('stroke', d => {
        const n = data.nodes[d.data];
        if (current.has(d.data) && n.o !== null) return CURRENT;
        return n.o !== null ? LEAF_COLOR : (PASS_COLORS[n.c] || '#888');
      })
      .attr('fill', d => {
        const n = data.nodes[d.data];
        const color = n.o !== null ? LEAF_COLOR : (PASS_COLORS[n.c] || '#888');
        const remaining = Math.max(0, 1 - explored[d.data] / n.m);
        return d3.interpolateRgb('#fffdf9', color)(folded.has(d.data) ? 1 : remaining);
      });
    node.select('text').text(d => {
      const n = data.nodes[d.data];
      return n.o !== null ? n.o : (folded.has(d.data) ? '…' : '');
    });
    node.select('title').text(d => {
      const n = data.nodes[d.data];
      const lines = [];
      if (n.p !== null) lines.push(`choice ${n.e} (probability ${n.q})`);
      if (n.c) lines.push(`next choice: ${n.c}`);
      lines.push(`mass ${n.m.toPrecision(3)}, unexplored ${Math.max(0, n.m - explored[d.data]).toPrecision(3)}`);
      if (n.o !== null) lines.push(`output: ${n.o}`);
      return lines.join('\n');
    });

    const covered = explored[0];
    els.status.textContent = `Traces explored: ${shown} of ${order.length} · ` +
      `unexplored mass: ${Math.max(0, 1 - covered).toFixed(4)}`;

    if (!userZoomed) fit(nodes, instant, nodes.find(d => current.has(d.data) && data.nodes[d.data].o !== null));
  }

  function fit(nodes, instant, focus) {
    const W = svg.node().clientWidth, H = svg.node().clientHeight;
    const xs = nodes.map(d => d.x), ys = nodes.map(d => d.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys) + 90;  // room for leaf labels
    const top = 30, pad = 16;
    const fitAll = Math.min(1.4, (W - 2 * pad) / Math.max(1, y1 - y0), (H - top - pad) / Math.max(1, x1 - x0));
    // Don't shrink past readability: if the whole tree won't fit, center on the latest trace.
    const k = Math.max(fitAll, 0.45);
    const tx = k > fitAll && (y1 - y0) * k > W - 2 * pad ? W / 2 - (focus ? focus.y : y0) * k : pad - y0 * k;
    const ty = k > fitAll
      ? top + (H - top - pad) / 2 - (focus ? focus.x : (x0 + x1) / 2) * k
      : top + ((H - top - pad) - (x1 - x0) * k) / 2 - x0 * k;
    const transform = d3.zoomIdentity.translate(tx, ty).scale(k);
    (instant ? svg : svg.transition('fit').duration(250)).call(zoom.transform, transform);
  }

  els.program.addEventListener('change', load);
  els.order.addEventListener('change', load);
  els.step.addEventListener('click', () => { stop(); step(); });
  els.play.addEventListener('click', play);
  els.all.addEventListener('click', showAll);
  els.reset.addEventListener('click', reset);
  load();
})();
