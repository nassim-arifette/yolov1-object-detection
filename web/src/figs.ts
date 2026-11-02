import { scaleBand, scaleLinear, select } from 'd3';
import { cellTargets, CLASSES, iou, nms, normalizedBox, type Box, type Detection, type VocRecord } from './yolo';
import { cssVar, el, onScreen, onThemeChange, reducedMotion, segmented } from './util';

export interface Example extends VocRecord { src: string; caption: string; dropped: number[] }
export interface Audit { split: string; images: number; objects: number; difficult: number; non_difficult_dropped_by_grid: number; images_with_collisions: number; by_class: Record<string, { objects: number; dropped_by_grid: number }> }
export interface Site { examples: Example[]; audit: Audit[]; grid_size: number }

const S = 7;
const base = import.meta.env.BASE_URL;
const label = (c: number) => CLASSES[c].replace('diningtable', 'dining table').replace('pottedplant', 'potted plant').replace('tvmonitor', 'tv monitor');

/** Draw an image, its grid and its annotations into an SVG in image pixel coordinates. */
function scene(svg: SVGSVGElement, ex: Example) {
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${ex.width} ${ex.height}`);
  el('image', { href: `${base}${ex.src}`, x: 0, y: 0, width: ex.width, height: ex.height }, svg);
  const layers = {
    cells: el('g', {}, svg), grid: el('g', {}, svg), boxes: el('g', {}, svg), marks: el('g', {}, svg), over: el('g', {}, svg),
  };
  const cw = ex.width / S, ch = ex.height / S;
  for (let i = 1; i < S; i++) {
    el('line', { class: 'gridline', x1: i * cw, x2: i * cw, y1: 0, y2: ex.height }, layers.grid);
    el('line', { class: 'gridline', y1: i * ch, y2: i * ch, x1: 0, x2: ex.width }, layers.grid);
  }
  return { ...layers, cw, ch };
}

const rectOf = (b: Box) => ({ x: b[0] - 1, y: b[1] - 1, width: b[2] - b[0] + 1, height: b[3] - b[1] + 1 });

function annotate(svg: SVGSVGElement, ex: Example, L: ReturnType<typeof scene>, opts: { stagger?: boolean } = {}) {
  const { cellOf, dropped } = cellTargets(ex);
  const drop = new Set(dropped);
  const fs = Math.max(11, ex.width / 34);
  const nodes: SVGElement[][] = [];
  ex.objects.forEach((o, i) => {
    if (o.difficult) return;
    const kept = !drop.has(i);
    const [cx, cy] = normalizedBox(o, ex);
    const [row, col] = cellOf[i]!;
    const cell = el('rect', { class: kept ? 'cell-taken' : 'cell-clash', x: col * L.cw, y: row * L.ch, width: L.cw, height: L.ch }, L.cells);
    const box = el('rect', { class: `box ${kept ? 'kept' : 'dropped'}`, ...rectOf(o.bbox) }, L.boxes);
    const dot = el('circle', { class: 'centre', cx: cx * ex.width, cy: cy * ex.height, r: fs * 0.42, fill: kept ? cssVar('--kept') : cssVar('--dropped') }, L.marks);
    const tagText = kept ? label(o.class_id) : `${label(o.class_id)}: no target`;
    const tag = el('text', { class: 'tag', x: o.bbox[0] + 3, y: o.bbox[1] + fs + 1, 'font-size': fs, fill: '#fff', stroke: kept ? cssVar('--kept') : cssVar('--dropped'), 'stroke-width': fs * 0.28 }, L.marks);
    tag.textContent = tagText;
    nodes.push([cell, box, dot, tag]);
  });
  if (opts.stagger) nodes.flat().forEach((n) => n.setAttribute('opacity', '0'));
  return nodes;
}

/** Hero: images with grid collisions, revealed step by step. */
export function initHero(site: Site) {
  const svg = document.getElementById('hero-svg') as unknown as SVGSVGElement;
  const root = document.getElementById('top')!;
  const caption = root.querySelector<HTMLElement>('[data-role=caption]')!;
  let i = 0, token = 0, visible = true;
  onScreen(root, (v) => { visible = v; });
  const sleep = (ms: number, t: number) => new Promise<boolean>((res) => {
    const start = performance.now();
    const tick = () => {
      if (t !== token) return res(false);
      if (!visible || document.hidden) return setTimeout(tick, 250);
      if (performance.now() - start >= ms) return res(true);
      setTimeout(tick, 40);
    };
    tick();
  });

  async function play(k: number) {
    const t = ++token;
    const ex = site.examples[k];
    const L = scene(svg, ex);
    const nodes = annotate(svg, ex, L, { stagger: !reducedMotion() });
    const dropped = ex.dropped.length, total = ex.objects.filter((o) => !o.difficult).length;
    caption.textContent = `${ex.caption} ${dropped} of ${total} objects get no training target.`;
    if (reducedMotion()) return;
    L.grid.setAttribute('opacity', '0');
    if (!(await sleep(350, t))) return;
    L.grid.setAttribute('opacity', '1');
    for (const [cell, box, dot, tag] of nodes) {
      for (const n of [box, dot, cell]) n.setAttribute('opacity', '1');
      if (!(await sleep(170, t))) return;
      tag.setAttribute('opacity', '1');
    }
    if (await sleep(4200, t)) { i = (i + 1) % site.examples.length; play(i); }
  }
  root.querySelector('[data-role=next]')!.addEventListener('click', () => { i = (i + 1) % site.examples.length; play(i); });
  root.querySelector('[data-role=prev]')!.addEventListener('click', () => { i = (i - 1 + site.examples.length) % site.examples.length; play(i); });
  onThemeChange(() => play(i));
  play(0);
}

/** Figure 1: hover a cell to see its training target. */
export function initCell(site: Site) {
  const fig = document.getElementById('fig-cell')!;
  const svg = fig.querySelector<SVGSVGElement>('[data-role=svg]')!;
  const panel = fig.querySelector<HTMLElement>('[data-role=panel]')!;
  const sel = fig.querySelector<HTMLSelectElement>('[data-ctl=image]')!;
  site.examples.forEach((e, k) => sel.add(new Option(e.caption, String(k))));
  let ex = site.examples[0];

  function describe(row: number, col: number) {
    const { selected, cellOf, dropped } = cellTargets(ex);
    const target = selected.get(`${row},${col}`);
    const here = ex.objects.map((o, i) => ({ o, i })).filter(({ i }) => cellOf[i] && cellOf[i]![0] === row && cellOf[i]![1] === col);
    panel.replaceChildren();
    const h = document.createElement('div');
    h.style.fontWeight = '700';
    h.textContent = `Cell row ${row + 1}, column ${col + 1}`;
    panel.append(h);
    const p = document.createElement('p'); p.className = 'small';
    p.textContent = here.length === 0 ? 'No object centre falls here: this cell is trained to predict “no object”.'
      : here.length === 1 ? `One object centre falls here: a ${label(here[0].o.class_id)}.`
        : `${here.length} object centres fall here: ${here.map(({ o }) => label(o.class_id)).join(', ')}. Only the first in annotation order becomes the target. The other${here.length > 2 ? 's get' : ' gets'} nothing.`;
    panel.append(p);
    const add = (title: string, values: { t: string; on: boolean }[], cols = 10) => {
      const l = document.createElement('div'); l.className = 'vec-label'; l.textContent = title;
      const v = document.createElement('div'); v.className = 'vec'; v.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      for (const { t, on } of values) { const c = document.createElement('div'); c.className = `v${on ? ' on' : ''}`; c.textContent = t; c.title = t; v.append(c); }
      panel.append(l, v);
    };
    add('20 class probabilities', CLASSES.map((c, k) => ({ t: label(k).slice(0, 4), on: !!target && target.class_id === k })));
    add('box target: object, x, y in the cell, w, h in the image', [
      { t: target ? '1' : '0', on: !!target },
      ...(['x', 'y', 'w', 'h'] as const).map((k) => ({ t: target ? target[k].toFixed(2) : '—', on: !!target })),
    ], 5);
    const n = document.createElement('p'); n.className = 'small';
    n.textContent = 'The cell predicts two boxes. In training, whichever overlaps this target more becomes responsible for it, and the other learns “no object”.';
    panel.append(n);
  }

  function draw() {
    const L = scene(svg, ex);
    annotate(svg, ex, L);
    const hot = el('rect', { class: 'cell-hot', width: L.cw, height: L.ch, x: -999, y: -999 }, L.over);
    for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) {
      const hit = el('rect', { class: 'cell-hit', x: c * L.cw, y: r * L.ch, width: L.cw, height: L.ch }, L.over);
      const go = () => { hot.setAttribute('x', String(c * L.cw)); hot.setAttribute('y', String(r * L.ch)); describe(r, c); };
      hit.addEventListener('pointerenter', go);
      hit.addEventListener('click', go);
    }
    // start on the first collision cell
    const { cellOf, dropped } = cellTargets(ex);
    const first = dropped.length ? cellOf[dropped[0]]! : [3, 3];
    hot.setAttribute('x', String(first[1] * L.cw)); hot.setAttribute('y', String(first[0] * L.ch));
    describe(first[0], first[1]);
  }
  sel.addEventListener('change', () => { ex = site.examples[+sel.value]; draw(); });
  onThemeChange(draw);
  draw();
}

/** Figure 2: per-class share of objects left without a target. */
export function initAudit(site: Site) {
  const fig = document.getElementById('fig-audit')!;
  const svgEl = fig.querySelector<SVGSVGElement>('[data-role=chart]')!;
  let split = 'all';
  const W = 760, rowH = 19, M = { l: 104, r: 120, t: 8, b: 8 };
  function draw() {
    const audits = split === 'all' ? site.audit : site.audit.filter((a) => a.split === split);
    const rows = CLASSES.map((_, k) => {
      const name = CLASSES[k];
      const objects = audits.reduce((s, a) => s + a.by_class[name].objects, 0);
      const dropped = audits.reduce((s, a) => s + a.by_class[name].dropped_by_grid, 0);
      return { k, objects, dropped, rate: objects ? dropped / objects : 0 };
    }).sort((a, b) => b.rate - a.rate);
    const H = M.t + M.b + rows.length * rowH;
    svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const svg = select(svgEl);
    svg.selectAll('*').remove();
    const x = scaleLinear().domain([0, Math.max(0.18, ...rows.map((r) => r.rate))]).range([M.l, W - M.r]);
    const y = scaleBand<number>().domain(rows.map((r) => r.k)).range([M.t, H - M.b]).padding(0.22);
    const total = rows.reduce((s, r) => s + r.objects, 0), lost = rows.reduce((s, r) => s + r.dropped, 0);
    const avg = lost / total;
    svg.append('line').attr('x1', x(avg)).attr('x2', x(avg)).attr('y1', M.t).attr('y2', H - M.b).attr('stroke', cssVar('--muted')).attr('stroke-dasharray', '3 3');
    svg.append('text').attr('x', x(avg) + 4).attr('y', H - M.b - 4).attr('font-weight', 700).text(`all classes ${(avg * 100).toFixed(1)} %`);
    for (const r of rows) {
      const yy = y(r.k)!;
      svg.append('text').attr('x', M.l - 8).attr('y', yy + y.bandwidth() / 2 + 4).attr('text-anchor', 'end').text(label(r.k));
      svg.append('rect').attr('x', M.l).attr('y', yy).attr('width', x(r.rate) - M.l).attr('height', y.bandwidth()).attr('rx', 2)
        .attr('fill', cssVar('--dropped')).attr('fill-opacity', 0.2 + 0.8 * Math.min(1, r.rate / 0.16));
      svg.append('text').attr('x', x(r.rate) + 6).attr('y', yy + y.bandwidth() / 2 + 4)
        .attr('stroke', cssVar('--surface')).attr('stroke-width', 4).attr('paint-order', 'stroke').attr('stroke-linejoin', 'round')
        .text(`${(r.rate * 100).toFixed(1)} %  (${r.dropped.toLocaleString('en')} of ${r.objects.toLocaleString('en')})`);
    }
  }
  segmented(fig.querySelector('[data-ctl=split]')!, (v) => { split = v; draw(); });
  onThemeChange(draw);
  draw();
}

/** Figure 3: drag a predicted box against a real annotation; compare the two confidence targets. */
export function initConf(site: Site) {
  const fig = document.getElementById('fig-conf')!;
  const svg = fig.querySelector<SVGSVGElement>('[data-role=svg]')!;
  const readout = fig.querySelector<HTMLElement>('[data-role=readout]')!;
  const pIn = fig.querySelector<HTMLInputElement>('[data-ctl=p]')!;
  const ex = site.examples[0];
  const gtIndex = ex.objects.findIndex((o) => CLASSES[o.class_id] === 'person');
  const gt = ex.objects[gtIndex].bbox;
  const L = scene(svg, ex);
  el('rect', { class: 'box kept', ...rectOf(gt) }, L.boxes);
  const w0 = gt[2] - gt[0], h0 = gt[3] - gt[1];
  let pred: Box = [gt[0] + w0 * 0.35, gt[1] - h0 * 0.15, gt[2] + w0 * 0.35, gt[3] - h0 * 0.15];
  const rect = el('rect', { class: 'pred' }, L.over);
  const handle = el('circle', { class: 'handle', r: Math.max(7, ex.width / 60) }, L.over);

  function update() {
    const b: Box = [Math.round(pred[0]), Math.round(pred[1]), Math.round(pred[2]), Math.round(pred[3])];
    const r = rectOf(b);
    for (const [k, v] of Object.entries(r)) rect.setAttribute(k, String(v));
    handle.setAttribute('cx', String(b[2])); handle.setAttribute('cy', String(b[3]));
    const v = iou(b, gt), p = +pIn.value;
    fig.querySelector('[data-out=p]')!.textContent = p.toFixed(2);
    readout.replaceChildren();
    const row = (k: string, val: string, strong = false) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd'); dd.textContent = val; if (strong) dd.style.color = cssVar('--kept');
      readout.append(dt, dd);
    };
    row('IoU with the annotation', v.toFixed(3), true);
    row('Confidence target, paper (IoU)', v.toFixed(3));
    row('Confidence target, control (1)', '1.000');
    row('Detection score if it learns IoU', (p * v).toFixed(3));
    row('Detection score if it learns 1', p.toFixed(3));
    row('Counts as a correct detection', v > 0.5 ? 'yes, IoU > 0.5' : 'no');
  }
  const toImage = (e: PointerEvent) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const q = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [q.x, q.y];
  };
  let mode: 'move' | 'size' | null = null, start = [0, 0], orig: Box = pred;
  const down = (m: 'move' | 'size') => (e: PointerEvent) => { mode = m; start = toImage(e); orig = [...pred] as Box; (e.target as Element).setPointerCapture(e.pointerId); e.preventDefault(); };
  rect.addEventListener('pointerdown', down('move'));
  handle.addEventListener('pointerdown', down('size'));
  const move = (e: PointerEvent) => {
    if (!mode) return;
    const [x, y] = toImage(e);
    const dx = x - start[0], dy = y - start[1];
    if (mode === 'move') pred = [orig[0] + dx, orig[1] + dy, orig[2] + dx, orig[3] + dy];
    else pred = [orig[0], orig[1], Math.max(orig[0] + 8, orig[2] + dx), Math.max(orig[1] + 8, orig[3] + dy)];
    pred = [Math.max(1, pred[0]), Math.max(1, pred[1]), Math.min(ex.width, pred[2]), Math.min(ex.height, pred[3])];
    update();
  };
  rect.addEventListener('pointermove', move);
  handle.addEventListener('pointermove', move);
  const up = () => { mode = null; };
  rect.addEventListener('pointerup', up); handle.addEventListener('pointerup', up);
  pIn.addEventListener('input', update);
  update();
}

/** Figure 4: class-specific NMS on illustrative candidates around real annotations. */
export function initNms(site: Site) {
  const fig = document.getElementById('fig-nms')!;
  const svg = fig.querySelector<SVGSVGElement>('[data-role=svg]')!;
  const tIn = fig.querySelector<HTMLInputElement>('[data-ctl=t]')!;
  const legend = fig.querySelector<HTMLElement>('[data-role=legend]')!;
  const ex = site.examples[1];
  const L = scene(svg, ex);
  // deterministic jitter
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const cands: Detection[] = [];
  ex.objects.forEach((o) => {
    if (o.difficult) return;
    const w = o.bbox[2] - o.bbox[0], h = o.bbox[3] - o.bbox[1];
    for (let k = 0; k < 5; k++) {
      const j = k === 0 ? 0.03 : 0.16;
      const b: Box = [o.bbox[0] + (rnd() - 0.5) * w * j * 2, o.bbox[1] + (rnd() - 0.5) * h * j * 2, o.bbox[2] + (rnd() - 0.5) * w * j * 2, o.bbox[3] + (rnd() - 0.5) * h * j * 2];
      const clipped: Box = [Math.max(1, b[0]), Math.max(1, b[1]), Math.min(ex.width, b[2]), Math.min(ex.height, b[3])];
      cands.push({ class_id: o.class_id, score: Math.round((0.35 + 0.6 * iou(clipped, o.bbox) * (0.85 + 0.15 * rnd())) * 100) / 100, bbox: clipped, id: cands.length });
    }
  });
  const classes = [...new Set(cands.map((c) => c.class_id))];
  const palette = [cssVar('--kept'), cssVar('--pred')];
  function draw() {
    L.boxes.replaceChildren();
    const t = +tIn.value;
    fig.querySelector('[data-out=t]')!.textContent = t.toFixed(2);
    const kept = new Set(nms(cands, t).map((d) => d.id));
    for (const d of [...cands].sort((a, b) => a.score - b.score)) {
      const on = kept.has(d.id);
      const color = palette[classes.indexOf(d.class_id) % palette.length];
      el('rect', { class: 'box', ...rectOf(d.bbox), stroke: color, 'stroke-opacity': on ? 1 : 0.35, 'stroke-dasharray': on ? '' : '4 4' }, L.boxes);
      if (on) {
        const fs = Math.max(11, ex.width / 36);
        el('text', { class: 'tag', x: d.bbox[0] + 3, y: d.bbox[1] + fs + 2, 'font-size': fs, fill: '#fff', stroke: color, 'stroke-width': fs * 0.28 }, L.boxes).textContent = `${label(d.class_id)} ${d.score.toFixed(2)}`;
      }
    }
    legend.replaceChildren();
    classes.forEach((c, k) => {
      const s = document.createElement('span'); const i = document.createElement('i'); i.style.background = palette[k % palette.length];
      const n = [...kept].filter((id) => cands[id!].class_id === c).length;
      s.append(i, `${label(c)}: ${n} kept of ${cands.filter((d) => d.class_id === c).length}`);
      legend.append(s);
    });
  }
  tIn.addEventListener('input', draw);
  onThemeChange(draw);
  draw();
}
