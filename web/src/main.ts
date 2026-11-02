import './style.css';
import { initAudit, initCell, initConf, initHero, initNms, type Site } from './figs';
import { whenVisible } from './util';

document.getElementById('theme-toggle')?.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch { /* storage unavailable */ }
  window.dispatchEvent(new Event('themechange'));
});

function failed(id: string, err: unknown) {
  console.error(err);
  const fig = document.getElementById(id);
  if (!fig) return;
  const p = document.createElement('p');
  p.className = 'small';
  p.textContent = 'This figure could not load its data. Serve the site over HTTP (for example with npm run preview).';
  fig.prepend(p);
}
const guard = (id: string, f: () => void) => { try { f(); } catch (e) { failed(id, e); } };

function stats(site: Site) {
  const objects = site.audit.reduce((s, a) => s + a.objects - a.difficult, 0);
  const dropped = site.audit.reduce((s, a) => s + a.non_difficult_dropped_by_grid, 0);
  const images = site.audit.reduce((s, a) => s + a.images, 0);
  const affected = site.audit.reduce((s, a) => s + a.images_with_collisions, 0);
  let worst = 0;
  for (const name of Object.keys(site.audit[0].by_class)) {
    const o = site.audit.reduce((s, a) => s + a.by_class[name].objects, 0);
    const d = site.audit.reduce((s, a) => s + a.by_class[name].dropped_by_grid, 0);
    if (o) worst = Math.max(worst, d / o);
  }
  const set = (k: string, v: string) => document.querySelectorAll(`[data-stat=${k}]`).forEach((n) => { n.textContent = v; });
  set('drop-pct', `${((dropped / objects) * 100).toFixed(1)} %`);
  set('dropped', dropped.toLocaleString('en'));
  set('objects', objects.toLocaleString('en'));
  set('worst', `${(worst * 100).toFixed(1)} %`);
  set('images', `${affected.toLocaleString('en')} of ${images.toLocaleString('en')} images`);
}

fetch(`${import.meta.env.BASE_URL}data/site.json`).then((r) => {
  if (!r.ok) throw new Error(`site.json: HTTP ${r.status}`);
  return r.json() as Promise<Site>;
}).then((site) => {
  stats(site);
  guard('top', () => initHero(site));
  const lazy: [string, () => void][] = [
    ['fig-cell', () => initCell(site)], ['fig-audit', () => initAudit(site)], ['fig-conf', () => initConf(site)], ['fig-nms', () => initNms(site)],
  ];
  for (const [id, init] of lazy) whenVisible(document.getElementById(id)!, () => guard(id, init));
}).catch((e) => ['top', 'fig-cell', 'fig-audit'].forEach((id) => failed(id, e)));
