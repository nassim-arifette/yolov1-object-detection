# Research website

A single English page in an interactive-article style. Sources live in `web/` (Vite + TypeScript, D3); `npm run build` writes `site/`, which is not tracked.

## Data

No model is run for the site. `tools/build_web_data.py` reads the outputs of `python download_voc.py --data-dir data` (verified VOC2007 archives, official splits, manifests, `grid_audit.json`) and writes:

- `web/public/data/site.json`: per-split and per-class grid-collision counts, after checking that `grid_audit.json` equals a fresh `protocol.audit_manifest` of each manifest, and the annotations of six example test images;
- `web/public/img/`: those six VOC2007 test images, unchanged;
- `web/test/fixtures/test_targets.json`: every test record with the objects `protocol.cell_targets` drops.

`web/test/yolo.test.ts` checks that the TypeScript encoder drops exactly the same objects on all 4,952 test images, and tests IoU and class-specific NMS against `metrics.py` conventions. These files are committed, so the Pages build needs neither VOC nor Python.

```bash
python download_voc.py --data-dir data
python tools/build_web_data.py
cd web
npm ci
npm test
npm run build   # -> ../site
cd ..
python tools/check_site.py
```

Training results will be added to the experiment section with their per-class AP, seeds, checkpoints and raw detections.

## Local preview and publishing

`cd web && npm run dev`, or `python -m http.server 8000 --directory site` after a build. `.github/workflows/pages.yml` installs, tests, builds, checks and deploys `site/`; in the repository settings choose **Pages → Source → GitHub Actions**.
