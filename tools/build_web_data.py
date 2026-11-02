"""Export VOC2007 grid-audit evidence and a few example images for the website.

Inputs are produced by `python download_voc.py --data-dir data` (verified
archives, official splits, manifests and grid_audit.json). Nothing here runs a
model. The export writes:

* web/public/data/site.json: per-split and per-class collision counts from
  grid_audit.json, and the annotations of the example images with the cell
  assignment computed by protocol.cell_targets;
* web/public/img/<id>.jpg: the example VOC2007 test images, unchanged;
* web/test/fixtures/test_targets.json: every test record with the indices
  protocol.cell_targets drops, so the TypeScript port can be checked on all of them.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol import CLASSES, audit_manifest, cell_targets, load_manifest  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "web/public"
FIXTURE = ROOT / "web/test/fixtures"
EXAMPLES = ["2007_005925", "2007_005677", "2007_000348", "2007_002676", "2007_000893", "2007_001575"]
CAPTIONS = {
    "2007_005925": "Two cyclists: each rider and bicycle share a cell.",
    "2007_005677": "A rider and her horse, with a second pair behind.",
    "2007_000348": "A motorbike and its passengers.",
    "2007_002676": "A herd of cows.",
    "2007_000893": "Bottles on a shelf.",
    "2007_001575": "A crowd of people.",
}


def compact(record):
    return {"image_id": record["image_id"], "width": record["width"], "height": record["height"],
            "objects": [{"class_id": o["class_id"], "bbox": o["bbox"], "difficult": o["difficult"]} for o in record["objects"]]}


def main():
    audits = json.loads((DATA / "manifests/grid_audit.json").read_text(encoding="utf-8"))
    manifests = {s: load_manifest(DATA / f"manifests/{s}.json") for s in ("train", "val", "test")}
    for audit in audits:
        if audit_manifest(manifests[audit["split"]]) != audit:
            raise SystemExit(f"grid_audit.json is stale for {audit['split']}; rerun prepare_voc.py")
    test = {r["image_id"]: r for r in manifests["test"]["records"]}
    OUT.joinpath("img").mkdir(parents=True, exist_ok=True)
    examples = []
    for image_id in EXAMPLES:
        record = test[image_id]
        selected, dropped = cell_targets(record)
        kept = {id(o) for o in record["objects"]} - {id(o) for o in dropped}
        shutil.copyfile(DATA / "VOCdevkit" / record["image"], OUT / "img" / f"{image_id}.jpg")
        examples.append({**compact(record), "src": f"img/{image_id}.jpg", "caption": CAPTIONS[image_id],
                         "dropped": [i for i, o in enumerate(record["objects"]) if id(o) not in kept]})
    fixture = []
    for record in manifests["test"]["records"]:
        _, dropped = cell_targets(record)
        ids = {id(o) for o in dropped}
        fixture.append({**compact(record), "dropped": [i for i, o in enumerate(record["objects"]) if id(o) in ids]})
    site = {"schema_version": 1, "classes": list(CLASSES), "grid_size": 7,
            "audit": audits, "examples": examples,
            "source": "VOC2007 official train/val/test splits, all XML annotations (download_voc.py)"}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data/site.json").write_text(json.dumps(site, separators=(",", ":")), encoding="utf-8")
    FIXTURE.mkdir(parents=True, exist_ok=True)
    (FIXTURE / "test_targets.json").write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    dropped = sum(a["non_difficult_dropped_by_grid"] for a in audits)
    print(f"{len(examples)} examples, {len(fixture)} test records, {dropped} objects dropped by the grid over all splits")


if __name__ == "__main__":
    main()
