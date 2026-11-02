"""Build explicit VOC splits without moving images or discarding XML annotations."""
import argparse
import json
from pathlib import Path
from protocol import CLASSES, audit_manifest, read_annotation, require_disjoint


def prepare(root, year, splits, out):
    manifests = []
    base = Path(root) / f"VOC{year}"
    for split in splits:
        ids = (base / "ImageSets/Main" / f"{split}.txt").read_text().split()
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError(f"{split}: empty split or duplicate image IDs")
        records = []
        for image_id in ids:
            image = f"VOC{year}/JPEGImages/{image_id}.jpg"
            if not (Path(root) / image).is_file():
                raise FileNotFoundError(Path(root) / image)
            records.append(read_annotation(base / "Annotations" / f"{image_id}.xml",
                                           f"{year}_{image_id}", image))
        manifests.append({"schema_version": 1, "dataset": f"VOC{year}", "split": split,
                          "classes": list(CLASSES), "records": records})
    require_disjoint(*manifests)
    Path(out).mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        (Path(out) / f"{manifest['split']}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [audit_manifest(m) for m in manifests]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", type=Path, required=True, help="Directory containing VOC2007")
    parser.add_argument("--year", default="2007")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    audits = prepare(args.voc_root, args.year, args.splits, args.out)
    (args.out / "grid_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    print(json.dumps(audits, indent=2))
