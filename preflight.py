"""Verify split separation, source annotations and grid losses without Torch."""
import argparse
import json
from pathlib import Path
from configuration import load_config
from protocol import audit_manifest, load_manifest, require_disjoint, sha256


def check(paths, voc_root=None, grid_size=7):
    manifests = [load_manifest(path) for path in paths]
    require_disjoint(*manifests)
    reports = []
    for path, data in zip(paths, manifests):
        if voc_root:
            for record in data["records"]:
                image = Path(voc_root) / record["image"]
                annotation = image.parent.parent / "Annotations" / (image.stem + ".xml")
                if not image.is_file():
                    raise FileNotFoundError(image)
                if sha256(annotation) != record["annotation_sha256"]:
                    raise ValueError(f"Source XML changed: {annotation}")
        reports.append({"manifest": str(path), "sha256": sha256(path),
                        **audit_manifest(data, grid_size)})
    return {"status": "verified", "inference_run": False, "manifests": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/voc2007.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    result = check(args.manifests, args.voc_root, config["model"]["split_size"])
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
