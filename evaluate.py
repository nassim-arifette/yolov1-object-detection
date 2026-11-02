"""Evaluate a selected checkpoint on complete, unaugmented VOC annotations."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from model import Yolov1
    from protocol import sha256
    from runtime import atomic_json, resolve_device, make_dataset, make_loader, eval_kwargs
    from utils import evaluate
    if args.out.exists():
        parser.error("--out already exists; preserve prior evaluations and choose a new directory")
    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved.get("schema_version") != 1:
        raise ValueError("Only revised, provenance-bearing checkpoints are supported")
    config = saved["config"]
    dataset = make_dataset(args.manifest, args.voc_root, config)
    ids = {r["image_id"] for r in dataset.records}
    if dataset.manifest["split"] == "test" and ids & (set(saved["train_ids"]) | set(saved["val_ids"])):
        raise ValueError("Test split overlaps training or checkpoint selection data")
    model = Yolov1(**config["model"]).to(device)
    model.load_state_dict(saved["model"])
    loader = make_loader(dataset, config, device)
    metrics, detections = evaluate(model, loader, device, **eval_kwargs(config))
    metrics.update({"checkpoint_sha256": sha256(args.checkpoint),
                    "manifest_sha256": sha256(args.manifest), "split": dataset.manifest["split"],
                    "checkpoint_epoch": saved["epoch"], "config": config})
    args.out.mkdir(parents=True)
    atomic_json(args.out / "metrics.json", metrics)
    atomic_json(args.out / "detections.json", detections)
    print(f"{dataset.manifest['split']} {metrics['ap_mode']} mAP@0.5: {metrics['map']}")


if __name__ == "__main__":
    main()
