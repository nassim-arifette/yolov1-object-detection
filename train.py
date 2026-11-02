"""Train on train, select on val, and leave test for evaluate.py."""
import argparse
from pathlib import Path
from configuration import load_config, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/voc2007.json"))
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.json"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--confidence-target", choices=("iou", "one"))
    parser.add_argument("--augmentation", choices=("none", "flip_color"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    for key in ("confidence_target", "augmentation"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    validate(config)
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    from runtime import train
    train(config, args.train_manifest, args.val_manifest, args.voc_root, args.out,
          args.device, args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
