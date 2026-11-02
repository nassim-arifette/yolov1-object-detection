"""Training and evaluation with split provenance and resumable checkpoints."""
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuration import validate
from dataset import VOCDataset
from loss import YoloLoss
from model import Yolov1
from protocol import CLASSES, audit_manifest, require_disjoint, sha256
from utils import evaluate


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_checkpoint(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp)
    temp.replace(path)


def make_dataset(path, root, config, train=False):
    m, t = config["model"], config["training"]
    return VOCDataset(path, root, S=m["split_size"], B=m["num_boxes"], C=m["num_classes"],
                      augmentation=t["augmentation"] if train else "none")


def make_loader(dataset, config, device, train=False, generator=None):
    t = config["training"]
    return DataLoader(dataset, batch_size=t["batch_size"], shuffle=train, drop_last=False,
                      num_workers=t["workers"], pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=generator)


def eval_kwargs(config):
    m = config["model"]
    return {"S": m["split_size"], "B": m["num_boxes"], "C": m["num_classes"], **config["evaluation"]}


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for intentional CPU checks")
    return device


def _rng_state(generator):
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
            int(np_state[2]), int(np_state[3]), float(np_state[4])],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def train(config, train_manifest, val_manifest, voc_root, out, device_name="cuda", resume=None,
          stop_after_epoch=None):
    validate(config)
    device = resolve_device(device_name)
    out = Path(out)
    if not resume and out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory or --resume")
    if resume and (Path(resume).resolve() != (out / "last.pt").resolve() or
                   not (out / "best.pt").is_file()):
        raise ValueError("Resume from last.pt in its original output directory containing best.pt")
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(seed)
    train_set = make_dataset(train_manifest, voc_root, config, train=True)
    val_set = make_dataset(val_manifest, voc_root, config)
    require_disjoint(train_set.manifest, val_set.manifest)
    if train_set.manifest["split"] != "train" or val_set.manifest["split"] != "val":
        raise ValueError("Checkpoint selection requires explicit train and val manifests; test is forbidden")
    manifests = {"train": sha256(train_manifest), "val": sha256(val_manifest)}
    source_root = Path(__file__).parent
    source_hashes = {p.name: sha256(p) for p in sorted(source_root.glob('*.py'))}
    train_loader = make_loader(train_set, config, device, True, generator)
    val_loader = make_loader(val_set, config, device)
    model = Yolov1(**config["model"]).to(device)
    t = config["training"]
    optimizer = torch.optim.SGD(model.parameters(), lr=t["learning_rate"],
                                momentum=t["momentum"], weight_decay=t["weight_decay"])
    loss_fn = YoloLoss(S=config["model"]["split_size"], B=config["model"]["num_boxes"],
                       C=config["model"]["num_classes"], confidence_target=t["confidence_target"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, history = 0, -1.0, []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("schema_version") != 1 or checkpoint["config"] != config or checkpoint["manifest_sha256"] != manifests:
            raise ValueError("Resume config/manifests differ from the checkpoint")
        if checkpoint.get("source_sha256") != source_hashes:
            raise ValueError("Resume source code differs from the checkpoint; start a new experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, history = checkpoint["epoch"], checkpoint["best_val_map"], checkpoint["history"]
        _restore_rng(checkpoint["rng"], generator)
    metadata = {"schema_version": 1, "config": config, "manifest_sha256": manifests,
                "source_sha256": source_hashes,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "cuda": torch.version.cuda,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "classes": CLASSES, "parameters": sum(p.numel() for p in model.parameters()),
                "grid_audits": [audit_manifest(x.manifest) for x in (train_set, val_set)],
                "resumed_from": str(resume) if resume else None,
                "status": "running", "completed_epochs": start}
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    out.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (out / f"attempt_{attempt:03d}.json").exists():
        attempt += 1
    metadata["attempt"] = attempt
    atomic_json(out / f"attempt_{attempt:03d}.json", {**metadata, "record_kind": "execution_start"})
    (out / f"requirements-freeze-{attempt:03d}.txt").write_text(freeze, encoding="utf-8")
    atomic_json(out / "run.json", metadata)
    if not (out / "requirements-freeze.txt").exists():
        (out / "requirements-freeze.txt").write_text(freeze, encoding="utf-8")
    total_epochs = t["epochs"]
    try:
        for epoch in range(start, min(total_epochs, stop_after_epoch or total_epochs)):
            warmup = t["warmup_epochs"]
            if epoch < warmup:
                multiplier = (epoch + 1) / warmup
            else:
                progress = (epoch - warmup) / max(1, total_epochs - warmup - 1)
                multiplier = t["minimum_lr_ratio"] + (1 - t["minimum_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group["lr"] = t["learning_rate"] * multiplier
            began, sum_loss, examples, skipped_updates = time.perf_counter(), 0.0, 0, 0
            model.train()
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    predictions = model(images)
                # The square-root coordinate loss stays float32 under AMP.
                loss = loss_fn(predictions.float(), targets.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in epoch {epoch + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=not amp)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skipped_updates += int(scaler.get_scale() < previous_scale)
                sum_loss += float(loss.detach()) * len(images); examples += len(images)
            metrics, _ = evaluate(model, val_loader, device, **eval_kwargs(config))
            score = metrics["map"]
            if score is None:
                raise ValueError("Validation has no non-difficult ground truths")
            row = {"epoch": epoch + 1, "train_loss_per_image": sum_loss / examples,
                   "val": metrics, "lr": optimizer.param_groups[0]["lr"],
                   "amp_skipped_updates": skipped_updates,
                   "seconds": time.perf_counter() - began}
            history.append(row)
            improved = score > best
            best = max(best, score)
            checkpoint = {"schema_version": 1, "config": config, "manifest_sha256": manifests,
                          "source_sha256": source_hashes,
                          "train_ids": [r["image_id"] for r in train_set.records],
                          "val_ids": [r["image_id"] for r in val_set.records],
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_val_map": best,
                          "history": history, "rng": _rng_state(generator)}
            if improved:
                atomic_checkpoint(out / "best.pt", checkpoint)
            atomic_checkpoint(out / "last.pt", checkpoint)
            atomic_json(out / "history.json", history)
            metadata["completed_epochs"] = epoch + 1
            atomic_json(out / "run.json", metadata)
            print(json.dumps({"epoch": epoch + 1, "loss": row["train_loss_per_image"], "val_map": score}), flush=True)
        metadata["status"] = "completed" if metadata["completed_epochs"] == total_epochs else "paused_at_epoch_boundary"
    except BaseException as exc:
        metadata.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(out / "run.json", metadata)
    return metadata
