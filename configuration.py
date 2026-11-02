"""Validate the experiment specification before loading large dependencies."""
import json
import math
from pathlib import Path


def validate(config):
    model, training, evaluation = config["model"], config["training"], config["evaluation"]
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    integers = [("seed", config["seed"])]
    integers += [(key, model[key]) for key in ("split_size", "num_boxes", "num_classes", "hidden_size")]
    integers += [(key, training[key]) for key in ("epochs", "batch_size", "workers", "warmup_epochs")]
    for key, value in integers:
        if type(value) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    floats = [("dropout", model["dropout"])]
    floats += [(key, training[key]) for key in ("learning_rate", "minimum_lr_ratio", "momentum", "weight_decay", "gradient_clip")]
    floats += [(key, evaluation[key]) for key in ("score_floor", "nms_iou")]
    for key, value in floats:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if type(training["amp"]) is not bool:
        raise ValueError("amp must be a boolean")
    if model["split_size"] != 7 or model["num_classes"] != 20 or model["num_boxes"] < 1:
        raise ValueError("VOC backbone requires S=7, C=20, B>=1")
    if model["hidden_size"] < 1 or not 0 <= model["dropout"] < 1:
        raise ValueError("Invalid head configuration")
    if min(training["epochs"], training["batch_size"]) < 1 or training["workers"] < 0:
        raise ValueError("epochs/batch_size must be positive, workers nonnegative")
    if training["augmentation"] not in ("none", "flip_color") or training["confidence_target"] not in ("iou", "one"):
        raise ValueError("Unknown ablation setting")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0 or training["gradient_clip"] <= 0:
        raise ValueError("Invalid optimizer configuration")
    if not 0 <= training["warmup_epochs"] < training["epochs"]:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if not 0 <= training["momentum"] < 1 or not 0 < training["minimum_lr_ratio"] <= 1:
        raise ValueError("Invalid momentum/minimum LR")
    if not 0 <= evaluation["score_floor"] < 1 or not 0 < evaluation["nms_iou"] <= 1:
        raise ValueError("Invalid evaluation thresholds")
    if evaluation["ap_mode"] not in ("voc2007", "integral"):
        raise ValueError("Unknown AP convention")
    return config


def load_config(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))
