"""VOC manifests and target geometry, independent of the training framework."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_annotation(path, image_id, image):
    node = ET.parse(path).getroot()
    width = int(node.findtext("size/width"))
    height = int(node.findtext("size/height"))
    if min(width, height) <= 0:
        raise ValueError(f"{path}: invalid image size")
    objects = []
    for obj in node.findall("object"):
        name = obj.findtext("name")
        if name not in CLASSES:
            raise ValueError(f"{path}: unknown class {name!r}")
        box = [float(obj.findtext(f"bndbox/{key}"))
               for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = box
        if not (all(math.isfinite(x) for x in box) and
                1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
            raise ValueError(f"{path}: invalid VOC box {box}")
        objects.append({"class_id": CLASSES.index(name), "bbox": box,
                        "difficult": bool(int(obj.findtext("difficult", "0")))})
    return {"image_id": image_id, "image": image, "width": width, "height": height,
            "annotation_sha256": sha256(path), "objects": objects}


def normalized_box(obj, record):
    """VOC 1-based inclusive pixels -> continuous normalized xywh edges."""
    x1, y1, x2, y2 = obj["bbox"]
    w, h = record["width"], record["height"]
    return [(x1 - 1 + x2) / (2 * w), (y1 - 1 + y2) / (2 * h),
            (x2 - x1 + 1) / w, (y2 - y1 + 1) / h]


def cell_targets(record, grid_size=7):
    """First non-difficult object in XML order wins; report every collision."""
    if not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("Grid size must be a positive integer")
    selected, dropped = {}, []
    for obj in record["objects"]:
        if obj["difficult"]:
            continue
        x, y, w, h = normalized_box(obj, record)
        row, col = min(grid_size - 1, int(y * grid_size)), min(grid_size - 1, int(x * grid_size))
        if (row, col) in selected:
            dropped.append(obj)
        else:
            selected[row, col] = [obj["class_id"], x * grid_size - col,
                                  y * grid_size - row, w, h]
    return selected, dropped


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("classes") != list(CLASSES):
        raise ValueError(f"{path}: unsupported manifest schema/classes")
    ids = [r["image_id"] for r in data["records"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: empty split or duplicate image IDs")
    return data


def require_disjoint(*manifests):
    seen = set()
    for manifest in manifests:
        ids = {r["image_id"] for r in manifest["records"]}
        if seen & ids:
            raise ValueError(f"Split overlap: {sorted(seen & ids)[:5]}")
        seen.update(ids)


def audit_manifest(data, grid_size=7):
    records = data["records"]
    total = difficult = collisions = affected = 0
    by_class = {name: {"objects": 0, "dropped_by_grid": 0} for name in CLASSES}
    for record in records:
        _, dropped = cell_targets(record, grid_size)
        total += len(record["objects"])
        difficult += sum(o["difficult"] for o in record["objects"])
        collisions += len(dropped)
        affected += bool(dropped)
        for obj in record["objects"]:
            if not obj["difficult"]:
                by_class[CLASSES[obj["class_id"]]]["objects"] += 1
        for obj in dropped:
            by_class[CLASSES[obj["class_id"]]]["dropped_by_grid"] += 1
    return {"split": data["split"], "images": len(records), "objects": total,
            "difficult": difficult, "grid_size": grid_size,
            "non_difficult_dropped_by_grid": collisions, "images_with_collisions": affected,
            "by_class": by_class}
