"""VOC detection matching and interpolated AP; no Torch or image dependency."""
from __future__ import annotations
import math


def iou(a, b):
    """VOC's 1-based inclusive pixel boxes."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    area_a = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1)
    area_b = max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(recall, precision, mode="voc2007"):
    if mode == "voc2007":
        return sum(max((p for r, p in zip(recall, precision) if r >= t / 10),
                       default=0.0) for t in range(11)) / 11
    if mode != "integral":
        raise ValueError(f"Unknown AP mode: {mode}")
    r, p = [0.0] + list(recall) + [1.0], [0.0] + list(precision) + [0.0]
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    return sum((r[i] - r[i - 1]) * p[i] for i in range(1, len(r)) if r[i] != r[i - 1])


def evaluate_detections(detections, records, num_classes=20, ap_mode="voc2007", threshold=0.5):
    if not 0 < threshold <= 1 or num_classes < 1 or ap_mode not in ("voc2007", "integral"):
        raise ValueError("Invalid evaluation configuration")
    known = {r["image_id"] for r in records}
    for d in detections:
        box = d["bbox"]
        if (d["image_id"] not in known or not 0 <= d["class_id"] < num_classes or
                len(box) != 4 or not all(math.isfinite(x) for x in box + [d["score"]]) or
                box[2] < box[0] or box[3] < box[1]):
            raise ValueError(f"Invalid detection: {d}")
    results = []
    for cls in range(num_classes):
        gt = {r["image_id"]: [o for o in r["objects"] if o["class_id"] == cls] for r in records}
        positives = sum(not o["difficult"] for objects in gt.values() for o in objects)
        seen, tp, fp, ignored = set(), [], [], 0
        ranked = sorted((d for d in detections if d["class_id"] == cls), key=lambda d: -d["score"])
        for d in ranked:
            objects = gt[d["image_id"]]
            overlaps = [iou(d["bbox"], obj["bbox"]) for obj in objects]
            best = max(range(len(objects)), key=lambda i: overlaps[i]) if objects else None
            if best is not None and overlaps[best] > threshold:
                if objects[best]["difficult"]:
                    ignored += 1
                    continue
                key = (d["image_id"], best)
                hit = key not in seen
                seen.add(key)
            else:
                hit = False
            tp.append(int(hit)); fp.append(int(not hit))
        recall, precision, true, false = [], [], 0, 0
        for t, f in zip(tp, fp):
            true += t; false += f
            recall.append(true / positives if positives else 0.0)
            precision.append(true / (true + false))
        results.append({"class_id": cls, "positives": positives, "detections": len(ranked),
                        "ignored_difficult": ignored, "tp": true, "fp": false,
                        "ap": average_precision(recall, precision, ap_mode) if positives else None})
    aps = [r["ap"] for r in results if r["ap"] is not None]
    return {"ap_mode": ap_mode, "iou_threshold": threshold, "iou_comparison": ">",
            "map": sum(aps) / len(aps) if aps else None,
            "classes_with_ground_truth": len(aps), "per_class": results}


def nms(detections, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("NMS threshold must be in (0, 1]")
    kept = []
    groups = {}
    for d in detections:
        groups.setdefault((d["image_id"], d["class_id"]), []).append(d)
    for group in groups.values():
        pending = sorted(group, key=lambda d: -d["score"])
        while pending:
            chosen = pending.pop(0)
            kept.append(chosen)
            pending = [d for d in pending if iou(chosen["bbox"], d["bbox"]) <= threshold]
    return kept
