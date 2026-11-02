"""Inference uses both box predictors and class-specific confidence scores."""
import torch
from metrics import evaluate_detections, nms


def decode_predictions(predictions, records, S=7, B=2, C=20, score_floor=0.001):
    values = predictions.detach().float().cpu().reshape(-1, S, S, C + 5 * B)
    if len(values) != len(records):
        raise ValueError("Prediction/record batch mismatch")
    if not torch.isfinite(values).all():
        raise ValueError("Model produced non-finite predictions")
    detections = []
    for grid, record in zip(values, records):
        width, height = record["width"], record["height"]
        for row in range(S):
            for col in range(S):
                cell = grid[row, col]
                probabilities = cell[:C].clamp_min(0)
                for box in cell[C:].reshape(B, 5):
                    confidence, x, y, w, h = box.tolist()
                    if confidence <= 0 or w <= 0 or h <= 0:
                        continue
                    x, y = (x + col) / S, (y + row) / S
                    coords = [max(1.0, (x - w / 2) * width + 1),
                              max(1.0, (y - h / 2) * height + 1),
                              min(float(width), (x + w / 2) * width),
                              min(float(height), (y + h / 2) * height)]
                    if coords[2] < coords[0] or coords[3] < coords[1]:
                        continue
                    for cls, prob in enumerate(probabilities.tolist()):
                        score = prob * confidence
                        if score >= score_floor:
                            detections.append({"image_id": record["image_id"], "class_id": cls,
                                               "score": score, "bbox": coords})
    return detections


@torch.no_grad()
def evaluate(model, loader, device, S=7, B=2, C=20, score_floor=0.001,
             nms_iou=0.5, ap_mode="voc2007"):
    previous_mode = model.training
    lookup = {r["image_id"]: r for r in loader.dataset.records}
    detections, visited = [], []
    model.eval()
    try:
        for images, _, ids in loader:
            records = [lookup[i] for i in ids]
            predictions = model(images.to(device))
            decoded = decode_predictions(predictions, records, S, B, C, score_floor)
            detections.extend(nms(decoded, nms_iou))
            visited.extend(ids)
    finally:
        model.train(previous_mode)
    if len(visited) != len(set(visited)) or set(visited) != set(lookup):
        raise ValueError("Evaluation loader did not visit every image exactly once")
    metrics = evaluate_detections(detections, loader.dataset.records, C, ap_mode)
    metrics.update({"images": len(visited), "score_floor": score_floor, "nms_iou": nms_iou})
    return metrics, detections
