"""YOLOv1 responsibility, coordinate and confidence losses."""
import torch
from torch import nn


def box_iou_xywh(a, b):
    """Broadcasted continuous IoU; negative predicted dimensions have zero area."""
    awh, bwh = a[..., 2:].clamp_min(0), b[..., 2:].clamp_min(0)
    low = torch.maximum(a[..., :2] - awh / 2, b[..., :2] - bwh / 2)
    high = torch.minimum(a[..., :2] + awh / 2, b[..., :2] + bwh / 2)
    overlap = (high - low).clamp_min(0).prod(-1)
    union = awh.prod(-1) + bwh.prod(-1) - overlap
    return overlap / union.clamp_min(1e-9)


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, confidence_target="iou",
                 lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        if min(S, B, C) < 1 or confidence_target not in ("iou", "one"):
            raise ValueError("Invalid grid dimensions or confidence target")
        self.S, self.B, self.C = S, B, C
        self.confidence_target = confidence_target
        self.lambda_coord, self.lambda_noobj = lambda_coord, lambda_noobj

    def components(self, predictions, target):
        pred = predictions.reshape(-1, self.S, self.S, self.C + 5 * self.B)
        target = target.reshape_as(pred)
        boxes = pred[..., self.C:].reshape(*pred.shape[:-1], self.B, 5)
        truth = target[..., self.C + 1:self.C + 5]
        exists = target[..., self.C] > 0
        # x/y use cell units; w/h use whole-image units, as in the paper.
        scale = pred.new_tensor([1 / self.S, 1 / self.S, 1, 1])
        overlaps = box_iou_xywh(boxes[..., 1:] * scale, truth.unsqueeze(-2) * scale)
        best = overlaps.detach().argmax(-1)
        responsible = torch.nn.functional.one_hot(best, self.B).bool() & exists.unsqueeze(-1)
        chosen = boxes[..., 1:].gather(-2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
        xy = ((chosen[..., :2] - truth[..., :2]).square() * exists.unsqueeze(-1)).sum()
        pred_wh = torch.sign(chosen[..., 2:]) * torch.sqrt(chosen[..., 2:].abs() + 1e-9)
        true_wh = torch.sqrt(truth[..., 2:].clamp_min(0))
        wh = ((pred_wh - true_wh).square() * exists.unsqueeze(-1)).sum()
        confidence = overlaps.detach() if self.confidence_target == "iou" else torch.ones_like(overlaps)
        object_loss = ((boxes[..., 0] - confidence).square() * responsible).sum()
        no_object_loss = (boxes[..., 0].square() * ~responsible).sum()
        class_loss = ((pred[..., :self.C] - target[..., :self.C]).square() * exists.unsqueeze(-1)).sum()
        return {"coordinate": (xy + wh) / len(pred), "object": object_loss / len(pred),
                "no_object": no_object_loss / len(pred), "class": class_loss / len(pred)}

    def forward(self, predictions, target):
        terms = self.components(predictions, target)
        return (self.lambda_coord * terms["coordinate"] + terms["object"] +
                self.lambda_noobj * terms["no_object"] + terms["class"])
