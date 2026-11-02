"""Autograd, decoding and resume checks; require the training dependency."""
import copy
import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-test.txt for training checks")
from PIL import Image
from configuration import load_config
from dataset import encode_target, flip_record
from loss import YoloLoss
from protocol import CLASSES
from utils import decode_predictions


def test_confidence_is_iou_and_target_is_detached():
    truth = torch.zeros(1, 1, 1, 11)
    truth[..., 0] = 1
    truth[..., 1:6] = torch.tensor([1, 0.5, 0.5, 0.4, 0.4])
    pred = truth.clone()
    pred[..., 2:6] = torch.tensor([0.5, 0.5, 0.2, 0.4])
    pred.requires_grad_()
    term = YoloLoss(S=1, B=2, C=1).components(pred, truth)["object"]
    assert term.item() == pytest.approx(0.25)
    term.backward()
    assert pred.grad[..., 1].item() == pytest.approx(1)
    assert pred.grad[..., 2:6].abs().sum().item() == 0


def test_non_responsible_predictor_in_object_cell_is_penalized():
    truth = torch.zeros(2, 1, 1, 11)
    truth[..., 0] = truth[..., 1] = 1
    truth[..., 2:6] = torch.tensor([0.5, 0.5, 0.4, 0.4])
    pred = truth.clone()
    pred[..., 6] = 0.7
    pred.requires_grad_()
    terms = YoloLoss(S=1, B=2, C=1).components(pred, truth)
    assert terms["no_object"].item() == pytest.approx(0.49)
    loss = YoloLoss(S=1, B=2, C=1)(pred, truth)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_unit_confidence_is_explicit_ablation():
    truth = torch.zeros(1, 1, 1, 6)
    truth[..., 0] = truth[..., 1] = 1
    truth[..., 2:6] = torch.tensor([0.5, 0.5, 0.4, 0.4])
    pred = truth.clone()
    pred[..., 4] = 0.2
    assert YoloLoss(S=1, B=1, C=1, confidence_target="one").components(pred, truth)["object"] == 0
    assert YoloLoss(S=1, B=1, C=1).components(pred, truth)["object"].item() == pytest.approx(0.25)


def test_decoder_keeps_both_boxes_and_multiplies_class_scores():
    pred = torch.tensor([[[[0.25, 0.5, 0.8, 0.2, 0.5, 0.2, 0.2,
                           0.4, 0.8, 0.5, 0.2, 0.2]]]])
    record = {"image_id": "x", "width": 100, "height": 100}
    decoded = decode_predictions(pred, [record], S=1, B=2, C=2)
    assert len(decoded) == 4
    assert [d["score"] for d in decoded] == pytest.approx([0.2, 0.4, 0.1, 0.2])


def test_grid_uses_image_relative_width_and_flip_is_involution():
    record = {"image_id": "x", "width": 100, "height": 100,
              "objects": [{"class_id": 1, "bbox": [11, 21, 30, 40], "difficult": False}]}
    target = encode_target(record)
    selected = target[..., 20] > 0
    assert target[..., 23][selected].item() == pytest.approx(0.2)
    assert flip_record(flip_record(record)) == record


def test_empty_targets_negative_predictions_have_finite_backward():
    pred = torch.full((3, 2, 2, 13), -0.1, requires_grad=True)
    target = torch.zeros_like(pred)
    loss = YoloLoss(S=2, B=2, C=3)(pred, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_epoch_resume_preserves_config_manifests_and_history(tmp_path, monkeypatch):
    import runtime
    config = load_config(Path(__file__).parents[1] / "configs/voc2007.json")
    config["training"].update(epochs=2, warmup_epochs=0, workers=0, batch_size=1, amp=False, augmentation="none")
    class Tiny(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1470))
        def forward(self, x):
            return self.net(x)
    monkeypatch.setattr(runtime, "Yolov1", Tiny)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "test fixture\n")
    paths = []
    for split in ("train", "val"):
        image = tmp_path / f"{split}.jpg"
        Image.new("RGB", (20, 20), "white").save(image)
        record = {"image_id": split, "image": image.name, "width": 20, "height": 20,
                  "objects": [{"class_id": 0, "bbox": [1, 1, 20, 20], "difficult": False}]}
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps({"schema_version": 1, "classes": list(CLASSES), "split": split, "records": [record]}))
        paths.append(path)
    out = tmp_path / "run"
    runtime.train(config, *paths, tmp_path, out, "cpu", stop_after_epoch=1)
    assert json.loads((out / "run.json").read_text())["status"] == "paused_at_epoch_boundary"
    runtime.train(config, *paths, tmp_path, out, "cpu", resume=out / "last.pt")
    assert len(json.loads((out / "history.json").read_text())) == 2
    changed = copy.deepcopy(config)
    changed["training"]["confidence_target"] = "one"
    with pytest.raises(ValueError, match="differ"):
        runtime.train(changed, *paths, tmp_path, out, "cpu", resume=out / "last.pt")
