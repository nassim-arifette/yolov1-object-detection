"""Small VOC examples with hand-computed outcomes; no Torch is required."""

import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import average_precision, evaluate_detections, iou, nms
from prepare_voc import prepare
from protocol import audit_manifest, cell_targets, load_manifest, normalized_box
from configuration import validate


def obj(box=(1, 1, 10, 10), cls=0, difficult=False):
    return {"class_id": cls, "bbox": list(box), "difficult": difficult}


def record(image_id="one", objects=None):
    return {"image_id": image_id, "width": 100, "height": 100,
            "objects": [obj()] if objects is None else objects}


def detection(box=(1, 1, 10, 10), score=0.9, image_id="one", cls=0):
    return {"image_id": image_id, "class_id": cls, "score": score, "bbox": list(box)}


@pytest.mark.parametrize("mode", ["voc2007", "integral"])
def test_perfect_detections_have_ap_one(mode):
    result = evaluate_detections([detection()], [record()], num_classes=2, ap_mode=mode)
    assert result["map"] == pytest.approx(1.0)
    assert result["classes_with_ground_truth"] == 1
    assert result["per_class"][0]["tp"] == 1
    assert result["per_class"][1]["ap"] is None


def test_duplicate_after_true_positive_is_false_positive():
    result = evaluate_detections([detection(score=0.9), detection(score=0.8)],
                                 [record()], num_classes=1)
    assert result["per_class"][0]["tp"] == 1
    assert result["per_class"][0]["fp"] == 1
    # A duplicate after full recall is a false positive but cannot lower
    # interpolated AP: precision 1 was already achieved at every recall level.
    assert result["map"] == pytest.approx(1.0)


def test_false_positive_before_true_positive_lowers_ap():
    result = evaluate_detections([
        detection(box=(50, 50, 60, 60), score=0.99), detection(score=0.9)
    ], [record()], num_classes=1)
    assert result["per_class"][0]["fp"] == 1
    assert result["map"] == pytest.approx(0.5)


def test_difficult_detection_is_ignored_not_counted_as_false_positive():
    records = [record(objects=[obj(), obj((50, 50, 60, 60), difficult=True)])]
    result = evaluate_detections([
        detection(box=(50, 50, 60, 60), score=0.99), detection(score=0.9)
    ], records, num_classes=1)
    cls = result["per_class"][0]
    assert (cls["positives"], cls["tp"], cls["fp"], cls["ignored_difficult"]) == (1, 1, 0, 1)
    assert result["map"] == pytest.approx(1.0)


def test_exact_half_iou_is_not_a_voc_true_positive():
    box = (1, 1, 20, 10)
    assert iou(box, (1, 1, 10, 10)) == pytest.approx(0.5)
    result = evaluate_detections([detection(box=box)], [record()], num_classes=1)
    assert result["per_class"][0]["tp"] == 0
    assert result["per_class"][0]["fp"] == 1
    assert result["map"] == 0


def test_eleven_point_ap_differs_from_integral_at_partial_recall():
    # Exactly one of three ground-truth objects is detected without a false
    # positive. Four VOC2007 recall thresholds are reached: 0, .1, .2, .3.
    records = [record(objects=[obj(), obj((20, 20, 30, 30)), obj((40, 40, 50, 50))])]
    discrete = evaluate_detections([detection()], records, num_classes=1, ap_mode="voc2007")
    integral = evaluate_detections([detection()], records, num_classes=1, ap_mode="integral")
    assert discrete["map"] == pytest.approx(4 / 11)
    assert integral["map"] == pytest.approx(1 / 3)


def test_integral_ap_uses_precision_envelope_not_raw_trapezoids():
    # Precision rises after an intervening false positive; interpolate upwards.
    assert average_precision([0.5, 0.5, 1.0], [1.0, 0.5, 2 / 3], "integral") == pytest.approx(5 / 6)


def test_no_predictions_is_zero_ap_when_ground_truth_exists():
    assert evaluate_detections([], [record()], num_classes=1)["map"] == 0


def test_grid_collision_does_not_remove_evaluation_ground_truth():
    source = record(objects=[obj((1, 1, 3, 3)), obj((8, 8, 10, 10)),
                             obj((4, 4, 5, 5), difficult=True)])
    original = copy.deepcopy(source)
    targets, dropped = cell_targets(source, grid_size=7)
    assert len(targets) == 1
    assert dropped == [source["objects"][1]]
    assert source == original
    audit = audit_manifest({"split": "train", "records": [source]})
    assert (audit["objects"], audit["difficult"], audit["non_difficult_dropped_by_grid"]) == (3, 1, 1)
    detections = [detection(box=(1, 1, 3, 3)), detection(box=(8, 8, 10, 10))]
    result = evaluate_detections(detections, [source], num_classes=1)
    assert result["per_class"][0]["positives"] == 2
    assert result["per_class"][0]["tp"] == 2
    assert result["map"] == pytest.approx(1.0)


@pytest.mark.parametrize("box, expected", [
    ((1, 1, 100, 100), [0.5, 0.5, 1.0, 1.0]),
    ((1, 1, 1, 1), [0.005, 0.005, 0.01, 0.01]),
    ((100, 100, 100, 100), [0.995, 0.995, 0.01, 0.01]),
])
def test_voc_inclusive_pixels_convert_to_normalized_edges(box, expected):
    assert normalized_box(obj(box), record()) == pytest.approx(expected)


def test_nms_keeps_distinct_classes_and_images():
    boxes = [detection(score=0.9), detection(score=0.8),
             detection(cls=1), detection(image_id="two")]
    kept = nms(boxes)
    assert len(kept) == 3
    assert {(d["image_id"], d["class_id"]) for d in kept} == {("one", 0), ("one", 1), ("two", 0)}


def write_voc_example(root, image_id, objects):
    base = root / "VOC2007"
    for directory in ("Annotations", "JPEGImages", "ImageSets/Main"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    # Preparation checks existence and XML, not image decoding. The fixture
    # deliberately needs neither PIL nor a training dependency.
    (base / "JPEGImages" / f"{image_id}.jpg").write_bytes(b"fixture")
    annotation = ET.Element("annotation")
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "100"
    for value in objects:
        node = ET.SubElement(annotation, "object")
        ET.SubElement(node, "name").text = "aeroplane"
        ET.SubElement(node, "difficult").text = str(int(value["difficult"]))
        box = ET.SubElement(node, "bndbox")
        for key, coordinate in zip(("xmin", "ymin", "xmax", "ymax"), value["bbox"]):
            ET.SubElement(box, key).text = str(coordinate)
    ET.ElementTree(annotation).write(base / "Annotations" / f"{image_id}.xml")


def test_prepare_preserves_first_image_empty_images_and_difficult_flags(tmp_path):
    root = tmp_path / "VOCdevkit"
    write_voc_example(root, "000001", [obj(), obj(difficult=True)])
    write_voc_example(root, "000002", [])
    write_voc_example(root, "000003", [obj()])
    lists = root / "VOC2007/ImageSets/Main"
    (lists / "train.txt").write_text("000001\n000002\n", encoding="utf-8")
    (lists / "val.txt").write_text("000003\n", encoding="utf-8")
    out = tmp_path / "manifests"
    audits = prepare(root, "2007", ["train", "val"], out)
    manifest = load_manifest(out / "train.json")
    assert [r["image_id"] for r in manifest["records"]] == ["2007_000001", "2007_000002"]
    assert manifest["records"][0]["objects"][1]["difficult"] is True
    assert manifest["records"][1]["objects"] == []
    assert len(manifest["records"][0]["annotation_sha256"]) == 64
    assert audits[0]["images"] == 2
    assert audits[0]["difficult"] == 1


def test_prepare_rejects_overlapping_splits_before_writing(tmp_path):
    root = tmp_path / "VOCdevkit"
    write_voc_example(root, "000001", [obj()])
    lists = root / "VOC2007/ImageSets/Main"
    for split in ("train", "val"):
        (lists / f"{split}.txt").write_text("000001\n", encoding="utf-8")
    out = tmp_path / "manifests"
    with pytest.raises(ValueError, match="Split overlap"):
        prepare(root, "2007", ["train", "val"], out)
    assert not out.exists()


def test_prepare_rejects_duplicate_image_ids(tmp_path):
    root = tmp_path / "VOCdevkit"
    write_voc_example(root, "000001", [obj()])
    (root / "VOC2007/ImageSets/Main/train.txt").write_text("000001\n000001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate image IDs"):
        prepare(root, "2007", ["train"], tmp_path / "manifests")


def baseline_config():
    path = Path(__file__).resolve().parents[1] / "configs/voc2007.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", ["learning_rate", "weight_decay", "gradient_clip"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_nonfinite_optimizer_parameters(key, value):
    config = baseline_config()
    config["training"][key] = value
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("section,key", [
    ("model", "num_boxes"), ("model", "hidden_size"),
    ("training", "epochs"), ("training", "batch_size"), ("training", "workers"),
])
def test_config_rejects_fractional_dimensions_and_counts(section, key):
    config = baseline_config()
    config[section][key] += 0.5
    with pytest.raises(ValueError):
        validate(config)


def test_config_rejects_fractional_seed():
    config = baseline_config()
    config["seed"] = 123.5
    with pytest.raises(ValueError):
        validate(config)


def test_checked_in_config_is_valid():
    assert validate(baseline_config())["schema_version"] == 1
