# One object per cell

A from-scratch YOLOv1 on Pascal VOC 2007, rebuilt so its numbers can be trusted. It looks at what the 7×7 grid costs and at what the confidence score should learn.

[**Open the interactive article →**](https://nassim-arifette.github.io/yolov1-object-detection/)

[![The website's hero: a VOC image with the 7×7 grid; a rider and her horse share a cell, so the rider gets no training target](docs/assets/site-hero.png)](https://nassim-arifette.github.io/yolov1-object-detection/)

## What it measures

### The grid's blind spot

YOLOv1 assigns each object to the cell that contains its centre, and a cell describes at most one object. If a second object lands in a cell that is already taken, it never becomes a training target. Over the official VOC 2007 train, validation and test lists, that happens to 1,745 of 24,640 non-difficult objects (7.1 %), in 1,141 of 9,963 images.

Some classes pay much more than others. 15.8 % of bottles and 11.0 % of people lose their cell, against 0.1 % of cats and not a single train. The audit only reads the annotations. It is not an AP loss, and it is not a hard recall ceiling either.

### What confidence should learn

In the paper, each box's confidence is trained towards its IoU with the object. A simpler control trains it towards 1. I compare the two with a frozen 2×2 protocol (confidence target × augmentation, three seeds each), everything else held equal. **Training is still in progress, and no new mAP is reported yet.**

| Condition | Confidence target | Augmentation |
|---|---|---|
| A | constant 1 | none |
| B | detached responsible-box IoU | none |
| C | constant 1 | horizontal flip and colour changes |
| D | detached responsible-box IoU | horizontal flip and colour changes |

The target effect is read from B − A and D − C, the augmentation effect from C − A and D − B. Every effect is reported with its seeds, even when it is zero or negative. The full protocol is in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## What you can do on the site

| Figure | Try this |
| --- | --- |
| What a cell has to say | Hover any cell of a real VOC image and see the exact training target it receives. |
| The grid's blind spot | The per-class share of objects left without a target, by split. |
| What confidence should learn | Drag a predicted box against a real annotation and compare the two confidence targets. |
| Non-maximum suppression | Move the threshold and watch class-specific suppression. |

The grid encoder, IoU and NMS on the site are TypeScript ports of the Python code in this repository. Tests check that they drop exactly the same objects on all 4,952 test images.

## What was fixed in the implementation

- Confidence now learns the detached IoU of the responsible box. That box is the predictor with the higher IoU; every other predictor gets the no-object penalty. A constant target is still available as a controlled ablation.
- Coordinates follow the paper, with centre offsets inside the cell and width and height in image units.
- At inference both boxes are kept. Detections are ranked by class score × confidence, then filtered with class-specific NMS.
- Evaluation reads every XML object, including the ones the training encoder drops. Difficult objects are ignored as in VOC, duplicates count as false positives, and AP is VOC 2007's 11-point interpolated version.
- Train, validation and test stay separate: the model trains on `train`, checkpoints are picked on `val`, and `test` is evaluated once. Manifests, hashes, settings, per-class AP and resumable checkpoints are all saved.

The network trains from scratch with BatchNorm and a 496-unit head, where the paper used ImageNet pretraining and a 4,096-unit head. This is a study on a limited compute budget, **not an exact reproduction of the paper's published mAP**.

## Run it

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python gpu_smoke.py
python download_voc.py --data-dir data          # verified VOC2007 archives, manifests, grid audit
python train.py --voc-root data/VOCdevkit --out outputs/iou_aug_seed123
python evaluate.py --checkpoint outputs/iou_aug_seed123/best.pt --manifest data/manifests/test.json --voc-root data/VOCdevkit --out outputs/iou_aug_seed123_test
```

Training needs a CUDA build of PyTorch. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) covers Colab, a one-epoch pipeline check and exact resumption. [`docs/LITERATURE.md`](docs/LITERATURE.md) lists what the paper and the authors' code actually specify.

## Repository layout

| Path | Role |
|---|---|
| `protocol.py`, `prepare_voc.py`, `preflight.py` | Frozen splits, full annotations and the grid-collision audit |
| `dataset.py`, `model.py`, `loss.py` | Image and target processing, network and objective |
| `metrics.py`, `utils.py` | VOC AP, class-specific NMS and prediction decoding |
| `train.py`, `runtime.py`, `evaluate.py` | Training, recovery and held-out evaluation |
| `configs/voc2007.json` | Recorded experiment defaults |
| `gpu_smoke.py`, `tests/` | Synthetic execution check and invariants |
| `web/` | Source of the website (Vite + TypeScript, D3). See [`docs/SITE.md`](docs/SITE.md) |

The repository contains no dataset, model weights or credentials.

## Historical material

The [poster](Poster_YoloV1.png), [report](yolo.pdf) and [figures](figures/) from the original course project are kept for the record. Their curves are training diagnostics, produced by code that has since been fixed, and they say nothing about generalisation. The videos comparing it with newer YOLO versions are only qualitative, so I removed the earlier claim that YOLOv3 was the best model.

Nassim Arifette, PhD student at École polytechnique. Master MVA, ENS Paris-Saclay.
