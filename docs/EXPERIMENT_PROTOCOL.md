# Frozen experiment protocol

This protocol was written before running the revised detector. It describes
planned measurements; it does not report an achieved mAP or a new detector.

## Question and scope

YOLOv1 predicts confidence that combines object presence and localization
quality. Does replacing a constant positive confidence target with the
paper's IoU target change detection AP under otherwise identical conditions?
How does the answer depend on simple augmentation?

Separately, quantify the targets that a one-object-per-cell encoder cannot
represent. This addresses a limitation explicitly discussed by the original
paper: nearby and small objects. The audit exposes that limitation; the
confidence ablation does not remove the grid capacity constraint.

## Data and selection

Use official VOC2007 `train`, `val` and `test` image lists separately. Prepare
all XML annotations once. Save manifest SHA-256 hashes and source XML hashes.
Training chooses the first non-difficult object in XML order in an occupied
cell; additional objects are counted by the collision audit. Difficult objects
are excluded from training targets. Evaluation retains every annotation and
uses the VOC difficult-object ignore rule.

Choose checkpoints by validation VOC2007 mAP@0.5, earliest epoch on ties.
Test is used only after configuration selection. Do not alter the loss,
thresholds or schedule in response to test results. An unexpected failure is
reported and diagnosed on training/validation before a new protocol is made.

## Comparisons

Use seeds 123, 124 and 125 for each condition:

| Condition | Positive confidence target | Augmentation |
|---|---|---|
| A | Constant one | None |
| B | Detached responsible-box IoU | None |
| C | Constant one | Horizontal flip and color changes |
| D | Detached responsible-box IoU | Horizontal flip and color changes |

The `one` condition is a deliberately modified control, not a faithful paper
baseline. All four use the corrected responsibility and no-object masks,
coordinate units, full-annotation evaluation, same architecture, optimizer,
schedule and image order seed. This separates the target intervention from
the broad repair of the historical implementation. Do not compare a new score
against the old training curve as an improvement measurement.

The defaults are 100 epochs, batch size 8, 448px images, a 496-unit head,
dropout 0.5, SGD momentum 0.9, learning rate 0.001 with five warmup epochs and
cosine decay, and no pretraining. These are a defined small-resource protocol,
not the original paper's complete training recipe. Stage the work: synthetic
GPU check, one-epoch validation of the pipeline, then full runs. A truncated
run must retain its completed epoch count and cannot be presented as a
100-epoch result. Runtime and memory are not yet measured.

## Measurements and interpretation

Report per-seed and mean/std VOC2007 11-point mAP@0.5, per-class AP, number of
evaluated images/classes, epochs, selected checkpoint and inference settings.
The optional integral AP is a different metric and must not be silently mixed
with 11-point AP. Count failed runs and explain them; do not keep only the
best seed. Preserve raw detections so matching can be recomputed independently.

The grid audit reports non-difficult objects dropped by the training encoder,
affected images and per-class counts. It is **not a hard recall ceiling**:
both predicted boxes can sometimes match different objects even though only
one object supplied a training target. It also does not establish AP loss
without evaluating the trained model on a defined subset.

The primary effects are paired B−A and D−C for the confidence target, C−A and
D−B for augmentation. Report all effects, including zero or negative ones.
Three seeds describe training variability; they do not by themselves prove
generalization beyond this dataset. A speed/accuracy comparison against newer
YOLO models would require shared images, labels, hardware and preprocessing;
the historical videos do not supply that comparison.
