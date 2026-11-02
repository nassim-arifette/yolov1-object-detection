# YOLOv1: primary sources and a bounded experiment

Checked against primary sources on 1 November 2026. This note defines what
the implementation should measure; it reports no new training result.

## What the original paper specifies

| Component | Specification |
|---|---|
| Prediction | A 7 x 7 grid, two boxes per cell, 20 shared conditional class scores; linear output |
| Box coordinates | Centre offsets within the cell; width and height relative to the image |
| Responsibility | The predictor with the highest current IoU receives the object's coordinate loss |
| Confidence | Target equals the responsible prediction's IoU; test-time class score multiplies confidence by conditional class score |
| Geometry loss | Squared errors, with square roots of width/height; coordinate weight 5, no-object weight 0.5 |
| Training | ImageNet pretraining of the first 20 convolutional layers; 24 convolutional and two connected layers for detection |
| Regularization | Dropout 0.5; scaling/translation up to 20%; exposure/saturation factors up to 1.5 |

The paper identifies localization errors and nearby small objects as
limitations. Its grid shares one class distribution, and downsampling yields
coarse features. Its squared-error objective is not identical to AP.
Source: [Redmon et al., CVPR 2016, Sections 2.1-2.4](https://arxiv.org/pdf/1506.02640).

Resize-only preprocessing departs from this recipe. Batch normalization,
a smaller head, Adam and training from scratch are implementation choices,
not established improvements without controlled results.

## Resolve implementation details with the authors' code

In the authors' `forward_detection_layer`, all predictors first receive a
zero-target confidence penalty. The responsible predictor's penalty is then
replaced. Consequently a non-responsible predictor still receives a no-object
penalty when its cell contains an object. With `rescore`, the confidence
gradient uses IoU as a scalar target, without a confidence-induced gradient
through the box coordinates. The implementation also uses box-distance as a
tie-break when all candidate IoUs are zero. These details matter when naming
what is reproduced. [Author implementation](https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c).

The current author repository's `yolov1.cfg` includes batch normalization,
three predictors and a local layer. It therefore differs from the paper's
architecture. Treat this mutable file as supplementary implementation evidence,
not an immutable specification of the 2016 experiment. Record a source commit
when importing code or weights. [Author configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg).

## Evaluation must identify the VOC protocol

VOC2007 uses **11-point interpolated AP**: at recall thresholds
`0, 0.1, ..., 1`, take the maximum precision at any greater or equal recall,
then average those eleven values. This is not the trapezoidal area under raw
precision/recall samples. [VOC authors' retrospective, Section 4.2](https://www.robots.ox.ac.uk/~vgg/publications/2010/Everingham10/everingham10.pdf).

VOC2010 instead integrates the piecewise-constant monotonically interpolated
precision envelope over all recall changes. If both implementations are
reported, name them separately; a different AP convention can change a score
without changing predictions. [VOC2010 development kit, Section 3.4.1](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2010/htmldoc/index.html).

For VOC detection, matching uses IoU **strictly greater than 0.5**. Repeated
detections of an already matched object are false positives; difficult objects
are ignored. Official image lists define the splits, and parameter selection
belongs to train/validation, excluding test. Pixel coordinates originate at
`(1, 1)`; conversion to continuous normalized boxes must be consistent.
[VOC2007 development kit, Sections 4 and 8](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/).

For this project, evaluate against the original annotations, including the
information needed to ignore difficult objects. Decoding a one-object-per-cell
training target as evaluation ground truth would silently remove objects
whose centres collide. That would reward the encoder's omissions. Keep the
full annotation list independently of the training tensor, and report how
many objects the chosen tensor cannot represent.

## Proposed contribution and experiment

The defensible scope is an **implementation fidelity and failure-analysis
study of localization confidence**, followed by an explicitly controlled
ablation. Correcting a loss or AP implementation is necessary engineering,
not a new object-detection method. No claim of improved mAP or resolution of
the original small-object limitation follows until a held-out experiment runs.

1. First validate invariants on synthetic examples: highest-IoU responsibility,
   zero confidence target for non-responsible boxes, finite gradients at empty
   cells, class-specific score ranking, duplicate matching, difficult-object
   handling, and hand-computed AP cases. Inventory annotation collisions per
   grid cell, and count excluded training objects by class and normalized area.
2. Freeze image lists, architecture, initialization, optimizer, schedule,
   number of updates, decoding, NMS and score threshold. Compare a constant-one
   confidence target with an IoU target. Call the former an intentional control
   reproducing the old simplification, not the faithful YOLOv1 baseline. Hold
   the non-responsible-box correction and evaluator fixed in both arms.
3. Run augmentation as a separate factor. A small 2 x 2 design (confidence
   target x augmentation) distinguishes the two effects and their interaction.
   Choose all settings on VOC2007 train/val before the final test comparison.
   Record every attempted seed, including divergent runs, rather than retaining
   only a best run. Three matched seeds is a useful initial budget, not a
   guarantee of statistical precision.
4. Report VOC2007 AP50 and per-class AP, plus a separately named stricter-IoU
   diagnostic and recall by predeclared normalized object-area bins. Archive
   predictions so metrics can be recomputed without rerunning inference.
   Compare loss and calibration only on matching conditions. Confidence plots
   alone do not measure detection accuracy or prove overfitting.

An optional later question is whether increasing grid resolution reduces
annotation collisions and improves crowded-object recall. That changes output
capacity and compute, so it needs its own matched experiment, latency/memory
measurement, and fixed evaluation annotations. It should not be mixed into
the initial confidence-target ablation.

The existing poster, PDF, training curves and illustrative videos are historical
artifacts. Without a recovered held-out prediction archive and complete run
configuration, they cannot establish a reproducible test mAP or comparative
accuracy across YOLO versions.
