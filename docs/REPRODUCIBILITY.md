# Reproduction and remaining validation

Run commands from this repository's root. The dependency ranges are not a
historical lockfile. Each actual run saves its package versions and source
hashes. No revised training checkpoint or mAP has been produced locally.

## Local checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

The local environment on 1 November 2026 had Python 3.11.0, NumPy, Pillow and
pytest but no PyTorch. CPU tests cover VOC matching/AP, difficult objects,
duplicates, full-annotation preservation, grid collisions, split separation,
archive extraction and configuration validation. The Torch test module is
explicitly skipped when Torch is unavailable; that is not a full-suite pass.
It checks loss gradients, IoU targets, unused predictors, decoding, target
units and checkpoint resumption once the dependency is present.

Latest local result: **40 CPU tests passed; one module containing seven Torch
tests was skipped**. Nine JavaScript assertions also passed for the site's
synthetic IoU, NMS and grid examples. Python compilation and JavaScript syntax
checks passed. No real VOC archive was downloaded or benchmark inference run
during these checks.

## Colab from the command line

Attach a GPU runtime using your Colab CLI, upload/clone this standalone
repository, and run inside that runtime:

```bash
nvidia-smi
python -m pip install -r requirements-test.txt
python -m pytest tests -q
python gpu_smoke.py --out outputs/gpu_smoke.json
python download_voc.py --data-dir data
```

Colab usually supplies a CUDA-enabled Torch build. If the test environment
instead has a CPU build, install the matching CUDA build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). GPU
commands fail if CUDA is unavailable; there is no silent CPU fallback.
`gpu_smoke.py` uses synthetic images, executes the actual backbone and one
optimization step, and records memory and device. Its loss is not a benchmark
measurement. Neither this smoke check nor real-data training has been run
in the current local verification.

The downloader uses official VOC2007 archives over HTTPS, verifies the known
MD5 values published in
[torchvision's VOC source](https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py),
records SHA-256 hashes and rejects archive paths outside `VOCdevkit/` or links.
If Oxford is unavailable, it falls back to the
[Pascal VOC archive mirror on Hugging Face](https://huggingface.co/datasets/DerrickUnleashed/Pascal_VOC/tree/dd873d3cf7b375ce99012a6c4d64729ab0d76602),
pinned to a commit. Both official MD5 and additional SHA-256 values must match:
the fallback changes the transfer host, not the archive or experiment protocol.
Receipts record the actual source URL, failed attempts and reuse of verified
cache files; a cache without known provenance is not attributed to Oxford.
It preserves XML, images, archives and official split files. Dataset access
conditions remain those of [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

If the data is already available:

```bash
python prepare_voc.py --voc-root data/VOCdevkit --out data/manifests
python preflight.py --manifests data/manifests/train.json data/manifests/val.json data/manifests/test.json --voc-root data/VOCdevkit --out outputs/preflight.json
```

Run one complete epoch to verify real-data loading, training, validation and
checkpointing while preserving the planned 100-epoch schedule:

```bash
python train.py --voc-root data/VOCdevkit --out outputs/iou_aug_seed123 --stop-after-epoch 1
```

Continue exactly that run:

```bash
python train.py --voc-root data/VOCdevkit --out outputs/iou_aug_seed123 --resume outputs/iou_aug_seed123/last.pt
```

Keep the original output directory, `best.pt` and `last.pt` together in
persistent storage. A dropped Colab runtime loses `/content`; download the
directory or use a persistent mounted path as `--out`. The CLI does not
silently allocate runtimes or mount Drive. Resume requires the same source,
configuration and manifest hashes. It restores optimizer, AMP scaler, random
states and loader generator at an epoch boundary. It is not recovery within
an unfinished epoch, and bitwise identity across hardware/library versions
is not claimed.

## Packaged GPU acceptance runner

To transfer the current local revision without first publishing it:

```bash
python tools/package_colab.py
```

Upload `colab_artifacts/yolov1_colab.tar.gz` to the GPU runtime, retain the
adjacent `package_manifest.json` locally, and extract the archive into a fresh
project directory. The archive embeds `colab_source_manifest.json`; the runner
verifies every listed source hash before executing code. The bundle includes
runtime source, configurations, tests and documentation, excluding datasets,
checkpoints, outputs and website assets.

Inside that extracted project:

```bash
python tools/colab_acceptance.py --install-deps \
  --data-dir /content/yolo_data \
  --out /content/yolo_outputs/acceptance_01 \
  --train-out /content/yolo_outputs/iou_aug_seed123
```

This checks CUDA, executes the complete Python suite without skipped tests,
runs the synthetic GPU smoke, downloads and verifies VOC2007, performs the
preflight, trains to epoch one, then verifies resumption by advancing to epoch
two. Both calls preserve the planned 100-epoch schedule. `--skip-resume-check`
stops after epoch one. A passed acceptance is not a completed training study.

The acceptance directory retains `acceptance.json`, one log per step,
`pytest.xml`, the smoke receipt and preflight. Failures set a failed status.
Choose a new acceptance directory for each attempt. If training already has
a checkpoint, resume it directly rather than rerunning this first-run wrapper.

The exact continuation command is stored in `acceptance.json`. For the paths
above it is:

```bash
python train.py --voc-root /content/yolo_data/VOCdevkit \
  --train-manifest /content/yolo_data/manifests/train.json \
  --val-manifest /content/yolo_data/manifests/val.json \
  --out /content/yolo_outputs/iou_aug_seed123 \
  --resume /content/yolo_outputs/iou_aug_seed123/last.pt
```

Use an already mounted persistent directory for both output paths when possible;
the runner does not mount Drive or allocate runtimes. Otherwise retrieve logs,
the full training directory, source package and data manifests before releasing
the VM. Never overwrite the packaged source between an epoch and its resumption.

## Ablations and final evaluation

Use a fresh directory for each seed/condition. For example:

```bash
python train.py --voc-root data/VOCdevkit --out outputs/one_noaug_seed123 --seed 123 --confidence-target one --augmentation none
python train.py --voc-root data/VOCdevkit --out outputs/iou_noaug_seed123 --seed 123 --confidence-target iou --augmentation none
```

Repeat the four protocol conditions at seeds 123, 124 and 125 only after the
smoke and pipeline checks succeed. Compare the fixed-budget runs; do not
silently reduce epochs for one condition to fit an expiring runtime.

```bash
python evaluate.py --checkpoint outputs/iou_aug_seed123/best.pt --manifest data/manifests/test.json --voc-root data/VOCdevkit --out outputs/iou_aug_seed123_test
```

The evaluator saves `metrics.json` and `detections.json` with the checkpoint
and manifest hashes. Old checkpoints lacking the revised schema are rejected:
coordinate units, training provenance and objective differ. Do not reinterpret
historical weights as models trained under this protocol.

## Outputs and completion criteria

`run.json` records runtime/configuration/provenance and completed epochs;
each `attempt_NNN.json` and matching `requirements-freeze-NNN.txt` preserves
the environment at the start of an execution, including each resumption;
`history.json` records losses, validation metrics, learning rates and skipped
AMP updates; `last.pt` permits continuation and `best.pt` holds the best
validation checkpoint. Interrupted runs keep an error and status. No result
should be cited from a run merely because its output directory exists.

The code/protocol work is reviewable now. Scientific completion still requires
a passing Torch suite and GPU smoke, a real VOC preflight, full runs and
held-out evaluation. Report zero or negative ablation effects as results;
the experiment does not require a performance gain to be informative.
