"""Exercise the actual 448px backbone/loss/backward on synthetic data; no mAP claim."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/gpu_smoke.json"))
    args = parser.parse_args()
    import torch
    from configuration import load_config
    from dataset import encode_target
    from loss import YoloLoss
    from model import Yolov1
    from runtime import atomic_json, resolve_device
    from utils import decode_predictions
    device = resolve_device("cuda")
    torch.manual_seed(123)
    config = load_config("configs/voc2007.json")
    model = Yolov1(**config["model"]).to(device)
    record = {"image_id": "synthetic", "width": 448, "height": 448,
              "objects": [{"class_id": 0, "bbox": [100, 100, 220, 240], "difficult": False}]}
    images = torch.rand(2, 3, 448, 448, device=device)
    target = encode_target(record).unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    predictions = model(images)
    loss = YoloLoss()(predictions, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite synthetic loss")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite gradient")
    optimizer.step()
    decode_predictions(predictions, [record, {**record, "image_id": "synthetic2"}])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, {"status": "passed", "synthetic": True, "benchmark_result": False,
                          "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(device),
                          "loss": float(loss.detach()), "output_shape": list(predictions.shape),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    print(f"Synthetic GPU smoke passed; receipt: {args.out}")


if __name__ == "__main__":
    main()
