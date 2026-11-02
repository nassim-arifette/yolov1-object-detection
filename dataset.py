"""VOC data: grid targets for training, complete XML records for evaluation."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps
from protocol import cell_targets, load_manifest


def flip_record(record):
    width = record["width"]
    objects = []
    for obj in record["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        objects.append({**obj, "bbox": [width + 1 - x2, y1, width + 1 - x1, y2]})
    return {**record, "objects": objects}


def encode_target(record, S=7, B=2, C=20):
    target = torch.zeros(S, S, C + 5 * B)
    selected, _ = cell_targets(record, S)
    for (row, col), (cls, x, y, w, h) in selected.items():
        target[row, col, cls] = 1
        target[row, col, C] = 1
        target[row, col, C + 1:C + 5] = torch.tensor([x, y, w, h])
    return target


class VOCDataset(Dataset):
    def __init__(self, manifest, voc_root, S=7, B=2, C=20, image_size=448,
                 augmentation="none"):
        if augmentation not in ("none", "flip_color"):
            raise ValueError("augmentation must be none or flip_color")
        self.manifest = load_manifest(manifest)
        self.records = self.manifest["records"]
        self.root = Path(voc_root)
        self.S, self.B, self.C = S, B, C
        self.image_size, self.augmentation = image_size, augmentation

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as source:
            image = source.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(f"Image/XML dimensions differ: {record['image_id']}")
        if self.augmentation == "flip_color":
            if random.random() < 0.5:
                image, record = ImageOps.mirror(image), flip_record(record)
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast):
                image = enhancer(image).enhance(random.uniform(0.7, 1.3))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return tensor, encode_target(record, self.S, self.B, self.C), record["image_id"]
