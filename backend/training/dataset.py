"""
BigEarthNet-style dataset loader for remote-sensing domain adaptation.

Expects a directory structure of:
    data_dir/
        optical/  *.tif   (Sentinel-2 multispectral patches)
        sar/      *.tif   (Sentinel-1 SAR patches, same patch IDs as optical/)
        labels.csv        (columns: patch_id, caption_or_labels)

`labels.csv` may hold either free-text captions (for LoRA VLM fine-tuning) or
comma-separated BigEarthNet land-cover class labels (auto-converted into a
templated caption, e.g. "urban,water" -> "An area containing urban and water
land cover."). This mirrors how BigEarthNet.txt (referenced in the problem
statement) pairs imagery with textual annotations.
"""
import os
import csv
from dataclasses import dataclass
from typing import List

from PIL import Image
from torch.utils.data import Dataset

from backend.utils.image_io import load_image


@dataclass
class RSTextPair:
    optical_path: str
    sar_path: str
    text: str


def _labels_to_caption(raw_label: str) -> str:
    if "." in raw_label or " " in raw_label:
        return raw_label  # already free text
    classes = [c.strip().replace("_", " ") for c in raw_label.split(",") if c.strip()]
    if not classes:
        return "A remote sensing image."
    if len(classes) == 1:
        return f"An area containing {classes[0]} land cover."
    return f"An area containing {', '.join(classes[:-1])} and {classes[-1]} land cover."


def load_bigearthnet_pairs(data_dir: str) -> List[RSTextPair]:
    labels_csv = os.path.join(data_dir, "labels.csv")
    optical_dir = os.path.join(data_dir, "optical")
    sar_dir = os.path.join(data_dir, "sar")
    pairs = []
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f"Expected {labels_csv} — see sample_data/README.md for the required BigEarthNet layout."
        )
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patch_id"]
            text = _labels_to_caption(row.get("caption_or_labels", ""))
            opt_path = os.path.join(optical_dir, f"{pid}.tif")
            sar_path = os.path.join(sar_dir, f"{pid}.tif")
            if os.path.exists(opt_path):
                pairs.append(RSTextPair(opt_path, sar_path if os.path.exists(sar_path) else "", text))
    return pairs


class BigEarthNetCLIPDataset(Dataset):
    """Yields (PIL image, caption) pairs for CLIP/BLIP LoRA fine-tuning. Uses the
    optical member of each pair (SAR patches feed the separate fusion-head training
    routine in finetune_bigearthnet.py::train_fusion_head)."""

    def __init__(self, data_dir: str, processor=None):
        self.pairs = load_bigearthnet_pairs(data_dir)
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        rs_img = load_image(pair.optical_path)
        pil_img = rs_img.to_pil()
        if self.processor is not None:
            enc = self.processor(images=pil_img, text=pair.text, return_tensors="pt",
                                  padding="max_length", truncation=True)
            return {k: v.squeeze(0) for k, v in enc.items()}
        return pil_img, pair.text
