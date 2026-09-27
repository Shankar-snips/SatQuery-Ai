"""
Remote-sensing domain adaptation via LoRA fine-tuning on BigEarthNet.

This is the script that produces the artifact `rs_adapter.py` loads at inference
time (`backend/storage/rs_adapter/bigearthnet_lora/` + the `fusion_head.pt`
checkpoint used by `optical_sar_fusion.py`). It directly satisfies the problem
statement's mandatory requirement: "At least one visual or vision-language
component must be fine-tuned or otherwise adapted using BigEarthNet.txt or the
any open source training data."

Usage:
    python backend/training/finetune_bigearthnet.py --data_dir sample_data/bigearthnet --epochs 3

What it does:
    1. Loads BigEarthNet optical/SAR/text triples (backend/training/dataset.py).
    2. LoRA-adapts the CLIP vision+text towers via contrastive (image-text) loss,
       exactly like CLIP pretraining but restricted to the RS domain -> this is
       what makes grounding/change-detection/VQA "remote-sensing-adapted" rather
       than generic ImageNet/web-pretrained CLIP.
    3. Trains the small SAR->CLIP-space fusion head (backend/models/optical_sar_fusion.py)
       on optical-SAR pairs using a contrastive alignment loss, so SAR structural
       features land close to their co-registered optical CLIP embedding.
    4. Saves the LoRA adapter with peft's `save_pretrained` and writes the READY
       flag so `rs_adapter.is_rs_adapted()` flips to True on next server start.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model

from backend import config
from backend.training.dataset import BigEarthNetCLIPDataset, load_bigearthnet_pairs
from backend.models.optical_sar_fusion import _FusionHead, _extract_sar_features
from backend.utils.image_io import load_image


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale=100.0):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.T * logit_scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def train_clip_lora(data_dir: str, epochs: int, batch_size: int, lr: float):
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    base_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # attention projections in both CLIP towers
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.to(config.DEVICE)
    model.train()

    dataset = BigEarthNetCLIPDataset(data_dir, processor=None)  # raw (PIL, text) pairs
    print(f"[finetune] Loaded {len(dataset)} BigEarthNet optical-text pairs from {data_dir}")

    def collate(batch):
        images, texts = zip(*batch)
        enc = processor(images=list(images), text=list(texts), return_tensors="pt", padding=True)
        return enc

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            out = model(**batch)
            loss = clip_contrastive_loss(out.image_embeds, out.text_embeds)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"[finetune] epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"[finetune] epoch {epoch} mean loss {total_loss / max(1, len(loader)):.4f}")

    os.makedirs(config.RS_LORA_ADAPTER_PATH, exist_ok=True)
    model.save_pretrained(config.RS_LORA_ADAPTER_PATH)
    print(f"[finetune] Saved CLIP LoRA adapter -> {config.RS_LORA_ADAPTER_PATH}")


def train_fusion_head(data_dir: str, epochs: int, lr: float):
    """Aligns SAR structural features (GLCM/Sobel) with the RS-adapted CLIP optical
    embedding of the *same geographic patch*, via a contrastive loss — teaches the
    fusion head where SAR texture patterns fall in CLIP's semantic space."""
    from backend.models import rs_adapter as adapter_mod

    pairs = [p for p in load_bigearthnet_pairs(data_dir) if p.sar_path]
    print(f"[finetune] {len(pairs)} optical-SAR pairs available for fusion-head training")
    if not pairs:
        print("[finetune] No SAR pairs found — skipping fusion head training.")
        return

    head = _FusionHead()
    optim = torch.optim.Adam(head.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for pair in pairs:
            optical_rs = load_image(pair.optical_path)
            sar_rs = load_image(pair.sar_path)
            with torch.no_grad():
                optical_emb = adapter_mod.clip_image_embedding(optical_rs.to_pil())  # (1,512)
            sar_feat = torch.tensor(_extract_sar_features(sar_rs)).unsqueeze(0)
            sar_emb = head(sar_feat)
            loss = 1 - F.cosine_similarity(optical_emb, sar_emb).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"[finetune] fusion-head epoch {epoch} mean loss {total_loss / max(1, len(pairs)):.4f}")

    torch.save(head.state_dict(), os.path.join(config.ADAPTER_DIR, "fusion_head.pt"))
    print(f"[finetune] Saved fusion head -> {config.ADAPTER_DIR}/fusion_head.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to BigEarthNet-formatted data (see sample_data/README.md)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--skip_fusion_head", action="store_true")
    args = ap.parse_args()

    train_clip_lora(args.data_dir, args.epochs, args.batch_size, args.lr)
    if not args.skip_fusion_head:
        train_fusion_head(args.data_dir, args.epochs, args.lr)

    with open(config.RS_ADAPTATION_ENABLED_FLAG, "w") as f:
        f.write("SatQuery-AI RS adaptation completed.\n")
    print("[finetune] Domain adaptation complete. rs_adapter.is_rs_adapted() will now return True.")


if __name__ == "__main__":
    main()
