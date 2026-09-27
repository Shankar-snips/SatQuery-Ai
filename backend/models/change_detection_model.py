"""
Bi-temporal change understanding (mandatory multi-image task).

Real pipeline:
  1. Split both co-registered images into a common patch grid.
  2. Embed every patch pair with the RS-adapted CLIP vision tower (Siamese use of
     one shared backbone, i.e. classic Siamese change-detection architecture).
  3. Cosine-distance each corresponding patch pair -> a per-patch change score.
  4. Threshold -> binary change mask -> spatial change map (the "reference mask"
     the problem statement mentions, generated here without needing ground truth).
  5. Caption the "before" and "after" content of the most-changed region with
     BLIP, and if the query is a yes/no or category question ("has built-up
     area increased?") route it through a closed-set CLIP comparison as well ->
     that becomes change-VQA.
"""
import numpy as np
import cv2
import torch
from PIL import Image

from backend import config
from backend.models import rs_adapter
from backend.models.captioning_model import caption_image
from backend.utils.confidence import cosine_similarity_confidence
from backend.utils.image_io import RSImage


def _grid_patches(w, h, n=6):
    pw, ph = w // n, h // n
    boxes = []
    for j in range(n):
        for i in range(n):
            boxes.append((i * pw, j * ph, (i + 1) * pw, (j + 1) * ph))
    return boxes


@torch.no_grad()
def _embed_patches(pil_img: Image.Image, boxes):
    model, processor = rs_adapter.get_clip()
    patches = [pil_img.crop(b) for b in boxes]
    inputs = processor(images=patches, return_tensors="pt").to(model.device)
    feats = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
    return feats.cpu().numpy()


@torch.no_grad()
def detect_change(image_t1: RSImage, image_t2: RSImage, question: str = None) -> dict:
    img1, img2 = image_t1.to_pil(), image_t2.to_pil()
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    w, h = img1.size

    boxes = _grid_patches(w, h, n=6)
    feats1 = _embed_patches(img1, boxes)
    feats2 = _embed_patches(img2, boxes)

    cos_sim = np.sum(feats1 * feats2, axis=1)          # per-patch cosine similarity
    change_score = 1.0 - cos_sim                        # 0 = identical, 2 = opposite
    changed_mask_flags = change_score > config.CHANGE_DETECTION_DIFF_THRESHOLD

    change_map = _build_change_map(w, h, boxes, change_score)
    overlay = _overlay_change(np.array(img2.convert("RGB")), change_map)

    pct_changed = float(changed_mask_flags.mean() * 100)
    max_change_idx = int(change_score.argmax())
    changed_region_box = boxes[max_change_idx]

    # describe before/after content of the most-changed patch for a grounded explanation
    before_crop = RSImage(path=image_t1.path, array=np.array(img1.crop(changed_region_box)).astype(np.float32) / 255.0,
                           band_count=3, width=0, height=0, is_geospatial=False)
    after_crop = RSImage(path=image_t2.path, array=np.array(img2.crop(changed_region_box)).astype(np.float32) / 255.0,
                          band_count=3, width=0, height=0, is_geospatial=False)
    before_desc = caption_image(before_crop)["answer"]
    after_desc = caption_image(after_crop)["answer"]

    answer = (
        f"{pct_changed:.1f}% of the scene shows significant change between the two dates. "
        f"The most affected region changed from '{before_desc}' to '{after_desc}'."
    )

    # directional yes/no change-VQA, e.g. "has the built-up area increased?"
    directional_answer = None
    if question:
        directional_answer = _directional_change_vqa(img1, img2, question)
        if directional_answer:
            answer = directional_answer["answer"] + " " + answer

    confidence = cosine_similarity_confidence(float(change_score[max_change_idx]), floor=0.0, ceil=2.0)

    return {
        "task": "change",
        "answer": answer,
        "percent_changed": pct_changed,
        "changed_region_bbox": changed_region_box,
        "confidence": confidence if not directional_answer else max(confidence, directional_answer["confidence"]),
        "evidence_image": overlay,
        "change_map": change_map,
        "rs_adapted": rs_adapter.is_rs_adapted(),
    }


def _build_change_map(w, h, boxes, scores):
    norm = (scores - scores.min()) / (scores.ptp() + 1e-8)
    change_map = np.zeros((h, w), dtype=np.float32)
    for (x0, y0, x1, y1), s in zip(boxes, norm):
        change_map[y0:y1, x0:x1] = s
    return change_map


def _overlay_change(base_rgb, change_map):
    heat = cv2.applyColorMap((change_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return cv2.addWeighted(base_rgb, 0.5, heat[:, :, ::-1], 0.5, 0)


@torch.no_grad()
def _directional_change_vqa(img1: Image.Image, img2: Image.Image, question: str):
    """Handles queries like 'has the built-up area increased, decreased, or remained
    unchanged?' by zero-shot scoring each image against the relevant land-cover class
    and comparing coverage estimates."""
    q = question.lower()
    topic = None
    for kw in ["built-up", "water", "vegetation", "forest", "urban"]:
        if kw in q:
            topic = kw
            break
    if topic is None:
        return None

    label = {"built-up": "urban / built-up area", "urban": "urban / built-up area",
              "water": "water body", "vegetation": "dense vegetation", "forest": "forest"}[topic]

    boxes = _grid_patches(*img1.size, n=6)
    txt_emb = rs_adapter.clip_text_embedding([label])
    f1 = _embed_patches(img1, boxes)
    f2 = _embed_patches(img2, boxes)
    sim1 = float((f1 @ txt_emb.cpu().numpy().T).mean())
    sim2 = float((f2 @ txt_emb.cpu().numpy().T).mean())
    delta = sim2 - sim1

    if abs(delta) < 0.01:
        verdict = "remained largely unchanged"
    elif delta > 0:
        verdict = "increased"
    else:
        verdict = "decreased"

    conf = cosine_similarity_confidence(abs(delta), floor=0.0, ceil=0.15)
    return {"answer": f"The {label} appears to have {verdict} between the two dates.", "confidence": conf}
