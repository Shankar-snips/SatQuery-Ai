"""
Text-guided region grounding (alternative to captioning for the mandatory 2nd
single-image task, and used standalone for queries like
"Highlight the water body referred to in the query.").

Real algorithm (no external detector needed, keeps the whole stack self-contained):
  1. Slide a grid of overlapping patches over the image.
  2. Embed every patch with the RS-adapted CLIP vision tower.
  3. Embed the query phrase with the CLIP text tower.
  4. Score every patch by cosine similarity to the phrase -> similarity heatmap.
  5. Threshold the top-K% of patches, take their union bounding box as the grounded
     region, and overlay both the heatmap and the box as the "visual evidence".
"""
import numpy as np
import cv2
import torch
from PIL import Image

from backend import config
from backend.models import rs_adapter
from backend.utils.confidence import cosine_similarity_confidence
from backend.utils.image_io import RSImage


def _sliding_patches(pil_img: Image.Image, patch_frac=0.2, stride_frac=0.1):
    w, h = pil_img.size
    pw, ph = int(w * patch_frac), int(h * patch_frac)
    sx, sy = max(1, int(w * stride_frac)), max(1, int(h * stride_frac))
    boxes = []
    for y in range(0, h - ph + 1, sy):
        for x in range(0, w - pw + 1, sx):
            boxes.append((x, y, x + pw, y + ph))
    if not boxes:
        boxes = [(0, 0, w, h)]
    return boxes


@torch.no_grad()
def ground_phrase(rs_image: RSImage, phrase: str) -> dict:
    pil_img = rs_image.to_pil()
    boxes = _sliding_patches(pil_img)
    patches = [pil_img.crop(b) for b in boxes]

    # batch-embed all patches through the shared RS-adapted CLIP tower
    model, processor = rs_adapter.get_clip()
    inputs = processor(images=patches, return_tensors="pt").to(model.device)
    patch_feats = torch.nn.functional.normalize(model.get_image_features(**inputs), dim=-1)
    text_feat = rs_adapter.clip_text_embedding([phrase])
    sims = (patch_feats @ text_feat.T).squeeze(1).cpu().numpy()  # cosine sim per patch, [-1,1]

    k = max(1, int(len(boxes) * config.GROUNDING_TOP_PATCH_FRACTION))
    top_idx = np.argsort(sims)[-k:]
    top_boxes = [boxes[i] for i in top_idx]

    xs0 = min(b[0] for b in top_boxes); ys0 = min(b[1] for b in top_boxes)
    xs1 = max(b[2] for b in top_boxes); ys1 = max(b[3] for b in top_boxes)
    union_box = (int(xs0), int(ys0), int(xs1), int(ys1))

    heatmap, overlay = _render_heatmap_and_box(pil_img, boxes, sims, union_box)

    best_sim = float(sims[top_idx].mean())
    confidence = cosine_similarity_confidence(best_sim)

    return {
        "task": "ground",
        "answer": f"Region most consistent with '{phrase}' highlighted (bbox {union_box}).",
        "bbox": union_box,
        "confidence": confidence,
        "evidence_image": overlay,   # numpy HxWx3 uint8, ready to save/serve
        "heatmap_image": heatmap,
        "rs_adapted": rs_adapter.is_rs_adapted(),
    }


def _render_heatmap_and_box(pil_img, boxes, sims, union_box):
    w, h = pil_img.size
    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    norm_sims = (sims - sims.min()) / (sims.ptp() + 1e-8)
    for (x0, y0, x1, y1), s in zip(boxes, norm_sims):
        heat[y0:y1, x0:x1] += s
        count[y0:y1, x0:x1] += 1
    count[count == 0] = 1
    heat = heat / count
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = np.array(pil_img.convert("RGB"))
    blended = cv2.addWeighted(base, 0.55, heat_color[:, :, ::-1], 0.45, 0)

    overlay = blended.copy()
    x0, y0, x1, y1 = union_box
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), max(2, w // 200))
    return blended, overlay
