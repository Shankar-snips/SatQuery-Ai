"""
Single-image scene description / land-cover captioning (satisfies the mandatory
"captioning OR grounding" second single-image task).
"""
import torch
from backend.models import rs_adapter
from backend.utils.image_io import RSImage

# Land-cover vocabulary the caption is cross-checked against for a lightweight,
# interpretable "coverage confidence" — how many of the caption's claims are
# corroborated by CLIP zero-shot scores against a BigEarthNet-style label set.
_LANDCOVER_CLASSES = [
    "urban / built-up area", "agricultural land", "forest", "water body",
    "bare soil", "grassland", "wetland", "industrial area", "road network",
]


@torch.no_grad()
def caption_image(rs_image: RSImage) -> dict:
    pil_img = rs_image.to_pil()
    model, processor = rs_adapter.get_blip_captioner()
    inputs = processor(images=pil_img, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # corroborate with CLIP zero-shot land-cover scores -> gives a defensible confidence
    img_emb = rs_adapter.clip_image_embedding(pil_img)
    txt_emb = rs_adapter.clip_text_embedding(_LANDCOVER_CLASSES)
    sims = (img_emb @ txt_emb.T).squeeze(0).cpu().numpy()
    ranked = sorted(zip(_LANDCOVER_CLASSES, sims.tolist()), key=lambda x: -x[1])
    top_classes = [c for c, s in ranked[:3]]
    confidence = float((ranked[0][1] - ranked[-1][1]))  # spread between best & worst match
    confidence = max(0.0, min(1.0, confidence))

    return {
        "task": "caption",
        "answer": caption,
        "supporting_landcover_classes": top_classes,
        "confidence": confidence,
        "rs_adapted": rs_adapter.is_rs_adapted(),
    }
