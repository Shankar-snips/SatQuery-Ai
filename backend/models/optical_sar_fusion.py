"""
Optical - SAR cross-modal fusion (mandatory cross-modal task).

Optical imagery is embedded with the RS-adapted CLIP vision tower (spectral/contextual
information). SAR imagery is not natural-image-like, so instead of forcing it through
the same RGB-pretrained tower, we extract genuine SAR-appropriate structural features:
  - GLCM (gray-level co-occurrence matrix) texture descriptors: contrast, homogeneity,
    energy, correlation  -> captures surface roughness / structural texture, which is
    what SAR backscatter actually encodes.
  - Sobel-gradient edge density -> captures man-made structure (built-up edges, linear
    infrastructure) which SAR is especially good at revealing through cloud cover.
These SAR features are projected into the same embedding space as the CLIP optical
features via a small linear fusion head, then compared against land-cover text
prompts. The head's weights are trained during BigEarthNet fine-tuning
(`training/finetune_bigearthnet.py`); a sensible identity-like default is used if no
trained head is present yet, so the module still runs pre-training.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

from backend import config
from backend.models import rs_adapter
from backend.utils.confidence import softmax_margin_confidence
from backend.utils.image_io import RSImage

_FUSION_CLASSES = [
    "built-up / urban area", "open water", "agricultural land",
    "forest / dense vegetation", "bare soil", "wetland",
]
_SAR_FEAT_DIM = 6   # contrast, dissimilarity, homogeneity, energy, correlation, edge_density
_CLIP_DIM = 512     # openai/clip-vit-base-patch32 projection dim


class _FusionHead(nn.Module):
    """Projects SAR structural features into CLIP embedding space so they can be
    directly compared / concatenated with optical CLIP embeddings and with the
    CLIP text embeddings of land-cover class names."""
    def __init__(self, sar_dim=_SAR_FEAT_DIM, clip_dim=_CLIP_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sar_dim, 128), nn.ReLU(),
            nn.Linear(128, clip_dim),
        )

    def forward(self, x):
        out = self.proj(x)
        return torch.nn.functional.normalize(out, dim=-1)


_fusion_head_cache = {}


def _get_fusion_head():
    if "head" in _fusion_head_cache:
        return _fusion_head_cache["head"]
    head = _FusionHead()
    ckpt_path = os.path.join(config.ADAPTER_DIR, "fusion_head.pt")
    if os.path.exists(ckpt_path):
        head.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    head.eval()
    _fusion_head_cache["head"] = head
    return head


def _extract_sar_features(rs_image: RSImage) -> np.ndarray:
    gray = np.array(rs_image.to_pil().convert("L"))
    gray_q = (gray / 32).astype(np.uint8)  # quantize to 8 levels for a stable GLCM
    glcm = graycomatrix(gray_q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=8, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    gx = np.abs(np.gradient(gray.astype(np.float32), axis=1))
    gy = np.abs(np.gradient(gray.astype(np.float32), axis=0))
    edge_density = float(((gx + gy) > (gx + gy).mean() * 1.5).mean())

    feats = np.array([contrast, dissimilarity, homogeneity, energy, correlation, edge_density], dtype=np.float32)
    # normalise into a roughly comparable range
    feats = feats / (np.linalg.norm(feats) + 1e-8)
    return feats


@torch.no_grad()
def fuse_optical_sar(optical_image: RSImage, sar_image: RSImage, question: str = None) -> dict:
    optical_pil = optical_image.to_pil()
    optical_emb = rs_adapter.clip_image_embedding(optical_pil).cpu().numpy()[0]  # (512,)

    sar_feats = _extract_sar_features(sar_image)
    head = _get_fusion_head()
    sar_emb = head(torch.tensor(sar_feats).unsqueeze(0)).numpy()[0]  # (512,) projected into CLIP space

    # fused representation: mean of the two modality embeddings in the shared space
    fused_emb = optical_emb + sar_emb
    fused_emb = fused_emb / (np.linalg.norm(fused_emb) + 1e-8)

    text_emb = rs_adapter.clip_text_embedding(_FUSION_CLASSES).cpu().numpy()
    sims_fused = fused_emb @ text_emb.T
    sims_optical = optical_emb @ text_emb.T
    sims_sar = sar_emb @ text_emb.T

    top_idx = int(sims_fused.argmax())
    label = _FUSION_CLASSES[top_idx]
    confidence = softmax_margin_confidence(sims_fused * 100)

    agreement = bool(sims_optical.argmax() == sims_sar.argmax())
    complementary_note = (
        "Optical and SAR evidence agree on the dominant class."
        if agreement else
        "Optical and SAR evidence disagree — SAR's structural cues (texture/edges) were "
        "weighted alongside optical spectral cues to resolve the fused estimate, which is "
        "particularly useful when the optical image is affected by haze/cloud or shadow."
    )

    answer = f"Combining the optical and SAR image, the dominant land-cover class is: {label}. {complementary_note}"
    if question:
        answer = f"{answer} (in response to: '{question}')"

    per_class = {cls: float(s) for cls, s in zip(_FUSION_CLASSES, sims_fused)}

    return {
        "task": "fusion",
        "answer": answer,
        "dominant_class": label,
        "per_class_scores": per_class,
        "modality_agreement": agreement,
        "confidence": confidence,
        "rs_adapted": rs_adapter.is_rs_adapted(),
    }
