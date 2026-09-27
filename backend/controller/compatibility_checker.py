"""
Validates uploads before any model runs: format, band/modality compatibility, and
(for pairs) approximate co-registration. Raises a structured `CompatibilityError`
so the agentic controller can return an actionable message to the user instead of
letting a specialist model crash or silently produce garbage.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

from backend import config
from backend.utils.image_io import RSImage, guess_modality


class CompatibilityError(Exception):
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def check_format(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in config.ALL_EXTENSIONS:
        raise CompatibilityError(
            f"Unsupported file extension '{ext}'. Supported: GeoTIFF/TIFF, or PNG/JPEG for benchmark datasets.",
            {"extension": ext},
        )


def check_pair_coregistration(img_a: RSImage, img_b: RSImage):
    """Confirms two images are close enough in size (proxy for co-registration when
    exact CRS/transform metadata isn't present, e.g. benchmark crops)."""
    if img_a.width == 0 or img_b.width == 0:
        return  # crops used internally, skip
    dw = abs(img_a.width - img_b.width) / max(img_a.width, img_b.width)
    dh = abs(img_a.height - img_b.height) / max(img_a.height, img_b.height)
    if dw > config.COREGISTRATION_MAX_DIM_MISMATCH_PCT or dh > config.COREGISTRATION_MAX_DIM_MISMATCH_PCT:
        raise CompatibilityError(
            "The two images differ too much in size to be treated as a co-registered pair. "
            "Please upload images covering the same geographic extent / resolution.",
            {"image_a_size": (img_a.width, img_a.height), "image_b_size": (img_b.width, img_b.height)},
        )
    if img_a.is_geospatial and img_b.is_geospatial and img_a.crs and img_b.crs:
        if img_a.crs != img_b.crs:
            raise CompatibilityError(
                f"CRS mismatch between the two images ({img_a.crs} vs {img_b.crs}). "
                "Please reproject to a common CRS before uploading.",
                {"crs_a": img_a.crs, "crs_b": img_b.crs},
            )


def check_cross_modal_pair(img_a: RSImage, img_b: RSImage) -> List[str]:
    """For fusion tasks, confirms one image is optical and the other SAR; returns
    [optical_modality, sar_modality] order info via the assigned `.modality` field."""
    mod_a = guess_modality(img_a)
    mod_b = guess_modality(img_b)
    img_a.modality, img_b.modality = mod_a, mod_b
    if {mod_a, mod_b} != {"optical", "sar"}:
        raise CompatibilityError(
            "Cross-modal fusion requires one optical/multispectral image and one SAR image. "
            f"Detected modalities: {mod_a}, {mod_b}.",
            {"modality_a": mod_a, "modality_b": mod_b},
        )
    return [mod_a, mod_b]


def check_single_image(img: RSImage):
    if img.width < 32 or img.height < 32:
        raise CompatibilityError("Image resolution too small for reliable analysis (min 32x32).")
