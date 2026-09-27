"""
Image I/O for SatQuery-AI.

Handles the two families of input the problem statement mandates:
  - GeoTIFF / TIFF geospatial imagery (multi-band, optionally georeferenced) via rasterio
  - PNG / JPEG for the prescribed public benchmark datasets (VRSBench, RSVQA, CDVQA)

Every loader returns a common `RSImage` structure so every downstream model works off
one interface regardless of source format or band count.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    _RASTERIO_AVAILABLE = True
except ImportError:  # keeps the demo runnable even before `pip install rasterio` completes
    _RASTERIO_AVAILABLE = False


@dataclass
class RSImage:
    path: str
    array: np.ndarray            # HxWxC, float32, 0..1
    band_count: int
    width: int
    height: int
    is_geospatial: bool
    crs: Optional[str] = None
    transform: Optional[object] = None
    modality: str = "optical"    # "optical" | "sar" — set by the caller / compatibility checker
    meta: dict = field(default_factory=dict)

    def to_pil(self) -> Image.Image:
        """RGB (or pseudo-RGB from first 3 bands / grayscale replication) preview for the VLM backbone."""
        arr = self.array
        if arr.shape[-1] >= 3:
            rgb = arr[:, :, :3]
        else:
            rgb = np.repeat(arr[:, :, :1], 3, axis=-1)
        rgb = np.clip(rgb, 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))


def _load_geotiff(path: str) -> RSImage:
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is not installed — run `pip install -r requirements.txt`.")
    with rasterio.open(path) as src:
        data = src.read()  # C x H x W
        data = np.transpose(data, (1, 2, 0)).astype(np.float32)  # H x W x C
        # robust per-band normalisation (2nd-98th percentile) — standard RS preprocessing
        norm = np.zeros_like(data)
        for b in range(data.shape[-1]):
            band = data[:, :, b]
            lo, hi = np.percentile(band, [2, 98])
            if hi - lo < 1e-6:
                hi = lo + 1e-6
            norm[:, :, b] = np.clip((band - lo) / (hi - lo), 0, 1)
        return RSImage(
            path=path,
            array=norm,
            band_count=data.shape[-1],
            width=src.width,
            height=src.height,
            is_geospatial=src.crs is not None,
            crs=str(src.crs) if src.crs else None,
            transform=src.transform,
            meta={"driver": src.driver, "dtype": str(src.dtypes[0])},
        )


def _load_benchmark_image(path: str) -> RSImage:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return RSImage(
        path=path,
        array=arr,
        band_count=3,
        width=img.width,
        height=img.height,
        is_geospatial=False,
        meta={"driver": "PIL"},
    )


def load_image(path: str) -> RSImage:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".tif", ".tiff", ".geotiff"}:
        try:
            return _load_geotiff(path)
        except (RasterioIOError, RuntimeError):
            # some ".tif" benchmark crops have no georeferencing at all — fall back gracefully
            return _load_benchmark_image(path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        return _load_benchmark_image(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def guess_modality(rs_image: RSImage) -> str:
    """
    Heuristic optical-vs-SAR detector used when the user doesn't label the upload:
    SAR imagery is typically single-band (amplitude) with a speckled, low-saturation
    texture, while optical/multispectral imagery has multiple correlated colour bands.
    """
    if rs_image.band_count == 1:
        return "sar"
    if rs_image.band_count >= 3:
        r, g, b = rs_image.array[:, :, 0], rs_image.array[:, :, 1], rs_image.array[:, :, 2]
        # near-identical bands (grayscale-like) + high local variance => likely SAR amplitude stored as RGB
        band_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        if band_corr > 0.98 and np.std(rs_image.array) > 0.2:
            return "sar"
    return "optical"
