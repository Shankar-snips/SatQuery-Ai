"""
Smoke tests. Run with:  pytest backend/tests/test_pipeline.py -v

These use small synthetic images (no network / dataset download required) so they
can run in CI or during the hackathon demo setup to confirm the pipeline wiring is
correct end-to-end, independent of whether the BigEarthNet adapter has been trained.
"""
import os
import numpy as np
from PIL import Image
import pytest

from backend import config


def _make_test_image(path, size=(128, 128), color=(80, 140, 90)):
    arr = np.ones((size[1], size[0], 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
    # add a bright "water-like" patch for grounding tests
    arr[20:60, 20:60] = (30, 60, 200)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def single_image(tmp_path):
    p = tmp_path / "scene.png"
    return _make_test_image(str(p))


@pytest.fixture
def bitemporal_images(tmp_path):
    p1 = tmp_path / "t1.png"
    p2 = tmp_path / "t2.png"
    _make_test_image(str(p1), color=(80, 140, 90))
    _make_test_image(str(p2), color=(180, 40, 40))  # simulate built-up growth
    return str(p1), str(p2)


def test_image_io_loads_png(single_image):
    from backend.utils.image_io import load_image
    rs_img = load_image(single_image)
    assert rs_img.band_count == 3
    assert rs_img.width == 128 and rs_img.height == 128


def test_compatibility_checker_rejects_bad_extension(tmp_path):
    from backend.controller.compatibility_checker import check_format, CompatibilityError
    bad = tmp_path / "scene.bmp"
    bad.write_bytes(b"not a real image")
    with pytest.raises(CompatibilityError):
        check_format(str(bad))


def test_query_classifier_routes_change_query():
    from backend.controller.query_classifier import classify_query
    ranked = classify_query("What changed between these two dates, and where did the change occur?")
    assert ranked[0]["task"] == "change"


def test_query_classifier_routes_fusion_query():
    from backend.controller.query_classifier import classify_query
    ranked = classify_query("Use the optical and SAR images together to identify built-up regions.")
    top_tasks = [r["task"] for r in ranked[:2]]
    assert "fusion" in top_tasks


@pytest.mark.slow
def test_end_to_end_single_image_vqa(single_image):
    """Requires model downloads (transformers/torch) — marked slow, skip offline."""
    from backend.controller.agentic_controller import run_query
    resp = run_query("What is the dominant land cover in this image?", [single_image], "single")
    assert resp.error is None
    assert resp.final_answer
    assert 0.0 <= resp.overall_confidence <= 1.0


@pytest.mark.slow
def test_end_to_end_change_detection(bitemporal_images):
    from backend.controller.agentic_controller import run_query
    p1, p2 = bitemporal_images
    resp = run_query("What changed between these two dates?", [p1, p2], "bi_temporal")
    assert resp.error is None
    assert any(r["task"] == "change" for r in resp.results)
