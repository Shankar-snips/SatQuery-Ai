"""
Calibrated confidence helpers shared by every specialist model so the agent can
report a single, comparable confidence number regardless of which tool ran.
"""
import numpy as np


def softmax_margin_confidence(logits: np.ndarray) -> float:
    """
    Confidence = softmax probability of the top class minus the second-best,
    rescaled to [0,1]. Rewards a model that is decisively picking one answer
    over near-ties (a proxy widely used for VQA / classification calibration).
    """
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    sorted_probs = np.sort(probs)[::-1]
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2
    # blend absolute top-1 probability with the margin so a confident-but-close call
    # still scores reasonably instead of collapsing to ~0
    return float(np.clip(0.5 * top1 + 0.5 * margin, 0.0, 1.0))


def cosine_similarity_confidence(sim: float, floor: float = -1.0, ceil: float = 1.0) -> float:
    """Rescale a cosine similarity (grounding / change detection) into a 0..1 confidence."""
    sim = float(np.clip(sim, floor, ceil))
    return float((sim - floor) / (ceil - floor))


def band_confidence(value: float) -> str:
    if value >= 0.7:
        return "high"
    if value >= 0.35:
        return "medium"
    return "low"
