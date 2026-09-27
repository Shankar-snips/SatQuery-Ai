"""
THE AGENT.

Implements exactly the orchestration loop required by the problem statement:
  - interpret the query and classify the requested task
  - check the number, modality, format, metadata, and compatibility of the input images
  - select one or more models/tools from a predefined registry (config.TASKS)
  - configure only permitted task parameters and execute the selected workflow
  - combine textual and spatial outputs, estimate confidence, and return visual evidence
  - provide an auditable execution summary (selected task, model/tool names, key parameters)

This module contains NO model weights itself — it only decides *what* to run and
*how to merge results*, which is the "agentic" contribution distinct from any single
VLM call.
"""
import time
import uuid
from typing import List, Optional

from backend import config
from backend.controller import query_classifier
from backend.controller.compatibility_checker import (
    CompatibilityError, check_format, check_single_image,
    check_pair_coregistration, check_cross_modal_pair,
)
from backend.utils.image_io import load_image, RSImage
from backend.models import vqa_model, captioning_model, grounding_model
from backend.models import change_detection_model, optical_sar_fusion


class AgentResponse:
    def __init__(self):
        self.execution_id = str(uuid.uuid4())[:8]
        self.trace: List[dict] = []          # auditable execution trace
        self.results: List[dict] = []          # raw per-tool results
        self.final_answer: str = ""
        self.overall_confidence: float = 0.0
        self.evidence_images: List = []
        self.error: Optional[str] = None

    def to_dict(self):
        return {
            "execution_id": self.execution_id,
            "trace": self.trace,
            "results": [
                {k: v for k, v in r.items() if k not in ("evidence_image", "heatmap_image", "change_map")}
                for r in self.results
            ],
            "final_answer": self.final_answer,
            "overall_confidence": self.overall_confidence,
            "error": self.error,
        }


def _log(resp: AgentResponse, step: str, **kwargs):
    resp.trace.append({"step": step, "timestamp": round(time.time(), 3), **kwargs})


def run_query(
    query: str,
    image_paths: List[str],
    input_mode: str,  # "single" | "cross_modal" | "bi_temporal"
) -> AgentResponse:
    resp = AgentResponse()
    _log(resp, "query_received", query=query, input_mode=input_mode, n_images=len(image_paths))

    # ---- 1. format + compatibility checking ----
    try:
        for p in image_paths:
            check_format(p)
        images: List[RSImage] = [load_image(p) for p in image_paths]

        if input_mode == "single":
            check_single_image(images[0])
        elif input_mode == "bi_temporal":
            if len(images) != 2:
                raise CompatibilityError("Bi-temporal analysis requires exactly 2 images.")
            check_pair_coregistration(images[0], images[1])
        elif input_mode == "cross_modal":
            if len(images) != 2:
                raise CompatibilityError("Cross-modal fusion requires exactly 2 images (optical + SAR).")
            check_pair_coregistration(images[0], images[1])
            check_cross_modal_pair(images[0], images[1])
        else:
            raise CompatibilityError(f"Unknown input_mode: {input_mode}")

    except CompatibilityError as e:
        resp.error = e.message
        _log(resp, "compatibility_check_failed", message=e.message, details=e.details)
        resp.final_answer = f"Cannot proceed: {e.message}"
        return resp

    _log(resp, "compatibility_check_passed",
         modalities=[getattr(im, "modality", "optical") for im in images],
         geospatial=[im.is_geospatial for im in images])

    # ---- 2. query classification ----
    ranked = query_classifier.classify_query(query)
    _log(resp, "query_classified", ranked_tasks=ranked)

    # ---- 3. gate classified tasks against what was actually uploaded ----
    candidate_tasks = query_classifier.split_compound_intents(query, ranked)
    candidate_tasks = _gate_tasks_by_input_mode(candidate_tasks, ranked, input_mode)
    _log(resp, "tasks_selected", tasks=candidate_tasks, input_mode=input_mode)

    if not candidate_tasks:
        resp.error = "Could not match the query to a supported task given the uploaded input."
        resp.final_answer = resp.error
        return resp

    # ---- 4. execute selected tool(s) ----
    for task in candidate_tasks:
        try:
            result = _execute_task(task, images, query)
            resp.results.append(result)
            if "evidence_image" in result:
                resp.evidence_images.append(result["evidence_image"])
            _log(resp, "tool_executed", task=task, model=_MODEL_NAMES[task],
                 confidence=result.get("confidence"), rs_adapted=result.get("rs_adapted"))
        except Exception as e:
            _log(resp, "tool_failed", task=task, error=str(e))
            resp.results.append({"task": task, "answer": f"[{task} tool failed: {e}]", "confidence": 0.0})

    # ---- 5. merge outputs + estimate overall confidence ----
    resp.final_answer = " \n".join(r["answer"] for r in resp.results if r.get("answer"))
    confidences = [r.get("confidence", 0.0) for r in resp.results]
    resp.overall_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    _log(resp, "outputs_merged", overall_confidence=resp.overall_confidence)

    return resp


_MODEL_NAMES = {
    "vqa": "BLIP-VQA (RS-adapted) / CLIP zero-shot",
    "caption": "BLIP-Captioning (RS-adapted)",
    "ground": "CLIP patch-grounding (RS-adapted)",
    "change": "Siamese CLIP change-detector (RS-adapted) + BLIP captioning",
    "fusion": "CLIP (optical) + GLCM/Sobel-SAR fusion head (RS-adapted)",
}


def _gate_tasks_by_input_mode(candidate_tasks, ranked, input_mode) -> List[str]:
    """Cross-checks the classified task(s) against what was actually uploaded,
    matching the 'compatibility of the input images' step in the problem statement."""
    valid_for_mode = {
        "single": {"vqa", "caption", "ground"},
        "bi_temporal": {"change"},
        "cross_modal": {"fusion", "vqa"},  # fusion primary; vqa allowed on the optical member
    }[input_mode]

    filtered = [t for t in candidate_tasks if t in valid_for_mode]
    if filtered:
        return filtered

    # fall back to the best-ranked task that IS valid for this input mode, rather than
    # failing outright on a slightly-off phrasing
    for entry in ranked:
        if entry["task"] in valid_for_mode:
            return [entry["task"]]

    # last resort: pick the canonical default task for this mode
    default = {"single": "vqa", "bi_temporal": "change", "cross_modal": "fusion"}[input_mode]
    return [default]


def _execute_task(task: str, images: List[RSImage], query: str) -> dict:
    if task == "vqa":
        target = images[0]
        return vqa_model.answer_question(target, query)
    if task == "caption":
        return captioning_model.caption_image(images[0])
    if task == "ground":
        phrase = _extract_grounding_phrase(query)
        return grounding_model.ground_phrase(images[0], phrase)
    if task == "change":
        return change_detection_model.detect_change(images[0], images[1], question=query)
    if task == "fusion":
        optical = next((im for im in images if getattr(im, "modality", "optical") == "optical"), images[0])
        sar = next((im for im in images if getattr(im, "modality", None) == "sar"), images[-1])
        return optical_sar_fusion.fuse_optical_sar(optical, sar, question=query)
    raise ValueError(f"Unregistered task: {task}")


def _extract_grounding_phrase(query: str) -> str:
    """Pulls the object/region phrase out of a grounding query, e.g.
    'Highlight the water body referred to in the query.' -> 'water body'."""
    import re
    m = re.search(r"highlight (the )?(.+?)( referred to.*)?$", query.lower())
    if m:
        return m.group(2).strip(" .")
    m = re.search(r"(?:where is|locate|point to) (the )?(.+?)[\.\?]?$", query.lower())
    if m:
        return m.group(2).strip(" .")
    return query
