"""
Classifies a natural-language query into one or more of the task registry entries
(`vqa`, `caption`, `ground`, `change`, `fusion`) so the agentic controller knows
which specialist tool(s) to invoke.

Two signals are combined:
  1. Zero-shot semantic scoring: the query is embedded with the CLIP text tower and
     compared to embedded task-prototype sentences (a real, model-driven signal, not
     just string matching).
  2. Deterministic keyword rules that hard-trigger on unambiguous phrasing (e.g. "and"
     joining two intents, explicit "SAR"/"optical" mentions, "changed"/"between the
     two dates"). These exist because task routing is safety-critical here (running
     the wrong tool wastes the whole pipeline) and because compound queries need
     more than a single top-1 zero-shot label.
"""
import re
from typing import List

import torch
from backend.models import rs_adapter

_TASK_PROTOTYPES = {
    "vqa": [
        "how many objects are in the image", "is there a water body",
        "what is the dominant land cover", "answer a question about the image",
    ],
    "caption": [
        "describe the land cover and major objects visible in this image",
        "give a scene description of the image", "summarise what is shown",
    ],
    "ground": [
        "highlight the water body referred to in the query",
        "point to the location of the object in the image", "show where the object is",
    ],
    "change": [
        "what changed between these two dates", "detect change over time",
        "has the area increased or decreased", "compare two images taken at different times",
    ],
    "fusion": [
        "use the optical and SAR images together", "combine radar and optical imagery",
        "cross modal analysis of optical and SAR data",
    ],
}

_KEYWORD_RULES = {
    "change": [r"\bchange(d|s)?\b", r"\bbetween\b.*\bdate", r"\bincreas", r"\bdecreas",
               r"\bbefore\b.*\bafter\b", r"\bcompare\b.*\btime"],
    "fusion": [r"\bsar\b", r"\bradar\b", r"\boptical\b.*\bsar\b", r"\bfuse\b", r"\bfusion\b",
               r"\bcombine\b.*\b(sar|radar|optical)"],
    "ground": [r"\bhighlight\b", r"\bpoint (out|to)\b", r"\bwhere is\b", r"\blocate\b", r"\bbounding box\b"],
    "caption": [r"\bdescribe\b", r"\bcaption\b", r"\bsummar", r"\bscene description\b"],
    "vqa": [r"\bhow many\b", r"\bis there\b", r"\bwhat is\b", r"\bdoes\b"],
}

_prototype_cache = {}


def _get_prototype_embeddings():
    if _prototype_cache:
        return _prototype_cache
    for task, sentences in _TASK_PROTOTYPES.items():
        emb = rs_adapter.clip_text_embedding(sentences)  # (N, dim)
        _prototype_cache[task] = emb.mean(dim=0, keepdim=True)  # mean prototype vector
    return _prototype_cache


@torch.no_grad()
def classify_query(query: str) -> List[dict]:
    """Returns a ranked list of {task, score} — the controller decides how many to act on."""
    query_lower = query.lower()

    # deterministic hits first (compound queries can hit >1 rule, which is intended)
    rule_hits = set()
    for task, patterns in _KEYWORD_RULES.items():
        for p in patterns:
            if re.search(p, query_lower):
                rule_hits.add(task)
                break

    prototypes = _get_prototype_embeddings()
    q_emb = rs_adapter.clip_text_embedding([query])  # (1, dim)
    scores = {}
    for task, proto in prototypes.items():
        sim = float((q_emb @ proto.T).squeeze())
        boost = 0.15 if task in rule_hits else 0.0
        scores[task] = sim + boost

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [{"task": t, "score": round(s, 4)} for t, s in ranked]


def split_compound_intents(query: str, ranked: List[dict], top_n_threshold: float = 0.03) -> List[str]:
    """If the query semantically/keyword-triggers more than one task within a small
    margin of the top score (or explicitly says 'and'), the controller should run
    multiple tools and merge results, matching queries like
    'describe this image and highlight the water body'."""
    if not ranked:
        return []
    top_score = ranked[0]["score"]
    selected = [ranked[0]["task"]]
    explicit_and = bool(re.search(r"\band\b", query.lower()))
    for entry in ranked[1:]:
        if explicit_and and (top_score - entry["score"]) <= top_n_threshold * 4:
            selected.append(entry["task"])
        elif (top_score - entry["score"]) <= top_n_threshold:
            selected.append(entry["task"])
    # de-dup while preserving order
    seen = set()
    out = []
    for t in selected:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:2]  # cap at 2 tools per query to keep the report readable
