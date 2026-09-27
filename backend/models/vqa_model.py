"""
Single-image Visual Question Answering (mandatory baseline task).

Wraps BLIP-VQA (RS-adapted via rs_adapter.get_blip_vqa) and additionally exposes a
CLIP zero-shot fallback for closed-set land-cover style questions
("is there a water body?", "is the area urban or rural?") where a short candidate-
answer list gives a much better-calibrated confidence than open-ended generation.
"""
import torch
from backend.models import rs_adapter
from backend.utils.confidence import softmax_margin_confidence
from backend.utils.image_io import RSImage

# Closed-set question patterns SatQuery-AI can answer with calibrated confidence
# via CLIP zero-shot classification instead of free-form generation.
_CLOSED_SET_TRIGGERS = {
    "is there": ["yes, it is present", "no, it is not present"],
    "how many": None,  # left to generative BLIP-VQA
    "built-up": ["mostly built-up / urban area", "mostly natural / non built-up area"],
    "water": ["a water body is clearly visible", "no water body is visible"],
    "vegetation": ["dense vegetation is present", "sparse or no vegetation is present"],
}


@torch.no_grad()
def answer_question(rs_image: RSImage, question: str) -> dict:
    pil_img = rs_image.to_pil()

    # 1) try closed-set CLIP zero-shot route for calibrated confidence
    q_lower = question.lower()
    for trigger, candidates in _CLOSED_SET_TRIGGERS.items():
        if trigger in q_lower and candidates:
            img_emb = rs_adapter.clip_image_embedding(pil_img)
            txt_emb = rs_adapter.clip_text_embedding(candidates)
            sims = (img_emb @ txt_emb.T).squeeze(0).cpu().numpy() * 100  # CLIP logit scale
            best_idx = int(sims.argmax())
            conf = softmax_margin_confidence(sims)
            return {
                "task": "vqa",
                "route": "clip_zero_shot",
                "answer": candidates[best_idx],
                "confidence": conf,
                "rs_adapted": rs_adapter.is_rs_adapted(),
            }

    # 2) generative BLIP-VQA for open-ended questions ("what is the dominant land cover?")
    model, processor = rs_adapter.get_blip_vqa()
    inputs = processor(images=pil_img, text=question, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=30, output_scores=True, return_dict_in_generate=True)
    answer = processor.decode(out.sequences[0], skip_special_tokens=True)

    # confidence from mean per-token top-vs-runner-up softmax margin across generation steps
    confs = []
    for step_scores in out.scores:
        top2 = torch.topk(step_scores[0], k=2).values
        probs = torch.softmax(top2, dim=-1)
        confs.append(float((probs[0] - probs[1]).clamp(0, 1)))
    confidence = float(sum(confs) / len(confs)) if confs else 0.5

    return {
        "task": "vqa",
        "route": "blip_generative",
        "answer": answer.strip(),
        "confidence": confidence,
        "rs_adapted": rs_adapter.is_rs_adapted(),
    }
