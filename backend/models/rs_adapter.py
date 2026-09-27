"""
Remote-sensing domain adaptation layer.

This is the single most important module for satisfying the problem statement's
"a general-purpose LLM/VLM cannot be expected to perform reliably without adaptation"
requirement: every specialist model in `backend/models/` pulls its vision-language
backbone from *here*, and here is where the BigEarthNet-trained LoRA adapter is
injected into the base CLIP/BLIP weights.

Design:
  - Base weights (`openai/clip-vit-base-patch32`, BLIP captioning/VQA) are loaded once
    and cached as process-wide singletons (agentic tools share one backbone).
  - If `backend/storage/rs_adapter/READY` exists (created by
    `backend/training/finetune_bigearthnet.py` after a successful run), the LoRA
    adapter weights at `RS_LORA_ADAPTER_PATH` are merged into the CLIP vision tower.
  - If not, the system runs on the base pretrained weights so the demo is never
    blocked on training, but `is_rs_adapted()` reports this honestly and the
    execution report shown to the user reflects it.
"""
import os
import threading
from functools import lru_cache

import torch
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForConditionalGeneration, BlipProcessor,
    BlipForQuestionAnswering,
)

from backend import config

_lock = threading.Lock()


def is_rs_adapted() -> bool:
    return os.path.exists(config.RS_ADAPTATION_ENABLED_FLAG)


@lru_cache(maxsize=1)
def get_clip():
    """Shared CLIP backbone (vision + text towers) used by grounding, change-detection
    and optical-SAR fusion. LoRA-merged with the BigEarthNet adapter when available."""
    with _lock:
        model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        if is_rs_adapted():
            model = _merge_lora(model, config.RS_LORA_ADAPTER_PATH)
        model.to(config.DEVICE)
        model.eval()
        return model, processor


@lru_cache(maxsize=1)
def get_blip_captioner():
    with _lock:
        model = BlipForConditionalGeneration.from_pretrained(config.BLIP_CAPTION_MODEL_NAME)
        processor = BlipProcessor.from_pretrained(config.BLIP_CAPTION_MODEL_NAME)
        if is_rs_adapted():
            model = _merge_lora(model, config.RS_LORA_ADAPTER_PATH + "_caption")
        model.to(config.DEVICE)
        model.eval()
        return model, processor


@lru_cache(maxsize=1)
def get_blip_vqa():
    with _lock:
        model = BlipForQuestionAnswering.from_pretrained(config.BLIP_VQA_MODEL_NAME)
        processor = BlipProcessor.from_pretrained(config.BLIP_VQA_MODEL_NAME)
        if is_rs_adapted():
            model = _merge_lora(model, config.RS_LORA_ADAPTER_PATH + "_vqa")
        model.to(config.DEVICE)
        model.eval()
        return model, processor


def _merge_lora(model, adapter_path: str):
    """Merge a trained LoRA adapter (produced by finetune_bigearthnet.py) into `model`,
    if that specific adapter directory exists. Falls back silently to the base model
    otherwise (e.g. only the shared CLIP adapter was trained, not the BLIP ones)."""
    if not os.path.isdir(adapter_path):
        return model
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    except Exception as e:  # keep the app alive even if a checkpoint is malformed
        print(f"[rs_adapter] Warning: failed to merge LoRA adapter at {adapter_path}: {e}")
    return model


@torch.no_grad()
def clip_image_embedding(pil_image) -> torch.Tensor:
    model, processor = get_clip()
    inputs = processor(images=pil_image, return_tensors="pt").to(config.DEVICE)
    feats = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(feats, dim=-1)


@torch.no_grad()
def clip_text_embedding(texts) -> torch.Tensor:
    model, processor = get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(config.DEVICE)
    feats = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(feats, dim=-1)
