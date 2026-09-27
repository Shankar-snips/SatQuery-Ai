"""
Central configuration for SatQuery-AI.
Every path / model name / threshold used across modules is defined here so the
agentic controller, the specialist models and the training script stay in sync.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "backend", "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
REPORT_DIR = os.path.join(STORAGE_DIR, "reports")
ADAPTER_DIR = os.path.join(BASE_DIR, "backend", "storage", "rs_adapter")  # LoRA weights land here

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

# --- Base pretrained checkpoints (downloaded once via HuggingFace on first run) ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
BLIP_VQA_MODEL_NAME = "Salesforce/blip-vqa-base"

# --- Remote-sensing adaptation (BigEarthNet LoRA) ---
RS_LORA_ADAPTER_PATH = os.path.join(ADAPTER_DIR, "bigearthnet_lora")
RS_ADAPTATION_ENABLED_FLAG = os.path.join(ADAPTER_DIR, "READY")  # created once fine-tuning succeeds

# --- Supported input formats ---
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
BENCHMARK_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALL_EXTENSIONS = GEOSPATIAL_EXTENSIONS | BENCHMARK_EXTENSIONS

# --- Task registry: the agentic controller only ever dispatches to these keys ---
TASKS = ["vqa", "caption", "ground", "change", "fusion"]

# --- Thresholds ---
COREGISTRATION_MAX_DIM_MISMATCH_PCT = 0.15   # pair images may differ in size by at most 15%
CHANGE_DETECTION_DIFF_THRESHOLD = 0.35        # cosine-distance threshold for "changed" patch
GROUNDING_TOP_PATCH_FRACTION = 0.12           # fraction of patches highlighted as the grounded region
CONFIDENCE_LOW_WATERMARK = 0.35               # below this, the report flags "low confidence"

# --- Device ---
def _resolve_device() -> str:
    if os.environ.get("SATQUERY_FORCE_CPU", "0") == "1":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = _resolve_device()

