"""
SatQuery-AI backend — FastAPI REST API.

Endpoints
---------
POST /api/upload        multipart file upload(s) -> stored paths + basic metadata
POST /api/query         {query, image_ids, input_mode} -> runs the agentic controller,
                         returns answer, trace, confidence, evidence image URLs, report URL
GET  /api/evidence/{id} serves a generated evidence image (heatmap/overlay/change-map)
GET  /api/report/{id}   downloads the generated PDF report
GET  /health            liveness check
"""
import os
import uuid
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image as PILImage

from backend import config
from backend.controller.agentic_controller import run_query
from backend.utils.report_generator import generate_report
from backend.models import rs_adapter

app = FastAPI(title="SatQuery-AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo setting — restrict in production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory registries for the demo (swap for a DB in production)
_UPLOADS: dict = {}      # image_id -> filesystem path
_EVIDENCE: dict = {}     # evidence_id -> filesystem path (rendered PNGs)
_REPORTS: dict = {}      # execution_id -> filesystem path (PDFs)


@app.get("/health")
def health():
    return {"status": "ok", "rs_adapted": rs_adapter.is_rs_adapted(), "device": config.DEVICE}


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    image_ids = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in config.ALL_EXTENSIONS:
            raise HTTPException(400, f"Unsupported extension: {ext}")
        image_id = str(uuid.uuid4())[:12]
        dest = os.path.join(config.UPLOAD_DIR, f"{image_id}{ext}")
        with open(dest, "wb") as out:
            out.write(await f.read())
        _UPLOADS[image_id] = dest
        image_ids.append({"image_id": image_id, "filename": f.filename})
    return {"uploaded": image_ids}


class QueryRequest(BaseModel):
    query: str
    image_ids: List[str]
    input_mode: str  # "single" | "cross_modal" | "bi_temporal"


@app.post("/api/query")
def submit_query(req: QueryRequest):
    paths = []
    for iid in req.image_ids:
        if iid not in _UPLOADS:
            raise HTTPException(404, f"Unknown image_id: {iid}")
        paths.append(_UPLOADS[iid])

    resp = run_query(req.query, paths, req.input_mode)

    evidence_urls = []
    evidence_arrays = []
    for idx, r in enumerate(resp.results):
        img_arr = r.get("evidence_image")
        if img_arr is not None:
            eid = f"{resp.execution_id}_{idx}"
            out_path = os.path.join(config.REPORT_DIR, f"evidence_{eid}.png")
            PILImage.fromarray(np.asarray(img_arr).astype("uint8")).save(out_path)
            _EVIDENCE[eid] = out_path
            evidence_urls.append(f"/api/evidence/{eid}")
            evidence_arrays.append(img_arr)

    report_path = generate_report(resp, evidence_arrays=evidence_arrays)
    _REPORTS[resp.execution_id] = report_path

    payload = resp.to_dict()
    payload["evidence_urls"] = evidence_urls
    payload["report_url"] = f"/api/report/{resp.execution_id}"
    return payload


@app.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str):
    path = _EVIDENCE.get(evidence_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Evidence image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/report/{execution_id}")
def get_report(execution_id: str):
    path = _REPORTS.get(execution_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "Report not found.")
    return FileResponse(path, media_type="application/pdf",
                         filename=f"satquery_report_{execution_id}.pdf")
