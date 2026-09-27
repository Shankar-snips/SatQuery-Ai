"""
Builds the downloadable PDF report the problem statement requires: query, the
auditable execution trace (selected task/model/parameters), the answer, visual
evidence image(s), and confidence — everything the agentic controller produced.
"""
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage

from backend import config


def generate_report(agent_response, evidence_arrays=None) -> str:
    styles = getSampleStyleSheet()
    out_path = os.path.join(config.REPORT_DIR, f"satquery_report_{agent_response.execution_id}.pdf")
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    story = []

    story.append(Paragraph("SatQuery-AI — Execution Report", styles["Title"]))
    story.append(Paragraph(f"Execution ID: {agent_response.execution_id}", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Answer", styles["Heading2"]))
    story.append(Paragraph(agent_response.final_answer.replace("\n", "<br/>"), styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Overall confidence: {agent_response.overall_confidence:.2f}", styles["Normal"]))
    story.append(Spacer(1, 10))

    if evidence_arrays:
        story.append(Paragraph("Visual Evidence", styles["Heading2"]))
        for i, arr in enumerate(evidence_arrays):
            if arr is None:
                continue
            tmp_path = os.path.join(config.REPORT_DIR, f"_tmp_evidence_{agent_response.execution_id}_{i}.png")
            PILImage.fromarray(np.asarray(arr).astype("uint8")).save(tmp_path)
            story.append(RLImage(tmp_path, width=140 * mm, height=140 * mm * arr.shape[0] / arr.shape[1]))
            story.append(Spacer(1, 6))

    story.append(Paragraph("Auditable Execution Trace", styles["Heading2"]))
    trace_rows = [["Step", "Details"]]
    for entry in agent_response.trace:
        step = entry.get("step", "")
        details = {k: v for k, v in entry.items() if k not in ("step", "timestamp")}
        trace_rows.append([step, str(details)])
    table = Table(trace_rows, colWidths=[45 * mm, 115 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Per-tool Results", styles["Heading2"]))
    for r in agent_response.results:
        clean = {k: v for k, v in r.items() if k not in ("evidence_image", "heatmap_image", "change_map")}
        story.append(Paragraph(str(clean), styles["Normal"]))
        story.append(Spacer(1, 4))

    doc.build(story)
    return out_path
