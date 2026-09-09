"""PDF report rendering for persisted ONI-SCAN inspection records."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.grading_engine import get_standard


LIMITATIONS = [
    "AI identifies visible characteristics from the submitted images.",
    "Visible rot refers to externally visible characteristics and does not establish internal rot.",
    "Final procurement classification requires verification against the applicable standard.",
]


def _text(value: Any, fallback: str = "Not available") -> str:
    if value is None or value == "":
        return fallback
    return str(value)


def _format_timestamp(value: Any) -> str:
    if not value:
        return "Not available"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return str(value)


def _paragraph(text: str, style) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


def build_inspection_pdf(record: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 14

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="ONI-SCAN Digital Quality Inspection Report",
        author="ONI-SCAN",
    )
    story = [
        _paragraph("ONI-SCAN", title_style),
        _paragraph("DIGITAL QUALITY INSPECTION REPORT", heading_style),
        Spacer(1, 5 * mm),
    ]

    identity = [
        ["Inspection ID", _text(record.get("inspection_id"))],
        ["Date / Time", _format_timestamp(record.get("timestamp"))],
        ["Uploaded filename", _text(record.get("uploaded_filename"))],
        ["Inspection status", _text(record.get("inspection_status"))],
    ]
    story.extend([_paragraph("Inspection", heading_style), _table(identity), Spacer(1, 4 * mm)])

    ai = record.get("ai_observations") or {}
    counts = ai.get("class_counts") or {}
    summary = [
        ["Total detected onions", _text(ai.get("total_detected"), "0")],
        ["Acceptable", _text(counts.get("acceptable"), "0")],
        ["Damaged", _text(counts.get("damaged"), "0")],
        ["Visible rot", _text(counts.get("visible_rot"), "0")],
        ["Sprouted", _text(counts.get("sprouted"), "0")],
        ["Average AI confidence", _format_percent(ai.get("average_confidence"))],
    ]
    story.extend([_paragraph("AI Observation Summary", heading_style), _table(summary), Spacer(1, 4 * mm)])

    measurements = record.get("size_measurements") or []
    calibration = record.get("calibration") or {}
    size_rows = [["Measurement", "Value"]]
    if measurements:
        for index, measurement in enumerate(measurements, start=1):
            size_rows.append([f"Detection {index}", _format_measurement(measurement)])
    else:
        size_rows.append(["Measurements", "No measured values recorded"])
    size_rows.append(["Calibration", _text(calibration.get("status"))])
    story.extend([_paragraph("Size Measurements", heading_style), _table(size_rows), Spacer(1, 4 * mm)])

    human = record.get("human_verification") or {}
    grading = record.get("grading_result") or {}
    standard = get_standard()
    story.extend([
        _paragraph("Human Verification", heading_style),
        _table([
            ["Verification status", _text(human.get("status"), "Pending")],
            ["Verified observations", _format_mapping(record.get("verified_observations") or {})],
        ]),
        Spacer(1, 4 * mm),
        _paragraph("Final Grading Result", heading_style),
        _table([
            ["Status", _text(grading.get("status"), "Pending verification")],
            ["Range", _text(grading.get("range"), "Pending")],
            ["Reasons", " ".join(str(reason) for reason in grading.get("reasons") or []) or "No grading result issued."],
        ]),
        Spacer(1, 4 * mm),
        _paragraph("Applicable Grading Standard", heading_style),
        _table([
            ["Standard", _text(standard.get("name"))],
            ["Standard ID", _text(standard.get("id"))],
            ["Source", _text((standard.get("source") or {}).get("url"))],
            ["Verified rules", "; ".join(str(rule.get("label")) for rule in (standard.get("rules") or {}).values())],
        ]),
        Spacer(1, 4 * mm),
        _paragraph("Important Limitations", heading_style),
    ])
    story.extend([_paragraph(f"- {limitation}", body_style) for limitation in LIMITATIONS])
    document.build(story)
    return buffer.getvalue()


def _format_percent(value: Any) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_measurement(measurement: Any) -> str:
    if not isinstance(measurement, dict):
        return _text(measurement)
    diameter = measurement.get("diameter_mm")
    return f"diameter_mm={diameter}" if diameter is not None else "Calibration required"


def _format_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "No verified observations recorded."
    return "; ".join(f"{key}={value}" for key, value in values.items() if value is not None) or "No verified observations recorded."


def _table(rows: list[list[str]]) -> Table:
    table = Table(rows, colWidths=[58 * mm, 112 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E9EEF8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#182238")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table
