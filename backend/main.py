from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, UnidentifiedImageError
from ultralytics import YOLO

from training.size_measurement import measure_bounding_box, pixels_per_mm
from backend.grading_engine import evaluate_grading, get_standard
from backend.inspection_storage import (
    InspectionStorageError,
    append_inspection,
    find_inspection,
    load_inspections,
    update_inspection,
)
from backend.report_generator import build_inspection_pdf

# --------------------------------------------------
# ONI-SCAN BACKEND
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MULTICLASS_MODEL_PATH = ROOT / "models" / "oni_scan_onion_quality" / "weights" / "best.pt"
SINGLE_CLASS_MODEL_PATH = ROOT / "models" / "oni_scan_bad_onion" / "weights" / "best.pt"
MODEL_PATH = MULTICLASS_MODEL_PATH if MULTICLASS_MODEL_PATH.exists() else SINGLE_CLASS_MODEL_PATH
INDIVIDUAL_ONION_MODEL_PATH = ROOT / "models" / "oni_scan_individual_onion" / "weights" / "best.pt"
SUPPORTED_QUALITY_CLASSES = {"acceptable", "bad-onion", "visible-rot", "sprouted"}
UPLOAD_DIR = ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SAMPLE_DIR = ROOT / "dataset" / "onion_quality" / "images" / "test"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_CONFIDENCE_THRESHOLD = 0.25
REPORT_MEDIA_TYPE = "application/pdf"
REPORT_LIMITATIONS = [
    "AI identifies visible characteristics from the submitted images.",
    "Visible rot refers to externally visible characteristics and does not establish internal rot.",
    "Final procurement classification requires verification against the applicable standard.",
]


def build_quality_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only classes and confidence values emitted by YOLO."""
    counts = {name: 0 for name in ("acceptable", "damaged", "visible_rot", "sprouted")}
    raw_counts: dict[str, int] = {}
    confidence_values = []
    for detection in detections:
        class_name = str(detection.get("class", "unknown"))
        raw_counts[class_name] = raw_counts.get(class_name, 0) + 1
        normalized_name = class_name.lower().replace("-", "_")
        if normalized_name in counts:
            counts[normalized_name] += 1
        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_values.append(float(confidence))

    total_detected = len(detections)
    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
    percentages = {
        name: round((count / total_detected) * 100, 2) if total_detected else None
        for name, count in counts.items()
    }
    return {
        "total_detected": total_detected,
        "average_confidence": round(average_confidence, 4) if average_confidence is not None else None,
        "class_counts": counts,
        "raw_class_counts": raw_counts,
        "percentages": percentages,
    }


def build_inspection_record(
    inspection_id: str,
    timestamp: str,
    filename: str,
    detections: list[dict[str, Any]],
    calibration: CalibrationConfig,
    result_image: str,
) -> dict[str, Any]:
    summary = build_quality_summary(detections)
    return {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "uploaded_filename": filename,
        "inspection_status": "Analysis Complete",
        "ai_observations": {
            **summary,
            "detections": detections,
            "confidence_values": [detection.get("confidence") for detection in detections],
        },
        "size_measurements": [detection.get("size") for detection in detections],
        "calibration": calibration,
        "human_verification": {"status": "Pending"},
        "verified_observations": {},
        "grading_result": {
            "status": "pending_verification",
            "range": None,
            "reasons": [],
        },
        "result_image": result_image,
        "report_limitations": REPORT_LIMITATIONS,
    }


def build_analytics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate dashboard statistics from persisted inspection records only."""
    observation_names = ("acceptable", "damaged", "visible_rot", "sprouted")
    observation_distribution = {name: 0 for name in observation_names}
    grading_distribution = {"Range-I": 0, "Range-II": 0, "Range-III": 0, "Pending Verification": 0}
    confidence_values: list[float] = []
    total_onions = 0
    pending_verification = 0
    trend_counts: dict[str, int] = {}

    for record in records:
        observations = record.get("ai_observations") or {}
        total_onions += int(observations.get("total_detected") or 0)
        for name in observation_names:
            observation_distribution[name] += int((observations.get("class_counts") or {}).get(name) or 0)
        for value in observations.get("confidence_values") or []:
            if isinstance(value, (int, float)):
                confidence_values.append(float(value))

        grading = record.get("grading_result") or {}
        grading_range = grading.get("range")
        if grading_range in ("Range-I", "Range-II", "Range-III") and grading.get("status") == "graded":
            grading_distribution[grading_range] += 1
        else:
            grading_distribution["Pending Verification"] += 1
            pending_verification += 1

        timestamp = str(record.get("timestamp") or "")
        date_key = timestamp[:10] if len(timestamp) >= 10 else "Unknown"
        trend_counts[date_key] = trend_counts.get(date_key, 0) + 1

    trend = [
        {"date": date, "inspections": trend_counts[date]}
        for date in sorted(trend_counts)
        if date != "Unknown"
    ]
    return {
        "total_inspections": len(records),
        "total_onions": total_onions,
        "average_ai_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "pending_verification": pending_verification,
        "observation_distribution": observation_distribution,
        "grading_distribution": grading_distribution,
        "inspection_trend": trend,
        "recent_inspections": records[:5],
    }


def sanitize_filename(name: str | None) -> str:
    """Keep uploaded names safe and filesystem-friendly."""
    fallback = "onion_sample"
    if not name:
        return fallback
    safe_name = Path(name).name
    sanitized = "".join(ch for ch in safe_name if ch.isalnum() or ch in {".", "_", "-"})
    return sanitized or fallback


def is_valid_image_bytes(data: bytes) -> bool:
    """Validate that the uploaded payload is a real image and not corrupted."""
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def annotation_color(class_name: str) -> tuple[int, int, int]:
    """Return the UI evidence color without changing the model class name."""
    normalized = str(class_name).lower().replace("-", "_")
    return {
        "acceptable": (69, 223, 145),
        "damaged": (245, 200, 75),
        "visible_rot": (255, 91, 103),
        "sprouted": (168, 121, 255),
        "bad_onion": (255, 157, 92),
    }.get(normalized, (114, 183, 255))


def save_annotated_image(
    source_path: Path,
    output_path: Path,
    detections: list[dict[str, Any]],
) -> None:
    """Render one evidence image from the detections produced by this inference."""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection.get("box") or []
        if len(box) != 4:
            continue
        coordinates = tuple(int(round(float(value))) for value in box)
        color = annotation_color(str(detection.get("class", "unknown")))
        draw.rectangle(coordinates, outline=color, width=4)
        label = f"{detection.get('class', 'unknown')} {float(detection.get('confidence', 0)) * 100:.1f}%"
        left, top, right, bottom = draw.textbbox((0, 0), label)
        label_width = right - left + 10
        label_height = bottom - top + 6
        label_top = max(0, coordinates[1] - label_height)
        draw.rectangle(
            (coordinates[0], label_top, coordinates[0] + label_width, label_top + label_height),
            fill=color,
        )
        draw.text((coordinates[0] + 5, label_top + 3), label, fill=(7, 9, 20))
    image.save(output_path, format="JPEG", quality=92)


# --------------------------------------------------
# LOAD AI MODEL
# --------------------------------------------------

print("Loading ONI-SCAN AI model...")

try:
    model = YOLO(str(MODEL_PATH))
    print("ONI-SCAN AI model loaded successfully.")
except Exception as exc:  # pragma: no cover - startup safety check
    model = None
    print(f"ONI-SCAN AI model failed to load: {exc}")

individual_onion_model = None
if INDIVIDUAL_ONION_MODEL_PATH.exists():
    try:
        individual_onion_model = YOLO(str(INDIVIDUAL_ONION_MODEL_PATH))
        print("ONI-SCAN individual onion detection model loaded successfully.")
    except Exception as exc:  # pragma: no cover - optional model startup safety
        print(f"ONI-SCAN individual onion detection model unavailable: {exc}")


class IndividualOnionRecord(TypedDict):
    onion_id: str
    bounding_box: list[float]
    detection_confidence: float
    ai_defects: list[dict[str, Any]]
    size: dict[str, Any] | None
    verification_status: str


class CalibrationConfig(TypedDict):
    """Per-image scale configuration; automatic reference detection is future work."""
    reference_width_mm: float | None
    reference_width_px: float | None
    pixels_per_mm: float | None
    calibrated: bool
    status: str
    automatic_reference_detection: str


def build_calibration_config(
    reference_width_mm: float | None,
    reference_width_px: float | None,
) -> CalibrationConfig:
    """Build a scale only from user/configured values actually supplied for this image."""
    if reference_width_mm is None and reference_width_px is None:
        return {
            "reference_width_mm": None, "reference_width_px": None, "pixels_per_mm": None,
            "calibrated": False, "status": "Calibration required",
            "automatic_reference_detection": "Not implemented (future functionality)",
        }
    if reference_width_mm is None or reference_width_px is None:
        return {
            "reference_width_mm": reference_width_mm, "reference_width_px": reference_width_px,
            "pixels_per_mm": None, "calibrated": False,
            "status": "Calibration required: provide known reference width in mm and its width in pixels.",
            "automatic_reference_detection": "Not implemented (future functionality)",
        }
    try:
        scale = pixels_per_mm(reference_width_px, reference_width_mm)
    except (TypeError, ValueError) as exc:
        return {
            "reference_width_mm": reference_width_mm, "reference_width_px": reference_width_px,
            "pixels_per_mm": None, "calibrated": False,
            "status": "Calibration unavailable: supplied reference values are invalid.",
            "automatic_reference_detection": "Not implemented (future functionality)",
        }
    return {
        "reference_width_mm": reference_width_mm, "reference_width_px": reference_width_px,
        "pixels_per_mm": round(scale, 6), "calibrated": True, "status": "Calibrated from supplied reference object.",
        "automatic_reference_detection": "Not implemented (future functionality)",
    }


def detect_reference_object_width_px(image_path: Path) -> float | None:
    """Future extension point for a trained/reference-marker detector.

    No reference-object model is installed in Phase 11, so this deliberately
    returns no value instead of inferring a scale from arbitrary image pixels.
    """
    del image_path
    return None


def get_model_capabilities() -> dict[str, Any]:
    """Describe available model roles without conflating their predictions."""
    quality_classes = get_model_classes()
    return {
        "quality_defect_detection": {
            "available": model is not None,
            "classes": quality_classes,
            "unsupported_classes": sorted(set(quality_classes) - SUPPORTED_QUALITY_CLASSES),
            "message": "Existing YOLO model detects trained defect classes only.",
        },
        "individual_onion_detection": {
            "available": individual_onion_model is not None,
            "classes": ["onion-bulb"] if individual_onion_model is not None else [],
            "message": (
                "Individual onion detection model ready."
                if individual_onion_model is not None
                else "Individual onion detection model not yet available."
            ),
        },
    }


def detect_individual_onions(
    image_path: Path,
    inspection_id: str,
    calibration: CalibrationConfig,
) -> list[IndividualOnionRecord]:
    """Run the dedicated onion detector when one is actually installed.

    The existing bad-onion model is intentionally never used here.
    """
    if individual_onion_model is None:
        return []

    results = individual_onion_model(str(image_path), conf=0.25, verbose=False)
    if not results:
        return []

    records: list[IndividualOnionRecord] = []
    result = results[0]
    for index, box in enumerate(getattr(result, "boxes", []), start=1):
        confidence = float(box.conf[0]) if len(box.conf) else 0.0
        raw_bounding_box = [float(value) for value in box.xyxy[0].tolist()]
        bounding_box = [round(value, 1) for value in raw_bounding_box]
        records.append(
            {
                "onion_id": f"{inspection_id}-onion-{index}",
                "bounding_box": bounding_box,
                "detection_confidence": round(confidence, 3),
                "ai_defects": [],
                "size": measure_bounding_box(
                    raw_bounding_box, calibration_pixels_per_mm=calibration["pixels_per_mm"]
                ),
                "verification_status": "pending",
            }
        )
    return records


def get_model_classes() -> list[str]:
    """Return the class names embedded in the loaded YOLO weights."""
    if model is None:
        return []

    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="ONI-SCAN API",
    description="AI-assisted onion quality assessment backend",
    version="1.0.0",
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/demo-samples")
def demo_samples():
    """List real repository test images for an operator-controlled demo picker."""
    if not DEMO_SAMPLE_DIR.is_dir():
        return {"success": True, "samples": []}
    samples = [
        {"name": path.name, "url": f"/demo-samples/{path.name}"}
        for path in sorted(DEMO_SAMPLE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ][:12]
    return {"success": True, "samples": samples}


@app.get("/demo-samples/{filename}")
def demo_sample(filename: str):
    """Serve only allowlisted files from the repository's test-image directory."""
    candidate = DEMO_SAMPLE_DIR / filename
    try:
        resolved_directory = DEMO_SAMPLE_DIR.resolve()
        resolved_candidate = candidate.resolve()
    except OSError:
        return JSONResponse(status_code=404, content={"success": False, "error": "Demo sample unavailable."})
    if (
        Path(filename).name != filename
        or resolved_candidate.parent != resolved_directory
        or not resolved_candidate.is_file()
        or resolved_candidate.suffix.lower() not in ALLOWED_EXTENSIONS
    ):
        return JSONResponse(status_code=404, content={"success": False, "error": "Demo sample unavailable."})
    return FileResponse(resolved_candidate)


@app.get("/app")
def serve_frontend():
    return FileResponse(ROOT / "index.html")


@app.get("/")
def home():
    classes = get_model_classes()
    return {
        "status": "online",
        "system": "ONI-SCAN",
        "ai_model": "loaded" if model is not None else "unavailable",
        "supports": classes,
        "model_classes": classes,
        "capabilities": get_model_capabilities(),
    }


@app.get("/grading-standard")
def grading_standard():
    """Expose the traceable, configured e-NAM onion standard without grading a lot."""
    return get_standard()


@app.post("/grade")
def grade_inspection(payload: dict[str, Any] = Body(default_factory=dict)):
    """Grade verified procurement observations independently from YOLO inference."""
    inspection_id = payload.get("inspection_id")
    if inspection_id:
        try:
            stored_record = find_inspection(str(inspection_id))
            if stored_record is None:
                return JSONResponse(status_code=404, content={"success": False, "error": "Inspection not found."})
            grading_payload = {
                **payload,
                "ai_observations": stored_record.get("ai_observations") or {},
            }
            result = evaluate_grading(grading_payload)
            update_inspection(
                str(inspection_id),
                {
                    "human_verification": result.get("human_verification", {"status": "Pending"}),
                    "verified_observations": result.get("human_verified_observations", {}),
                    "grading_result": result.get("grading", {}),
                    "inspection_status": (
                        "Graded" if result.get("grading", {}).get("status") == "graded" else "Verification Updated"
                    ),
                },
            )
        except InspectionStorageError as exc:
            return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    else:
        result = evaluate_grading(payload)
    return result


@app.get("/inspections")
def inspections():
    try:
        return {"success": True, "inspections": load_inspections()}
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/analytics")
def analytics():
    try:
        return {"success": True, **build_analytics(load_inspections())}
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.get("/inspections/{inspection_id}")
def inspection_detail(inspection_id: str):
    try:
        record = find_inspection(inspection_id)
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    if record is None:
        return JSONResponse(status_code=404, content={"success": False, "error": "Inspection not found."})
    return {"success": True, "inspection": record}


@app.get("/inspections/{inspection_id}/report")
def inspection_report(inspection_id: str):
    try:
        record = find_inspection(inspection_id)
        if record is None:
            return JSONResponse(status_code=404, content={"success": False, "error": "Inspection not found."})
        pdf_bytes = build_inspection_pdf(record)
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "error": "The inspection report could not be generated."})

    return Response(
        content=pdf_bytes,
        media_type=REPORT_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="ONI-SCAN-{inspection_id}.pdf"'},
    )


@app.post("/inspections/{inspection_id}/finalize")
def finalize_inspection(inspection_id: str):
    try:
        record = find_inspection(inspection_id)
        if record is None:
            return JSONResponse(status_code=404, content={"success": False, "error": "Inspection not found."})
        grading = record.get("grading_result") or {}
        verification = record.get("human_verification") or {}
        missing: list[str] = []
        if str(verification.get("status", "")).lower() != "verified":
            missing.append("human verification")
        if grading.get("status") != "graded":
            missing.append("completed grading result")
        if missing:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Cannot finalize inspection. Missing: " + ", ".join(missing) + "."},
            )
        finalized = update_inspection(
            inspection_id,
            {"inspection_status": "Finalized", "finalized_at": datetime.now(timezone.utc).isoformat()},
        )
        return {"success": True, "inspection": finalized}
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


@app.post("/analyze")
async def analyze(
    file: UploadFile | None = File(default=None),
    reference_width_mm: float | None = Form(default=None),
    reference_width_px: float | None = Form(default=None),
):
    if file is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No image file was uploaded."},
        )

    if file.filename is None or not file.filename.strip():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "The uploaded file is missing a valid filename."},
        )

    content_type = (file.content_type or "").lower()
    file_ext = Path(file.filename).suffix.lower()

    if content_type not in ALLOWED_MIME_TYPES and file_ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Only JPG, PNG, BMP, and WEBP image files are allowed."},
        )

    uploaded_bytes = await file.read()

    if not uploaded_bytes:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "The uploaded image is empty."},
        )

    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "The uploaded image is too large. Please use a smaller file."},
        )

    if not is_valid_image_bytes(uploaded_bytes):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "The uploaded file is not a valid image or the file is corrupted."},
        )

    if model is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "The AI model is unavailable. Please check the model file and service configuration."},
        )

    safe_name = sanitize_filename(file.filename)
    safe_upload_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = UPLOAD_DIR / safe_upload_name

    try:
        with image_path.open("wb") as buffer:
            buffer.write(uploaded_bytes)
    except OSError as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "The uploaded image could not be stored. Please try again."},
        )

    try:
        results = model(str(image_path), conf=MODEL_CONFIDENCE_THRESHOLD, verbose=False)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "AI inference failed. Please try again or contact the service administrator."},
        )

    if not results or len(results) == 0:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "AI inference returned no results."},
        )

    result = results[0]
    inspection_id = f"inspection-{uuid.uuid4().hex}"
    detected_reference_width_px = (
        reference_width_px
        if reference_width_px is not None
        else detect_reference_object_width_px(image_path)
    )
    calibration = build_calibration_config(reference_width_mm, detected_reference_width_px)
    individual_detection_error = None
    try:
        individual_onions = detect_individual_onions(image_path, inspection_id, calibration)
    except Exception as exc:  # pragma: no cover - optional model runtime safety
        individual_onions = []
        individual_detection_error = "Individual onion detection failed; quality analysis remains available."

    detections = []

    try:
        for box in getattr(result, "boxes", []):
            confidence_value = float(box.conf[0]) if len(box.conf) > 0 else 0.0
            class_id = int(box.cls[0]) if len(box.cls) > 0 else 0
            class_name = result.names.get(class_id, "unknown") if hasattr(result, "names") and result.names else "unknown"
            raw_bounding_box = [float(value) for value in box.xyxy[0].tolist()]
            bounding_box = [round(value, 1) for value in raw_bounding_box]
            detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence_value, 3),
                    "box": bounding_box,
                    "size": measure_bounding_box(
                        raw_bounding_box, calibration_pixels_per_mm=calibration["pixels_per_mm"]
                    ),
                }
            )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "The AI detection result could not be read."},
        )

    try:
        result_filename = f"{uuid.uuid4().hex}_result.jpg"
        result_path = UPLOAD_DIR / result_filename
        save_annotated_image(image_path, result_path, detections)
    except (OSError, ValueError):
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "The annotated result image could not be saved."},
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    inspection_record = build_inspection_record(
        inspection_id=inspection_id,
        timestamp=timestamp,
        filename=safe_name,
        detections=detections,
        calibration=calibration,
        result_image=f"/uploads/{result_filename}",
    )
    try:
        append_inspection(inspection_record)
    except InspectionStorageError as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})

    quality = build_quality_summary(detections)

    return JSONResponse(
        content={
            "success": True,
            "filename": safe_name,
            "result_image": f"/uploads/{result_filename}",
            "annotated_image": f"/uploads/{result_filename}",
            "detections": detections,
            "count": len(detections),
            "total_detected": quality["total_detected"],
            "average_confidence": quality["average_confidence"],
            "confidence_threshold": MODEL_CONFIDENCE_THRESHOLD,
            "quality": {
                "counts": quality["class_counts"],
                "percentages": quality["percentages"],
                "raw_class_counts": quality["raw_class_counts"],
            },
            "model_classes": get_model_classes(),
            "inspection_id": inspection_id,
            "timestamp": timestamp,
            "individual_onions": individual_onions,
            "individual_detection_error": individual_detection_error,
            "calibration": calibration,
            "size_rule": {
                "configurable": True,
                "threshold_mm": None,
                "decision": None,
                "status": "No verified procurement size standard configured.",
            },
            "size_measurement_notice": (
                "Size measurement is an AI-assisted estimate based on image calibration. "
                "Final procurement classification requires verification against the applicable standard."
            ),
            "unsupported_model_classes": sorted(set(get_model_classes()) - SUPPORTED_QUALITY_CLASSES),
            "capabilities": get_model_capabilities(),
        }
    )
