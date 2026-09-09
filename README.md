# ONI-SCAN

ONI-SCAN is an AI-assisted onion quality assessment and procurement inspection application for the Smart India Hackathon demo. It combines real YOLO visual detection, image geometry, optional image calibration, human verification, configured grading rules, inspection history, analytics, and PDF reports.

## Problem and Solution

Manual onion inspection is slow and difficult to trace consistently. ONI-SCAN provides visible, evidence-backed assistance: an inspector uploads an onion sample, reviews the actual detections and measurements, supplies verified procurement observations, and finalizes a traceable digital inspection report.

The system does not treat AI output as an official procurement decision. Human verification is required before grading and finalization.

## Architecture

- `index.html`: single-page inspection dashboard and workflow UI.
- `backend/main.py`: FastAPI application, upload validation, YOLO inference, annotation rendering, calibration, persistence, and API routes.
- `backend/grading_engine.py`: deterministic verification-first grading against `backend/grading_standard.json`.
- `backend/inspection_storage.py`: JSON-backed persistent inspection history.
- `backend/report_generator.py`: PDF report generation from stored inspection data.
- `training/size_measurement.py`: deterministic pixel geometry and calibration calculations.
- `backend/data/inspections.json`: persisted inspection records.
- `backend/uploads/`: uniquely named uploaded and annotated inspection images.

## Technology Stack

- Python, FastAPI, Uvicorn
- Ultralytics YOLO and PyTorch
- Pillow for image validation and evidence annotation
- ReportLab for PDF reports
- HTML, CSS, and browser JavaScript
- JSON persistence for the current prototype

## AI Pipeline

1. The backend validates the uploaded image and size limit.
2. The configured YOLO model runs once at the configured confidence threshold.
3. The response preserves the model's class name, confidence, and `[x1, y1, x2, y2]` box.
4. A unique annotated image is rendered from those real detections.
5. Pixel dimensions are calculated from each real box.
6. Physical diameter is calculated only when valid image-specific calibration is supplied.

The currently loaded model reports its actual class as `bad-onion`. The backend does not rename model classes or invent unsupported defect categories.

## Dataset and Training

Prepared onion-quality data and dataset policy are documented in `dataset/onion_quality/README.md`. Training and validation scripts are under `training/`. Trained weights are stored under `models/`. Dataset labels must be manually verified according to the documented class policy; filenames are not treated as labels.

## Calibration

Calibration requires a known reference width in millimetres and its measured width in pixels in the same image plane:

`pixels_per_mm = reference_width_px / reference_width_mm`

Without valid calibration, pixel measurements remain available but `diameter_mm` is `null`. Zero, negative, missing, or malformed values do not produce physical measurements.

## Grading and Human Verification

AI observations are retained for traceability and are not silently converted into official procurement parameters. An inspector supplies verified observations and selects `Range-I`, `Range-II`, or `Range-III`. The deterministic grading engine applies the configured rules and only issues a range when required verified fields and rule checks pass.

## API Endpoints

- `GET /`: service and model capabilities.
- `GET /app`: frontend.
- `POST /analyze`: validate, infer, measure, annotate, and persist an image inspection.
- `POST /grade`: store human verification and evaluate configured grading rules.
- `POST /inspections/{inspection_id}/finalize`: finalize only verified, graded inspections.
- `GET /inspections`: inspection history.
- `GET /inspections/{inspection_id}`: stored inspection detail.
- `GET /inspections/{inspection_id}/report`: PDF report from stored data.
- `GET /analytics`: analytics calculated from stored records.
- `GET /grading-standard`: configured grading standard metadata.
- `GET /demo-samples`: allowlisted real test images for the live demo picker.
- `GET /demo-samples/{filename}`: serve an allowlisted test image only.

## Run Locally on Windows

Use the repository virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app` in a browser. The `start_oni.bat` launcher is also available when the environment's `python` command is configured correctly.

## Demo Workflow

1. Select `START NEW INSPECTION`.
2. Upload an image or choose a real `SAMPLE IMAGE`.
3. Start AI inspection and review the original and annotated evidence.
4. Review class, confidence, boxes, and pixel measurements.
5. Add reference measurements when physical size is required.
6. Enter only human-verified procurement observations.
7. Submit verification and configured grading.
8. Finalize the inspection after required information is complete.
9. View or download the stored digital report.
10. Open history and analytics without deleting prior inspections.

## Known Limitations

- The current loaded model reports `bad-onion`; defect-specific categories are not claimed unless emitted by the model.
- RGB analysis identifies visible characteristics. Internal defects that are not externally visible cannot be reliably determined from a standard RGB image alone.
- Automatic reference-object detection is not implemented; calibration is manual.
- Individual onion detection is optional and unavailable unless its separate trained model is installed.
- JSON persistence is suitable for the current prototype/demo, not concurrent production deployment.
- Final procurement classification requires qualified human verification against the configured standard.
