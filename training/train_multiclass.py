"""Train the future ONI-SCAN four-class model after dataset approval.

This script is intentionally explicit and conservative for a CPU laptop. It
refuses to start until the prepared dataset passes validation and all four
classes appear in the labels. It is never invoked automatically.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "dataset" / "multiclass" / "data.yaml"
BASE_MODEL = ROOT / "yolo11n.pt"
VALIDATOR = ROOT / "training" / "validate_multiclass_dataset.py"
OUTPUT_DIR = ROOT / "models"


def validate_before_training() -> None:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Missing dataset configuration: {DATASET_YAML}")
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Missing pretrained model: {BASE_MODEL}")
    result = subprocess.run([sys.executable, str(VALIDATOR)], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("Multiclass dataset validation failed; training was not started.")


def main() -> None:
    validate_before_training()
    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        batch=2,
        workers=0,
        device="cpu",
        project=str(OUTPUT_DIR),
        name="oni_scan_multiclass",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
