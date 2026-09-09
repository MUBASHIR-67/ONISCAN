"""Train ONI-SCAN with the real four-class onion quality dataset.

This script refuses to train when the dataset is missing or when the labels do
not contain all configured classes. It must never turn the existing
single-class model into fabricated multi-class predictions.
"""

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "raw" / "onion_quality"
DATA_YAML = DATASET / "data.yaml"
BASE_MODEL = ROOT / "yolo11n.pt"
OUTPUT_DIR = ROOT / "models"
EXPECTED_CLASSES = {"0", "1", "2", "3"}


def label_ids() -> set[str]:
    ids: set[str] = set()
    for label_file in DATASET.rglob("labels/*.txt"):
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                ids.add(line.split()[0])
    return ids


def validate_dataset() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing dataset config: {DATA_YAML}")
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Missing base model: {BASE_MODEL}")

    image_files = list(DATASET.rglob("images/*"))
    ids = label_ids()
    if not image_files or not ids:
        raise RuntimeError(
            "Add real onion images and YOLO labels under dataset/raw/onion_quality "
            "before training. No synthetic data is generated."
        )
    missing = EXPECTED_CLASSES - ids
    if missing:
        raise RuntimeError(
            "The dataset is not yet four-class: missing label ids "
            f"{sorted(missing)}. Expected 0=acceptable, 1=bad-onion, "
            "2=visible-rot, 3=sprouted."
        )


if __name__ == "__main__":
    validate_dataset()
    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=80,
        imgsz=640,
        batch=8,
        project=str(OUTPUT_DIR),
        name="oni_scan_onion_quality",
        exist_ok=True,
    )