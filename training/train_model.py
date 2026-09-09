from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset" / "raw" / "bad_onion" / "data.yaml"
MODEL = ROOT / "yolo11n.pt"

print("=" * 60)
print("             ONI-SCAN MODEL TRAINING")
print("=" * 60)

print(f"\nDataset: {DATASET}")
print(f"Base model: {MODEL}")

model = YOLO(str(MODEL))

results = model.train(
    data=str(DATASET),
    epochs=5,
    imgsz=416,
    batch=4,
    patience=10,
    project=str(ROOT / "models"),
    name="oni_scan_bad_onion",
    exist_ok=True
)

print("\n" + "=" * 60)
print("             TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest model should be here:")
print(ROOT / "models" / "oni_scan_bad_onion" / "weights" / "best.pt")
