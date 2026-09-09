from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

print("=" * 60)
print("          ONI-SCAN DATASET PREPARATION")
print("=" * 60)

# Create required directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("\n[OK] Dataset folders are ready.")

# Look for files already placed in raw/
files = [p for p in RAW_DIR.rglob("*") if p.is_file()]

print(f"[INFO] Files currently in dataset/raw: {len(files)}")

if not files:
    print("\n[WAITING]")
    print("dataset/raw is empty.")
    print("No real training data has been added yet.")
    print("\nONI-SCAN will NOT create fake images or fake labels.")
    print("Add a legitimate onion dataset before training.")
else:
    print("\nFiles found:")
    for file in files[:30]:
        print(f"  - {file.relative_to(RAW_DIR)}")

    if len(files) > 30:
        print(f"  ... and {len(files) - 30} more")

# Create a dataset information file
info = {
    "project": "ONI-SCAN",
    "purpose": "Onion quality assessment",
    "status": "awaiting_real_dataset",
    "classes_planned": [
        "acceptable",
        "damaged",
        "visible_rot",
        "sprouted"
    ],
    "note": (
        "Classes must be verified against the actual dataset annotations "
        "before training."
    )
}

info_file = PROCESSED_DIR / "dataset_info.json"

with open(info_file, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=4)

print(f"\n[OK] Created: {info_file.relative_to(ROOT)}")

print("\n" + "=" * 60)
print("DATASET PREPARATION CHECK COMPLETE")
print("=" * 60)