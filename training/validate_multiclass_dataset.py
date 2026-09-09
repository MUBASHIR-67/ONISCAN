"""Validate the prepared ONI-SCAN four-class YOLO dataset.

This validator never repairs, relabels, downloads, or creates data. It fails
when an image/label pair or annotation is not valid for the configured schema.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "multiclass"
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("acceptable", "damaged", "visible_rot", "sprouted")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_label(label_path: Path, split: str, class_counts: Counter[str]) -> list[str]:
    errors: list[str] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: expected 5 YOLO fields")
            continue

        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: class id and coordinates must be numeric")
            continue

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: invalid class id {class_id}")
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: coordinates must be finite")
        elif not all(0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: coordinates must be between 0 and 1")
        elif coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            errors.append(f"{label_path.relative_to(ROOT)}:{line_number}: width and height must be positive")

        if 0 <= class_id < len(CLASS_NAMES):
            class_counts[CLASS_NAMES[class_id]] += 1

    return errors


def validate_split(split: str) -> tuple[dict[str, int], list[str]]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split
    errors: list[str] = []
    counts = Counter[str]()

    if not image_dir.is_dir():
        errors.append(f"missing image directory: {image_dir.relative_to(ROOT)}")
    if not label_dir.is_dir():
        errors.append(f"missing label directory: {label_dir.relative_to(ROOT)}")
    if errors:
        return {"images": 0, "annotations": 0, **{name: 0 for name in CLASS_NAMES}}, errors

    images = image_files(image_dir)
    image_stems = {image.stem for image in images}
    labels = sorted(label_dir.glob("*.txt"))
    label_stems = {label.stem for label in labels}

    for image in images:
        label_path = label_dir / f"{image.stem}.txt"
        if not label_path.exists():
            errors.append(f"missing label for image: {image.relative_to(ROOT)}")
    for label in labels:
        if label.stem not in image_stems:
            errors.append(f"label has no matching image: {label.relative_to(ROOT)}")

    for label in labels:
        if label.stem in image_stems:
            errors.extend(validate_label(label, split, counts))

    annotation_count = sum(counts.values())
    return {
        "images": len(images),
        "annotations": annotation_count,
        **{name: counts[name] for name in CLASS_NAMES},
    }, errors


def main() -> int:
    print("ONI-SCAN MULTICLASS DATASET VALIDATION")
    print(f"Dataset: {DATASET.relative_to(ROOT)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    all_errors: list[str] = []
    totals = Counter[str]()
    split_counts: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        counts, errors = validate_split(split)
        split_counts[split] = counts
        totals.update(counts)
        print(f"{split:>5}: {counts['images']} images, {counts['annotations']} annotations")
        for name in CLASS_NAMES:
            print(f"       {name}: {counts[name]}")
        all_errors.extend(errors)

    if all_errors:
        print("\nINVALID DATASET")
        for error in all_errors:
            print(f"- {error}")
        print(f"\nFound {len(all_errors)} validation error(s).")
        return 1

    if totals["images"] == 0:
        print("\nINVALID DATASET")
        print("- no images are present; add reviewed real data before training")
        return 1

    print("\nVALID DATASET")
    print(f"Total images: {totals['images']}")
    print(f"Total annotations: {totals['annotations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
