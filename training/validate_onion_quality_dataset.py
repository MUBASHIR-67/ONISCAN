"""Fail-fast validation for the manual ONI-SCAN onion-quality dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "onion_quality"
MANIFEST = DATASET / "manifest.csv"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANIFEST_COLUMNS = {
    "source_dataset",
    "original_filename",
    "new_filename",
    "original_class",
    "verified_oni_scan_class",
    "license_source",
    "annotation_status",
    "reviewer_status",
    "source_split",
    "split",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(errors: list[str]) -> list[dict[str, str]]:
    if not MANIFEST.exists():
        fail(errors, f"Missing manifest: {MANIFEST}")
        return []
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = MANIFEST_COLUMNS - fields
        if missing:
            fail(errors, f"Manifest is missing columns: {sorted(missing)}")
        return list(reader)


def validate_dataset() -> int:
    errors: list[str] = []
    rows = read_manifest(errors)
    manifest_by_path: dict[str, dict[str, str]] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[str] = Counter()
    seen_hashes: dict[str, tuple[str, str]] = {}
    actual_paths: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        relative = row.get("new_filename", "").replace("\\", "/")
        if relative in manifest_by_path:
            fail(errors, f"Duplicate manifest new_filename at row {row_number}: {relative}")
        manifest_by_path[relative] = row
        split = row.get("split", "")
        if split not in SPLITS:
            fail(errors, f"Invalid manifest split at row {row_number}: {split}")
        source_key = f"{row.get('source_dataset', '')}|{row.get('original_filename', '')}"
        source_splits[source_key].add(split)
        verified = row.get("verified_oni_scan_class", "")
        if verified and verified not in {"acceptable", "bad-onion", "visible-rot", "sprouted"}:
            fail(errors, f"Invalid verified class at row {row_number}: {verified}")

    for split in SPLITS:
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        if not image_dir.is_dir():
            fail(errors, f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            fail(errors, f"Missing label directory: {label_dir}")

        images = [p for p in image_dir.iterdir() if p.is_file()] if image_dir.is_dir() else []
        labels = [p for p in label_dir.iterdir() if p.is_file()] if label_dir.is_dir() else []
        image_stems = {p.stem for p in images}

        for image in images:
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                fail(errors, f"Unsupported image file: {image}")
            relative = image.relative_to(DATASET).as_posix()
            actual_paths.add(relative)
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                fail(errors, f"Missing label file for image: {image}")
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as exc:
                fail(errors, f"Corrupted image {image}: {exc}")
            digest = image_hash(image)
            previous = seen_hashes.get(digest)
            if previous and previous[1] != split:
                fail(errors, f"Duplicate image hash across splits: {image} and {previous[0]}")
            else:
                seen_hashes[digest] = (relative, split)

            manifest_row = manifest_by_path.get(relative)
            if manifest_row is None:
                fail(errors, f"Image missing from manifest: {relative}")

        for label in labels:
            if label.suffix.lower() != ".txt":
                fail(errors, f"Unexpected file in label directory: {label}")
            if label.stem not in image_stems:
                fail(errors, f"Label has no matching image: {label}")
            try:
                lines = label.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                fail(errors, f"Unreadable label file {label}: {exc}")
                continue
            for line_number, line in enumerate(lines, start=1):
                values = line.split()
                if len(values) != 5:
                    fail(errors, f"Invalid YOLO field count in {label}:{line_number}")
                    continue
                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError:
                    fail(errors, f"Non-numeric YOLO annotation in {label}:{line_number}")
                    continue
                if class_id not in range(4):
                    fail(errors, f"Class ID outside 0-3 in {label}:{line_number}")
                width, height = coordinates[2:]
                if not all(0 <= value <= 1 for value in coordinates):
                    fail(errors, f"Unnormalized box in {label}:{line_number}")
                if width <= 0 or height <= 0:
                    fail(errors, f"Non-positive box dimensions in {label}:{line_number}")
                class_counts[str(class_id)] += 1

    for relative in actual_paths - set(manifest_by_path):
        fail(errors, f"Dataset image is absent from manifest: {relative}")
    for source_key, splits in source_splits.items():
        if len(splits) > 1:
            fail(errors, f"Source appears in multiple splits: {source_key} -> {sorted(splits)}")

    print("Class distribution:")
    for class_id, name in enumerate(("acceptable", "bad-onion", "visible-rot", "sprouted")):
        print(f"  {class_id} {name}: {class_counts[str(class_id)]} objects")
    print(f"Images: {len(actual_paths)}")
    print(f"Manifest rows: {len(rows)}")

    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(validate_dataset())