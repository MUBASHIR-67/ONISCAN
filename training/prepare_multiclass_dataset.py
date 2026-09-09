"""Prepare verified source datasets for the ONI-SCAN multiclass dataset.

The default mode is a dry-run. Copying is opt-in with --execute and requires a
complete, non-ambiguous mapping for every source class used by the source
annotations. No downloads, synthetic files, or automatic class guesses occur.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset"
TARGET_ROOT = DATASET_ROOT / "multiclass"
MAPPING_PATH = ROOT / "training" / "class_mapping.json"
TARGET_CLASSES = ("acceptable", "damaged", "visible_rot", "sprouted")
TARGET_IDS = {name: index for index, name in enumerate(TARGET_CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def load_mapping() -> dict:
    data = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    if data.get("target_classes") != list(TARGET_CLASSES):
        raise ValueError(f"{MAPPING_PATH} must define exactly {TARGET_CLASSES}")
    return data


def parse_source_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing source data.yaml: {data_yaml}")
    text = data_yaml.read_text(encoding="utf-8")
    match = re.search(r"^names:\s*(\[.*\])\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read names from {data_yaml}")
    names = ast.literal_eval(match.group(1))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Invalid names in {data_yaml}")
    return names


def source_split_dirs(source_root: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for directory in sorted(source_root.iterdir()):
        if not directory.is_dir() or directory.name.lower() not in SPLIT_ALIASES:
            continue
        split = SPLIT_ALIASES[directory.name.lower()]
        image_dir = directory / "images"
        label_dir = directory / "labels"
        if image_dir.is_dir() and label_dir.is_dir():
            pairs.append((split, directory))
    return pairs


def prepare_source(source_name: str, execute: bool, mapping_data: dict) -> dict[str, int]:
    source = mapping_data.get("sources", {}).get(source_name)
    if not source:
        raise ValueError(f"Unknown source dataset in mapping: {source_name}")

    source_root = ROOT / source["source_path"]
    names = parse_source_names(source_root / "data.yaml")
    mapping = source.get("mapping", {})
    unresolved = [name for name in names if mapping.get(name) not in TARGET_IDS]
    if unresolved:
        raise ValueError(
            f"Source {source_name} has unresolved classes: {unresolved}. "
            "Manual review and an explicit mapping are required before --execute."
        )

    split_dirs = source_split_dirs(source_root)
    if not split_dirs:
        raise ValueError(f"No YOLO train/val/test split directories found under {source_root}")

    summary = Counter[str]()
    for split, split_root in split_dirs:
        images = sorted(path for path in (split_root / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        for image_path in images:
            label_path = split_root / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise ValueError(f"Missing label for source image: {image_path}")

            output_stem = f"{source_name}__{image_path.stem}"
            output_image = TARGET_ROOT / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
            output_label = TARGET_ROOT / "labels" / split / f"{output_stem}.txt"
            converted_lines: list[str] = []
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = raw_line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label {label_path}:{line_number}")
                source_id = int(fields[0])
                if source_id < 0 or source_id >= len(names):
                    raise ValueError(f"Invalid source class id {source_id} in {label_path}:{line_number}")
                target_name = mapping[names[source_id]]
                converted_lines.append(" ".join([str(TARGET_IDS[target_name]), *fields[1:]]))
                summary[target_name] += 1

            summary[split] += 1
            if execute:
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, output_image)
                output_label.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")

    return dict(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="Mapped source name; may be repeated")
    parser.add_argument("--execute", action="store_true", help="Copy and convert after all mappings pass validation")
    args = parser.parse_args()

    mapping_data = load_mapping()
    sources = args.source or list(mapping_data.get("sources", {}))
    print("ONI-SCAN MULTICLASS PREPARATION")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    blocked = False

    for source_name in sources:
        print(f"\nSource: {source_name}")
        try:
            summary = prepare_source(source_name, args.execute, mapping_data)
        except (FileNotFoundError, ValueError) as error:
            blocked = True
            print(f"BLOCKED: {error}")
            continue
        print(json.dumps(summary, indent=2))

    if blocked:
        print("\nPreparation is blocked until every selected source has verified mappings and YOLO pairs.")
        return 1 if args.execute else 0

    if not args.execute:
        print("\nNo files were copied. Add reviewed mappings, then rerun with --execute.")
    else:
        print("\nPreparation completed. Run training/validate_multiclass_dataset.py before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
