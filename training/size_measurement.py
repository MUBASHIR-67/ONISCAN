"""Calibration-based geometry helpers for ONI-SCAN measurements.

Pixel geometry is available for every YOLO box.  Physical dimensions are only
returned when a valid image-specific reference scale is supplied.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Sequence


def bounding_box_dimensions_px(box: Sequence[float]) -> tuple[float, float]:
    """Return the non-negative width and height of an ``[x1, y1, x2, y2]`` box."""
    if len(box) != 4:
        raise ValueError("A bounding box must contain exactly four coordinates.")
    x1, y1, x2, y2 = (float(value) for value in box)
    return abs(x2 - x1), abs(y2 - y1)


def approximate_diameter_px(width_px: float, height_px: float) -> float:
    """Estimate a circular object's diameter from its bounding-box axes.

    The root-mean-square axis is used so neither axis is silently discarded.
    This is an image-geometry estimate, not a physical measurement by itself.
    """
    width, height = float(width_px), float(height_px)
    if width < 0 or height < 0:
        raise ValueError("Pixel dimensions cannot be negative.")
    return sqrt((width * width + height * height) / 2)


def pixels_per_mm(reference_width_px: float, reference_width_mm: float) -> float:
    """Calculate image scale from a reference object of known width."""
    reference_px, reference_mm = float(reference_width_px), float(reference_width_mm)
    if reference_px <= 0 or reference_mm <= 0:
        raise ValueError("Reference widths in pixels and millimetres must be greater than zero.")
    return reference_px / reference_mm


def pixels_to_mm(pixel_value: float, calibration_pixels_per_mm: float) -> float:
    """Convert a pixel dimension to millimetres using a valid scale."""
    scale = float(calibration_pixels_per_mm)
    if scale <= 0:
        raise ValueError("Calibration pixels-per-mm must be greater than zero.")
    return float(pixel_value) / scale


def measure_bounding_box(
    box: Sequence[float],
    *,
    calibration_pixels_per_mm: float | None = None,
) -> dict[str, Any]:
    """Return real pixel geometry and, only when calibrated, physical diameter."""
    width_px, height_px = bounding_box_dimensions_px(box)
    diameter_px = approximate_diameter_px(width_px, height_px)
    calibrated = calibration_pixels_per_mm is not None
    diameter_mm = pixels_to_mm(diameter_px, calibration_pixels_per_mm) if calibrated else None
    return {
        "width_px": round(width_px, 2),
        "height_px": round(height_px, 2),
        "diameter_px": round(diameter_px, 2),
        "diameter_mm": round(diameter_mm, 2) if diameter_mm is not None else None,
        "calibrated": calibrated,
    }
