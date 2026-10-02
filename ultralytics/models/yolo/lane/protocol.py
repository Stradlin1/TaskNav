# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


# These values mirror scripts/manual_fix_56anchors_v11_class1_789_update.py.
# The test suite imports that script and checks that the two definitions do not drift apart.
MANUAL_ROW_ANCHORS = 56
MANUAL_X_BINS = 640
MANUAL_Y_START = 1.0
MANUAL_CROP_RATIO = 2.0 / 3.0
MANUAL_Y_END = 1.0 - MANUAL_CROP_RATIO
MANUAL_ABSENT_X = -1.0
MANUAL_Y_ATOL = 1.1e-6  # Labels are written with six decimal places.


def manual_y_anchors(dtype=np.float32) -> np.ndarray:
    """Return the fixed bottom-to-top row anchors emitted by the manual annotation tool."""
    return np.linspace(MANUAL_Y_START, MANUAL_Y_END, MANUAL_ROW_ANCHORS, dtype=dtype)


def validate_manual_geometry(row_anchors: int, y_start: float, y_end: float) -> None:
    """Reject training geometry that disagrees with the manual annotation protocol."""
    if int(row_anchors) != MANUAL_ROW_ANCHORS:
        raise ValueError(
            f"Manual lane protocol requires row_anchors={MANUAL_ROW_ANCHORS}, got {row_anchors}."
        )
    if not math.isclose(float(y_start), MANUAL_Y_START, rel_tol=0.0, abs_tol=MANUAL_Y_ATOL):
        raise ValueError(f"Manual lane protocol requires y_start={MANUAL_Y_START}, got {y_start}.")
    if not math.isclose(float(y_end), MANUAL_Y_END, rel_tol=0.0, abs_tol=MANUAL_Y_ATOL):
        raise ValueError(f"Manual lane protocol requires y_end={MANUAL_Y_END}, got {y_end}.")


def parse_manual_label(text: str, label_path: str | Path, num_lanes: int) -> list[tuple[int, np.ndarray]]:
    """Parse and strictly validate one manual label file.

    Each non-empty line must be exactly::

        task_id x0 y0 x1 y1 ... x55 y55

    Rows are ordered from image bottom to the top edge of the lower two thirds.
    Valid x values are normalized to [0, 1]; exactly -1 marks an absent row.
    """
    label_path = Path(label_path)
    expected_y = manual_y_anchors(dtype=np.float64)
    expected_fields = 1 + 2 * MANUAL_ROW_ANCHORS
    seen_tasks: set[int] = set()
    records: list[tuple[int, np.ndarray]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if len(fields) != expected_fields:
            raise ValueError(
                f"Manual lane label {label_path}:{line_number} has {len(fields)} fields; "
                f"expected {expected_fields} (task_id plus {MANUAL_ROW_ANCHORS} x/y pairs)."
            )

        try:
            task_value = float(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"Manual lane label {label_path}:{line_number} has invalid task_id {fields[0]!r}."
            ) from exc
        if not math.isfinite(task_value) or not task_value.is_integer():
            raise ValueError(f"Manual lane label {label_path}:{line_number} has non-integer task_id {fields[0]!r}.")
        task_id = int(task_value)
        if not 0 <= task_id < int(num_lanes):
            raise ValueError(
                f"Manual lane label {label_path}:{line_number} has task_id={task_id}; "
                f"expected 0..{int(num_lanes) - 1}."
            )
        if task_id in seen_tasks:
            raise ValueError(f"Manual lane label {label_path}:{line_number} repeats task_id={task_id}.")
        seen_tasks.add(task_id)

        xs = np.empty(MANUAL_ROW_ANCHORS, dtype=np.float32)
        for row in range(MANUAL_ROW_ANCHORS):
            x_text = fields[1 + 2 * row]
            y_text = fields[2 + 2 * row]
            try:
                x = float(x_text)
                y = float(y_text)
            except ValueError as exc:
                raise ValueError(
                    f"Manual lane label {label_path}:{line_number} row {row} contains a non-numeric x/y value."
                ) from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"Manual lane label {label_path}:{line_number} row {row} contains NaN or Inf: x={x}, y={y}."
                )
            if x != MANUAL_ABSENT_X and not 0.0 <= x <= 1.0:
                raise ValueError(
                    f"Manual lane label {label_path}:{line_number} row {row} has x={x}; "
                    f"expected -1 or a normalized value in [0, 1]."
                )
            if not math.isclose(y, float(expected_y[row]), rel_tol=0.0, abs_tol=MANUAL_Y_ATOL):
                raise ValueError(
                    f"Manual lane label {label_path}:{line_number} row {row} has y={y}; "
                    f"expected {expected_y[row]:.9f} in bottom-to-top order."
                )
            xs[row] = x
        records.append((task_id, xs))

    return records
