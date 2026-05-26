"""Participant naming and result export helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import generated_results_dir, participant_data_dir


def make_abbrev(full_name: str) -> str:
    """Derive an abbreviation from a participant name.

    "First Last" -> "F.L". Falls back to the raw name if it cannot be split.
    """
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return ".".join(p[0].upper() for p in parts)
    return full_name.strip().replace(" ", "_") or "anonymous"


def _write_results_file(
    csv_path: Path,
    participant_name: str,
    condition: str,
    elapsed: float | None,
    rows: list[tuple[str, float, float, float]],
) -> None:
    """Write one condition CSV in the experiment's expected format."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Participant: {participant_name}"])
        if elapsed is not None:
            writer.writerow([f"Time: {elapsed:.2f}"])
        else:
            writer.writerow(["Time:"])
        writer.writerow([f"Condition: {condition}"])
        writer.writerow([])
        writer.writerow(["mask_png", "x", "y", "z"])
        for name, x_csv, y_csv, z_csv in rows:
            writer.writerow([
                name,
                f"{x_csv:.6f}",
                f"{y_csv:.6f}",
                f"{z_csv:.6f}",
            ])


def export_results_csv(experiment: Any) -> Path | None:
    """Export placed experiment points to a condition-specific CSV file."""
    condition = experiment.current_condition or "unknown"
    condition_key = condition.lower()

    raw_name = experiment.name_input.text().strip() or "anonymous"
    participant_name = make_abbrev(raw_name) if raw_name != "anonymous" else "anonymous"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{participant_name}_{ts}.csv"

    elapsed = None
    if experiment.start_time is not None:
        elapsed = (datetime.now() - experiment.start_time).total_seconds()

    rows = []
    for name, xn, yn, zn in experiment._collect_combined_points_norm():
        if condition_key == "2d":
            x_csv, y_csv, z_csv = xn, yn, 0.0
        else:
            x_csv, y_csv, z_csv = xn, zn, yn
        rows.append((name, x_csv, y_csv, z_csv))

    raw_csv_path = generated_results_dir() / condition_key / csv_name
    participant_csv_path = (
        participant_data_dir() / participant_name / condition_key / csv_name
    )

    try:
        _write_results_file(raw_csv_path, participant_name, condition, elapsed, rows)
        _write_results_file(
            participant_csv_path, participant_name, condition, elapsed, rows
        )
    except Exception as exc:
        experiment.logger.log_session_event(f"Failed to write CSV: {exc}")
        return None

    experiment.logger.log_session_event(
        f"Exported {condition.upper()} CSV: {raw_csv_path}"
    )
    return raw_csv_path
