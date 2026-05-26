"""Participant naming and result export helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import app_data_dir


def make_abbrev(full_name: str) -> str:
    """Derive an abbreviation from a participant name.

    "First Last" -> "F.L". Falls back to the raw name if it cannot be split.
    """
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return ".".join(p[0].upper() for p in parts)
    return full_name.strip().replace(" ", "_") or "anonymous"


def export_results_csv(experiment: Any) -> Path | None:
    """Export placed experiment points to a condition-specific CSV file."""
    results_root = app_data_dir() / "results"
    condition = experiment.current_condition or "unknown"
    condition_dir = results_root / condition.lower()

    try:
        condition_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        experiment.logger.log_session_event(f"Failed to create results directory: {exc}")
        return None

    raw_name = experiment.name_input.text().strip() or "anonymous"
    participant_name = make_abbrev(raw_name) if raw_name != "anonymous" else "anonymous"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = condition_dir / f"{participant_name}_{ts}.csv"

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"Participant: {participant_name}"])
            if experiment.start_time is not None:
                elapsed = (datetime.now() - experiment.start_time).total_seconds()
                writer.writerow([f"Time: {elapsed:.2f}"])
            else:
                writer.writerow(["Time:"])
            writer.writerow([f"Condition: {condition}"])
            writer.writerow([])
            writer.writerow(["mask_png", "x", "y", "z"])

            rows = experiment._collect_combined_points_norm()
            for name, xn, yn, zn in rows:
                if experiment.current_condition == "2d":
                    x_csv, y_csv, z_csv = xn, yn, 0.0
                else:
                    x_csv, y_csv, z_csv = xn, zn, yn
                writer.writerow([
                    name,
                    f"{x_csv:.6f}",
                    f"{y_csv:.6f}",
                    f"{z_csv:.6f}",
                ])
    except Exception as exc:
        experiment.logger.log_session_event(f"Failed to write CSV: {exc}")
        return None

    return csv_path
