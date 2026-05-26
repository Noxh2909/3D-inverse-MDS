"""Experiment session logging."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import QPlainTextEdit

from .export import make_abbrev
from .paths import app_data_dir


class Logger:
    """Handles experiment event logging to console widget and CSV files."""

    def __init__(self, console_box: QPlainTextEdit):
        self.console_box = console_box
        self.start_time: Optional[datetime] = None
        self._name_provider = None

    def set_name_provider(self, provider):
        """Set a callable returning the current participant name string."""
        self._name_provider = provider

    def log_to_console(self, text: str):
        """Append a timestamped message to the console widget."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        self.console_box.appendPlainText(entry)

    def write_log_to_file(self, log_path: str, text: str):
        """Append a line to the given log file."""
        try:
            with open(log_path, "a", encoding="utf-8", newline="") as f:
                f.write(text + "\n")
        except Exception as exc:
            self.log_to_console(f"Failed to write log: {exc}")

    def log_session_event(self, event: str):
        """Log a session event with timestamp and elapsed time."""
        if not self.start_time:
            return
        elapsed = (datetime.now() - self.start_time).total_seconds()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        name = "anonymous"
        if self._name_provider:
            raw = self._name_provider() or "anonymous"
            name = make_abbrev(raw) if raw != "anonymous" else "anonymous"
        name = name or "anonymous"

        log_dir = app_data_dir() / "logs" / name
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{name}_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.write_log_to_file(str(log_path), f"{timestamp},{elapsed:.2f},{event}")
        self.log_to_console(event)
