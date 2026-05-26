"""Path helpers for the 3D inverse MDS application."""

from __future__ import annotations

import os
import pathlib
import shutil
import sys

APP_NAME = "3D inverse MDS"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def app_resource_dir() -> pathlib.Path:
    """Return the directory containing bundled application resources."""
    if getattr(sys, "frozen", False):
        return pathlib.Path(
            getattr(sys, "_MEIPASS", pathlib.Path(sys.executable).resolve().parent)
        )
    return pathlib.Path(__file__).resolve().parent.parent


def app_data_dir() -> pathlib.Path:
    """Return a writable base directory for logs and result exports."""
    if not getattr(sys, "frozen", False):
        return app_resource_dir()

    home = pathlib.Path.home()

    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME

    if sys.platform.startswith("win"):
        base = pathlib.Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return base / APP_NAME

    base = pathlib.Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    return base / APP_NAME


def stimuli_dir() -> pathlib.Path:
    """Return the active stimulus folder used by the experiment."""
    if not getattr(sys, "frozen", False):
        return app_resource_dir() / "pictures"

    writable_dir = app_data_dir() / "pictures"
    bundled_dir = app_resource_dir() / "pictures"

    if writable_dir.exists():
        return writable_dir

    try:
        writable_dir.mkdir(parents=True, exist_ok=True)
        if bundled_dir.is_dir():
            for path in bundled_dir.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    target = writable_dir / path.name
                    if not target.exists():
                        shutil.copy2(path, target)
    except OSError:
        return bundled_dir

    return writable_dir
