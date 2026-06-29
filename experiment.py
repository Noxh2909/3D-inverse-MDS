"""Application entry point for the 3D inverse MDS GUI."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from ui.launcher import LauncherWindow


def main() -> int:
    """Start the experiment hub application."""
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyleSheet(
        ""
        "ExperimentWindow { background-color: #141414; }"
        "ExperimentWindow > QWidget { background-color: #141414; }"
    )
    try:
        app.setFont(QFont("SF Pro Text", 12))
    except Exception:
        try:
            app.setFont(QFont(".SF NS Text", 12))
        except Exception:
            pass

    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
