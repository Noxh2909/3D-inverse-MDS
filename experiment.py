"""
3D Inverse MDS Experiment
=========================

PySide6/pyqtgraph-based GUI application for a 3D inverse Multidimensional
Scaling (MDS) experiment. Participants place stimulus tokens in a 2D or 3D
coordinate space based on perceived similarity.

Usage::

    python experiment.py

The application presents a condition dialog (2D or 3D), then a fullscreen
experiment window. Participants complete one condition, results are exported
to CSV, then the second condition begins automatically.
"""

import sys
import os
import random
import pathlib
import csv
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QCheckBox, QMainWindow, QSizePolicy, QGridLayout,
    QScrollArea, QPlainTextEdit, QSlider, QDialog, QRadioButton,
)
from PySide6.QtGui import (
    QVector3D, QDrag, QCursor, QPixmap, QFont, QKeySequence, QShortcut,
)
from PySide6.QtCore import Qt, QTimer, QMimeData, QPoint, QObject
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLGridItem


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

AXIS_LEN = 10.0
TICK_STEP = 1.0
TICK_SIZE = 0.10
LABEL_WIDTH = 80
INPUT_WIDTH = 120
POINT_SIZE = 16
Z_ALIGN_EPS = 0.5
LABEL_OVER_POINT_MARGIN = 8
IMAGE_OVER_POINT_MARGIN = 32
IMAGE_MAX_WH = 40
IMAGE_CONTAINER_WH = 120
LABEL_SCREEN_MARGIN = 24
VIS_DOT_THRESHOLD = 0.0
CUBE_WIDTH = 1
LATTICE_WIDTH = 1
BTN_SPACING = 8
ROW_SPACING = 15
CONTROL_H = 28
GAP_H = 12
PREVIEW_TOP_OFFSET = 22
ACTIONS_TOP_OFFSET = -16
SCENE_TOP_OFFSET = 22
SCENE_BOTTOM_OFFSET = 22
SCENE_FIXED_HEIGHT = 910
TOKEN_CONTAINER_W = 140

POINT_COLOR = np.array([[1.0, 1.0, 0.0, 1.0]])
CUBE_COLOR = (120, 120, 120, 0.2)
LATTICE_COLOR = (120, 120, 120, 0.2)
PLANE_OFFSETS = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}

ALIGN_OK_HTML = "<span style='color:#7CFC00'>✅ {partner}</span>"
ALIGN_BAD_HTML = "<span style='color:#ff6666'>❌ {partner}</span>"


# ═══════════════════════════════════════════════════════════════════════════
# Pure Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _category_of(pid: str) -> str:
    """Extract category (number prefix) from a point id like '3.1'."""
    return pid.split('.')[0]


def _partner_of(pid: str) -> Optional[str]:
    """Return the partner point id (e.g. '3.1' -> '3.2')."""
    if '.' not in pid:
        return None
    cat = _category_of(pid)
    return f"{cat}.2" if pid.endswith('.1') else f"{cat}.1"


def _token_style(placed: bool) -> str:
    """Return CSS style for token labels based on placed state."""
    if placed:
        return (
            "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; "
            "border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
        )
    return (
        "QLabel { color: #eee; background: #444; border: 1px solid #999; "
        "border-radius: 4px; padding: 2px 6px; } "
        "QLabel:hover { background: #555; }"
    )


def _token_style_mode(mode: str) -> str:
    """Return CSS for named token modes: 'placed', 'disabled', or default active."""
    if mode == 'placed':
        return (
            "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; "
            "border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
        )
    if mode == 'disabled':
        return (
            "QLabel { color: #aaa; background: #333; border: 1px solid #666; "
            "border-radius: 4px; padding: 2px 6px; }"
        )
    return (
        "QLabel { color: #eee; background: #444; border: 1px solid #999; "
        "border-radius: 4px; padding: 2px 6px; } "
        "QLabel:hover { background: #555; }"
    )


# ── Axis / Grid Building Helpers ──────────────────────────────────────────

def _axis_segment(p0, p1, color=(1, 1, 1, 1), width=3):
    """Create a single GL line segment between two 3D points."""
    pts = np.array([p0, p1], dtype=float)
    return GLLinePlotItem(pos=pts, color=color, width=width, mode='lines')


def _auto_tick_step(length: float) -> float:
    """Choose a readable tick step based on axis length."""
    if length <= 10:
        return 1.0
    if length <= 20:
        return 2.0
    return max(1.0, round(length / 10.0, 1))


def _build_axis_solid(axis: str, length: float, color=(1, 1, 1, 1), width=3):
    """Build solid line items for a single axis."""
    endpoints = {
        'x': ((0, 0, 0), (length, 0, 0)),
        'y': ((0, 0, 0), (0, length, 0)),
        'z': ((0, 0, 0), (0, 0, length)),
    }
    p0, p1 = endpoints[axis]
    return [_axis_segment(p0, p1, color=color, width=width)]


def _build_axis_ticks(axis: str, length: float, tick_step: float = 0,
                      tick_size: float = TICK_SIZE, color=(1, 1, 1, 0.9),
                      width=2):
    """Build tick mark line items along an axis."""
    items = []
    if not tick_step:
        tick_step = _auto_tick_step(length)
    t = float(tick_step)
    while t < length + 1e-9:
        if axis == 'x':
            items.append(_axis_segment((t, 0, 0), (t, 0, tick_size),
                                       color=color, width=width))
        elif axis == 'y':
            items.append(_axis_segment((0, t, 0), (0, t, tick_size),
                                       color=color, width=width))
        else:
            items.append(_axis_segment((0, 0, t), (tick_size, 0, t),
                                       color=color, width=width))
            items.append(_axis_segment((0, 0, t), (0, tick_size, t),
                                       color=color, width=width))
        t += tick_step
    return items


def _make_edge(p0, p1, color=CUBE_COLOR, width=CUBE_WIDTH):
    """Create a single wireframe edge between two points."""
    pts = np.array([p0, p1], dtype=float)
    return GLLinePlotItem(pos=pts, color=color, width=width, mode='lines')


# ═══════════════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════════════

class Logger:
    """Handles experiment event logging to console widget and CSV files.

    Attributes:
        console_box: QPlainTextEdit widget for displaying log messages.
        start_time: Timestamp when the experiment session started.
    """

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
            with open(log_path, 'a', encoding='utf-8', newline='') as f:
                f.write(text + '\n')
        except Exception as e:
            self.log_to_console(f"Failed to write log: {e}")

    def log_session_event(self, event: str):
        """Log a session event with timestamp and elapsed time."""
        if not self.start_time:
            return
        elapsed = (datetime.now() - self.start_time).total_seconds()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        name = "anonymous"
        if self._name_provider:
            name = self._name_provider() or "anonymous"
        name = name.strip().replace(" ", "_") or "anonymous"

        base = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base, "logs", name)
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(
            log_dir,
            f"{name}_log_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv",
        )
        self.write_log_to_file(log_path, f"{timestamp},{elapsed:.2f},{event}")
        self.log_to_console(event)


# ═══════════════════════════════════════════════════════════════════════════
# FileHandler
# ═══════════════════════════════════════════════════════════════════════════

class FileHandler:
    """Handles image discovery, loading, and category assignment.

    Images are loaded from a ``pictures_test`` folder relative to the script,
    randomly assigned to token categories, and stored at two resolutions
    (original for preview, scaled for UI overlays).
    """

    def __init__(self, point_tokens, images_by_cat: dict, images_orig: dict,
                 png_name_by_cat: dict, image_max_wh: int):
        self.point_tokens = point_tokens
        self.images_by_cat = images_by_cat
        self.images_orig = images_orig
        self.png_name_by_cat = png_name_by_cat
        self.image_max_wh = image_max_wh

    def pictures_dir(self) -> str:
        """Return the absolute path to the images folder."""
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "pictures_test")

    def count_images(self) -> int:
        """Count image files in the pictures folder."""
        folder = self.pictures_dir()
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
        count = 0
        try:
            for p in pathlib.Path(folder).iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    count += 1
        except FileNotFoundError:
            pass
        return count

    def token_categories(self) -> list[str]:
        """Return category names derived from token ids."""
        cats = []
        try:
            for t in self.point_tokens:
                cat = t.pid.split('.')[0]
                if cat not in cats:
                    cats.append(cat)
        except Exception:
            pass
        if not cats:
            n = max(1, self.count_images())
            cats = [f"{i}. image" for i in range(1, n + 1)]
        return cats

    def load_images_for_categories(self):
        """Load and assign images to categories, populating the shared dicts."""
        folder = self.pictures_dir()
        exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

        paths = []
        try:
            for p in pathlib.Path(folder).iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    paths.append(p)
        except FileNotFoundError:
            self.images_by_cat.clear()
            self.images_orig.clear()
            self.png_name_by_cat.clear()
            return

        if not paths:
            self.images_by_cat.clear()
            self.images_orig.clear()
            self.png_name_by_cat.clear()
            return

        random.shuffle(paths)
        cats = self.token_categories()
        if not cats:
            return

        if len(paths) > len(cats):
            paths = paths[:len(cats)]
        if len(cats) > len(paths):
            cats = cats[:len(paths)]

        self.images_by_cat.clear()
        self.images_orig.clear()
        self.png_name_by_cat.clear()

        for i, cat in enumerate(cats):
            path = paths[i]
            pm_orig = QPixmap(str(path))
            if pm_orig.isNull():
                continue
            self.images_orig[cat] = pm_orig
            pm_scaled = pm_orig.scaled(
                self.image_max_wh, self.image_max_wh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.images_by_cat[cat] = pm_scaled
            self.png_name_by_cat[cat] = path.name


# ═══════════════════════════════════════════════════════════════════════════
# ConditionDialog
# ═══════════════════════════════════════════════════════════════════════════

class ConditionDialog(QDialog):
    """Modal dialog for selecting the initial experimental condition (2D/3D)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Condition")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Please select the experimental condition:")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        self.radio_2d = QRadioButton("2D Condition")
        self.radio_3d = QRadioButton("3D Condition")
        self.radio_2d.setStyleSheet("font-size: 14px; padding: 10px;")
        self.radio_3d.setStyleSheet("font-size: 14px; padding: 10px;")
        self.radio_2d.setChecked(True)
        layout.addWidget(self.radio_2d)
        layout.addWidget(self.radio_3d)

        btn = QPushButton("Start Experiment")
        btn.setStyleSheet(
            "QPushButton { background: #00cc66; color: white; border: none; "
            "border-radius: 6px; padding: 12px 24px; font-size: 14px; "
            "font-weight: bold; } "
            "QPushButton:hover { background: #00b359; }"
        )
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setFixedSize(400, 250)

    def get_condition(self) -> str:
        """Return the selected condition string ('2d' or '3d')."""
        return "2d" if self.radio_2d.isChecked() else "3d"


# ═══════════════════════════════════════════════════════════════════════════
# Widget Helpers
# ═══════════════════════════════════════════════════════════════════════════

class _PointLabelFilter(QObject):
    """Event filter for overlay point labels (passthrough)."""

    def eventFilter(self, obj, ev):
        return False


class _ViewResizeFilter(QObject):
    """Repositions overlay checkboxes when the scene view resizes."""

    def __init__(self, experiment: 'ExperimentWindow', parent=None):
        super().__init__(parent)
        self.exp = experiment

    def eventFilter(self, obj, ev):
        if ev.type() == ev.Type.Resize:
            exp = self.exp
            w, h = obj.width(), obj.height()
            try:
                exp.cb_lock.move(w - exp.cb_lock.width() - 20,
                                 h - exp.cb_lock.height() - 20)
                exp.cb_lock.raise_()
            except Exception:
                pass
            try:
                exp.cb_stimuli.move(20, h - exp.cb_stimuli.height() - 20)
                exp.cb_stimuli.raise_()
            except Exception:
                pass
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SceneView
# ═══════════════════════════════════════════════════════════════════════════

class SceneView(GLViewWidget):
    """OpenGL scene view with drag-and-drop point placement and hover preview.

    Handles:
    - Drag-and-drop of stimulus tokens onto the 3D/2D scene
    - Picking and dragging of already-placed points
    - Hover preview when cursor is near a placed point
    - Scroll-wheel height adjustment for placed points (3D mode)
    """

    def __init__(self, experiment: 'ExperimentWindow', **kwargs):
        super().__init__(**kwargs)
        self.exp = experiment
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.dragging_pid: Optional[str] = None
        self.selected_pid: Optional[str] = None
        self.drag_offset = (0.0, 0.0)
        self.drag_pending = False
        self.drag_z0: Optional[float] = None
        self.drag_plane_origin = None
        self.drag_plane_normal = None
        self.freeze_xy_after_scroll = False
        self.freeze_xy_pos = None
        self.hover_pid: Optional[str] = None

    # ── Drag & Drop ───────────────────────────────────────────────────────

    def dragEnterEvent(self, ev):
        """Accept drags carrying point-id MIME data."""
        if ev.mimeData().hasFormat('application/x-point-id'):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        """Continue accepting point-id drags."""
        if ev.mimeData().hasFormat('application/x-point-id'):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        """Place a token at the ray-plane intersection point."""
        exp = self.exp
        if not ev.mimeData().hasFormat('application/x-point-id'):
            ev.ignore()
            return
        try:
            pid = ev.mimeData().data(
                'application/x-point-id').data().decode('utf-8')
        except Exception:
            ev.ignore()
            return

        pos = ev.position().toPoint()
        px, py = int(pos.x()), int(pos.y())
        p0, p1 = exp.screen_to_world_ray(px, py)
        if p0 is None or p1 is None:
            ev.ignore()
            return

        d = QVector3D(p1.x() - p0.x(), p1.y() - p0.y(), p1.z() - p0.z())

        if exp.current_condition == "2d":
            if abs(d.y()) < 1e-9:
                ev.ignore()
                return
            t = (0.0 - p0.y()) / d.y()
            if t < 0:
                ev.ignore()
                return
            hit = p0 + d * t
            x, y, z = exp.clamp_to_cube(float(hit.x()), 0.0, float(hit.z()))
            exp.set_point_position(pid, (x, 0.0, z))
        else:
            zmid = AXIS_LEN * 0.5
            if abs(d.z()) < 1e-9:
                ev.ignore()
                return
            t = (zmid - p0.z()) / d.z()
            if t < 0:
                ev.ignore()
                return
            hit = p0 + d * t
            x, y, z = exp.clamp_to_cube(
                float(hit.x()), float(hit.y()), float(hit.z()))
            exp.set_point_position(pid, (x, y, z))

        exp.update_helper_lines(pid)
        exp.mark_token_placed(pid)
        ev.acceptProposedAction()

    # ── Mouse Events ──────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        """Pick placed points for dragging on left-click."""
        exp = self.exp
        pos = ev.position().toPoint()
        mx, my = pos.x(), pos.y()

        hit_pid, best_d2 = None, 1e9
        for pid, (_item, coords) in exp.placed_points.items():
            pr = exp.project_point(coords)
            if pr is None:
                continue
            px, py = pr
            d2 = (px - mx) ** 2 + (py - my) ** 2
            if d2 < best_d2:
                best_d2 = d2
                hit_pid = pid

        pick_r = float(max(16.0, POINT_SIZE + 6.0))

        if ev.button() == Qt.MouseButton.LeftButton:
            if hit_pid is not None and best_d2 <= pick_r * pick_r:
                self.dragging_pid = hit_pid
                exp.update_helper_lines(hit_pid)
                self.drag_pending = False
                self.drag_offset = (mx, my)
                item, coords_old = exp.placed_points.get(
                    hit_pid, (None, [0, 0, AXIS_LEN * 0.5]))
                self.drag_z0 = float(coords_old[2])
                self.freeze_xy_after_scroll = False
                self.freeze_xy_pos = None
                exp._raise_point_overlays(hit_pid)
                return

        if not exp.lock_camera:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Handle dragging, hover preview, rotation detection, label updates."""
        exp = self.exp

        # Tutorial: detect view rotation (left-click drag, no point selected)
        if (ev.buttons() & Qt.MouseButton.LeftButton
                and self.dragging_pid is None):
            exp._check_tutorial_rotation()

        # Keep overlay labels positioned
        exp.position_axis_labels()
        exp.update_all_point_labels()

        # Point dragging
        if self.dragging_pid is not None:
            self._handle_drag_move(ev)
            return

        # Hover detection
        self._handle_hover(ev)

        # Camera interaction
        if not exp.lock_camera:
            super().mouseMoveEvent(ev)

    def _handle_drag_move(self, ev):
        """Project dragged point onto the appropriate plane."""
        exp = self.exp
        pos = ev.position().toPoint()
        mx, my = pos.x(), pos.y()

        pid = self.dragging_pid
        if pid is None:
            return

        if self.freeze_xy_after_scroll:
            dx = abs(mx - self.drag_offset[0])
            dy = abs(my - self.drag_offset[1])
            if dx < 3 and dy < 3 and self.freeze_xy_pos is not None:
                fx, fy = self.freeze_xy_pos
                zp = (0.0 if exp.current_condition == "2d"
                      else float(self.drag_z0 or AXIS_LEN * 0.5))
                if exp.current_condition == "2d":
                    exp.set_point_position(pid, (fx, 0.0, zp))
                else:
                    exp.set_point_position(pid, (fx, fy, zp))
                cat = pid.split('.')[0]
                exp.set_preview_for_category(cat)
                return
            else:
                self.freeze_xy_after_scroll = False
                self.freeze_xy_pos = None

        p0, p1 = exp.screen_to_world_ray(int(mx), int(my))
        if p0 is None or p1 is None:
            return

        zp = (0.0 if exp.current_condition == "2d"
              else float(self.drag_z0 or AXIS_LEN * 0.5))
        d = QVector3D(float(p1.x() - p0.x()), float(p1.y() - p0.y()),
                      float(p1.z() - p0.z()))

        if exp.current_condition == "2d":
            if abs(d.y()) > 1e-9:
                t = (0.0 - p0.y()) / d.y()
                if t >= 0:
                    hit = p0 + d * t
                    x, _, z = exp.clamp_to_cube(
                        float(hit.x()), 0.0, float(hit.z()))
                    exp.set_point_position(pid, (x, 0.0, z))
        else:
            if abs(d.z()) > 1e-9:
                t = (zp - p0.z()) / d.z()
                if t >= 0:
                    hit = p0 + d * t
                    x, y, _ = exp.clamp_to_cube(
                        float(hit.x()), float(hit.y()), zp)
                    exp.set_point_position(pid, (x, y, zp))

        cat = pid.split('.')[0]
        exp.set_preview_for_category(cat)

    def _handle_hover(self, ev):
        """Detect nearest placed point under cursor and show hover preview."""
        exp = self.exp
        try:
            pos = ev.position().toPoint()
            mx, my = pos.x(), pos.y()
            hit_pid, best_d2 = None, 1e9
            for pid, (_item, coords) in exp.placed_points.items():
                if not exp.is_point_visible_world(coords):
                    continue
                proj = exp.project_point(coords)
                if proj is None:
                    continue
                px, py = proj
                d2 = (px - mx) ** 2 + (py - my) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    hit_pid = pid

            pick_r = float(max(16.0, POINT_SIZE + 6.0))
            prev_hover = self.hover_pid
            self.hover_pid = (hit_pid if hit_pid and best_d2 <= pick_r ** 2
                              else None)

            if self.hover_pid != prev_hover:
                if prev_hover is not None:
                    exp.update_helper_lines(prev_hover)
                if self.hover_pid is not None:
                    exp.update_helper_lines(self.hover_pid)

            if self.hover_pid is not None:
                cat = self.hover_pid.split('.')[0]
                exp.set_preview_for_category(cat)
                item, _ = exp.placed_points[self.hover_pid]
                item.setData(color=np.array([[1, 1, 1, 1]]),
                             size=POINT_SIZE + 6)
                for pid, (itm, _) in exp.placed_points.items():
                    if pid != self.hover_pid:
                        itm.setData(color=POINT_COLOR, size=POINT_SIZE)
            else:
                for pid, (item, _) in exp.placed_points.items():
                    item.setData(color=POINT_COLOR, size=POINT_SIZE)
        except Exception:
            pass

    def wheelEvent(self, ev):
        """Adjust point height (Z-axis) via scroll wheel."""
        exp = self.exp
        pid = self.dragging_pid or self.hover_pid
        if pid is None:
            return

        if pid in exp.placed_points:
            exp._check_tutorial_height_adjust()

        delta = ev.angleDelta().y() / 120.0
        item, coords = exp.placed_points.get(pid, (None, None))
        if coords is None:
            return

        x, y, z = coords
        z += float(delta) * 0.3
        x, y, z = exp.clamp_to_cube(x, y, z)

        if self.drag_plane_origin is not None:
            self.drag_plane_origin.setZ(z)
        self.drag_z0 = z
        self.freeze_xy_after_scroll = True
        self.freeze_xy_pos = (x, y)
        exp.set_point_position(pid, (x, y, z))
        exp.update_helper_lines(pid)

    def keyPressEvent(self, ev):
        """Forward key events to base class."""
        super().keyPressEvent(ev)

    def mouseReleaseEvent(self, ev):
        """End point dragging on left-button release."""
        if ev.button() == Qt.MouseButton.LeftButton:
            exp = self.exp
            self.drag_pending = False
            self.drag_z0 = None
            self.freeze_xy_after_scroll = False
            self.freeze_xy_pos = None
            if self.dragging_pid in exp.helper_lines:
                for it in exp.helper_lines[self.dragging_pid]:
                    try:
                        self.removeItem(it)
                    except Exception:
                        pass
                exp.helper_lines.pop(self.dragging_pid, None)
            self.dragging_pid = None
            return
        super().mouseReleaseEvent(ev)

    def leaveEvent(self, ev):
        """Clear hover state when cursor leaves the view."""
        self.hover_pid = None
        super().leaveEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════
# DraggableToken
# ═══════════════════════════════════════════════════════════════════════════

class DraggableToken(QLabel):
    """Draggable stimulus token label for the token dock panel.

    Each token represents a stimulus that can be dragged into the scene view.
    Supports hover preview and drag-and-drop with image attachment.
    """

    def __init__(self, pid: str, experiment: 'ExperimentWindow', parent=None):
        super().__init__(pid, parent)
        self.pid = pid
        self.exp = experiment
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setStyleSheet(_token_style_mode('disabled'))
        self.setFixedSize(110, 30)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        """Show image preview when hovering over token."""
        try:
            cat = self.pid.split('.')[0]
            self.exp.show_hover_preview_over_dock(cat)
        except Exception:
            pass
        try:
            self.exp.check_hover_cb.setChecked(True)
            self.raise_()
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        """Keep last preview active on leave."""
        super().leaveEvent(ev)

    def mouseMoveEvent(self, ev):
        """Keep preview visible while hovering over token."""
        try:
            cat = self.pid.split('.')[0]
            self.exp.show_hover_preview_over_dock(cat)
            self.raise_()
        except Exception:
            pass
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        """Initiate drag-and-drop for token placement."""
        if ev.button() != Qt.MouseButton.LeftButton or not self.isEnabled():
            return

        exp = self.exp
        exp.is_dock_drag = True
        cat = self.pid.split('.')[0]
        exp.show_hover_preview_over_dock(cat)

        try:
            self.raise_()
            for suffix in ("", ".1", ".2"):
                key = f"{cat}{suffix}"
                if key in exp.image_labels:
                    exp.image_labels[key].raise_()
                if key in exp.point_labels:
                    exp.point_labels[key].raise_()
        except Exception:
            pass

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData('application/x-point-id', self.pid.encode('utf-8'))
        drag.setMimeData(mime)

        pm = exp.images_by_cat.get(cat)
        if pm is not None and not pm.isNull():
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, int(pm.height() * 0.8)))

        drag.exec(Qt.DropAction.MoveAction)
        exp.is_dock_drag = False


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentWindow
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentWindow(QMainWindow):
    """Main experiment window encapsulating all UI, state, and logic.

    Manages the full experiment lifecycle:

    1. UI initialization with tutorial checklist
    2. Token drag-and-drop placement in 2D/3D space
    3. CSV result export
    4. Automatic condition switching (2D <-> 3D)

    Args:
        condition: Initial condition ('2d' or '3d').
    """

    def __init__(self, condition: str):
        super().__init__()
        self.current_condition = condition
        self.setUpdatesEnabled(False)

        self._init_state()
        self._build_main_layout()
        self._build_info_section()
        self._build_scene_view()
        self._build_checklist()
        self._build_control_panel()
        self._build_token_dock()
        self._build_buttons()
        self._build_console()
        self._build_final_layout()
        self._connect_signals()
        self._setup_shortcuts()
        self._load_images()

        QTimer.singleShot(0, self._finalize_startup)

    # ══════════════════════════════════════════════════════════════════════
    # State Initialization
    # ══════════════════════════════════════════════════════════════════════

    def _init_state(self):
        """Initialize all experiment state variables."""
        self.placed_points: Dict[str, tuple] = {}
        self.point_labels: Dict[str, QLabel] = {}
        self.pair_lines: Dict[str, GLLinePlotItem] = {}
        self.helper_lines: dict = {}
        self.points: list = []
        self.cube_items: list = []
        self.point_tokens: list = []
        self.lattice_items: list = []
        self.current_plane = 'xy'
        self.placement_phase = 1
        self.lock_camera = False
        self.images_by_cat: Dict[str, QPixmap] = {}
        self.images_orig: Dict[str, QPixmap] = {}
        self.png_name_by_cat: dict = {}
        self.image_labels: Dict[str, QLabel] = {}
        self.hover_preview_label: Optional[QLabel] = None
        self.is_dock_drag = False
        self.experiment_running = False
        self.start_time: Optional[datetime] = None
        self.rotation_done = False
        self.height_adjust_done = False
        self.submitted_conditions: set = set()
        self._header_label: Optional[QLabel] = None
        self._header_y = 10
        self._is_collapsed = False
        self.axis_items = {"x": [], "y": [], "z": []}
        self.axis_tick_labels = {'x': [], 'y': [], 'z': []}
        self._point_label_filter = _PointLabelFilter()

    # ══════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════

    def _build_main_layout(self):
        """Build the main window structure: central widget, layout, columns."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._main_row = QHBoxLayout()
        self._main_row.setContentsMargins(0, 0, 0, 0)
        self._main_row.setSpacing(GAP_H)
        root.addLayout(self._main_row, 1)

        self.left_col = QFrame()
        self.left_col.setFrameShape(QFrame.Shape.StyledPanel)
        self.left_col.setStyleSheet(
            "QFrame { border: 0px solid #666; border-radius: 8px; }")
        self.left_col.setMinimumWidth(120)
        self.left_v = QVBoxLayout(self.left_col)
        self.left_v.setContentsMargins(10, 10, 10, 20)
        self.left_v.setSpacing(12)
        self.left_v.addStretch(1)
        self._main_row.addWidget(self.left_col, 0)

    def _build_info_section(self):
        """Build participant info: title, condition, name, age, separator."""
        # Preview container
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.setSpacing(6)
        self.left_v.addWidget(self.preview_container)

        self.preview_box = QLabel()
        self.preview_layout.addWidget(self.preview_box)
        self.preview_box.setFixedSize(230, 210)
        self.preview_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_box.setStyleSheet(
            "QLabel { background: rgba(255,255,255,0.1); color: #ddd; "
            "border: 1px solid #888; border-radius: 6px; }")
        self.preview_box.setText("Image Preview")

        self.preview_label = QLabel("Preview:")
        self.preview_layout.addWidget(self.preview_label)
        self.preview_label.setStyleSheet(
            "color: #fff; font-size: 14px; font-weight: 600; "
            "background: transparent;")
        self.preview_label.adjustSize()
        self.preview_label.show()

        self.actions_row = QWidget(self.left_col)
        actions_h = QHBoxLayout(self.actions_row)
        actions_h.setContentsMargins(0, 0, 0, 0)
        actions_h.setSpacing(8)

        # Title label (absolute positioned on main window)
        self.title_label = QLabel("Inverse MDS", parent=self)
        self.title_label.setStyleSheet(
            "color: white; font-size: 26px; font-weight: 700; "
            "background: transparent;")
        self.title_label.adjustSize()
        self.title_label.move(20, 30)
        self.title_label.raise_()

        # Separator below title
        line_top = QFrame(parent=self)
        line_top.setFrameShape(QFrame.Shape.HLine)
        line_top.setFrameShadow(QFrame.Shadow.Plain)
        line_top.setStyleSheet(
            "color: gray; background-color: transparent;")
        line_top.setFixedWidth(380)
        line_top.move(20, 60)

        # Participant label
        self.name_label = QLabel("Participant:", parent=self)
        self.name_label.setStyleSheet(
            "color: white; font-size: 16px; background: transparent; "
            "font-weight: bold;")
        self.name_label.adjustSize()
        self.name_label.move(20, 90)
        self.name_label.raise_()

        # Participant name input
        self.name_input = QLineEdit(parent=self)
        self.name_input.setPlaceholderText("First name, Last name")
        self.name_input.setFixedWidth(200)
        self.name_input.setStyleSheet(
            "color: #000000; background: #f5f5f5; border: 1px solid black; "
            "border-radius: 6px; padding: 4px 8px;")
        self.name_input.move(20, 115)
        self.name_input.raise_()

        # Age label
        self.age_label = QLabel("Age:", parent=self)
        self.age_label.setStyleSheet(
            "color: white; font-size: 16px; background: transparent; "
            "font-weight: bold;")
        self.age_label.adjustSize()
        self.age_label.move(230, 90)
        self.age_label.raise_()

        # Age slider
        self.age_slider = QSlider(Qt.Orientation.Horizontal, parent=self)
        self.age_slider.setMinimum(18)
        self.age_slider.setMaximum(90)
        self.age_slider.setValue(24)
        self.age_slider.setFixedWidth(160)
        self.age_slider.move(230, 115)
        self.age_slider.raise_()

        self.age_value = QLabel("24", parent=self)
        self.age_value.setStyleSheet(
            "color: white; font-size: 16px; background: transparent;")
        self.age_value.adjustSize()
        self.age_value.move(270, 90)
        self.age_value.raise_()

        # Separator
        middle_line = QFrame(parent=self)
        middle_line.setFrameShape(QFrame.Shape.HLine)
        middle_line.setFrameShadow(QFrame.Shadow.Plain)
        middle_line.setStyleSheet(
            "color: gray; background-color: transparent;")
        middle_line.setFixedWidth(380)
        middle_line.move(20, 150)

    def _build_scene_view(self):
        """Build the 3D scene view, axes, and grids."""
        self.view = SceneView(experiment=self)
        self.view.setBackgroundColor('k')
        self.view.setCameraPosition(distance=8)
        self.view.setCameraParams(fov=60)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)

        view_wrap = QWidget()
        vw_lay = QVBoxLayout(view_wrap)
        vw_lay.setContentsMargins(0, 20, 20, 20)
        vw_lay.setSpacing(0)
        vw_lay.addWidget(self.view)
        view_wrap.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self._main_row.addWidget(view_wrap, 1)

        # Condition label (overlay on scene view)
        self.condition_label = QLabel("", parent=self.view)
        self.condition_label.setStyleSheet(
            "color: white; font-size: 16px; background: transparent; "
            "font-weight: bold;")
        self.condition_label.adjustSize()
        self.condition_label.move(10, 10)
        self.condition_label.raise_()

        # Build and add axes
        self._build_axes()
        for items in self.axis_items.values():
            for it in items:
                self.view.addItem(it)

        # Axis labels (overlay on view)
        self.axis_label_x = QLabel("", parent=self.view)
        self.axis_label_y = QLabel("", parent=self.view)
        self.axis_label_z = QLabel("", parent=self.view)
        for lab, col in [(self.axis_label_x, "#d33"),
                         (self.axis_label_y, "#0a0"),
                         (self.axis_label_z, "#33d")]:
            lab.setStyleSheet(
                f"color: {col}; font-size: 16px; font-weight: 500; "
                f"background: transparent;")
            lab.raise_()
            lab.show()

        # Grids
        self.yz_grid = GLGridItem()
        self.yz_grid.setSize(x=AXIS_LEN, y=AXIS_LEN)
        self.yz_grid.rotate(90, 0, 1, 0)
        self.yz_grid.translate(PLANE_OFFSETS['yz'],
                               AXIS_LEN * 0.5, AXIS_LEN * 0.5)

        self.xz_grid = GLGridItem()
        self.xz_grid.setSize(x=AXIS_LEN, y=AXIS_LEN)
        self.xz_grid.rotate(90, 1, 0, 0)
        self.xz_grid.translate(AXIS_LEN * 0.5,
                               PLANE_OFFSETS['xz'], AXIS_LEN * 0.5)

    def _build_checklist(self):
        """Build the tutorial checklist overlay."""
        self.collapse_btn = QPushButton("", parent=self)
        self.collapse_btn.setFixedSize(20, 20)
        self.collapse_btn.setCursor(
            QCursor(Qt.CursorShape.PointingHandCursor))
        self.collapse_btn.hide()
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                color: white; background: transparent; border: none;
                font-size: 12px; padding: 0px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.1); }
        """)
        self.collapse_btn.move(170, 180)
        self.collapse_btn.raise_()

        self.status_label = QLabel("Tutorial Checklist:", parent=self)
        self.status_label.setStyleSheet(
            "color: white; font-size: 16px; background: transparent; "
            "font-weight: bold; border: none;")
        self.status_label.adjustSize()
        self.status_label.move(20, 180)
        self.status_label.raise_()

        # Step 1: Set Name
        self.set_name_label = QLabel(
            "1. Set Name and Age:", parent=self)
        self.set_name_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.set_name_label.adjustSize()
        self.set_name_label.move(20, 210)
        self.set_name_label.raise_()

        self.name_cb = QCheckBox("", parent=self)
        self.name_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.name_cb.setEnabled(False)
        self.name_cb.move(355, 205)
        self.name_cb.raise_()

        # Step 2: Hover
        self.check_hover_label = QLabel(
            "2. Hover over Stimuli to preview images:", parent=self)
        self.check_hover_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.check_hover_label.adjustSize()
        self.check_hover_label.move(20, 240)
        self.check_hover_label.raise_()

        self.check_hover_cb = QCheckBox("", parent=self)
        self.check_hover_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.check_hover_cb.setEnabled(False)
        self.check_hover_cb.move(355, 235)
        self.check_hover_cb.raise_()

        # Step 3: Rotate
        self.rotate_and_adjust_label = QLabel(parent=self)
        self.rotate_and_adjust_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.rotate_and_adjust_label.adjustSize()
        self.rotate_and_adjust_label.move(20, 270)
        self.rotate_and_adjust_label.raise_()

        self.rotate_and_adjust_cb = QCheckBox("", parent=self)
        self.rotate_and_adjust_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.rotate_and_adjust_cb.setEnabled(False)
        self.rotate_and_adjust_cb.move(355, 265)
        self.rotate_and_adjust_cb.raise_()

        # Step 4: Drag Stimuli
        self.stimuli_drag_label = QLabel(parent=self)
        self.stimuli_drag_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.stimuli_drag_label.adjustSize()
        self.stimuli_drag_label.move(20, 300)
        self.stimuli_drag_label.raise_()

        self.stimuli_cb = QCheckBox("", parent=self)
        self.stimuli_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.stimuli_cb.setEnabled(False)
        self.stimuli_cb.move(355, 295)
        self.stimuli_cb.raise_()

        # Step 5: Adjust height
        self.adjust_token_height_label = QLabel(
            '5. Hold Point and use "Wheel" to adjust height:', parent=self)
        self.adjust_token_height_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.adjust_token_height_label.adjustSize()
        self.adjust_token_height_label.move(20, 330)
        self.adjust_token_height_label.raise_()

        self.adjust_token_height_cb = QCheckBox("", parent=self)
        self.adjust_token_height_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.adjust_token_height_cb.setEnabled(False)
        self.adjust_token_height_cb.move(355, 325)
        self.adjust_token_height_cb.raise_()

        # Step 6: Start
        self.start_label = QLabel(
            '6. Press "Start" to start the Experiment:', parent=self)
        self.start_label.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.start_label.adjustSize()
        self.start_label.move(20, 360)
        self.start_label.raise_()

        self.start_cb = QCheckBox("", parent=self)
        self.start_cb.setStyleSheet(
            "color: lightgray; font-size: 14px; background: transparent;")
        self.start_cb.setEnabled(False)
        self.start_cb.move(355, 355)
        self.start_cb.raise_()

        # Counter
        self.counter_label = QLabel("(0/5)", parent=self)
        self.counter_label.setStyleSheet(
            "color: lightgray; font-size: 12px; background: transparent;")
        self.counter_label.adjustSize()
        self.counter_label.move(375, 331)
        self.counter_label.raise_()

        # Bottom separator
        self.line_bottom = QFrame(parent=self)
        self.line_bottom.setFrameShape(QFrame.Shape.HLine)
        self.line_bottom.setFrameShadow(QFrame.Shadow.Plain)
        self.line_bottom.setStyleSheet(
            "color: gray; background-color: transparent;")
        self.line_bottom.setFixedWidth(380)
        self.line_bottom.move(20, 385)

    def _build_control_panel(self):
        """Build axis input panel and overlay checkboxes."""
        self.panel = QFrame(parent=self.left_col)
        self.left_v.addWidget(self.panel)
        self.panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.panel.setStyleSheet("""
            QFrame { background: none; border-radius: 8px; }
            QLabel { color: white; }
            QLineEdit { color: #000000; background: lightgray;
                        border: 1px solid black; border-radius: 4px;
                        padding: 1px 6px; font-size: 12px; }
            QPushButton { color: #000000; background: lightgray;
                          border: 1px solid black; border-radius: 4px;
                          padding: 4px 8px; }
            QPushButton:pressed { background: #e5e5e5; }
        """)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(0, 8, 0, 10)
        panel_layout.setSpacing(ROW_SPACING)

        # Hidden axis input rows
        def make_row(caption, default_text):
            row = QWidget(self.panel)
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(ROW_SPACING)
            lab = QLabel(caption, row)
            lab.setFixedWidth(LABEL_WIDTH)
            lab.setAlignment(Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignVCenter)
            edit = QLineEdit(row)
            edit.setText(default_text)
            edit.setFixedHeight(CONTROL_H)
            h.addWidget(lab)
            h.addWidget(edit)
            h.addStretch(1)
            return row, edit, lab, h

        row_x, self.edit_x, self.lab_x, _ = make_row("X:", "X")
        self.edit_x.setReadOnly(True)
        self.edit_x.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
        row_y, self.edit_y, self.lab_y, _ = make_row("Y:", "Y")
        self.edit_y.setReadOnly(True)
        self.edit_y.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
        row_z, self.edit_z, self.lab_z, _ = make_row("Z:", "Z")
        self.edit_z.setReadOnly(True)
        self.edit_z.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))

        row_x.hide()
        row_y.hide()
        row_z.hide()

        for lab in (self.lab_x, self.lab_y, self.lab_z):
            lab.setAlignment(Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignVCenter)
        for edit in (self.edit_x, self.edit_y, self.edit_z):
            edit.setFixedWidth(INPUT_WIDTH)
            edit.setFixedHeight(CONTROL_H)

        # Hint row (hidden)
        hint_row = QWidget(self.panel)
        hint_box = QHBoxLayout(hint_row)
        hint_box.setContentsMargins(0, 0, 0, 0)
        hint_box.setSpacing(ROW_SPACING)
        spacer = QWidget(hint_row)
        spacer.setFixedWidth(LABEL_WIDTH)
        hint_box.addWidget(spacer)
        hint_box.addWidget(QLabel("Press ⏎ to apply", hint_row))
        hint_box.addStretch(1)
        panel_layout.addWidget(hint_row)
        hint_row.hide()

        self.panel.adjustSize()
        self.panel.setMinimumSize(150, 250)
        self.panel.show()

        # Lock View checkbox (overlay on scene view)
        self.cb_lock = QCheckBox("Lock View (^L)", parent=self.view)
        self.cb_lock.setStyleSheet(
            "QCheckBox { color: #ffffff; background: rgba(0,0,0,120); }")
        self.cb_lock.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cb_lock.setChecked(True)
        self.cb_lock.show()
        self.cb_lock.raise_()

        # Show Stimuli checkbox (overlay on scene view)
        self.cb_stimuli = QCheckBox("Show Stimuli (^B)", parent=self.view)
        self.cb_stimuli.setStyleSheet(
            "QCheckBox { color: #ffffff; background: rgba(0,0,0,120); }")
        self.cb_stimuli.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cb_stimuli.setChecked(True)
        self.cb_stimuli.show()
        self.cb_stimuli.raise_()

        # Resize filter for repositioning overlays
        self.view.installEventFilter(
            _ViewResizeFilter(experiment=self, parent=self.view))

    def _build_token_dock(self):
        """Build the token dock with draggable stimulus tokens."""
        self.tokens_label = QLabel("Stimuli:")
        self.left_v.addWidget(self.tokens_label)
        self.tokens_label.setStyleSheet(
            "color: #fff; font-size: 14px; font-weight: 600; "
            "background: transparent;")
        self.tokens_label.adjustSize()
        self.tokens_label.show()

        self.point_dock = QFrame()
        self.left_v.addWidget(self.point_dock)
        self.point_dock.setFrameShape(QFrame.Shape.StyledPanel)
        self.point_dock.setStyleSheet(
            "QFrame { background: rgba(0,0,0,120); border: 1px solid #777; "
            "border-radius: 6px; }")

        self.point_dock_layout = QGridLayout(self.point_dock)
        self.point_dock_layout.setContentsMargins(10, 8, 8, 8)
        self.point_dock_layout.setHorizontalSpacing(6)
        self.point_dock_layout.setVerticalSpacing(8)
        self.point_dock_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Count images to determine token count
        img_count = FileHandler(
            point_tokens=[], images_by_cat={}, images_orig={},
            png_name_by_cat={}, image_max_wh=IMAGE_MAX_WH,
        ).count_images()
        img_count = max(1, img_count)

        for i in range(1, img_count + 1):
            t = DraggableToken(f"{i}. Stimulus", experiment=self,
                               parent=self.point_dock)
            t.setMinimumWidth(110)
            t.setFixedHeight(26)
            self.point_tokens.append(t)
            self.point_dock_layout.addWidget(t, i - 1, 0)

        self.point_dock.adjustSize()
        self.point_dock.show()

    def _build_buttons(self):
        """Build action buttons: Reset, Submit, Start, Set View."""
        self.btn_reset = QPushButton("Reset  (^R)", self.left_col)
        self.btn_submit = QPushButton("Submit  (^↩)", self.left_col)
        self.btn_start = QPushButton("Start", self.left_col)
        self.btn_start.setDisabled(True)
        self.btn_grid = QPushButton("Set View (^D)", self.left_col)

        for btn in (self.btn_reset, self.btn_submit,
                    self.btn_start, self.btn_grid):
            btn.setFixedSize(110, 25)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        btn_style = """
            QPushButton {
                color: #000000; background: #f5f5f5;
                border: 1px solid black; border-radius: 6px;
                padding: 4px 8px;
            }
            QPushButton:pressed { background: grey; }
            QPushButton:disabled {
                background: #e0e0e0; color: #888888;
                border: 1px solid black;
            }
        """
        self.btn_reset.setStyleSheet(btn_style)
        self.btn_submit.setStyleSheet(
            btn_style + "QPushButton { background: #00cc66; "
            "border: solid lightgray; }")
        self.btn_start.setStyleSheet(
            btn_style + "QPushButton { background: #00cc66; "
            "border: solid lightgray; }")
        self.btn_grid.setStyleSheet(btn_style)
        self.btn_grid.setCheckable(True)
        self.btn_submit.setEnabled(False)

    def _build_console(self):
        """Build the event terminal / console log box and Logger instance."""
        self.event_terminal_label = QLabel("Event Terminal:")
        self.event_terminal_label.setStyleSheet(
            "color: #fff; font-size: 14px; font-weight: 600; "
            "background: transparent;")
        self.event_terminal_label.adjustSize()

        self.console_box = QPlainTextEdit()
        self.console_box.setReadOnly(True)
        self.console_box.setStyleSheet("""
            QPlainTextEdit {
                background: rgba(255,255,255,0.1); color: #ddd;
                border-radius: 6px; font-size: 12px;
            }
        """)
        self.console_box.setFixedHeight(100)
        self.console_box.setFixedWidth(self.preview_box.width())

        self.logger = Logger(self.console_box)
        self.logger.set_name_provider(
            lambda: self.name_input.text().strip())

    def _build_final_layout(self):
        """Assemble all widgets into the final layout structure."""
        # Image row: tokens + preview side by side
        image_row = QWidget(self.left_col)
        image_h = QHBoxLayout(image_row)
        image_h.setContentsMargins(0, 0, 0, 0)
        image_h.setSpacing(GAP_H)

        # Token column
        token_col = QWidget(image_row)
        token_v = QVBoxLayout(token_col)
        token_v.setContentsMargins(0, 0, 0, 0)
        token_v.setSpacing(6)
        token_v.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        token_scroll = QScrollArea(parent=token_col)
        token_scroll.setWidget(self.point_dock)
        token_scroll.setWidgetResizable(False)
        token_scroll.setFrameShape(QFrame.Shape.NoFrame)
        token_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        token_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        token_scroll.setFixedWidth(TOKEN_CONTAINER_W)
        token_scroll.setFixedHeight(415)
        token_scroll.setStyleSheet(
            "QScrollArea { background: transparent; } "
            "QScrollBar:vertical { background: #222; width: 8px; } "
            "QScrollBar::handle:vertical { background: #555; "
            "min-height: 10px; border-radius: 4px; } "
            "QScrollBar::handle:vertical:hover { background: #777; } "
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            "{ height: 0px; } "
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical "
            "{ background: none; }")
        token_scroll.setSizePolicy(QSizePolicy.Policy.Fixed,
                                   QSizePolicy.Policy.Fixed)

        token_v.addWidget(self.tokens_label, 0,
                          Qt.AlignmentFlag.AlignLeft)
        token_v.addWidget(token_scroll, 0,
                          Qt.AlignmentFlag.AlignLeft)

        # Preview column
        preview_col = QWidget(image_row)
        preview_v = QVBoxLayout(preview_col)
        preview_v.setContentsMargins(0, PREVIEW_TOP_OFFSET, 0, 0)
        preview_v.setSpacing(6)

        self.preview_label.setParent(preview_col)
        self.preview_label.move(0, 0)
        self.preview_box.setParent(preview_col)
        preview_v.addWidget(self.preview_box, 0,
                            Qt.AlignmentFlag.AlignTop)
        preview_v.addSpacing(ACTIONS_TOP_OFFSET)
        self.actions_row.setParent(preview_col)
        preview_v.addWidget(self.actions_row, 0)

        # Position buttons absolutely in preview column
        self.btn_start.setParent(preview_col)
        self.btn_start.move(0, 240)
        self.btn_reset.setParent(preview_col)
        self.btn_reset.move(120, 240)
        self.btn_grid.setParent(preview_col)
        self.btn_grid.move(0, 275)
        self.btn_submit.setParent(preview_col)
        self.btn_submit.move(120, 275)

        # Console
        self.event_terminal_label.setParent(preview_col)
        self.event_terminal_label.move(0, 320)
        self.console_box.setParent(preview_col)
        self.console_box.move(0, 340)

        image_h.addWidget(token_col, 0)
        image_h.addWidget(preview_col, 0)
        self.left_v.addWidget(image_row, 0)

    def _connect_signals(self):
        """Connect all widget signals to their handlers."""
        self.age_slider.valueChanged.connect(
            lambda v: self.age_value.setText(str(v)))
        self.name_input.textChanged.connect(self.update_progress_counter)
        self.check_hover_cb.toggled.connect(self.update_progress_counter)
        self.rotate_and_adjust_cb.toggled.connect(
            self.update_progress_counter)
        self.stimuli_cb.toggled.connect(self.update_progress_counter)
        self.adjust_token_height_cb.toggled.connect(
            self.update_progress_counter)

        self.collapse_btn.clicked.connect(self.toggle_checklist)
        self.cb_lock.toggled.connect(self._toggle_lock)
        self.btn_reset.clicked.connect(self.reset_all_points)
        self.btn_start.clicked.connect(self.start_experiment)
        self.btn_grid.toggled.connect(self.set_view_default)
        self.btn_submit.clicked.connect(self.export_results)

        for e in (self.edit_x, self.edit_y, self.edit_z):
            e.returnPressed.connect(self.apply_labels)

        self.update_token_states()
        self.update_submit_state()

        # Initial states
        self.btn_grid.setChecked(False)
        self.cb_lock.setChecked(False)

    def _setup_shortcuts(self):
        """Register keyboard shortcuts."""
        try:
            sc = QShortcut(QKeySequence("Meta+B"), self)
            sc.activated.connect(
                lambda: self.cb_stimuli.setChecked(
                    not self.cb_stimuli.isChecked()))
            sc = QShortcut(QKeySequence("Meta+R"), self)
            sc.activated.connect(self.reset_all_points)
            sc = QShortcut(QKeySequence("Meta+D"), self)
            sc.activated.connect(
                lambda: self.btn_grid.setChecked(
                    not self.btn_grid.isChecked()))
            sc = QShortcut(QKeySequence("Meta+Return"), self)
            sc.activated.connect(self.export_results)
            sc = QShortcut(QKeySequence("Meta+L"), self)
            sc.activated.connect(
                lambda: self.cb_lock.setChecked(
                    not self.cb_lock.isChecked()))
        except Exception:
            pass

    def _load_images(self):
        """Load images via FileHandler and apply axis labels."""
        self.file_handler = FileHandler(
            point_tokens=self.point_tokens,
            images_by_cat=self.images_by_cat,
            images_orig=self.images_orig,
            png_name_by_cat=self.png_name_by_cat,
            image_max_wh=IMAGE_MAX_WH,
        )
        self.file_handler.load_images_for_categories()
        self.apply_labels()
        self.position_axis_labels()
        self.update_token_states()
        self.update_submit_state()

    # ══════════════════════════════════════════════════════════════════════
    # Startup & Resize
    # ══════════════════════════════════════════════════════════════════════

    def _finalize_startup(self):
        """Apply condition defaults and show the window fullscreen."""
        self.apply_condition_defaults()
        self.update_label()
        self.update_progress_counter()
        self.position_axis_labels()

        if self.current_condition == "2d":
            self.rotate_and_adjust_cb.setChecked(True)
            self.adjust_token_height_cb.setChecked(True)
            self.cb_lock.setChecked(True)
            self.btn_grid.setDisabled(True)
            self.cb_lock.hide()
            self.set_view_xy()
            self.hide_z_axis()
        else:
            self.cb_lock.setChecked(False)
            self.cb_lock.show()
            self.set_view_default()
            self.show_z_axis()

        # Re-position labels after camera is set
        self.position_axis_labels()

        self.condition_label.setText(
            f"{self.current_condition.upper()} Condition")
        self.condition_label.adjustSize()
        self.condition_label.raise_()

        self.adjustSize()
        self.setUpdatesEnabled(True)
        self._show_fullscreen_on_current_screen()

    def _show_fullscreen_on_current_screen(self):
        """Show the window fullscreen on the screen under the cursor."""
        app = QApplication.instance()  # type: ignore[assignment]
        if not isinstance(app, QApplication):
            return
        screen = app.screenAt(QCursor.pos()) or app.primaryScreen()
        if self.windowHandle():
            self.windowHandle().setScreen(screen)
        self.setGeometry(screen.geometry())
        self.showFullScreen()

    def resizeEvent(self, ev):
        """Handle window resize: reposition overlays and labels."""
        super().resizeEvent(ev)
        self.position_axis_labels()
        self.update_all_point_labels()
        self.update_token_states()
        self._reposition_header()

        self.cb_lock.move(
            self.view.width() - self.cb_lock.width() - 20,
            self.view.height() - self.cb_lock.height() - 20)
        self.cb_lock.raise_()

    # ══════════════════════════════════════════════════════════════════════
    # Camera Helpers
    # ══════════════════════════════════════════════════════════════════════

    def cube_center(self):
        """Return the center of the cube as QVector3D."""
        half = float(AXIS_LEN) / 2.0
        return QVector3D(half, half, half)

    def fit_distance_for_extent(self, extent: float, margin: float = 2):
        """Compute camera distance for a given visible extent."""
        w = max(1, self.view.width())
        h = max(1, self.view.height())
        vfov_deg = float(self.view.opts.get('fov', 60))
        vfov = np.deg2rad(vfov_deg)
        aspect = w / h
        hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect)
        half = extent / 2.0
        d_v = half / np.tan(vfov / 2.0)
        d_h = half / np.tan(hfov / 2.0)
        return float(max(d_v, d_h) * margin)

    def set_view_xy(self, offset_x=0, offset_y=0, offset_z=0):
        """Top-down view onto XZ plane (for 2D condition)."""
        self.view.opts['ortho'] = True  # type: ignore[assignment]
        cx = AXIS_LEN * 0.5 + offset_x
        cy = AXIS_LEN * 0.5 + offset_y
        cz = AXIS_LEN * 0.5 + offset_z
        self.view.opts['center'] = QVector3D(cx, cy, cz)  # type: ignore[assignment]
        self.view.setCameraPosition(
            distance=self.fit_distance_for_extent(6),
            elevation=0, azimuth=90)

    def set_view_default(self):
        """Reset camera to the default 3D perspective view."""
        try:
            self.view.opts['ortho'] = False  # type: ignore[assignment]
        except Exception:
            pass
        self.view.opts['center'] = self.cube_center()  # type: ignore[assignment]
        self.view.setCameraPosition(
            distance=self.fit_distance_for_extent(14),
            elevation=35.3, azimuth=45)
        self.position_axis_labels()
        self.logger.log_session_event("Set Default View")
        self.update_all_point_labels()

    # ══════════════════════════════════════════════════════════════════════
    # Projection / Raycasting
    # ══════════════════════════════════════════════════════════════════════

    def _get_proj_matrix(self):
        """Return the projection matrix, compatible with pyqtgraph 0.13 and 0.14."""
        vp = self.view.getViewport()
        try:
            return self.view.projectionMatrix(vp, vp)
        except TypeError:
            return self.view.projectionMatrix(vp) # type: ignore[call-arg]

    def project_point(self, p):
        """Project a world 3D point to 2D view pixel coordinates (x, y)."""
        try:
            vm = self.view.viewMatrix()
            pm = self._get_proj_matrix()
            m = pm * vm
            v = QVector3D(float(p[0]), float(p[1]), float(p[2]))
            ndc = m.map(v)
            w = max(1, self.view.width())
            h = max(1, self.view.height())
            px = int((float(ndc.x()) + 1.0) * 0.5 * w)
            py = int((1.0 - (float(ndc.y()) + 1.0) * 0.5) * h)
            return px, py
        except Exception:
            return None

    def screen_to_world_ray(self, px: int, py: int):
        """Return (near_world, far_world) QVector3D for a screen pixel."""
        w = max(1, self.view.width())
        h = max(1, self.view.height())
        nx = 2.0 * px / w - 1.0
        ny = 1.0 - 2.0 * py / h
        vm = self.view.viewMatrix()
        pm = self._get_proj_matrix()
        m = pm * vm
        try:
            inv = m.inverted()[0]
        except Exception:
            return None, None
        near_w = inv.map(QVector3D(nx, ny, -1.0))
        far_w = inv.map(QVector3D(nx, ny, 1.0))
        return near_w, far_w

    def intersect_with_plane(self, p0, p1, plane: str):
        """Intersect ray (p0->p1) with xy/xz/yz plane. Return (x,y,z)."""
        d = QVector3D(float(p1.x() - p0.x()), float(p1.y() - p0.y()),
                      float(p1.z() - p0.z()))
        if plane == 'xy':
            if abs(d.z()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['xy']) - p0.z()) / d.z()
        elif plane == 'xz':
            if abs(d.y()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['xz']) - p0.y()) / d.y()
        else:
            if abs(d.x()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['yz']) - p0.x()) / d.x()
        if t < 0:
            return None
        hit = p0 + d * t
        return float(hit.x()), float(hit.y()), float(hit.z())

    def intersect_with_plane_t(self, p0, p1, plane: str):
        """Like intersect_with_plane but returns (t, (x,y,z))."""
        d = QVector3D(float(p1.x() - p0.x()), float(p1.y() - p0.y()),
                      float(p1.z() - p0.z()))
        if plane == 'xy':
            if abs(d.z()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['xy']) - p0.z()) / d.z()
        elif plane == 'xz':
            if abs(d.y()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['xz']) - p0.y()) / d.y()
        else:
            if abs(d.x()) < 1e-9:
                return None
            t = (float(PLANE_OFFSETS['yz']) - p0.x()) / d.x()
        if t < 0:
            return None
        hit = p0 + d * t
        return float(t), (float(hit.x()), float(hit.y()), float(hit.z()))

    def clamp_to_cube(self, x, y, z):
        """Clamp world coordinates to [0, AXIS_LEN]."""
        L = AXIS_LEN
        return (max(0, min(L, x)), max(0, min(L, y)), max(0, min(L, z)))

    def camera_position_vec3(self):
        """Return camera position as numpy array or None."""
        try:
            p = self.view.cameraPosition()
            return np.array([float(p.x()), float(p.y()), float(p.z())])
        except Exception:
            return None

    def camera_forward_vec3(self):
        """Compute normalized forward vector from camera to center."""
        try:
            cp = self.camera_position_vec3()
            ctr_raw = self.view.opts.get('center')
            if cp is None or ctr_raw is None:
                return None
            if isinstance(ctr_raw, QVector3D):
                ctr = ctr_raw
            else:
                try:
                    x_a = getattr(ctr_raw, 'x', None)
                    if x_a is not None and callable(x_a):
                        ctr = QVector3D(
                            float(x_a()),  # type: ignore[arg-type]
                            float(getattr(ctr_raw, 'y')()), # type: ignore[union-attr]
                            float(getattr(ctr_raw, 'z')()))  # type: ignore[union-attr]
                    else:
                        ctr = QVector3D(float(ctr_raw[0]),  # type: ignore[index]
                                        float(ctr_raw[1]),  # type: ignore[index]
                                        float(ctr_raw[2]))  # type: ignore[index]
                except Exception:
                    ctr = QVector3D(float(ctr_raw[0]),  # type: ignore[index]
                                    float(ctr_raw[1]),  # type: ignore[index]
                                    float(ctr_raw[2]))  # type: ignore[index]
            c = np.array([float(ctr.x()), float(ctr.y()), float(ctr.z())])
            f = c - cp
            n = np.linalg.norm(f)
            if n <= 1e-9:
                return None
            return f / n
        except Exception:
            return None

    def is_point_visible_world(self, coords):
        """Return False if a world point is behind camera or offscreen."""
        try:
            cp = self.camera_position_vec3()
            fwd = self.camera_forward_vec3()
            pr = self.project_point(coords)
            if pr is None:
                return False
            x, y = pr
            w, h = self.view.width(), self.view.height()
            m = LABEL_SCREEN_MARGIN
            within = (m <= x <= w - m) and (m <= y <= h - m)
            if cp is None or fwd is None:
                return within
            pt = np.array([float(coords[0]), float(coords[1]),
                           float(coords[2])])
            if float(np.dot(pt - cp, fwd)) <= VIS_DOT_THRESHOLD:
                return False
            return within
        except Exception:
            return True

    def choose_plane_and_hit(self, px: int, py: int):
        """Choose best plane hit for screen pixel, return (plane, hit)."""
        p0, p1 = self.screen_to_world_ray(px, py)
        if p0 is None or p1 is None:
            return None
        ortho = bool(self.view.opts.get('ortho', False))
        planes_to_try = ('yz', 'xz')

        candidates = []
        for pl in planes_to_try:
            r = self.intersect_with_plane_t(p0, p1, pl)
            if r is not None:
                t, hit = r
                if ortho and pl == self.current_plane:
                    t = t - 1e-9
                candidates.append((t, pl, hit))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        _, pl, hit = candidates[0]
        return pl, hit

    # ══════════════════════════════════════════════════════════════════════
    # Point Management
    # ══════════════════════════════════════════════════════════════════════

    def ensure_point_item(self, pid: str):
        """Create or return a GLScatterPlotItem for a point sprite."""
        from pyqtgraph.opengl import GLScatterPlotItem
        if pid in self.placed_points:
            return self.placed_points[pid][0]
        item = GLScatterPlotItem(
            pos=np.array([[0.0, 0.0, 0.0]]),
            size=POINT_SIZE, color=POINT_COLOR, pxMode=True)
        self.view.addItem(item)
        self.placed_points[pid] = (item, [0.0, 0.0, 0.0])
        return item

    def set_point_position(self, pid: str, coords):
        """Set world position of a point and update all overlays."""
        item = self.ensure_point_item(pid)
        x, y, z = map(float, coords)
        if self.current_condition == "2d":
            y = 0.0
        pos = np.array([[x, y, z]], dtype=float)
        item.setData(pos=pos)
        self.placed_points[pid] = (item, [x, y, z])
        self.update_point_label(pid)
        self.update_image_label(pid)
        self.update_point_color(pid)
        self.update_helper_lines(pid)
        if '.' in pid:
            self.update_pair_line(_category_of(pid))
        try:
            self.stimuli_cb.setChecked(True)
        except Exception:
            pass

    def reset_all_points(self):
        """Remove all placed points, overlays, and reset token states."""
        for pid, (it, _) in list(self.placed_points.items()):
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.placed_points.clear()

        for cat, line in list(self.pair_lines.items()):
            try:
                self.view.removeItem(line)
            except Exception:
                pass
        self.pair_lines.clear()

        for pid, lab in list(self.point_labels.items()):
            try:
                lab.hide()
                lab.deleteLater()
            except Exception:
                pass
        self.point_labels.clear()

        for pid, lab in list(self.image_labels.items()):
            try:
                lab.hide()
                lab.deleteLater()
            except Exception:
                pass
        self.image_labels.clear()

        self.hide_hover_preview()

        for t in self.point_tokens:
            t.setProperty('placed', False)
            t.setStyleSheet(_token_style(False))
            t.show()

        self.placement_phase = 1
        self.update_token_states()
        self.update_submit_state()
        self.logger.log_session_event("Reset all placed points")

    # ══════════════════════════════════════════════════════════════════════
    # Point Overlays
    # ══════════════════════════════════════════════════════════════════════

    def ensure_point_label(self, pid: str) -> QLabel:
        """Ensure an overlay QLabel exists for a point id and return it."""
        if pid in self.point_labels:
            return self.point_labels[pid]
        lab = QLabel(pid, parent=self.view)
        lab.setProperty('pid', pid)
        lab.installEventFilter(self._point_label_filter)
        lab.setStyleSheet(
            "color: #ffffff; background: rgba(0,0,0,140); "
            "border: 1px solid #666; border-radius: 4px; "
            "padding: 1px 4px; font-size: 12px; font-weight: 600;")
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.hide()
        self.point_labels[pid] = lab
        return lab

    def ensure_image_label(self, pid: str) -> Optional[QLabel]:
        """Return or create an image overlay label for a token id."""
        cat = _category_of(pid) if '.' in pid else pid
        pm = self.images_by_cat.get(cat)
        if pm is None or pm.isNull():
            return None
        if pid in self.image_labels:
            lab = self.image_labels[pid]
            lab.setPixmap(pm)
            return lab
        lab = QLabel(parent=self.view)
        lab.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lab.setPixmap(pm)
        lab.hide()
        lab.lower()
        self.image_labels[pid] = lab
        return lab

    def update_image_label(self, pid: str):
        """Position image overlay above a placed point (debug mode only)."""
        if not self._debug_on():
            if pid in self.image_labels:
                try:
                    self.image_labels[pid].hide()
                except Exception:
                    pass
            return
        if pid not in self.placed_points:
            if pid in self.image_labels:
                try:
                    self.image_labels[pid].hide()
                except Exception:
                    pass
            return
        lab = self.ensure_image_label(pid)
        if lab is None:
            return
        _, coords = self.placed_points[pid]
        if not self.is_point_visible_world(coords):
            if pid in self.image_labels:
                try:
                    self.image_labels[pid].hide()
                except Exception:
                    pass
            return
        pr = self.project_point(coords)
        if pr is None:
            lab.hide()
            return
        px, py = pr
        lab.adjustSize()
        lab.move(int(px - lab.width() // 2),
                 int(py - lab.height() - IMAGE_OVER_POINT_MARGIN))
        lab.show()

    def update_point_label(self, pid: str):
        """Update overlay label text and position for a placed point."""
        if pid not in self.placed_points:
            return
        self.ensure_point_label(pid)
        item, coords = self.placed_points[pid]
        if not self.is_point_visible_world(coords):
            lab = self.point_labels[pid]
            try:
                lab.hide()
            except Exception:
                pass
            return
        pr = self.project_point(coords)
        lab = self.point_labels[pid]
        if pr is None:
            lab.hide()
            return
        px, py = pr
        try:
            num = pid.split('.')[0]
        except Exception:
            num = pid
        lab.setText(num)
        lab.adjustSize()
        lab.move(int(px - lab.width() // 2),
                 int(py - lab.height() - LABEL_OVER_POINT_MARGIN))
        lab.show()

    def update_all_point_labels(self):
        """Reposition all placed point labels and image overlays."""
        for pid in list(self.placed_points.keys()):
            self.update_point_label(pid)
        for pid in list(self.placed_points.keys()):
            self.update_image_label(pid)

    def update_point_color(self, pid: str):
        """Set both points of a category green if Z values match."""
        if pid not in self.placed_points or '.' not in pid:
            return
        partner = _partner_of(pid)
        if not partner or partner not in self.placed_points:
            return
        _, c_self = self.placed_points[pid]
        _, c_part = self.placed_points[partner]
        color = np.array([[1.0, 1.0, 0.0, 1.0]])
        if abs(float(c_self[2]) - float(c_part[2])) <= Z_ALIGN_EPS:
            color = np.array([[0.0, 1.0, 0.0, 1.0]])
        self.placed_points[pid][0].setData(color=color)
        self.placed_points[partner][0].setData(color=color)

    def _raise_point_overlays(self, pid: str):
        """Raise point label and image overlays for a point and its pair."""
        try:
            if pid in self.point_labels:
                self.point_labels[pid].raise_()
            if pid in self.image_labels:
                self.image_labels[pid].raise_()
            cat = pid.split('.')[0]
            for suffix in ("", ".1", ".2"):
                key = f"{cat}{suffix}"
                if key in self.image_labels:
                    self.image_labels[key].raise_()
                if key in self.point_labels:
                    self.point_labels[key].raise_()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Helper Lines (dashed 3D guides)
    # ══════════════════════════════════════════════════════════════════════

    def update_helper_lines(self, pid):
        """Draw or remove dashed helper lines for a point."""
        if self.current_condition == "2d":
            return

        hover = getattr(self.view, "hover_pid", None)
        for other_pid in list(self.helper_lines.keys()):
            if other_pid != hover:
                for it in self.helper_lines[other_pid]:
                    try:
                        self.view.removeItem(it)
                    except Exception:
                        pass
                self.helper_lines.pop(other_pid, None)

        if pid is None or hover != pid or pid not in self.placed_points:
            return

        # Remove old lines for this pid
        if pid in self.helper_lines:
            for it in self.helper_lines[pid]:
                try:
                    self.view.removeItem(it)
                except Exception:
                    pass
            self.helper_lines.pop(pid, None)

        _, (x, y, z) = self.placed_points[pid]

        def dashed(p0, p1, dash=0.4, gap=0.35):
            v = np.array(p1) - np.array(p0)
            length = float(np.linalg.norm(v))
            if length <= 1e-9:
                return []
            d = v / length
            out = []
            s = 0.0
            while s < length:
                e = min(length, s + dash)
                a = np.array(p0) + d * s
                b = np.array(p0) + d * e
                item = GLLinePlotItem(
                    pos=np.vstack([a, b]),
                    color=(1, 1, 1, 0.4), width=1, mode='lines')
                out.append(item)
                s += dash + gap
            return out

        segs = []
        segs += dashed((x, y, z), (x, y, 0.0))
        segs += dashed((x, y, z), (0.0, y, z))
        segs += dashed((x, y, z), (x, 0.0, z))
        segs += dashed((0.0, y, z), (0.0, 0.0, z))
        segs += dashed((0.0, y, z), (0.0, y, 0.0))
        segs += dashed((x, 0.0, z), (0.0, 0.0, z))
        segs += dashed((x, 0.0, z), (x, 0.0, 0.0))

        self.helper_lines[pid] = segs
        for it in segs:
            try:
                self.view.addItem(it)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════
    # Preview / Hover
    # ══════════════════════════════════════════════════════════════════════

    def ensure_hover_preview(self) -> QLabel:
        """Ensure the hover preview label exists and return it."""
        if self.hover_preview_label is None:
            lab = QLabel(parent=self.view)
            lab.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lab.hide()
            self.hover_preview_label = lab
        return self.hover_preview_label

    def set_preview_for_category(self, cat: Optional[str]):
        """Update the preview box with a scaled image for a category."""
        if not cat:
            self.preview_box.setText("Image Preview")
            self.preview_box.setPixmap(QPixmap())
            return
        pm_orig = self.images_orig.get(cat) or self.images_by_cat.get(cat)
        if pm_orig is None or pm_orig.isNull():
            self.preview_box.setText("Image Preview")
            self.preview_box.setPixmap(QPixmap())
            return
        pm_scaled = pm_orig.scaled(
            self.preview_box.width() - 12,
            self.preview_box.height() - 12,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.preview_box.setPixmap(pm_scaled)

    def _show_hover_preview_over_dock_impl(self, cat: str):
        """Show a larger preview above the token dock."""
        pm_orig = self.images_orig.get(cat) or self.images_by_cat.get(cat)
        if pm_orig is None or pm_orig.isNull():
            self.hide_hover_preview()
            return
        lab = self.ensure_hover_preview()
        pm_big = pm_orig.scaled(
            IMAGE_CONTAINER_WH, IMAGE_CONTAINER_WH,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        lab.setPixmap(pm_big)
        lab.adjustSize()
        dock_cx = self.point_dock.x() + self.point_dock.width() // 2
        top_y = self.point_dock.y()
        lab.move(int(dock_cx - lab.width() // 2),
                 int(top_y - lab.height() - IMAGE_OVER_POINT_MARGIN))
        lab.show()
        lab.lower()

    def show_hover_preview_over_dock(self, cat: str):
        """Show hover preview and update the preview box."""
        self._show_hover_preview_over_dock_impl(cat)
        self.set_preview_for_category(cat)

    def hide_hover_preview(self):
        """Hide the hover preview label (no-op to keep last preview)."""
        return

    # ══════════════════════════════════════════════════════════════════════
    # Axes & Grids
    # ══════════════════════════════════════════════════════════════════════

    def _build_axes(self):
        """Build axis line items with ticks for all three axes."""
        step = _auto_tick_step(AXIS_LEN)
        self.axis_items["x"] = (
            _build_axis_solid('x', AXIS_LEN, color=(1, 0, 0, 1), width=3)
            + _build_axis_ticks('x', AXIS_LEN, tick_step=step,
                                color=(1, 0, 0, 0.9), width=2))
        self.axis_items["y"] = (
            _build_axis_solid('y', AXIS_LEN, color=(0, 1, 0, 1), width=3)
            + _build_axis_ticks('y', AXIS_LEN, tick_step=step,
                                color=(0, 1, 0, 0.9), width=2))
        self.axis_items["z"] = (
            _build_axis_solid('z', AXIS_LEN, color=(0, 0, 1, 1), width=3)
            + _build_axis_ticks('z', AXIS_LEN, tick_step=step,
                                color=(0, 0, 1, 0.9), width=2))

    def show_z_axis(self):
        """Show the Y-axis (green), orthogonal to the 2D XZ plane."""
        for it in self.axis_items["y"]:
            try:
                self.view.addItem(it)
            except Exception:
                pass
            try:
                it.setData(color=(0, 1, 0, 1))
            except Exception:
                pass
        try:
            self.axis_label_y.show()
            self.axis_label_y.raise_()
        except Exception:
            pass

    def hide_z_axis(self):
        """Hide the Y-axis (green), orthogonal to the 2D XZ plane."""
        for it in self.axis_items["y"]:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
            try:
                it.setData(color=(0, 1, 0, 0))
            except Exception:
                pass
        try:
            self.axis_label_y.hide()
        except Exception:
            pass

    def show_plane_grids(self):
        """Show the YZ and XZ plane grids."""
        try:
            self.view.addItem(self.yz_grid)
        except Exception:
            pass
        try:
            self.view.addItem(self.xz_grid)
        except Exception:
            pass

    def hide_plane_grids(self):
        """Hide the YZ and XZ plane grids."""
        try:
            self.view.removeItem(self.yz_grid)
        except Exception:
            pass
        try:
            self.view.removeItem(self.xz_grid)
        except Exception:
            pass

    def show_cube(self):
        """Show cube wireframe around the coordinate space."""
        if not self.cube_items:
            self.cube_items = self._build_cube_wireframe(AXIS_LEN)
        for it in self.cube_items:
            try:
                self.view.addItem(it)
            except Exception:
                pass

    def hide_cube(self):
        """Hide the cube wireframe."""
        for it in self.cube_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass

    def show_lattice(self, step: float):
        """Show a 3D lattice grid inside the cube."""
        self.hide_lattice()
        self.lattice_items = self._build_lattice_grid(AXIS_LEN, step)
        for it in self.lattice_items:
            try:
                self.view.addItem(it)
            except Exception:
                pass

    def hide_lattice(self):
        """Hide the lattice grid."""
        for it in self.lattice_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self.lattice_items = []

    @staticmethod
    def _build_cube_wireframe(length: float):
        """Return line items forming a wireframe cube of given size."""
        items = []
        vals = [0.0, float(length)]
        for y in vals:
            for z in vals:
                items.append(_make_edge((0, y, z), (length, y, z)))
        for x in vals:
            for z in vals:
                items.append(_make_edge((x, 0, z), (x, length, z)))
        for x in vals:
            for y in vals:
                items.append(_make_edge((x, y, 0), (x, y, length)))
        return items

    @staticmethod
    def _build_lattice_grid(length: float, step: float):
        """Return GLLinePlotItem list for a 3D lattice grid."""
        step = max(1e-3, float(step))
        ticks = np.arange(0.0, float(length) + 1e-9, step, dtype=float)
        items = []
        for y in ticks:
            for z in ticks:
                items.append(_make_edge((0, y, z), (length, y, z),
                                        color=LATTICE_COLOR,
                                        width=LATTICE_WIDTH))
        for x in ticks:
            for z in ticks:
                items.append(_make_edge((x, 0, z), (x, length, z),
                                        color=LATTICE_COLOR,
                                        width=LATTICE_WIDTH))
        for x in ticks:
            for y in ticks:
                items.append(_make_edge((x, y, 0), (x, y, length),
                                        color=LATTICE_COLOR,
                                        width=LATTICE_WIDTH))
        return items

    # ══════════════════════════════════════════════════════════════════════
    # Axis Labels & Header
    # ══════════════════════════════════════════════════════════════════════

    def position_axis_labels(self):
        """Position axis labels at the visible ends of the three axes."""
        axis_pts = {
            'x': (AXIS_LEN, 0.0, 0.0),
            'y': (0.0, AXIS_LEN, 0.0),
            'z': (0.0, 0.0, AXIS_LEN),
        }
        label_map = {
            'x': self.axis_label_x,
            'y': self.axis_label_y,
            'z': self.axis_label_z,
        }
        for axis, world_pt in axis_pts.items():
            pr = self.project_point(world_pt)
            if pr is None:
                continue
            px, py = pr
            lab = label_map[axis]
            lab.adjustSize()
            if axis == 'x':
                lab.move(px - lab.width() // 2 - 20, py - lab.height() + 7)
            elif axis == 'y':
                lab.move(px - lab.width() // 2 + 20, py - lab.height() + 7)
            else:
                lab.move(px - 6, py - 25)
            lab.raise_()

    def position_header(self, text: str, y: Optional[int] = None):
        """Display a centered header text at top of the view."""
        if y is not None:
            self._header_y = y
        if self._header_label is None:
            self._header_label = QLabel(parent=self.view)
            self._header_label.setStyleSheet(
                "color: #c13535; font-size: 20px; font-weight: 400; "
                "background: rgba(0,0,0,0.7); padding: 10px 20px; "
                "border-radius: 10px;")
        self._header_label.setText("Hint:" + text)
        self._header_label.adjustSize()
        vw = self.view.width() if self.view.width() > 0 else 800
        x = (vw - self._header_label.width()) // 2
        self._header_label.move(x, self._header_y)
        self._header_label.raise_()
        self._header_label.show()

    def _reposition_header(self):
        """Re-center header label when view resizes."""
        if self._header_label is not None and self._header_label.isVisible():
            vw = self.view.width() if self.view.width() > 0 else 800
            x = (vw - self._header_label.width()) // 2
            self._header_label.move(x, self._header_y)

    # ══════════════════════════════════════════════════════════════════════
    # Token Management
    # ══════════════════════════════════════════════════════════════════════

    def mark_token_placed(self, pid: str):
        """Update a token as placed (hide/green) and update UI state."""
        for t in self.point_tokens:
            if t.pid == pid:
                t.setProperty('placed', True)
                t.setStyleSheet(_token_style_mode('placed'))
                t.hide()
                self.ensure_image_label(pid)
                break
        self.update_token_states()
        self.update_submit_state()

    def mark_token_unplaced(self, pid: str):
        """Mark a token as unplaced and update UI state."""
        for t in self.point_tokens:
            if t.pid == pid:
                t.setProperty('placed', False)
                t.show()
                break
        self.update_token_states()

        if pid is None:
            for pid2, segs in list(self.helper_lines.items()):
                for it in segs:
                    try:
                        self.view.removeItem(it)
                    except Exception:
                        pass
            self.helper_lines.clear()
            return

        if pid in self.placed_points:
            it, _ = self.placed_points[pid]
            it.setData(size=POINT_SIZE + 10)
            it.setData(color=np.array([[1.0, 1.0, 1.0, 1.0]]))

    def update_token_states(self):
        """Update enabled/disabled state and styles of all tokens."""
        for t in self.point_tokens:
            placed = bool(t.property('placed'))
            if placed:
                t.setStyleSheet(_token_style_mode('placed'))
                t.setEnabled(False)
            else:
                t.setEnabled(True)
                t.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                t.setStyleSheet(_token_style_mode('active'))
        self.hide_hover_preview()

    def remove_placed_point(self, pid: str):
        """Remove a placed point's sprite and overlays."""
        if pid in self.placed_points:
            try:
                it, _ = self.placed_points[pid]
                self.view.removeItem(it)
            except Exception:
                pass
            self.placed_points.pop(pid, None)
        if pid in self.point_labels:
            try:
                self.point_labels[pid].hide()
                self.point_labels[pid].deleteLater()
            except Exception:
                pass
            self.point_labels.pop(pid, None)
        if pid in self.image_labels:
            try:
                self.image_labels[pid].hide()
                self.image_labels[pid].deleteLater()
            except Exception:
                pass
            self.image_labels.pop(pid, None)
        self.update_pair_line(_category_of(pid))
        self.update_submit_state()

    def update_pair_line(self, cat: str):
        """Draw or update the line between .1 and .2 points of a category."""
        if not self._debug_on():
            if cat in self.pair_lines:
                line = self.pair_lines[cat]
                try:
                    self.view.removeItem(line)
                except Exception:
                    pass
                try:
                    line.setData(pos=np.empty((0, 3), dtype=float))
                except Exception:
                    pass
                del self.pair_lines[cat]
            return

        pid1, pid2 = f"{cat}.1", f"{cat}.2"
        has1 = pid1 in self.placed_points
        has2 = pid2 in self.placed_points
        if has1 and has2:
            _, c1 = self.placed_points[pid1]
            _, c2 = self.placed_points[pid2]
            pts = np.array([c1, c2], dtype=float)
            if cat in self.pair_lines:
                self.pair_lines[cat].setData(pos=pts)
            else:
                line = GLLinePlotItem(
                    pos=pts, color=(1, 1, 1, 1), width=1, mode='lines')
                self.pair_lines[cat] = line
            try:
                self.view.addItem(self.pair_lines[cat])
            except Exception:
                pass
        else:
            if cat in self.pair_lines:
                line = self.pair_lines[cat]
                try:
                    self.view.removeItem(line)
                except Exception:
                    pass
                try:
                    line.setData(pos=np.empty((0, 3), dtype=float))
                except Exception:
                    pass
                del self.pair_lines[cat]

    def update_submit_state(self):
        """Enable submit button only when all tokens are placed."""
        try:
            total = len(self.point_tokens)
            placed = len(self.placed_points)
            started = bool(self.start_cb.isChecked())
            self.btn_submit.setEnabled(placed == total and started)
        except Exception:
            pass

    def _debug_on(self) -> bool:
        """Return True if 'Show Stimuli' debug mode is enabled."""
        return bool(self.cb_stimuli.isChecked())

    # ══════════════════════════════════════════════════════════════════════
    # Tutorial Progress
    # ══════════════════════════════════════════════════════════════════════

    def _check_tutorial_rotation(self):
        """Mark rotation tutorial step as completed."""
        if not self.rotation_done:
            self.rotation_done = True
            self.rotate_and_adjust_cb.setChecked(True)
            self.update_progress_counter()

    def _check_tutorial_height_adjust(self):
        """Mark height-adjustment tutorial step as completed."""
        if not self.height_adjust_done:
            self.height_adjust_done = True
            self.adjust_token_height_cb.setChecked(True)
            self.update_progress_counter()

    def update_progress_counter(self):
        """Update the tutorial progress counter and start button state."""
        completed = 0
        name_ok = bool(self.name_input.text().strip())
        self.name_cb.setChecked(name_ok)
        if self.name_cb.isChecked():
            completed += 1
        if self.check_hover_cb.isChecked():
            completed += 1
        if self.rotate_and_adjust_cb.isChecked():
            completed += 1
        if self.stimuli_cb.isChecked():
            completed += 1
        if self.adjust_token_height_cb.isChecked():
            completed += 1
        self.counter_label.setText(f"({completed}/5)")
        self.counter_label.adjustSize()
        self._update_start_button_state()

    def _update_start_button_state(self):
        """Enable Start button when all tutorial steps are done."""
        self.btn_start.setEnabled(
            not self.experiment_running
            and bool(self.name_input.text().strip())
            and self.check_hover_cb.isChecked()
            and self.rotate_and_adjust_cb.isChecked()
            and self.stimuli_cb.isChecked()
            and self.adjust_token_height_cb.isChecked())

    def toggle_checklist(self, checked=None):
        """Toggle tutorial checklist between collapsed and expanded."""
        if self._is_collapsed:
            self.collapse_btn.setText("\u25BC")
            self.status_label.show()
            self.set_name_label.show()
            self.name_cb.show()
            self.check_hover_label.show()
            self.check_hover_cb.show()
            self.stimuli_drag_label.show()
            self.stimuli_cb.show()
            self.rotate_and_adjust_label.show()
            self.rotate_and_adjust_cb.show()
            self.adjust_token_height_label.show()
            self.adjust_token_height_cb.show()
            self.start_label.show()
            self.start_cb.show()
            self.counter_label.show()
            self.line_bottom.move(20, 385)
            self._is_collapsed = False
        else:
            self.collapse_btn.setText("\u25B6")
            self.status_label.show()
            self.set_name_label.hide()
            self.name_cb.hide()
            self.check_hover_label.hide()
            self.check_hover_cb.hide()
            self.stimuli_drag_label.hide()
            self.stimuli_cb.hide()
            self.rotate_and_adjust_label.hide()
            self.rotate_and_adjust_cb.hide()
            self.adjust_token_height_label.hide()
            self.adjust_token_height_cb.hide()
            self.start_label.hide()
            self.start_cb.hide()
            self.counter_label.hide()
            self.line_bottom.move(20, 200)
            self._is_collapsed = True

    def update_label(self):
        """Update checklist step labels based on current condition."""
        if self.current_condition == "2d":
            self.rotate_and_adjust_label.setText(
                '3. View Rotation in this instance is forbidden:')
            self.stimuli_drag_label.setText(
                '4. Drag Stimuli into 2D space, adjust position:')
            self.adjust_token_height_label.setText(
                '5. Hold and Drag Stimuli to alter point position:')
        else:
            self.rotate_and_adjust_label.setText(
                '3. Hold and Drag "left-click" to rotate view:')
            self.stimuli_drag_label.setText(
                '4. Drag and Drop Stimuli into the 3D space:')
            self.adjust_token_height_label.setText(
                '5. Hold Point and use "Wheel" to adjust height:')
        self.adjust_token_height_label.adjustSize()
        self.rotate_and_adjust_label.adjustSize()
        self.stimuli_drag_label.adjustSize()

    # ══════════════════════════════════════════════════════════════════════
    # Experiment Flow
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_lock(self, checked: bool):
        """Toggle camera lock state."""
        self.lock_camera = bool(checked)
        if checked:
            self.logger.log_session_event("camera locked")
        else:
            self.logger.log_session_event("camera unlocked")

    def start_experiment(self):
        """Start the experiment session."""
        if self.experiment_running:
            return
        if not self.name_input.text().strip():
            return

        self.experiment_running = True
        self.start_time = datetime.now()
        self.logger.start_time = self.start_time

        self.start_cb.setChecked(True)
        self.cb_lock.setChecked(True)
        self.name_cb.setChecked(True)
        self.check_hover_cb.setChecked(True)
        self.stimuli_cb.setChecked(True)
        self.rotate_and_adjust_cb.setChecked(True)
        self.adjust_token_height_cb.setChecked(True)

        self.btn_start.setDisabled(True)
        self.name_input.setDisabled(True)
        self.age_slider.setDisabled(True)

        if not self._is_collapsed:
            self.toggle_checklist(None)
        self.collapse_btn.show()
        self.reset_all_points()
        self.apply_condition_defaults()
        self.update_progress_counter()

        if self.current_condition == "2d":
            self.set_view_xy()
        else:
            self.set_view_default()
        self.logger.log_session_event("Experiment started")

    def apply_condition_defaults(self):
        """Apply default tutorial states for the current condition."""
        if self.current_condition == "2d":
            self.rotation_done = True
            self.height_adjust_done = True
            self.rotate_and_adjust_cb.setChecked(True)
            self.adjust_token_height_cb.setChecked(True)

    def apply_labels(self):
        """Apply axis labels from edit fields to overlay labels."""
        tx = self.edit_x.text().strip()
        ty = self.edit_y.text().strip()
        tz = self.edit_z.text().strip()
        self.axis_label_x.setText(tx)
        self.axis_label_y.setText(tz)   # swapped
        self.axis_label_z.setText(ty)   # swapped
        for lab in (self.axis_label_x, self.axis_label_y,
                    self.axis_label_z):
            lab.show()
            lab.raise_()
        self.position_axis_labels()

    def _collect_combined_points_norm(self):
        """Collect and normalize placed point coordinates for export."""
        data = []
        half = float(AXIS_LEN) * 0.5
        for pid, (_item, coords) in self.placed_points.items():
            x, y, z = map(float, coords)
            cat = pid.split('.')[0]
            png_name = self.png_name_by_cat.get(cat, pid)
            if self.current_condition == "2d":
                xn = (x - half) / half
                yn = (z - half) / half
                zn = 0.0
            else:
                xn = (x - half) / half
                yn = (y - half) / half
                zn = (z - half) / half
            data.append((png_name, xn, yn, zn))
        data.sort(key=lambda t: t[0])
        return data

    def export_results(self):
        """Export placed points to CSV and switch to the next condition."""
        base = os.path.dirname(os.path.abspath(__file__))
        results_root = os.path.join(base, "results")
        condition = self.current_condition or "unknown"
        condition_dir = os.path.join(results_root, condition.lower())

        try:
            os.makedirs(condition_dir, exist_ok=True)
        except Exception as e:
            self.logger.log_session_event(
                f"Failed to create results directory: {e}")
            return

        participant_name = (self.name_input.text().strip().replace(" ", "_")
                            or "anonymous")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(condition_dir, f"{participant_name}_{ts}.csv")

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([f"Participant: "
                            f"{participant_name.replace('_', ' ')}"])
                if self.start_time is not None:
                    elapsed = (datetime.now()
                               - self.start_time).total_seconds()
                    w.writerow([f"Time: {elapsed:.2f}"])
                else:
                    w.writerow(["Time:"])
                w.writerow([f"Condition: {condition}"])
                w.writerow([])
                w.writerow(["mask_png", "x", "y", "z"])

                rows = self._collect_combined_points_norm()
                for name, xn, yn, zn in rows:
                    if self.current_condition == "2d":
                        x_csv, y_csv, z_csv = xn, yn, 0.0
                    else:
                        x_csv, y_csv, z_csv = xn, zn, yn
                    w.writerow([
                        name,
                        f"{x_csv:.6f}",
                        f"{y_csv:.6f}",
                        f"{z_csv:.6f}",
                    ])
        except Exception as e:
            self.logger.log_session_event(f"Failed to write CSV: {e}")
            return

        # ── Condition switching ───────────────────────────────────────
        self.submitted_conditions.add(self.current_condition)

        if self.submitted_conditions == {"2d", "3d"}:
            self.experiment_running = False
            QApplication.quit()
            return

        if self.current_condition == "2d":
            self.current_condition = "3d"
            self.condition_label.setText("3D Condition")
            self.condition_label.adjustSize()
            self.condition_label.raise_()
            self.rotation_done = False
            self.height_adjust_done = False
            self.cb_lock.setChecked(False)
            self.cb_lock.show()
            self.btn_grid.setDisabled(False)
            self.show_z_axis()
            try:
                self.axis_label_y.show()
                self.axis_label_y.raise_()
            except Exception:
                pass
            self.set_view_default()
            self.position_axis_labels()
            self.update_label()
            self.update_progress_counter()
            self.reset_all_points()
            self.logger.log_session_event("submitted 2D, switched to 3D")
        else:
            self.current_condition = "2d"
            self.condition_label.setText("2D Condition")
            self.condition_label.adjustSize()
            self.condition_label.raise_()
            self.rotation_done = True
            self.height_adjust_done = True
            self.cb_lock.setChecked(True)
            self.cb_lock.hide()
            self.btn_grid.setDisabled(True)
            self.hide_z_axis()
            try:
                self.axis_label_y.hide()
            except Exception:
                pass
            self.set_view_xy()
            self.position_axis_labels()
            self.update_label()
            self.update_progress_counter()
            self.reset_all_points()
            self.logger.log_session_event("submitted 3D, switched to 2D")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    try:
        app.setFont(QFont("SF Pro Text", 12))
    except Exception:
        try:
            app.setFont(QFont(".SF NS Text", 12))
        except Exception:
            pass

    dialog = ConditionDialog()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    window = ExperimentWindow(condition=dialog.get_condition())
    sys.exit(app.exec())
