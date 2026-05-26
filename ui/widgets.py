"""Reusable widgets and drawing helpers for the experiment UI."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QMimeData, QPoint, QObject
from PySide6.QtGui import QCursor, QDrag, QPixmap, QVector3D
from PySide6.QtWidgets import QLabel
from pyqtgraph.opengl import GLLinePlotItem, GLViewWidget

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
CUBE_COLOR = (0.56, 0.56, 0.60, 0.34)
GRID_COLOR = (0.56, 0.56, 0.60, 0.28)
HELPER_LINE_COLOR = (0.88, 0.88, 0.92, 0.45)
LATTICE_COLOR = (0.56, 0.56, 0.60, 0.24)
AXIS_X_COLOR = (0.86, 0.18, 0.18, 1.0)
AXIS_X_TICK_COLOR = (0.86, 0.18, 0.18, 0.90)
AXIS_Y_COLOR = (0.12, 0.72, 0.20, 1.0)
AXIS_Y_TICK_COLOR = (0.12, 0.72, 0.20, 0.90)
AXIS_Z_COLOR = (0.22, 0.38, 0.92, 1.0)
AXIS_Z_TICK_COLOR = (0.22, 0.38, 0.92, 0.90)
PLANE_OFFSETS = {"xy": 0.0, "xz": 0.0, "yz": 0.0}

ALIGN_OK_HTML = "<span style='color:#7CFC00'>✅ {partner}</span>"
ALIGN_BAD_HTML = "<span style='color:#ff6666'>❌ {partner}</span>"


def _category_of(pid: str) -> str:
    """Extract category (number prefix) from a point id like '3.1'."""
    return pid.split(".")[0]


def _partner_of(pid: str) -> Optional[str]:
    """Return the partner point id (e.g. '3.1' -> '3.2')."""
    if "." not in pid:
        return None
    cat = _category_of(pid)
    return f"{cat}.2" if pid.endswith(".1") else f"{cat}.1"


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
    if mode == "placed":
        return (
            "QLabel { color: #111; background: #7CFC00; border: 1px solid #3a3; "
            "border-radius: 4px; padding: 2px 6px; font-weight: 600; }"
        )
    if mode == "disabled":
        return (
            "QLabel { color: #aaa; background: #333; border: 1px solid #666; "
            "border-radius: 4px; padding: 2px 6px; }"
        )
    return (
        "QLabel { color: #eee; background: #444; border: 1px solid #999; "
        "border-radius: 4px; padding: 2px 6px; } "
        "QLabel:hover { background: #555; }"
    )


def _gl_color(color):
    """Return an OpenGL-safe RGBA color in the 0..1 range."""
    if isinstance(color, np.ndarray):
        arr = np.ascontiguousarray(color, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 3 and np.nanmax(arr[:3]) > 1.0:
            arr = arr.copy()
            arr[:3] /= 255.0
        elif arr.ndim >= 2 and arr.shape[-1] >= 3 and np.nanmax(arr[..., :3]) > 1.0:
            arr = arr.copy()
            arr[..., :3] /= 255.0
        return np.clip(arr, 0.0, 1.0)

    values = tuple(float(v) for v in color)
    if len(values) >= 3 and max(values[:3]) > 1.0:
        alpha = values[3] if len(values) > 3 else 1.0
        values = (values[0] / 255.0, values[1] / 255.0, values[2] / 255.0, alpha)
    return tuple(max(0.0, min(1.0, value)) for value in values)


def _set_line_color(item, color) -> None:
    """Apply normalized line color to an existing GL line item."""
    item.setData(color=_gl_color(color))


def _safe_line_width(width: float) -> float:
    """Use line widths supported reliably by macOS OpenGL contexts."""
    return 1.0 if float(width) > 1.0 else float(width)


def _line_item(p0, p1, color=(1, 1, 1, 1), width=1):
    pts = np.ascontiguousarray([p0, p1], dtype=np.float32)
    return GLLinePlotItem(
        pos=pts,
        color=_gl_color(color),
        width=_safe_line_width(width),
        mode="lines",
    )


def _line_segments_item(segments, color=(1, 1, 1, 1), width=1):
    """Create one GL item for many independent line segments."""
    pts = []
    for p0, p1 in segments:
        pts.extend((p0, p1))
    if not pts:
        pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    arr = np.ascontiguousarray(pts, dtype=np.float32)
    return GLLinePlotItem(
        pos=arr,
        color=_gl_color(color),
        width=_safe_line_width(width),
        mode="lines",
    )


def _axis_segment(p0, p1, color=(1, 1, 1, 1), width=3):
    """Create a single GL line segment between two 3D points."""
    return _line_item(p0, p1, color=color, width=width)


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
        "x": ((0, 0, 0), (length, 0, 0)),
        "y": ((0, 0, 0), (0, length, 0)),
        "z": ((0, 0, 0), (0, 0, length)),
    }
    p0, p1 = endpoints[axis]
    return [_axis_segment(p0, p1, color=color, width=width)]


def _build_axis_ticks(
    axis: str,
    length: float,
    tick_step: float = 0,
    tick_size: float = TICK_SIZE,
    color=(1, 1, 1, 0.9),
    width=2,
):
    """Build tick mark line items along an axis."""
    segments = []
    if not tick_step:
        tick_step = _auto_tick_step(length)
    t = float(tick_step)
    while t < length + 1e-9:
        if axis == "x":
            segments.append(((t, 0, 0), (t, 0, tick_size)))
        elif axis == "y":
            segments.append(((0, t, 0), (0, t, tick_size)))
        else:
            segments.append(((0, 0, t), (tick_size, 0, t)))
            segments.append(((0, 0, t), (0, tick_size, t)))
        t += tick_step
    return [_line_segments_item(segments, color=color, width=width)]


def _make_edge(p0, p1, color=CUBE_COLOR, width=CUBE_WIDTH):
    """Create a single wireframe edge between two points."""
    return _line_item(p0, p1, color=color, width=width)


# ═══════════════════════════════════════════════════════════════════════════
# Widget Helpers
# ═══════════════════════════════════════════════════════════════════════════


class _PointLabelFilter(QObject):
    """Event filter for overlay point labels (passthrough)."""

    def eventFilter(self, obj, ev):
        return False


class _ViewResizeFilter(QObject):
    """Repositions overlay checkboxes when the scene view resizes."""

    def __init__(self, experiment: "ExperimentWindow", parent=None): #type: ignore
        super().__init__(parent)
        self.exp = experiment

    def eventFilter(self, obj, ev):
        if ev.type() == ev.Type.Resize:
            exp = self.exp
            w, h = obj.width(), obj.height()
            try:
                exp.cb_lock.move(
                    w - exp.cb_lock.width() - 20, h - exp.cb_lock.height() - 20
                )
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

    def __init__(self, experiment: "ExperimentWindow", **kwargs): #type: ignore
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

    def initializeGL(self):
        """Initialize OpenGL and reset pyqtgraph line shaders for this context."""
        try:
            GLLinePlotItem._shaderProgram = None
        except Exception:
            pass
        super().initializeGL()

    # ── Drag & Drop ───────────────────────────────────────────────────────

    def dragEnterEvent(self, ev):
        """Accept drags carrying point-id MIME data."""
        if ev.mimeData().hasFormat("application/x-point-id"):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        """Continue accepting point-id drags."""
        if ev.mimeData().hasFormat("application/x-point-id"):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        """Place a token at the ray-plane intersection point."""
        exp = self.exp
        if not ev.mimeData().hasFormat("application/x-point-id"):
            ev.ignore()
            return
        try:
            pid = ev.mimeData().data("application/x-point-id").data().decode("utf-8")
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
            x, y, z = exp.clamp_to_cube(float(hit.x()), float(hit.y()), float(hit.z()))
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
                    hit_pid, (None, [0, 0, AXIS_LEN * 0.5])
                )
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
        if ev.buttons() & Qt.MouseButton.LeftButton and self.dragging_pid is None:
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
                zp = (
                    0.0
                    if exp.current_condition == "2d"
                    else float(self.drag_z0 or AXIS_LEN * 0.5)
                )
                if exp.current_condition == "2d":
                    exp.set_point_position(pid, (fx, 0.0, zp))
                else:
                    exp.set_point_position(pid, (fx, fy, zp))
                cat = pid.split(".")[0]
                exp.set_preview_for_category(cat)
                return
            else:
                self.freeze_xy_after_scroll = False
                self.freeze_xy_pos = None

        p0, p1 = exp.screen_to_world_ray(int(mx), int(my))
        if p0 is None or p1 is None:
            return

        zp = (
            0.0
            if exp.current_condition == "2d"
            else float(self.drag_z0 or AXIS_LEN * 0.5)
        )
        d = QVector3D(
            float(p1.x() - p0.x()), float(p1.y() - p0.y()), float(p1.z() - p0.z())
        )

        if exp.current_condition == "2d":
            if abs(d.y()) > 1e-9:
                t = (0.0 - p0.y()) / d.y()
                if t >= 0:
                    hit = p0 + d * t
                    x, _, z = exp.clamp_to_cube(float(hit.x()), 0.0, float(hit.z()))
                    exp.set_point_position(pid, (x, 0.0, z))
        else:
            if abs(d.z()) > 1e-9:
                t = (zp - p0.z()) / d.z()
                if t >= 0:
                    hit = p0 + d * t
                    x, y, _ = exp.clamp_to_cube(float(hit.x()), float(hit.y()), zp)
                    exp.set_point_position(pid, (x, y, zp))

        cat = pid.split(".")[0]
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
            self.hover_pid = hit_pid if hit_pid and best_d2 <= pick_r**2 else None

            if self.hover_pid != prev_hover:
                if prev_hover is not None:
                    exp.update_helper_lines(prev_hover)
                if self.hover_pid is not None:
                    exp.update_helper_lines(self.hover_pid)

            if self.hover_pid is not None:
                cat = self.hover_pid.split(".")[0]
                exp.set_preview_for_category(cat)
                item, _ = exp.placed_points[self.hover_pid]
                item.setData(color=np.array([[1, 1, 1, 1]]), size=POINT_SIZE + 6)
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

    def __init__(self, pid: str, experiment: "ExperimentWindow", parent=None): #type: ignore
        super().__init__(pid, parent)
        self.pid = pid
        self.exp = experiment
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setStyleSheet(_token_style_mode("disabled"))
        self.setFixedSize(110, 30)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        """Show image preview when hovering over token."""
        try:
            cat = self.pid.split(".")[0]
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
            cat = self.pid.split(".")[0]
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
        cat = self.pid.split(".")[0]
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
        mime.setData("application/x-point-id", self.pid.encode("utf-8"))
        drag.setMimeData(mime)

        pm = exp.images_by_cat.get(cat)
        if pm is not None and not pm.isNull():
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, int(pm.height() * 0.8)))

        drag.exec(Qt.DropAction.MoveAction)
        exp.is_dock_drag = False
