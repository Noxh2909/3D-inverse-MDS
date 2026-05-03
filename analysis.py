"""
analysis.py – Class-based analysis pipeline for 3D inverse-MDS experiment data.

Loads participant embedding CSVs, runs Procrustes alignment, computes inter-subject
consistency metrics (RMSE, Spearman, kNN preservation), and generates 
plots for both aggregated and per-participant analysis.

Usage:
    python analysis.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import proj3d  # noqa: F401 – required for 3D projection
from PIL import Image, ImageOps
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnalysisSelection:
    """Stores which analysis modules should be executed."""

    metrics: frozenset[str] = frozenset()

    def wants(self, *names: str) -> bool:
        """Return True when one of the requested modules should run."""
        if not self.metrics:
            return True
        return any(name in self.metrics for name in names)


@dataclass
class AnalysisConfig:
    """Central configuration for the analysis pipeline."""

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Set to True for anonymous participant labels (Participant 1, 2, …).
    anonymous: bool = True

    # Restrict analysis to specific participant indices (1-based).
    # Empty list means ALL participants.
    participants: list[int] = field(default_factory=list)

    # Scale Procrustes dissimilarity matrices to [0, 1].
    normalize_procrustes: bool = True

    # Plot font controls.
    font_size: int = 18
    title_font_size: int = 12
    scale_number_size: int = 12
    stimulus_number_size: int = 16
    matrix_number_size: int = 10
     
    rdm_axis_font_size: int = 30
    rdm_scale_font_size: int = 40
    rdm_legend_font_size: int = 25
    
    pro_rdm_axis_font_size: int = 20
    pro_rdm_scale_font_size: int = 20
    pro_rdm_legend_font_size: int = 20
    
    spr_rdm_axis_font_size: int = 20
    spr_rdm_scale_font_size: int = 20
    spr_rdm_legend_font_size: int = 20

    arrangement_axis_font_size: int = 25

    # Select which analysis modules should run.
    selection: AnalysisSelection = field(default_factory=AnalysisSelection)

    # Per-stimulus border colours (1-based stimulus index → hex).
    stimulus_border_colors: dict[int, str] = field(default_factory=lambda: {
        1: "#6929c4", 2: "#1192e8", 3: "#005d5d", 4: "#9f1853",
        5: "#fa4d56", 6: "#570408", 7: "#198038", 8: "#002d9c",
        9: "#ee538b", 10: "#b28600", 11: "#009d9a", 12: "#012749",
    })

    # Derived paths (computed in __post_init__).
    final_results_dir: Path = field(init=False)
    pictures_dir: Path = field(init=False)
    analysis_dir: Path = field(init=False)
    general_dir: Path = field(init=False)
    general_2d_dir: Path = field(init=False)
    general_3d_dir: Path = field(init=False)
    detailed_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.final_results_dir = self.base_dir / "final_results"
        self.pictures_dir = self.base_dir / "pictures"
        self.analysis_dir = self.base_dir / "analysis"
        self.general_dir = self.analysis_dir / "general"
        self.general_2d_dir = self.general_dir / "2d"
        self.general_3d_dir = self.general_dir / "3d"
        self.detailed_dir = self.analysis_dir / "detailed"


# ──────────────────────────────────────────────────────────────────────
# Progress helpers
# ──────────────────────────────────────────────────────────────────────


def _progress(iterable, **kwargs):
    """Wrap iterable with tqdm if available, otherwise return as-is."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


def _set_progress_label(progress, label: str) -> None:
    """Set the postfix string on a tqdm progress bar (no-op if unavailable)."""
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(label)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_embedding(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load stimulus names and 3D coordinates from an experiment CSV.

    Returns:
        Tuple of (stimulus_names, coordinates_array) where coordinates_array
        has shape (n_stimuli, 3).
    """
    names: list[str] = []
    coords: list[list[float]] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith(("mask_png", "Participant")):
                continue
            if len(row) != 4:
                continue
            names.append(row[0])
            coords.append([float(row[1]), float(row[2]), float(row[3])])

    return names, np.asarray(coords)


# ──────────────────────────────────────────────────────────────────────
# Stimulus visualisation helpers
# ──────────────────────────────────────────────────────────────────────


class StimulusVisualizer:
    """Handles loading, rendering and labelling of stimulus images in plots."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.cfg = config

    # ── index / label helpers ──

    @staticmethod
    def stimulus_index(name: str) -> int | None:
        """Extract 1-based stimulus index from a filename like 'Stimuli_00.png'."""
        stem = Path(name).stem.replace("Stimuli_", "")
        try:
            return int(stem) + 1
        except ValueError:
            return None

    @staticmethod
    def stimulus_label(name: str) -> str:
        """Convert e.g. 'Stimuli_00.png' → '01' (1-based, zero-padded)."""
        idx = StimulusVisualizer.stimulus_index(name)
        if idx is None:
            return Path(name).stem
        return f"{idx:02d}"

    # ── image loading ──

    def _add_border(self, image: Image.Image, name: str,
                    border_width: int = 10) -> Image.Image:
        """Add a coloured border to a stimulus image."""
        idx = self.stimulus_index(name)
        colour = self.cfg.stimulus_border_colors.get(idx, "#000000") if idx else "#000000"
        return ImageOps.expand(image, border=border_width, fill=colour)

    def load_image(self, name: str, zoom: float = 0.35,
                   bordered: bool = False, border_width: int = 5) -> OffsetImage | None:
        """Load a stimulus image and return it as a matplotlib OffsetImage."""
        path = self.cfg.pictures_dir / name
        if not path.exists():
            return None
        img = Image.open(path).convert("RGBA")
        if bordered:
            img = self._add_border(img, name, border_width)
        return OffsetImage(np.asarray(img), zoom=zoom)

    def create_stimulus_image(self, name: str, zoom: float = 0.35,
                              border_width: int = 5) -> OffsetImage | None:
        """Create a bordered stimulus thumbnail for plot annotation."""
        return self.load_image(name, zoom=zoom, bordered=True, border_width=border_width)

    def stimulus_size_pixels(self, name: str, border_width: int) -> tuple[int, int]:
        """Return (width, height) in pixels of a stimulus image including border."""
        path = self.cfg.pictures_dir / name
        if not path.exists():
            return 42 + 2 * border_width, 42 + 2 * border_width
        with Image.open(path) as img:
            w, h = img.size
        return w + 2 * border_width, h + 2 * border_width

    def stimulus_size_points(self, name: str, zoom: float,
                             border_width: int) -> tuple[float, float]:
        """Return (width, height) in figure points."""
        w, h = self.stimulus_size_pixels(name, border_width)
        return w * zoom, h * zoom

    # ── label placement ──

    @staticmethod
    def _label_box_size(name: str, label_size: int) -> tuple[float, float]:
        """Estimate (width, height) of a text label box in points."""
        label = StimulusVisualizer.stimulus_label(name)
        width = max(18.0, len(label) * label_size * 0.72 + 6.0)
        height = label_size + 6.0
        return width, height

    @staticmethod
    def _pixels_to_points(fig, values):
        """Convert pixel coordinates to figure points."""
        return np.asarray(values, dtype=float) * (72.0 / fig.dpi)

    @staticmethod
    def _rect_from_center(cx, cy, w, h):
        hw, hh = w / 2.0, h / 2.0
        return cx - hw, cy - hh, cx + hw, cy + hh

    @staticmethod
    def _rect_from_anchor(ax, ay, w, h, ha, va):
        if ha == "center":
            x0, x1 = ax - w / 2, ax + w / 2
        elif ha == "left":
            x0, x1 = ax, ax + w
        else:
            x0, x1 = ax - w, ax

        if va == "center":
            y0, y1 = ay - h / 2, ay + h / 2
        elif va == "bottom":
            y0, y1 = ay, ay + h
        else:
            y0, y1 = ay - h, ay

        return x0, y0, x1, y1

    @staticmethod
    def _rect_overlap_area(r1, r2, padding: float = 0.0) -> float:
        """Compute overlap area between two axis-aligned rectangles."""
        x0 = max(r1[0] - padding, r2[0] - padding)
        y0 = max(r1[1] - padding, r2[1] - padding)
        x1 = min(r1[2] + padding, r2[2] + padding)
        y1 = min(r1[3] + padding, r2[3] + padding)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    @staticmethod
    def _direction_offsets(img_w: float, img_h: float, gap: float = 6.0):
        """Return candidate label offsets: below, left, right, above."""
        hw, hh = img_w / 2.0, img_h / 2.0
        return [
            ((0.0, -(hh + gap)), "center", "top"),
            ((-(hw + gap), 0.0), "right", "center"),
            (((hw + gap), 0.0), "left", "center"),
            ((0.0, hh + gap), "center", "bottom"),
        ]

    def compute_label_placements(self, points: np.ndarray, image_sizes,
                                 names: list[str], label_size: int = 11):
        """Place labels to minimise overlap with images and other labels.

        Returns a list of (offset, ha, va) tuples parallel to *points*.
        """
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return []
        if len(points) == 1:
            iw, ih = image_sizes[0]
            offset, ha, va = self._direction_offsets(iw, ih)[0]
            return [(offset, ha, va)]

        image_rects = [
            self._rect_from_center(pt[0], pt[1], iw, ih)
            for pt, (iw, ih) in zip(points, image_sizes)
        ]

        # Sort by nearest-neighbour distance (tightest first).
        nearest = []
        for i, pt in enumerate(points):
            others = np.delete(points, i, axis=0)
            dists = np.linalg.norm(others - pt, axis=1)
            nearest.append(np.min(dists) if len(dists) else np.inf)

        placements: list = [None] * len(points)
        placed_rects: list = []

        for idx in np.argsort(nearest):
            pt = points[idx]
            iw, ih = image_sizes[idx]
            lw, lh = self._label_box_size(names[idx], label_size)
            candidates = self._direction_offsets(iw, ih)
            best, best_pen = None, None

            for cand in candidates:
                offset, ha, va = cand
                anchor = (pt[0] + offset[0], pt[1] + offset[1])
                rect = self._rect_from_anchor(anchor[0], anchor[1], lw, lh, ha, va)

                pen = 0.0
                for j, ir in enumerate(image_rects):
                    if j == idx:
                        continue
                    pen += self._rect_overlap_area(rect, ir, padding=2.0)
                for lr in placed_rects:
                    pen += 1.5 * self._rect_overlap_area(rect, lr, padding=2.0)

                if pen == 0.0:
                    best = cand
                    break
                if best_pen is None or pen < best_pen:
                    best, best_pen = cand, pen

            if best is None:
                best = candidates[0]
            placements[idx] = best
            offset, ha, va = best
            anchor = (pt[0] + offset[0], pt[1] + offset[1])
            placed_rects.append(
                self._rect_from_anchor(anchor[0], anchor[1], lw, lh, ha, va)
            )

        return placements

    # ── annotation helpers ──

    def add_stimulus_label(self, ax, x: float, y: float, name: str,
                           placement, label_size: int = 11) -> None:
        """Annotate a data point with its stimulus number."""
        offset, ha, va = placement
        ax.annotate(
            self.stimulus_label(name),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=label_size, fontweight="bold", color="black",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 0.12},
            zorder=12, clip_on=False,
        )

    def add_stimuli_2d(self, ax, names, x_coords, y_coords,
                       zoom: float = 0.20, border_width: int = 5,
                       label_size: int = 11) -> None:
        """Overlay stimulus thumbnails and labels on a 2D axes."""
        fig = ax.figure
        fig.canvas.draw()

        data_pts = np.column_stack((x_coords, y_coords))
        display_pts = self._pixels_to_points(fig, ax.transData.transform(data_pts))
        img_sizes = [self.stimulus_size_points(n, zoom, border_width) for n in names]
        placements = self.compute_label_placements(display_pts, img_sizes, names,
                                                   label_size=label_size)

        for x, y, name, plc in zip(x_coords, y_coords, names, placements):
            img = self.create_stimulus_image(name, zoom=zoom, border_width=border_width)
            if img is not None:
                ab = AnnotationBbox(img, (x, y), frameon=False,
                                    box_alignment=(0.5, 0.5), pad=0.0, zorder=5)
                ax.add_artist(ab)
            self.add_stimulus_label(ax, x, y, name, plc, label_size=label_size)

    def add_projected_stimuli_3d(self, ax, names, coords,
                                 zoom: float = 0.12, border_width: int = 5,
                                 label_size: int = 11) -> None:
        """Project stimulus thumbnails onto 3D axes and place labels."""
        fig = ax.figure
        fig.canvas.draw()

        proj_pts = []
        for x, y, z in coords:
            xp, yp, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            proj_pts.append((xp, yp))

        proj_pts_arr = np.asarray(proj_pts, dtype=float)
        display_pts = self._pixels_to_points(fig, ax.transData.transform(proj_pts_arr))
        img_sizes = [self.stimulus_size_points(n, zoom, border_width) for n in names]
        placements = self.compute_label_placements(display_pts, img_sizes, names,
                                                   label_size=label_size)

        for name, (x, y, z), (xp, yp), plc in zip(names, coords, proj_pts, placements):
            img = self.create_stimulus_image(name, zoom=zoom, border_width=border_width)
            if img is not None:
                ab = AnnotationBbox(img, (xp, yp), xycoords="data", frameon=False,
                                    pad=0.0, box_alignment=(0.5, 0.5), zorder=10)
                ax.add_artist(ab)
            self.add_stimulus_label(ax, xp, yp, name, plc, label_size=label_size)

    @staticmethod
    def add_depth_guides(ax, coords) -> None:
        """Draw subtle dashed vertical lines from each point to the floor."""
        z_min, _ = ax.get_zlim()
        for x, y, z in coords:
            ax.plot([x, x], [y, y], [z_min, z],
                    linestyle="--", linewidth=0.8, color="black", alpha=0.42, zorder=1)


# ──────────────────────────────────────────────────────────────────────
# Statistical analysis
# ──────────────────────────────────────────────────────────────────────


class StatisticalAnalyzer:
    """Procrustes alignment, consistency metrics, and distance computations."""

    @staticmethod
    def generalized_procrustes(coords_list: list[np.ndarray],
                               n_iter: int = 10
                               ) -> tuple[np.ndarray | None, list[np.ndarray]]:
        """Perform Generalised Procrustes Analysis on a list of configurations.

        Returns:
            (mean_shape, aligned_configs) – the consensus shape and all aligned
            configurations after iterative re-alignment.
        """
        if len(coords_list) == 0:
            return None, []

        aligned = [coords_list[0].copy()]
        reference = coords_list[0].copy()

        for c in coords_list[1:]:
            _, aligned_c, _ = procrustes(reference, c)
            aligned.append(aligned_c)

        for _ in range(n_iter):
            mean_shape = np.mean(aligned, axis=0)
            aligned = [procrustes(mean_shape, c)[1] for c in aligned]

        return np.mean(aligned, axis=0), aligned

    @staticmethod
    def procrustes_rmse(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Compute normalised RMSE between two Procrustes-aligned configurations."""
        mtx1, mtx2, _ = procrustes(coords1, coords2)
        n, p = mtx1.shape
        return float(np.sqrt(((mtx1 - mtx2) ** 2).sum() / (n * p)))

    @staticmethod
    def leave_one_out_consensus(d_matrices: list[np.ndarray]) -> list[np.ndarray]:
        """Return one LOO consensus distance matrix per participant.

        Each consensus matrix is the mean of all *other* participants' matrices.
        """
        stack = np.asarray(d_matrices, dtype=float)
        if len(stack) < 2:
            return [d.copy() for d in stack]
        total = np.sum(stack, axis=0)
        return [(total - stack[i]) / (len(stack) - 1) for i in range(len(stack))]

    @staticmethod
    def knn_overlap(d1: np.ndarray, d2: np.ndarray, k: int) -> float:
        """Compute mean k-nearest-neighbour overlap between two distance matrices."""
        n = d1.shape[0]
        overlaps = []
        for i in range(n):
            nn1 = set(np.argsort(d1[i])[1:k + 1])
            nn2 = set(np.argsort(d2[i])[1:k + 1])
            overlaps.append(len(nn1 & nn2) / k)
        return float(np.mean(overlaps))

    @staticmethod
    def extract_spearman_rho(result) -> float:
        """Robustly extract a scalar Spearman rho from scipy's return value."""
        rho_val = getattr(cast(Any, result), "statistic", result[0])
        arr = np.asarray(rho_val)
        if arr.ndim == 0:
            rho = float(arr)
        elif arr.shape == (2, 2):
            rho = float(arr[0, 1])
        else:
            rho = float(arr.reshape(-1)[0])
        return 0.0 if np.isnan(rho) else rho

    @staticmethod
    def condensed_rdm(coords: np.ndarray) -> np.ndarray:
        """Return the condensed pairwise-distance vector (upper triangle, no diagonal)."""
        return pdist(np.asarray(coords), metric="euclidean")


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────


class PlotFactory:
    """Generates all publication-quality plots used by the analysis pipeline."""

    def __init__(self, config: AnalysisConfig, visualizer: StimulusVisualizer,
                 analyzer: StatisticalAnalyzer) -> None:
        self.cfg = config
        self.vis = visualizer
        self.stats = analyzer

    # ── helper ──

    @staticmethod
    def _coords_to_unit_scale(coords: np.ndarray) -> np.ndarray:
        """Map centered coordinates to [0, 1] without changing aspect ratios."""
        arr = np.asarray(coords, dtype=float)
        if arr.size == 0:
            return arr.copy()
        max_abs = np.nanmax(np.abs(arr))
        if not np.isfinite(max_abs) or max_abs <= 0:
            return np.full_like(arr, 0.5, dtype=float)
        return np.clip(arr / (2.0 * max_abs) + 0.5, 0.0, 1.0)

    @staticmethod
    def _normalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize two non-negative vectors by their shared maximum."""
        max_val = max(float(np.max(a)), float(np.max(b))) if len(a) and len(b) else 0.0
        if max_val <= 0:
            return a.copy(), b.copy()
        return a / max_val, b / max_val

    def _set_unit_ticks(self, ax) -> None:
        ticks = np.linspace(0, 1, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if hasattr(ax, "set_zticks"):
            ax.set_zticks(ticks)

    def _set_signed_unit_ticks(self, ax, avoid_2d_corner_overlap: bool = False,
                               avoid_3d_corner_overlap: bool = False) -> None:
        ticks = np.linspace(-1, 1, 5)
        tick_labels = [f"{tick:.1f}" for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if avoid_2d_corner_overlap and not hasattr(ax, "zaxis"):
            ax.set_xticklabels(["", *tick_labels[1:]])
            ax.set_yticklabels(tick_labels)
        if hasattr(ax, "set_zticks"):
            ax.set_zticks(ticks)
        if avoid_3d_corner_overlap and hasattr(ax, "zaxis"):
            # Hide the last x-tick label (1.0) that visually collides with the
            # last y-tick label (1.0) at the shared front-right corner.
            ax.set_xticklabels([*tick_labels[:-1], ""])
        ax.tick_params(axis="both", pad=8)

    def _style_colorbar(self, cbar, value_format: str | None = None,
                        tick_size: int | None = None,
                        label_size: int | None = None) -> None:
        tick_size = tick_size or self.cfg.scale_number_size
        label_size = label_size or self.cfg.font_size
        setattr(cbar.ax, "_analysis_tick_label_size", tick_size)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.ax.yaxis.label.set_fontsize(label_size)
        if value_format is not None:
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(value_format))

    def _add_rdm_colorbar(self, ax, im, ticks, value_format: str | None,
                          label: str = "", axis_font_size: int | None = None,
                          scale_font_size: int | None = None) -> Any:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.2%", pad=0.35)
        cbar = ax.figure.colorbar(im, cax=cax, ticks=ticks)
        if label:
            cbar.set_label(label)
        self._style_colorbar(
            cbar,
            value_format,
            tick_size=scale_font_size or self.cfg.rdm_scale_font_size,
            label_size=axis_font_size or self.cfg.rdm_axis_font_size,
        )
        return cbar

    def _add_metric_legend(self, ax, label: str,
                           legend_font_size: int | None = None) -> None:
        handle = Line2D([], [], linestyle="none", label=label)
        ax.legend(handles=[handle], loc="upper right", frameon=True,
                  handlelength=0, handletextpad=0, borderpad=0.35)
        setattr(ax, "_analysis_legend_font_size",
                legend_font_size or self.cfg.rdm_legend_font_size)

    def _style_figure_fonts(self, fig) -> None:
        for ax in fig.axes:
            ax.set_title("")
            ax.title.set_fontsize(self.cfg.title_font_size)
            ax.xaxis.label.set_fontsize(self.cfg.font_size)
            ax.yaxis.label.set_fontsize(self.cfg.font_size)
            ax.xaxis.labelpad = max(ax.xaxis.labelpad, 8)
            ax.yaxis.labelpad = max(ax.yaxis.labelpad, 8)
            tick_size = getattr(ax, "_analysis_tick_label_size",
                                self.cfg.scale_number_size)
            ax.tick_params(axis="both", labelsize=tick_size, pad=8)
            if hasattr(ax, "zaxis"):
                ax.zaxis.label.set_fontsize(self.cfg.font_size)
                ax.zaxis.labelpad = max(ax.zaxis.labelpad, 8)
                ax.zaxis.set_tick_params(labelsize=tick_size, pad=6)
            legend = ax.get_legend()
            if legend is not None:
                legend_size = getattr(ax, "_analysis_legend_font_size",
                                      max(7, self.cfg.font_size - 3))
                for text in legend.get_texts():
                    text.set_fontsize(legend_size)

    def _save(self, fig, path: Path, dpi: int = 200, **kwargs) -> None:
        """Save a figure and close it."""
        self._style_figure_fonts(fig)
        plt.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        plt.close(fig)

    # ── per-participant arrangements ──

    def participant_arrangement_2d(self, names, coords, label: str,
                                   out_path: Path) -> None:
        """2D scatter of stimuli as placed by one participant."""
        fig, ax = plt.subplots(figsize=(8, 8))
        x, y = coords[:, 0], coords[:, 1]
        ax.scatter(x, y, alpha=0.0)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        setattr(ax, "_analysis_tick_label_size", self.cfg.arrangement_axis_font_size)
        self._set_signed_unit_ticks(ax, avoid_2d_corner_overlap=True)

        self.vis.add_stimuli_2d(ax, names, x, y, zoom=0.40,
                                border_width=8,
                                label_size=self.cfg.stimulus_number_size)
        ax.set(xlabel="X", ylabel="Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        self._save(fig, out_path)

    def participant_arrangement_3d(self, names, coords, label: str,
                                   out_path: Path) -> None:
        """3D scatter of stimuli as placed by one participant."""
        fig = plt.figure(figsize=(11, 10))
        ax = fig.add_subplot(111, projection="3d")
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        ax.scatter(x, y, z, s=0, alpha=0)  # type: ignore
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        setattr(ax, "_analysis_tick_label_size", self.cfg.arrangement_axis_font_size)
        self._set_signed_unit_ticks(ax, avoid_3d_corner_overlap=True)
        ax.set_box_aspect((1, 1, 1))
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20

        self.vis.add_depth_guides(ax, coords)
        self.vis.add_projected_stimuli_3d(
            ax, names, coords, zoom=0.40, border_width=8,
            label_size=self.cfg.stimulus_number_size)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        plt.tight_layout()
        self._save(fig, out_path)

    # ── Procrustes mean shape ──

    def procrustes_arrangement_2d(self, aligned, mean_shape, participant_names,
                                  stimulus_names, out_path: Path) -> None:
        """2D scatter of the Procrustes mean shape with stimulus images."""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], alpha=0.0)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        self._set_signed_unit_ticks(ax, avoid_2d_corner_overlap=True)

        self.vis.add_stimuli_2d(ax, stimulus_names, mean_shape[:, 0],
                                mean_shape[:, 1], zoom=0.32,
                                border_width=8,
                                label_size=self.cfg.stimulus_number_size)
        ax.set(xlabel="X", ylabel="Y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        self._save(fig, out_path)

    def procrustes_arrangement_3d(self, aligned, mean_shape, participant_names,
                                  stimulus_names, out_path: Path) -> None:
        """3D scatter of the Procrustes mean shape with stimulus images."""
        fig = plt.figure(figsize=(12, 11))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], mean_shape[:, 2], #type: ignore
                   s=0, alpha=0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        self._set_signed_unit_ticks(ax, avoid_3d_corner_overlap=True)
        ax.set_box_aspect((1, 1, 1))
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20

        self.vis.add_depth_guides(ax, mean_shape)
        self.vis.add_projected_stimuli_3d(
            ax, stimulus_names, mean_shape, zoom=0.40, border_width=8,
            label_size=self.cfg.stimulus_number_size)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        plt.tight_layout()
        self._save(fig, out_path)

    # ── dissimilarity matrix (per-participant) ──

    def dissimilarity_matrix(self, names, d_matrix: np.ndarray,
                             csv_name: str, out_dir: Path) -> None:
        """Heatmap of a single participant's pairwise Euclidean distances."""
        n = len(names)
        fig, ax = plt.subplots(figsize=(1.2 * n, 1.2 * n))
        border_width = 6

        mask = np.triu(np.ones_like(d_matrix, dtype=bool), k=0)
        masked = np.ma.array(d_matrix, mask=mask)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white", alpha=0)

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        self._add_rdm_colorbar(ax, im, ticks=[0.0, 0.5, 1.0],
                               value_format="%.1f")
        fig.canvas.draw()

        # Stimulus-image axis ticks.
        origin = np.asarray(ax.transData.transform((0, 0)), dtype=float)
        step_x = np.asarray(ax.transData.transform((1, 0)), dtype=float)
        step_y = np.asarray(ax.transData.transform((0, 1)), dtype=float)
        cell_w = abs(step_x[0] - origin[0])
        cell_h = abs(step_y[1] - origin[1])
        target_px = 0.30 * min(cell_w, cell_h)
        x_trans = ax.get_xaxis_transform()
        y_trans = ax.get_yaxis_transform()

        for i, name in enumerate(names):
            sw, sh = self.vis.stimulus_size_pixels(name, border_width)
            z = min(target_px / sw, target_px / sh)
            img_x = self.vis.create_stimulus_image(name, zoom=z,
                                                   border_width=border_width)
            img_y = self.vis.create_stimulus_image(name, zoom=z,
                                                   border_width=border_width)
            if img_x is None:
                continue

            lbl = self.vis.stimulus_label(name)

            ab_x = AnnotationBbox(img_x, (i, -0.015), xycoords=x_trans,
                                  frameon=False, box_alignment=(0.5, 1.0),
                                  annotation_clip=False)
            ax.add_artist(ab_x)
            ax.text(i, -0.1, lbl, transform=x_trans, ha="center", va="top",
                    fontsize=self.cfg.rdm_axis_font_size, fontweight="bold",
                    color="black", clip_on=False)

            if img_y is not None:
                ab_y = AnnotationBbox(img_y, (-0.015, i), xycoords=y_trans,
                                      frameon=False, box_alignment=(1, 0.5),
                                      annotation_clip=False)
                ax.add_artist(ab_y)
                ax.text(-0.1, i, lbl, transform=y_trans, ha="right", va="center",
                        fontsize=self.cfg.rdm_axis_font_size, fontweight="bold",
                        color="black", clip_on=False)

        self._save(fig, out_dir / f"{csv_name}.png")

    # ── intersubject consistency bar chart ──

    def intersubject_consistency(self, disparities, participant_names,
                                 condition: str, out_path: Path) -> None:
        """Bar chart of per-participant Procrustes disparities from the mean."""
        fig, ax = plt.subplots(figsize=(max(8, len(disparities) * 0.8), 6))
        x = np.arange(len(disparities))
        ax.bar(x, disparities, color="steelblue", edgecolor="black")

        mean_d = np.mean(disparities)
        ax.axhline(mean_d, color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_d:.4f}")
        ax.set(xlabel="P", ylabel="Procrustes Disparity")
        ax.set_xticks(x)
        ax.set_xticklabels(participant_names, rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        self._save(fig, out_path)

    # ── Procrustes dissimilarity matrix ──

    def procrustes_dissimilarity_matrix(self, coords_list, participant_names,
                                        condition: str, out_path: Path,
                                        normalize: bool = True) -> None:
        """Pairwise Procrustes RMSE matrix between all participants."""
        n = len(coords_list)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rmse = self.stats.procrustes_rmse(coords_list[i], coords_list[j])
                D[i, j] = D[j, i] = rmse

        off_diag = ~np.eye(n, dtype=bool)
        mean_d = np.mean(D[off_diag])
        max_d = np.max(D[off_diag]) if np.any(off_diag) else 0.0

        if normalize and max_d > 0:
            vals = D / max_d
            cb_label = "Normalized disparity"
            disp_max = 1.0
        else:
            vals = D.copy()
            cb_label = "Disparity"
            disp_max = np.max(vals[off_diag]) if np.any(off_diag) else 1.0
            if disp_max <= 0:
                disp_max = 1.0

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
        mask = np.triu(np.ones_like(D, dtype=bool), k=0)
        masked = np.ma.array(vals, mask=mask)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white", alpha=0)

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=disp_max)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(participant_names, rotation=45, ha="right")
        ax.set_yticklabels(participant_names)
        setattr(ax, "_analysis_tick_label_size", self.cfg.pro_rdm_axis_font_size)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        cbar_ticks = [0.0, 0.5, 1.0] if normalize else None
        cbar_label = "" if normalize else cb_label
        self._add_rdm_colorbar(
            ax, im, ticks=cbar_ticks, value_format="%.1f" if normalize else None,
            label=cbar_label,
            axis_font_size=self.cfg.pro_rdm_axis_font_size,
            scale_font_size=self.cfg.pro_rdm_scale_font_size,
        )
        self._add_metric_legend(
            ax, f"Mean disparity = {mean_d:.4f}",
            legend_font_size=self.cfg.pro_rdm_legend_font_size)
        plt.tight_layout()
        self._save(fig, out_path)

    # ── Spearman consistency heatmap ──

    def spearman_consistency_heatmap(self, coords_list, participant_names,
                                     condition: str, out_path: Path) -> None:
        """Spearman correlation heatmap based on pairwise distances."""
        n = len(coords_list)
        corr = np.ones((n, n))
        dist_vecs = [self.stats.condensed_rdm(c) for c in coords_list]

        for i in range(n):
            for j in range(i + 1, n):
                rho = self.stats.extract_spearman_rho(
                    spearmanr(dist_vecs[i], dist_vecs[j]))
                corr[i, j] = corr[j, i] = rho

        mean_rho = np.mean(corr[~np.eye(n, dtype=bool)])

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
        masked = np.ma.array(corr, mask=mask)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color="white", alpha=0)

        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(participant_names, rotation=45, ha="right")
        ax.set_yticklabels(participant_names)
        setattr(ax, "_analysis_tick_label_size", self.cfg.spr_rdm_axis_font_size)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        self._add_rdm_colorbar(
            ax, im, ticks=[-1.0, 0.0, 1.0], value_format="%.1f",
            axis_font_size=self.cfg.spr_rdm_axis_font_size,
            scale_font_size=self.cfg.spr_rdm_scale_font_size,
        )
        self._add_metric_legend(
            ax, f"Mean ρ = {mean_rho:.4f}",
            legend_font_size=self.cfg.spr_rdm_legend_font_size)
        plt.tight_layout()
        self._save(fig, out_path)

    # ── Shepard diagram (global) ──

    def shepard_global(self, coords_list, stimulus_names,
                       condition: str, out_path: Path) -> None:
        """Aggregate Shepard diagram using leave-one-out consensus."""
        d_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
        d_loo = self.stats.leave_one_out_consensus(d_mats)

        all_ref, all_emb = [], []
        for D, D_loo in zip(d_mats, d_loo):
            all_ref.extend(squareform(D_loo, checks=False))
            all_emb.extend(squareform(D, checks=False))

        ref = np.asarray(all_ref)
        emb = np.asarray(all_emb)
        plot_ref, plot_emb = self._normalize_pair(ref, emb)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(plot_ref, plot_emb, alpha=0.45, s=18)

        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Identity (y = x)")

        # Binned mean trend.
        bins = np.linspace(lo, hi, 25)
        centres, means = [], []
        for k in range(len(bins) - 1):
            m = (plot_ref >= bins[k]) & (plot_ref < bins[k + 1])
            if np.sum(m) > 0:
                centres.append((bins[k] + bins[k + 1]) / 2)
                means.append(np.mean(plot_emb[m]))

        # R² against fitted regression.
        r2 = 0.0
        if len(plot_ref) > 1:
            coeffs = np.polyfit(plot_ref, plot_emb, 1)
            pred = np.polyval(coeffs, plot_ref)
            ss_res = np.sum((plot_emb - pred) ** 2)
            ss_tot = np.sum((plot_emb - np.mean(plot_emb)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.plot(centres, means, linewidth=3,
                label=f"Mean agreement trend (R²={r2:.3f})")
        ax.set(xlabel="Normalized consensus distance",
               ylabel="Normalized participant distance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self._set_unit_ticks(ax)
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path, dpi=300)

    # ── kNN preservation curves ──

    def _compute_knn_scores(self, coords_list) -> tuple[list[int], list[float]]:
        """Compute mean kNN overlap scores for k = 1 … n-1."""
        d_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
        d_loo = self.stats.leave_one_out_consensus(d_mats)
        n = d_mats[0].shape[0]
        ks = list(range(1, n))
        scores = []
        for k in ks:
            vals = [self.stats.knn_overlap(D, Dc, k)
                    for D, Dc in zip(d_mats, d_loo)]
            scores.append(np.mean(vals))
        return ks, scores

    def knn_curve(self, coords_list, condition: str, out_path: Path) -> None:
        """Aggregate kNN preservation curve for one condition."""
        ks, scores = self._compute_knn_scores(coords_list)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks, scores, marker="o")
        ax.set(xlabel="k Nearest Neighbours", ylabel="Neighbour preservation")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path, dpi=300)

    def knn_combined(self, coords_2d, coords_3d, out_path: Path) -> None:
        """Combined 2D vs 3D kNN preservation curve."""
        ks_2d, sc_2d = self._compute_knn_scores(coords_2d)
        ks_3d, sc_3d = self._compute_knn_scores(coords_3d)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ks_2d, sc_2d, marker="o", label="2D")
        ax.plot(ks_3d, sc_3d, marker="s", label="3D")
        ax.set(xlabel="k Nearest Neighbours", ylabel="Neighbour preservation")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path, dpi=300)

    def knn_individual(self, coords_list, participant_names,
                       condition: str, out_path: Path) -> None:
        """Per-participant kNN curves overlaid on one plot."""
        d_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
        d_loo = self.stats.leave_one_out_consensus(d_mats)
        n = d_mats[0].shape[0]
        ks = list(range(1, n))

        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab10", len(coords_list))

        for idx, (D, Dc, name) in enumerate(zip(d_mats, d_loo, participant_names)):
            scores = [self.stats.knn_overlap(D, Dc, k) for k in ks]
            ax.plot(ks, scores, marker=".", markersize=4,
                    label=name, color=cmap(idx), alpha=0.8)

        ax.set(xlabel="k Nearest Neighbours", ylabel="Neighbour preservation")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=max(7, self.cfg.font_size - 3), loc="lower right")
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path, dpi=300)

    # ── axis variance (dimension usage) ──

    def axis_variance(self, coords_list, condition: str, out_path: Path,
                      participant_names: list[str] | None = None) -> None:
        """Bar plot of mean per-axis variance with individual participant dots."""
        vars_all = []
        for c in coords_list:
            v = np.var(c, axis=0)
            v = v / np.sum(v)
            vars_all.append(v)

        vars_all = np.asarray(vars_all)
        n_dims = vars_all.shape[1]
        dims = ["X", "Y", "Z"][:n_dims]
        mean_v = np.mean(vars_all, axis=0)

        if participant_names is None:
            participant_names = [f"P {i + 1}" for i in range(len(vars_all))]

        fig, ax = plt.subplots(figsize=(6, 5))
        x_pos = np.arange(n_dims)
        ax.bar(x_pos, mean_v, width=0.5, color="lightsteelblue",
               edgecolor="black", label="Mean", zorder=2)

        # Jittered participant dots.
        pt_x = np.tile(x_pos, (len(vars_all), 1)).astype(float)
        for d in range(n_dims):
            order = np.argsort(vars_all[:, d])
            offsets = np.linspace(-0.08, 0.08, len(vars_all))
            pt_x[order, d] += offsets

        cmap = plt.get_cmap("tab10", len(vars_all))
        for p, pname in enumerate(participant_names):
            ax.scatter(pt_x[p], vars_all[p], color=cmap(p), edgecolor="black",
                       s=55, zorder=5, alpha=0.9, label=pname)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(dims)
        ax.set(ylabel="Variance ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
        legend_font_size = max(7, self.cfg.font_size - 4)
        legend_cols = min(len(participant_names) + 1, 6)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=legend_cols,
            fontsize=legend_font_size,
            frameon=True,
            borderaxespad=0,
            borderpad=0.25,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.55,
            labelspacing=0.2,
            markerscale=0.8,
        )
        setattr(ax, "_analysis_legend_font_size", legend_font_size)
        plt.tight_layout()
        self._save(fig, out_path, dpi=300)

    # ── cross-dimensional RDM similarity ──

    def crossdimensional_rdm_similarity(self, embeddings_2d, embeddings_3d,
                                        out_path: Path) -> None:
        """Bar chart comparing each participant's 2D vs 3D RDM (Spearman ρ)."""
        by_2d = {e[0]: e for e in embeddings_2d}
        by_3d = {e[0]: e for e in embeddings_3d}
        common = sorted(set(by_2d) & set(by_3d))
        if not common:
            return

        labels, rhos = [], []
        for pid in common:
            e2, e3 = by_2d[pid], by_3d[pid]
            names_2d, coords_2d = e2[3], e2[4]
            names_3d, coords_3d = e3[3], e3[4]
            c3_map = dict(zip(names_3d, coords_3d))
            shared = [n for n in names_2d if n in c3_map]
            if len(shared) < 2:
                continue

            ord_2d = np.asarray([coords_2d[names_2d.index(n)] for n in shared])
            ord_3d = np.asarray([c3_map[n] for n in shared])
            rdm2 = self.stats.condensed_rdm(ord_2d)
            rdm3 = self.stats.condensed_rdm(ord_3d)
            rho = self.stats.extract_spearman_rho(spearmanr(rdm2, rdm3))
            labels.append(e2[2])
            rhos.append(rho)

        if not rhos:
            return

        fig, ax = plt.subplots(figsize=(max(6, len(rhos) * 1.2), 5))
        bars = ax.bar(np.arange(len(rhos)), rhos,
                      color="slateblue", edgecolor="black")

        for bar, rho in zip(bars, rhos):
            y_off = 0.03 if rho >= 0 else -0.05
            va = "bottom" if rho >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, rho + y_off,
                    f"{rho:.3f}", ha="center", va=va,
                    fontsize=self.cfg.matrix_number_size)

        ax.set_xticks(np.arange(len(rhos)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set(ylabel="Spearman ρ")
        ax.set_ylim(-1, 1)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path)

    # ── per-participant detail plots ──

    def shepard_single(self, v_ref, v_embed, label: str,
                       condition: str, out_path: Path) -> None:
        """Shepard diagram for a single participant against LOO consensus."""
        plot_ref, plot_embed = self._normalize_pair(v_ref, v_embed)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(plot_ref, plot_embed, alpha=0.5, s=20)

        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Identity")

        r2 = 0.0
        if len(plot_ref) > 1:
            coeffs = np.polyfit(plot_ref, plot_embed, 1)
            pred = np.polyval(coeffs, plot_ref)
            ss_res = np.sum((plot_embed - pred) ** 2)
            ss_tot = np.sum((plot_embed - np.mean(plot_embed)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        r2_handle = Line2D([], [], linestyle="none", label=f"R\u00b2 = {r2:.3f}")
        ax.set(xlabel="Normalized consensus distance",
               ylabel="Normalized participant distance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self._set_unit_ticks(ax)
        ax.legend(handles=[ax.get_lines()[0], r2_handle])
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path)

    def axis_variance_single(self, variance_ratios, label: str,
                             condition: str, out_path: Path) -> None:
        """Axis variance bar chart for a single participant."""
        n_dims = len(variance_ratios)
        dims = ["X", "Y", "Z"][:n_dims]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(dims, variance_ratios, color="steelblue", edgecolor="black")
        for i, val in enumerate(variance_ratios):
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center",
                    fontsize=self.cfg.matrix_number_size)

        ax.set(ylabel="Variance ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path)

    def rdm_scatter_single(self, coords_2d, coords_3d,
                           label: str, out_path: Path) -> None:
        """Scatter of 2D vs 3D pairwise distances for a single participant."""
        rdm_2d = pdist(coords_2d, metric="euclidean")
        rdm_3d = pdist(coords_3d, metric="euclidean")
        rho = self.stats.extract_spearman_rho(spearmanr(rdm_2d, rdm_3d))
        plot_2d, plot_3d = self._normalize_pair(rdm_2d, rdm_3d)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(plot_2d, plot_3d, alpha=0.5, s=20, color="slateblue")

        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Identity")

        rho_handle = Line2D([], [], linestyle="none",
                            label=f"Spearman \u03c1 = {rho:.3f}")
        ax.legend(handles=[ax.get_lines()[0], rho_handle])

        ax.set(xlabel="Normalized 2D pairwise distance",
               ylabel="Normalized 3D pairwise distance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self._set_unit_ticks(ax)
        ax.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        self._save(fig, out_path)


# ──────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────────────────────────────


# Embedding tuple: (participant_index, folder_name, display_label, names, coords)
Embedding = tuple[int, str, str, list[str], np.ndarray]


class AnalysisPipeline:
    """Orchestrates the full analysis workflow: load → analyse → plot."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.cfg = config or AnalysisConfig()
        self.vis = StimulusVisualizer(self.cfg)
        self.stats = StatisticalAnalyzer()
        self.plots = PlotFactory(self.cfg, self.vis, self.stats)

        self.embeddings_2d: list[Embedding] = []
        self.embeddings_3d: list[Embedding] = []

    # ── data loading ──

    def load_all_participants(self) -> None:
        """Scan final_results/ and load all 2D/3D embeddings."""
        self.embeddings_2d.clear()
        self.embeddings_3d.clear()

        participant_dirs = sorted(
            d for d in self.cfg.final_results_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        selected_participants = set(self.cfg.participants)

        for idx, p_dir in enumerate(participant_dirs, start=1):
            if selected_participants and idx not in selected_participants:
                continue
            label = (f"Participant {idx}" if self.cfg.anonymous
                     else p_dir.name)

            for cond in ("2d", "3d"):
                cond_dir = p_dir / cond
                if not cond_dir.exists():
                    continue
                for csv_file in sorted(cond_dir.glob("*.csv")):
                    names, coords = load_embedding(csv_file)
                    if cond == "2d":
                        coords = coords[:, :2]
                    if len(coords) < 2:
                        continue
                    emb: Embedding = (idx, p_dir.name, label, names, coords)
                    if cond == "2d":
                        self.embeddings_2d.append(emb)
                    else:
                        self.embeddings_3d.append(emb)

    # ── condition-level analysis ──

    def _collect_condition_steps(self, embeddings: list[Embedding],
                                  condition: str, out_dir: Path) -> list[tuple[str, Any]]:
        """Collect plot-generation steps for one condition (2d or 3d)."""
        if len(embeddings) < 2:
            return []

        coords_list = [e[4] for e in embeddings]
        stimulus_names = embeddings[0][3]
        participant_names = [
            f"P {e[0]}" if self.cfg.anonymous else e[1]
            for e in embeddings
        ]

        mean_shape = None
        aligned: list[np.ndarray] = []
        disparities: list[float] = []
        if self.cfg.selection.wants("procrustes"):
            mean_shape, aligned = self.stats.generalized_procrustes(coords_list)
            assert mean_shape is not None, "Need at least one embedding for Procrustes"
            disparities = [
                self.stats.procrustes_rmse(mean_shape, a) for a in aligned
            ]

        steps: list[Any] = [
            ("Intersubject consistency",
             lambda: self.plots.intersubject_consistency(
                 disparities, participant_names, condition,
                 out_dir / f"intersubject_consistency_{condition}.png"))
            if self.cfg.selection.wants("procrustes") else None,
            ("Procrustes dissimilarity matrix",
             lambda: self.plots.procrustes_dissimilarity_matrix(
                 coords_list, participant_names, condition,
                 out_dir / f"procrustes_dissimilarity_matrix_{condition}.png",
                 normalize=self.cfg.normalize_procrustes))
            if self.cfg.selection.wants("procrustes") else None,
            ("Spearman consistency heatmap",
             lambda: self.plots.spearman_consistency_heatmap(
                 coords_list, participant_names, condition,
                 out_dir / f"spearman_consistency_heatmap_{condition}.png"))
            if self.cfg.selection.wants("spearman") else None,
            ("Shepard diagram",
             lambda: self.plots.shepard_global(
                 coords_list, stimulus_names, condition,
                 out_dir / f"shepard_{condition}.png"))
            if self.cfg.selection.wants("shepard") else None,
            ("kNN preservation",
             lambda: self.plots.knn_curve(
                 coords_list, condition,
                 out_dir / f"knn_preservation_{condition}.png"))
            if self.cfg.selection.wants("knn") else None,
            ("Axis variance",
             lambda: self.plots.axis_variance(
                 coords_list, condition,
                 out_dir / f"axis_variance_{condition}.png",
                 participant_names))
            if self.cfg.selection.wants("axis_variance") else None,
            ("Procrustes arrangement",
             lambda: (
                 self.plots.procrustes_arrangement_2d(
                     aligned, mean_shape, participant_names, stimulus_names,
                     out_dir / f"procrustes_arrangement_{condition}.png")
                 if condition == "2d" else
                 self.plots.procrustes_arrangement_3d(
                     aligned, mean_shape, participant_names, stimulus_names,
                     out_dir / f"procrustes_arrangement_{condition}.png")))
            if self.cfg.selection.wants("procrustes") else None,
            ("Per-participant kNN",
             lambda: self.plots.knn_individual(
                 coords_list, participant_names, condition,
                 out_dir / f"knn_individual_{condition}.png"))
            if self.cfg.selection.wants("knn") else None,
        ]
        return [step for step in steps if step is not None]

    # ── detailed per-participant analysis ──

    def _collect_detailed_steps(self) -> list[tuple[str, Any]]:
        """Collect per-participant plot-generation steps."""
        selected = {e[0] for e in self.embeddings_2d} | {e[0] for e in self.embeddings_3d}
        want_dissimilarity = self.cfg.selection.wants("dissimilarity")
        want_arrangements = self.cfg.selection.wants("arrangements")
        want_shepard = self.cfg.selection.wants("shepard")
        want_axis_variance = self.cfg.selection.wants("axis_variance")
        want_rdm_similarity = self.cfg.selection.wants("rdm_similarity")

        if not any((
            want_dissimilarity,
            want_arrangements,
            want_shepard,
            want_axis_variance,
            want_rdm_similarity,
        )):
            return []

        all_steps: list[tuple[str, Any]] = []

        for cond, all_emb in [("2d", self.embeddings_2d),
                               ("3d", self.embeddings_3d)]:
            if len(all_emb) < 1:
                continue

            d_mats: list[np.ndarray] = []
            d_loo: list[np.ndarray] = []
            if want_dissimilarity or want_shepard:
                coords_list = [e[4] for e in all_emb]
                d_mats = [squareform(pdist(c, metric="euclidean"))
                          for c in coords_list]
            if want_shepard:
                d_loo = self.stats.leave_one_out_consensus(d_mats)

            for idx, emb in enumerate(all_emb):
                pid, pname, plabel, names, coords = emb
                if pid not in selected:
                    continue

                p_dir = (self.cfg.detailed_dir /
                         plabel.replace(" ", "_") / cond)
                p_dir.mkdir(parents=True, exist_ok=True)

                if want_dissimilarity:
                    def _make_dissim(names=names, coords=coords,
                                     cond=cond, p_dir=p_dir):
                        D = squareform(pdist(coords, metric="euclidean"))
                        D_norm = D / D.max() if D.max() > 0 else D.copy()
                        self.plots.dissimilarity_matrix(
                            names, D_norm, f"dissimilarity_{cond}", p_dir)
                    all_steps.append(
                        (f"Dissimilarity {plabel} ({cond.upper()})", _make_dissim))

                if want_arrangements:
                    def _make_arr(names=names, coords=coords, plabel=plabel,
                                  p_dir=p_dir, cond=cond):
                        if cond == "2d":
                            self.plots.participant_arrangement_2d(
                                names, coords, plabel,
                                p_dir / f"arrangement_{cond}.png")
                        else:
                            self.plots.participant_arrangement_3d(
                                names, coords, plabel,
                                p_dir / f"arrangement_{cond}.png")
                    all_steps.append(
                        (f"Arrangement {plabel} ({cond.upper()})", _make_arr))

                if want_shepard and d_mats and d_loo:
                    v_ref = squareform(d_loo[idx], checks=False)
                    v_emb = squareform(d_mats[idx], checks=False)
                    def _make_shep(v_ref=v_ref, v_emb=v_emb, plabel=plabel,
                                   cond=cond, p_dir=p_dir):
                        self.plots.shepard_single(
                            v_ref, v_emb, plabel, cond,
                            p_dir / f"shepard_{cond}.png")
                    all_steps.append(
                        (f"Shepard {plabel} ({cond.upper()})", _make_shep))

                if want_axis_variance:
                    def _make_axvar(coords=coords, plabel=plabel,
                                    cond=cond, p_dir=p_dir):
                        v = np.var(coords, axis=0)
                        v = v / np.sum(v)
                        self.plots.axis_variance_single(
                            v, plabel, cond,
                            p_dir / f"axis_variance_{cond}.png")
                    all_steps.append(
                        (f"Axis variance {plabel} ({cond.upper()})", _make_axvar))

        # Cross-dimensional RDM scatter per participant.
        if want_rdm_similarity:
            by_2d = {e[0]: e for e in self.embeddings_2d}
            by_3d = {e[0]: e for e in self.embeddings_3d}
            common = sorted(selected & set(by_2d) & set(by_3d))

            for pid in common:
                e2, e3 = by_2d[pid], by_3d[pid]
                plabel = e2[2]
                names_2d, coords_2d = e2[3], e2[4]
                names_3d, coords_3d = e3[3], e3[4]

                c3_map = dict(zip(names_3d, coords_3d))
                shared = [n for n in names_2d if n in c3_map]
                if len(shared) < 2:
                    continue

                ord_2d = np.asarray([coords_2d[names_2d.index(n)] for n in shared])
                ord_3d = np.asarray([c3_map[n] for n in shared])

                p_dir = self.cfg.detailed_dir / plabel.replace(" ", "_")
                def _make_rdm(ord_2d=ord_2d, ord_3d=ord_3d,
                               plabel=plabel, p_dir=p_dir):
                    self.plots.rdm_scatter_single(
                        ord_2d, ord_3d, plabel, p_dir / "rdm_scatter.png")
                all_steps.append((f"RDM scatter {plabel}", _make_rdm))

        return all_steps

    # ── main entry point ──

    def run(self) -> None:
        """Execute the complete analysis pipeline."""
        # Create output directories.
        for d in (self.cfg.analysis_dir, self.cfg.general_dir,
                  self.cfg.general_2d_dir, self.cfg.general_3d_dir,
                  self.cfg.detailed_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Phase 1: Load data (silent).
        self.load_all_participants()

        # Phase 2 + 3: Collect all plot steps.
        all_steps: list[tuple[str, Any]] = []
        all_steps += self._collect_condition_steps(
            self.embeddings_2d, "2d", self.cfg.general_2d_dir)
        all_steps += self._collect_condition_steps(
            self.embeddings_3d, "3d", self.cfg.general_3d_dir)

        if (self.cfg.selection.wants("knn") and
                len(self.embeddings_2d) >= 2 and len(self.embeddings_3d) >= 2):
            all_steps.append(("kNN 2D vs 3D", lambda: self.plots.knn_combined(
                [e[4] for e in self.embeddings_2d],
                [e[4] for e in self.embeddings_3d],
                self.cfg.general_dir / "knn_preservation_2d_vs_3d.png")))

        if (self.cfg.selection.wants("rdm_similarity") and
                self.embeddings_2d and self.embeddings_3d):
            all_steps.append(("RDM similarity 2D vs 3D",
                               lambda: self.plots.crossdimensional_rdm_similarity(
                                   self.embeddings_2d, self.embeddings_3d,
                                   self.cfg.general_dir / "rdm_similarity_2d_vs_3d.png")))

        all_steps += self._collect_detailed_steps()

        # Execute all steps with a single progress bar.
        progress = _progress(all_steps, desc="Generating plots",
                             unit="plot", dynamic_ncols=True)
        for name, fn in progress:
            _set_progress_label(progress, name)
            fn()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def _parse_participants(raw_value: str) -> list[int]:
    """Parse participant lists like '1,2,3' or '[1, 2, 3]'."""
    value = raw_value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Analysis for the 3D inverse MDS project. "
            "If no metric flags are provided, the full analysis suite is executed."
        )
    )
    parser.add_argument(
        "--participants",
        type=_parse_participants,
        default=[],
        help="Participant list, e.g. 1,2,3 or [1,2,3].",
    )
    parser.add_argument(
        "--normalize-procrustes",
        dest="normalize_procrustes",
        action="store_true",
        default=True,
        help="Normalize the Procrustes dissimilarity matrix to [0, 1].",
    )
    parser.add_argument(
        "--raw-procrustes",
        dest="normalize_procrustes",
        action="store_false",
        help="Use raw Procrustes dissimilarity values instead of [0, 1].",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Base font size for plot labels.",
    )
    parser.add_argument(
        "--title-font-size",
        type=int,
        default=None,
        help="Font size for plot titles.",
    )
    parser.add_argument(
        "--scale-number-size",
        type=int,
        default=None,
        help="Font size for ordinary axis scale numbers.",
    )
    parser.add_argument(
        "--stimulus-number-size",
        type=int,
        default=None,
        help="Font size for ordinary stimulus number labels.",
    )
    parser.add_argument(
        "--rdm-axis-font-size", "--rdm-stimulus-number-size",
        dest="rdm_axis_font_size",
        type=int,
        default=None,
        help="Axis/tick/stimulus font size for Spearman, Procrustes and dissimilarity RDM plots.",
    )
    parser.add_argument(
        "--rdm-scale-font-size", "--rdm-scale-number-size",
        dest="rdm_scale_font_size",
        type=int,
        default=None,
        help="Colorbar scale-number font size for Spearman, Procrustes and dissimilarity RDM plots.",
    )
    parser.add_argument(
        "--rdm-legend-font-size",
        type=int,
        default=None,
        help="Legend font size for Spearman and Procrustes RDM mean values.",
    )
    parser.add_argument(
        "--pro-rdm-axis-font-size",
        type=int,
        default=None,
        help="Axis/tick font size for Procrustes dissimilarity RDM plots.",
    )
    parser.add_argument(
        "--pro-rdm-scale-font-size",
        type=int,
        default=None,
        help="Colorbar scale-number font size for Procrustes dissimilarity RDM plots.",
    )
    parser.add_argument(
        "--pro-rdm-legend-font-size",
        type=int,
        default=None,
        help="Legend font size for Procrustes mean disparity.",
    )
    parser.add_argument(
        "--spr-rdm-axis-font-size",
        type=int,
        default=None,
        help="Axis/tick font size for Spearman consistency RDM plots.",
    )
    parser.add_argument(
        "--spr-rdm-scale-font-size",
        type=int,
        default=None,
        help="Colorbar scale-number font size for Spearman consistency RDM plots.",
    )
    parser.add_argument(
        "--spr-rdm-legend-font-size",
        type=int,
        default=None,
        help="Legend font size for Spearman mean rho.",
    )
    parser.add_argument(
        "--arrangement-axis-font-size",
        type=int,
        default=None,
        help="Axis/tick font size for per-participant arrangement plots (2D and 3D).",
    )
    parser.add_argument(
        "--matrix-number-size",
        type=int,
        default=None,
        help="Font size for numeric annotations outside RDM heatmaps.",
    )
    parser.add_argument(
        "--named-participants",
        action="store_true",
        help="Use folder names instead of anonymous participant labels.",
    )
    parser.add_argument(
        "--procrustes", "--procruster",
        dest="procrustes",
        action="store_true",
        help="Run Procrustes-based outputs.",
    )
    parser.add_argument(
        "--spearman", "--spreamean",
        dest="spearman",
        action="store_true",
        help="Generate Spearman consistency plots.",
    )
    parser.add_argument(
        "--shepard", "--shaprd",
        dest="shepard",
        action="store_true",
        help="Generate global and individual Shepard diagrams.",
    )
    parser.add_argument(
        "--knn",
        action="store_true",
        help="Generate kNN preservation plots.",
    )
    parser.add_argument(
        "--axis-variance",
        dest="axis_variance",
        action="store_true",
        help="Generate axis-variance plots.",
    )
    parser.add_argument(
        "--rdm-similarity",
        dest="rdm_similarity",
        action="store_true",
        help="Generate 2D-vs-3D RDM comparisons.",
    )
    parser.add_argument(
        "--arrangements",
        action="store_true",
        help="Generate participant arrangement plots.",
    )
    parser.add_argument(
        "--dissimilarity",
        action="store_true",
        help="Generate participant dissimilarity matrices.",
    )
    return parser.parse_args(argv)


def _selection_from_args(args: argparse.Namespace) -> AnalysisSelection:
    """Build the selection object from parsed command line flags."""
    metrics = {
        name for name in (
            "procrustes",
            "spearman",
            "shepard",
            "knn",
            "axis_variance",
            "rdm_similarity",
            "arrangements",
            "dissimilarity",
        )
        if getattr(args, name)
    }
    return AnalysisSelection(frozenset(metrics))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis pipeline."""
    args = parse_args(argv)

    def cfg_value(name: str):
        value = getattr(args, name)
        return value if value is not None else getattr(AnalysisConfig, name)

    config = AnalysisConfig(
        anonymous=not args.named_participants,
        participants=args.participants,
        normalize_procrustes=args.normalize_procrustes,
        font_size=cfg_value("font_size"),
        title_font_size=cfg_value("title_font_size"),
        scale_number_size=cfg_value("scale_number_size"),
        stimulus_number_size=cfg_value("stimulus_number_size"),
        rdm_axis_font_size=cfg_value("rdm_axis_font_size"),
        rdm_scale_font_size=cfg_value("rdm_scale_font_size"),
        rdm_legend_font_size=cfg_value("rdm_legend_font_size"),
        pro_rdm_axis_font_size=cfg_value("pro_rdm_axis_font_size"),
        pro_rdm_scale_font_size=cfg_value("pro_rdm_scale_font_size"),
        pro_rdm_legend_font_size=cfg_value("pro_rdm_legend_font_size"),
        spr_rdm_axis_font_size=cfg_value("spr_rdm_axis_font_size"),
        spr_rdm_scale_font_size=cfg_value("spr_rdm_scale_font_size"),
        spr_rdm_legend_font_size=cfg_value("spr_rdm_legend_font_size"),
        arrangement_axis_font_size=cfg_value("arrangement_axis_font_size"),
        matrix_number_size=cfg_value("matrix_number_size"),
        selection=_selection_from_args(args),
    )
    pipeline = AnalysisPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
