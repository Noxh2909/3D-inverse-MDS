import csv
from typing import Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D, proj3d
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from PIL import Image, ImageOps

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _progress(iterable, **kwargs):
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


def _set_progress_label(progress, label):
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(label)


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FINAL_RESULTS_DIR = BASE_DIR / "final_results"
PICTURES_DIR = BASE_DIR / "pictures"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

GENERAL_DIR = ANALYSIS_DIR / "general"
GENERAL_2D_DIR = GENERAL_DIR / "2d"
GENERAL_3D_DIR = GENERAL_DIR / "3d"
DETAILED_DIR = ANALYSIS_DIR / "detailed"

# Set to True for anonymous participant labels (Participant 1, 2, 3...)
# Set to False to use actual folder names
ANONYMOUS = True

STIMULUS_BORDER_COLORS = {
    1: "#6929c4",
    2: "#1192e8",
    3: "#005d5d",
    4: "#9f1853",
    5: "#fa4d56",
    6: "#570408",
    7: "#198038",
    8: "#002d9c",
    9: "#ee538b",
    10: "#b28600",
    11: "#009d9a",
    12: "#012749",
}

# Restrict per-participant detailed analysis to specific participant indices.
# Example: [4, 7] will generate analysis/detailed/ only for Participant 4 and 7.
# Leave empty [] to generate detailed analysis for ALL participants.
PARTICIPANTS = []

# Set to True to scale Procrustes dissimilarity matrices to [0, 1].
# Set to False to show raw disparities.
NORMALIZE_PROCRUSTES_DISSIMILARITY = False


# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------


def load_embedding(csv_path):
    names = []
    coords = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("mask_png"):
                continue
            if row[0].startswith("Participant"):
                continue
            if len(row) != 4:
                continue

            names.append(row[0])
            coords.append([float(row[1]), float(row[2]), float(row[3])])

    return names, np.asarray(coords)


# -------------------------------------------------
# IMAGE HELPERS
# -------------------------------------------------


def _stimulus_index(name):
    stem = Path(name).stem.replace("Stimuli_", "")
    try:
        return int(stem) + 1
    except ValueError:
        return None


def _stimulus_label(name):
    """Convert e.g. 'Stimuli_00.png' -> '01' (1-based)."""
    stimulus_index = _stimulus_index(name)
    if stimulus_index is None:
        return Path(name).stem
    return f"{stimulus_index:02d}"


def _add_stimulus_border(image, name, border_width=10):
    stimulus_index = _stimulus_index(name)
    if stimulus_index is None:
        border_color = "#000000"
    else:
        border_color = STIMULUS_BORDER_COLORS.get(stimulus_index, "#000000")
    return ImageOps.expand(image, border=border_width, fill=border_color)


def load_image(name, zoom=0.35, bordered=False, border_width=5):
    path = PICTURES_DIR / name
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    if bordered:
        img = _add_stimulus_border(img, name, border_width=border_width)
    arr = np.asarray(img)
    return OffsetImage(arr, zoom=zoom)


def create_stimulus_image(name, zoom=0.35, border_width=5):
    return load_image(name, zoom=zoom, bordered=True, border_width=border_width)


def _stimulus_size_pixels(name, border_width):
    path = PICTURES_DIR / name
    if not path.exists():
        return 42 + 2 * border_width, 42 + 2 * border_width

    with Image.open(path) as image:
        width, height = image.size

    return width + 2 * border_width, height + 2 * border_width


def _stimulus_size_points(name, zoom, border_width):
    width, height = _stimulus_size_pixels(name, border_width)
    return width * zoom, height * zoom


def _label_box_size_points(name, label_size):
    label = _stimulus_label(name)
    width = max(18.0, len(label) * label_size * 0.72 + 6.0)
    height = label_size + 6.0
    return width, height


def _pixels_to_points(fig, values):
    return np.asarray(values, dtype=float) * (72.0 / fig.dpi)


def _rect_from_center(center_x, center_y, width, height):
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _rect_from_anchor(anchor_x, anchor_y, width, height, ha, va):
    if ha == "center":
        x_min = anchor_x - width / 2.0
        x_max = anchor_x + width / 2.0
    elif ha == "left":
        x_min = anchor_x
        x_max = anchor_x + width
    else:
        x_min = anchor_x - width
        x_max = anchor_x

    if va == "center":
        y_min = anchor_y - height / 2.0
        y_max = anchor_y + height / 2.0
    elif va == "bottom":
        y_min = anchor_y
        y_max = anchor_y + height
    else:
        y_min = anchor_y - height
        y_max = anchor_y

    return (x_min, y_min, x_max, y_max)


def _rect_intersection_area(rect1, rect2, padding=0.0):
    x_min = max(rect1[0] - padding, rect2[0] - padding)
    y_min = max(rect1[1] - padding, rect2[1] - padding)
    x_max = min(rect1[2] + padding, rect2[2] + padding)
    y_max = min(rect1[3] + padding, rect2[3] + padding)

    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)


def _label_direction_offsets(image_width, image_height, gap=6.0):
    half_width = image_width / 2.0
    half_height = image_height / 2.0
    return [
        ((0.0, -(half_height + gap)), "center", "top"),
        ((-(half_width + gap), 0.0), "right", "center"),
        (((half_width + gap), 0.0), "left", "center"),
        ((0.0, half_height + gap), "center", "bottom"),
    ]


def _candidate_label_rect(point, label_width, label_height, candidate):
    offset, ha, va = candidate
    anchor_x = point[0] + offset[0]
    anchor_y = point[1] + offset[1]
    return _rect_from_anchor(anchor_x, anchor_y, label_width, label_height, ha, va)


def _candidate_penalty(candidate_rect, point_index, image_rects, placed_label_rects):
    penalty = 0.0

    for other_index, image_rect in enumerate(image_rects):
        if other_index == point_index:
            continue
        penalty += _rect_intersection_area(candidate_rect, image_rect, padding=2.0)

    for label_rect in placed_label_rects:
        penalty += 1.5 * _rect_intersection_area(
            candidate_rect, label_rect, padding=2.0
        )

    return penalty


def compute_label_placements(points, image_sizes, names, label_size=11):
    """Place labels below by default and move them left, right, or above on collision."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []
    if len(points) == 1:
        image_width, image_height = image_sizes[0]
        offset, ha, va = _label_direction_offsets(image_width, image_height)[0]
        return [(offset, ha, va)]

    image_rects = [
        _rect_from_center(point[0], point[1], image_width, image_height)
        for point, (image_width, image_height) in zip(points, image_sizes)
    ]

    nearest_distances = []
    for index, point in enumerate(points):
        others = np.delete(points, index, axis=0)
        distances = np.linalg.norm(others - point, axis=1)
        nearest_distances.append(np.min(distances) if len(distances) else np.inf)

    placements = [None] * len(points)
    placed_label_rects = []

    for index in np.argsort(nearest_distances):
        point = points[index]
        image_width, image_height = image_sizes[index]
        label_width, label_height = _label_box_size_points(names[index], label_size)
        candidates = _label_direction_offsets(image_width, image_height)
        best_candidate = None
        best_penalty = None

        for candidate in candidates:
            candidate_rect = _candidate_label_rect(
                point, label_width, label_height, candidate
            )
            penalty = _candidate_penalty(
                candidate_rect, index, image_rects, placed_label_rects
            )

            if penalty == 0.0:
                best_candidate = candidate
                best_penalty = penalty
                break

            if best_penalty is None or penalty < best_penalty:
                best_candidate = candidate
                best_penalty = penalty

        placements[index] = best_candidate
        placed_label_rects.append(
            _candidate_label_rect(point, label_width, label_height, best_candidate)
        )

    return placements


def add_stimulus_label(ax, x_coord, y_coord, name, placement, label_size=11):
    offset, ha, va = placement
    ax.annotate(
        _stimulus_label(name),
        (x_coord, y_coord),
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=label_size,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.12},
        zorder=12,
        clip_on=False,
    )


def add_projected_stimuli_3d(
    ax, names, coords, zoom=0.12, border_width=5, label_size=11
):
    """Project stimulus thumbnails onto a 3D axes and place labels beside them."""
    fig = ax.figure
    fig.canvas.draw()

    projected_points = []
    for x_coord, y_coord, z_coord in coords:
        x_proj, y_proj, _ = proj3d.proj_transform(
            x_coord, y_coord, z_coord, ax.get_proj()
        )
        projected_points.append((x_proj, y_proj))

    projected_points = np.asarray(projected_points, dtype=float)
    display_points = ax.transData.transform(projected_points)
    display_points = _pixels_to_points(fig, display_points)
    image_sizes = [_stimulus_size_points(name, zoom, border_width) for name in names]
    label_placements = compute_label_placements(
        display_points,
        image_sizes,
        names,
        label_size=label_size,
    )

    for name, (x_coord, y_coord, z_coord), (x_proj, y_proj), placement in zip(
        names,
        coords,
        projected_points,
        label_placements,
    ):
        image = create_stimulus_image(name, zoom=zoom, border_width=border_width)
        if image is not None:
            annotation = AnnotationBbox(
                image,
                (x_proj, y_proj),
                xycoords="data",
                frameon=False,
                pad=0.0,
                box_alignment=(0.5, 0.5),
                zorder=10,
            )
            ax.add_artist(annotation)

        add_stimulus_label(ax, x_proj, y_proj, name, placement, label_size=label_size)


def add_depth_guides_3d(ax, coords):
    """Add one subtle dashed vertical line from each stimulus down to the floor."""
    z_min, _ = ax.get_zlim()

    for x_coord, y_coord, z_coord in coords:
        ax.plot(
            [x_coord, x_coord],
            [y_coord, y_coord],
            [z_min, z_coord],
            linestyle="--",
            linewidth=0.8,
            color="black",
            alpha=0.42,
            zorder=1,
        )


def remap_coords_for_3d_axes(coords):
    """Keep original 3D axis order."""
    return coords[:, [0, 1, 2]]


def add_stimuli_2d(
    ax, names, x_coords, y_coords, zoom=0.20, border_width=5, label_size=11
):
    fig = ax.figure
    fig.canvas.draw()

    data_points = np.column_stack((x_coords, y_coords))
    display_points = ax.transData.transform(data_points)
    display_points = _pixels_to_points(fig, display_points)
    image_sizes = [_stimulus_size_points(name, zoom, border_width) for name in names]
    label_placements = compute_label_placements(
        display_points,
        image_sizes,
        names,
        label_size=label_size,
    )

    for x_coord, y_coord, name, placement in zip(
        x_coords, y_coords, names, label_placements
    ):
        image = create_stimulus_image(name, zoom=zoom, border_width=border_width)
        if image is not None:
            annotation = AnnotationBbox(
                image,
                (x_coord, y_coord),
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0.0,
                zorder=5,
            )
            ax.add_artist(annotation)

        add_stimulus_label(ax, x_coord, y_coord, name, placement, label_size=label_size)


# -------------------------------------------------
# PER-PARTICIPANT STIMULUS ARRANGEMENT (2D scatter / 3D cube)
# -------------------------------------------------


def plot_participant_arrangement_2d(names, coords, participant_label, out_path):
    """2D scatter of stimuli as placed by one participant, with stimulus images."""
    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = coords[:, 0], coords[:, 1]
    ax.scatter(x, y, alpha=0.0)  # invisible dots, images replace them

    pad = 0.22
    rx = x.max() - x.min() if x.max() != x.min() else 1
    ry = y.max() - y.min() if y.max() != y.min() else 1
    ax.set_xlim(x.min() - pad * rx, x.max() + pad * rx)
    ax.set_ylim(y.min() - pad * ry, y.max() + pad * ry)
    add_stimuli_2d(ax, names, x, y, zoom=0.40, border_width=8, label_size=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Stimulus Arrangement 2D – {participant_label}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_participant_arrangement_3d(names, coords, participant_label, out_path):
    """3D cube scatter of stimuli as placed by one participant, with images."""
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")
    remapped_coords = remap_coords_for_3d_axes(coords)
    x, y, z = remapped_coords[:, 0], remapped_coords[:, 1], remapped_coords[:, 2]

    ax.scatter(x, y, z, s=0, alpha=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_zlim(-1, 1)
    add_depth_guides_3d(ax, remapped_coords)
    add_projected_stimuli_3d(
        ax, names, remapped_coords, zoom=0.40, border_width=8, label_size=12
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Stimulus Arrangement 3D – {participant_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------------------------------
# OVERALL PROCRUSTES ARRANGEMENT (all aligned + mean)
# -------------------------------------------------


def plot_procrustes_arrangement_2d(
    aligned_list, mean_shape, participant_names, stimulus_names, out_path
):
    """2D scatter showing only the Procrustes mean shape with stimulus images."""
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], alpha=0.0)

    pad = 0.22
    rx = mean_shape[:, 0].max() - mean_shape[:, 0].min() or 1
    ry = mean_shape[:, 1].max() - mean_shape[:, 1].min() or 1
    ax.set_xlim(mean_shape[:, 0].min() - pad * rx, mean_shape[:, 0].max() + pad * rx)
    ax.set_ylim(mean_shape[:, 1].min() - pad * ry, mean_shape[:, 1].max() + pad * ry)
    add_stimuli_2d(
        ax,
        stimulus_names,
        mean_shape[:, 0],
        mean_shape[:, 1],
        zoom=0.32,
        border_width=8,
        label_size=12,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Procrustes Mean Shape (2D)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_procrustes_arrangement_3d(
    aligned_list, mean_shape, participant_names, stimulus_names, out_path
):
    """3D cube showing only the Procrustes mean shape with stimulus images."""
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection="3d")
    remapped_mean_shape = remap_coords_for_3d_axes(mean_shape)

    ax.scatter(
        remapped_mean_shape[:, 0],
        remapped_mean_shape[:, 1],
        remapped_mean_shape[:, 2],
        s=0,
        alpha=0,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_zlim(-1, 1)
    add_depth_guides_3d(ax, remapped_mean_shape)
    add_projected_stimuli_3d(
        ax,
        stimulus_names,
        remapped_mean_shape,
        zoom=0.40,
        border_width=8,
        label_size=12,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Procrustes Mean Shape (3D)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------------------------------
# PLOT MATRIX WITH IMAGES + VALUES
# -------------------------------------------------


def plot_dissimilarity_matrix(names, D, csv_name, out_dir):
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.2 * n, 1.2 * n))
    axis_border_width = 6

    lower_triangle_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    masked_D = np.ma.array(D, mask=lower_triangle_mask)
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0)

    im = ax.imshow(masked_D, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Dissimilarity Matrix (Euclidean Distance)")
    fig.canvas.draw()

    origin = np.asarray(ax.transData.transform((0, 0)), dtype=float)
    step_x = np.asarray(ax.transData.transform((1, 0)), dtype=float)
    step_y = np.asarray(ax.transData.transform((0, 1)), dtype=float)
    cell_width_px = abs(step_x[0] - origin[0])
    cell_height_px = abs(step_y[1] - origin[1])
    target_cell_px = 0.30 * min(cell_width_px, cell_height_px)
    x_axis_transform = ax.get_xaxis_transform()
    y_axis_transform = ax.get_yaxis_transform()

    # --- values inside cells ---
    for i in range(n):
        for j in range(i + 1):
            ax.text(
                j,
                i,
                f"{D[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if D[i, j] > D.max() * 0.5 else "black",
                fontsize=10,
            )

    # --- image ticks ---
    for i, name in enumerate(names):
        stim_width_px, stim_height_px = _stimulus_size_pixels(name, axis_border_width)
        axis_image_zoom = min(
            target_cell_px / stim_width_px, target_cell_px / stim_height_px
        )
        img_x = create_stimulus_image(
            name, zoom=axis_image_zoom, border_width=axis_border_width
        )
        img_y = create_stimulus_image(
            name, zoom=axis_image_zoom, border_width=axis_border_width
        )
        if img_x is None:
            continue

        stimulus_label = _stimulus_label(name)

        # X axis
        ab_x = AnnotationBbox(
            img_x,
            (i, -0.015),
            xycoords=x_axis_transform,
            frameon=False,
            box_alignment=(0.5, 1.0),
            annotation_clip=False,
        )
        ax.add_artist(ab_x)
        ax.text(
            i,
            -0.1,
            stimulus_label,
            transform=x_axis_transform,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        # Y axis
        if img_y is not None:
            ab_y = AnnotationBbox(
                img_y,
                (-0.015, i),
                xycoords=y_axis_transform,
                frameon=False,
                box_alignment=(1, 0.5),
                annotation_clip=False,
            )
            ax.add_artist(ab_y)
            ax.text(
                -0.1,
                i,
                stimulus_label,
                transform=y_axis_transform,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
                clip_on=False,
            )

    out = out_dir / f"{csv_name}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------


def analyze_csv(csv_path, out_dir):
    names, coords = load_embedding(csv_path)
    if len(coords) < 2:
        print("Not enough points:", csv_path)
        return

    D = squareform(pdist(coords, metric="euclidean"))
    if D.max() > 0:
        D = D / D.max()
    csv_name = csv_path.stem
    plot_dissimilarity_matrix(names, D, csv_name, out_dir)


# -------------------------------------------------
# PROCRUSTES ANALYSIS
# -------------------------------------------------


def generalized_procrustes_analysis(coords_list):
    """
    Perform Generalized Procrustes Analysis (GPA) on a list of coordinate arrays.
    Returns the mean shape and all aligned configurations.
    """
    if len(coords_list) == 0:
        return None, []

    # Start with the first configuration as reference
    aligned = [coords_list[0].copy()]
    reference = coords_list[0].copy()

    # Align all other configurations to the reference
    for coords in coords_list[1:]:
        _, aligned_coords, _ = procrustes(reference, coords)
        aligned.append(aligned_coords)

    # Iterative alignment to mean
    for _ in range(10):  # 10 iterations usually sufficient
        # Compute mean shape
        mean_shape = np.mean(aligned, axis=0)

        # Re-align all to mean
        new_aligned = []
        for coords in aligned:
            _, aligned_coords, _ = procrustes(mean_shape, coords)
            new_aligned.append(aligned_coords)
        aligned = new_aligned

    # Final mean shape
    mean_shape = np.mean(aligned, axis=0)
    return mean_shape, aligned


def compute_procrustes_distance(coords1, coords2):
    """Compute normalised RMSE between two Procrustes-aligned configurations."""
    mtx1, mtx2, _ = procrustes(coords1, coords2)
    n, p = mtx1.shape
    rmse = np.sqrt(((mtx1 - mtx2) ** 2).sum() / (n * p))
    return rmse


def plot_intersubject_consistency(disparities, participant_names, condition, out_path):
    """Plot bar chart of Procrustes disparities for each participant."""
    fig, ax = plt.subplots(figsize=(max(8, len(disparities) * 0.8), 6))

    x = np.arange(len(disparities))
    bars = ax.bar(x, disparities, color="steelblue", edgecolor="black")

    # Add mean line
    mean_disp = np.mean(disparities)
    ax.axhline(
        mean_disp,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_disp:.4f}",
    )

    ax.set_xlabel("Participant")
    ax.set_ylabel("Procrustes Disparity")
    ax.set_title(
        f"Intersubject Consistency - {condition.upper()}\n(lower = more consistent)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(participant_names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------------------------------
# SHEPARD DIAGRAM (GLOBAL)
# -------------------------------------------------


def _leave_one_out_consensus_matrices(D_mats):
    """Return one consensus distance matrix per participant excluding their own data."""
    D_stack = np.asarray(D_mats, dtype=float)

    if len(D_stack) < 2:
        return [D.copy() for D in D_stack]

    total = np.sum(D_stack, axis=0)
    return [(total - D_stack[i]) / (len(D_stack) - 1) for i in range(len(D_stack))]


def plot_shepard_global(coords_list, stimulus_names, condition, out_path):
    """Aggregate Shepard diagram across participants using leave-one-out consensus."""

    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus_mats = _leave_one_out_consensus_matrices(D_mats)

    all_ref = []
    all_embed = []

    for D, D_consensus in zip(D_mats, D_consensus_mats):
        all_ref.extend(squareform(D_consensus, checks=False))
        all_embed.extend(squareform(D, checks=False))

    v_ref_rep = np.asarray(all_ref)
    all_embed = np.asarray(all_embed)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(v_ref_rep, all_embed, alpha=0.45, s=18)

    # identity line (perfect reconstruction)
    min_v = min(v_ref_rep.min(), all_embed.min())
    max_v = max(v_ref_rep.max(), all_embed.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Identity (y = x)",
    )

    # trend line following the mass of the data (binned mean)
    bins = np.linspace(min_v, max_v, 25)

    bin_centers = []
    bin_means = []

    for i in range(len(bins) - 1):
        mask = (v_ref_rep >= bins[i]) & (v_ref_rep < bins[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(np.mean(all_embed[mask]))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)

    # global R² (coefficient of determination against fitted regression line)
    if len(v_ref_rep) > 1:
        coeffs = np.polyfit(v_ref_rep, all_embed, 1)
        predicted = np.polyval(coeffs, v_ref_rep)
        ss_res = np.sum((all_embed - predicted) ** 2)
        ss_tot = np.sum((all_embed - np.mean(all_embed)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r2 = 0.0

    ax.plot(
        bin_centers, bin_means, linewidth=3, label=f"Mean agreement trend (R²={r2:.3f})"
    )

    ax.set_xlabel("Consensus distance")
    ax.set_ylabel("Participant distance")
    ax.set_title(f"Shepard-like Agreement Plot ({condition.upper()})")
    ax.legend()

    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


# -------------------------------------------------
# KNN PRESERVATION CURVE
# -------------------------------------------------


def knn_overlap(D1, D2, k):
    """Compute kNN overlap between two distance matrices."""
    n = D1.shape[0]

    overlap = []

    for i in range(n):
        nn1 = np.argsort(D1[i])[1 : k + 1]
        nn2 = np.argsort(D2[i])[1 : k + 1]

        overlap.append(len(set(nn1).intersection(set(nn2))) / k)

    return np.mean(overlap)


def plot_knn_curve(coords_list, condition, out_path):

    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus_mats = _leave_one_out_consensus_matrices(D_mats)

    n = D_mats[0].shape[0]

    ks = range(1, n)

    scores = []

    for k in ks:
        vals = []
        for D, D_consensus in zip(D_mats, D_consensus_mats):
            vals.append(knn_overlap(D, D_consensus, k))
        scores.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(list(ks), scores, marker="o")

    ax.set_xlabel("k Nearest Neighbours")
    ax.set_ylabel("Neighbour preservation")
    ax.set_title(f"kNN Preservation Curve ({condition.upper()})")

    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


def _compute_knn_scores(coords_list):
    """Return (ks, scores) for a set of participant coordinate lists."""
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus_mats = _leave_one_out_consensus_matrices(D_mats)
    n = D_mats[0].shape[0]
    ks = list(range(1, n))
    scores = []
    for k in ks:
        vals = [
            knn_overlap(D, D_consensus, k)
            for D, D_consensus in zip(D_mats, D_consensus_mats)
        ]
        scores.append(np.mean(vals))
    return ks, scores


def plot_knn_combined(coords_list_2d, coords_list_3d, out_path):
    """Combined kNN preservation curve with 2D and 3D in one plot."""
    ks_2d, scores_2d = _compute_knn_scores(coords_list_2d)
    ks_3d, scores_3d = _compute_knn_scores(coords_list_3d)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks_2d, scores_2d, marker="o", label="2D")
    ax.plot(ks_3d, scores_3d, marker="s", label="3D")

    ax.set_xlabel("k Nearest Neighbours")
    ax.set_ylabel("Neighbour preservation")
    ax.set_title("kNN Preservation Curve (2D vs 3D)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------------------------------
# PER-PARTICIPANT PLOTS
# -------------------------------------------------


def plot_knn_individual(coords_list, participant_names, condition, out_path):
    """Per-participant kNN curves overlaid on a single plot."""
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus_mats = _leave_one_out_consensus_matrices(D_mats)
    n = D_mats[0].shape[0]
    ks = list(range(1, n))

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("tab10", len(coords_list))

    for idx, (D, D_consensus, name) in enumerate(
        zip(D_mats, D_consensus_mats, participant_names)
    ):
        scores = [knn_overlap(D, D_consensus, k) for k in ks]
        ax.plot(
            ks, scores, marker=".", markersize=4,
            label=name, color=cmap(idx), alpha=0.8,
        )

    ax.set_xlabel("k Nearest Neighbours")
    ax.set_ylabel("Neighbour preservation")
    ax.set_title(f"kNN Preservation per Participant ({condition.upper()})")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_shepard_single(v_ref, v_embed, participant_label, condition, out_path):
    """Shepard diagram for a single participant against LOO consensus."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(v_ref, v_embed, alpha=0.5, s=20)

    min_v = min(v_ref.min(), v_embed.min())
    max_v = max(v_ref.max(), v_embed.max())
    ax.plot(
        [min_v, max_v], [min_v, max_v],
        "r--", linewidth=1.5, label="Identity",
    )

    if len(v_ref) > 1:
        coeffs = np.polyfit(v_ref, v_embed, 1)
        predicted = np.polyval(coeffs, v_ref)
        ss_res = np.sum((v_embed - predicted) ** 2)
        ss_tot = np.sum((v_embed - np.mean(v_embed)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r2 = 0.0

    ax.set_xlabel("Consensus distance")
    ax.set_ylabel("Participant distance")
    ax.set_title(f"Shepard – {participant_label} ({condition.upper()})\nR² = {r2:.3f}")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_axis_variance_single(variance_ratios, participant_label, condition, out_path):
    """Axis variance bar chart for a single participant."""
    n_dims = len(variance_ratios)
    dims = ["X", "Y", "Z"][:n_dims]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(dims, variance_ratios, color="steelblue", edgecolor="black")
    for i_dim, val in enumerate(variance_ratios):
        ax.text(i_dim, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("Variance ratio")
    ax.set_title(f"Axis Variance – {participant_label} ({condition.upper()})")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_rdm_scatter_single(coords_2d, coords_3d, participant_label, out_path):
    """Scatter of 2D vs 3D pairwise distances for a single participant."""
    rdm_2d = pdist(coords_2d, metric="euclidean")
    rdm_3d = pdist(coords_3d, metric="euclidean")
    rho = _extract_spearman_rho(spearmanr(rdm_2d, rdm_3d))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rdm_2d, rdm_3d, alpha=0.5, s=20, color="slateblue")

    min_v = min(rdm_2d.min(), rdm_3d.min())
    max_v = max(rdm_2d.max(), rdm_3d.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, alpha=0.5)

    ax.set_xlabel("2D pairwise distance")
    ax.set_ylabel("3D pairwise distance")
    ax.set_title(f"2D vs 3D RDM – {participant_label}\nSpearman ρ = {rho:.3f}")
    ax.grid(True, linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------------------------------
# AXIS VARIANCE (DIMENSION USAGE)
# -------------------------------------------------


def plot_axis_variance(coords_list, condition, out_path, participant_names=None):
    """Bar plot of mean per-axis variance with individual participant dots."""
    vars_all = []
    for coords in coords_list:
        v = np.var(coords, axis=0)
        v = v / np.sum(v)
        vars_all.append(v)

    vars_all = np.asarray(vars_all)  # (n_participants, n_dims)
    n_dims = vars_all.shape[1]
    dims = ["X", "Y", "Z"][:n_dims]
    mean_vars = np.mean(vars_all, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))

    x_pos = np.arange(n_dims)
    ax.bar(
        x_pos,
        mean_vars,
        width=0.5,
        color="lightsteelblue",
        edgecolor="black",
        label="Mean",
        zorder=2,
    )

    # participant dots share one fixed vertical line per axis and are distinguished by color
    if participant_names is None:
        participant_names = [f"Participant {i+1}" for i in range(len(vars_all))]

    point_x = np.tile(x_pos, (len(vars_all), 1)).astype(float)
    for d in range(n_dims):
        order = np.argsort(vars_all[:, d])
        offsets = np.linspace(-0.08, 0.08, len(vars_all))
        point_x[order, d] += offsets

    cmap = plt.cm.get_cmap("tab10", len(vars_all))
    for p, pname in enumerate(participant_names):
        color = cmap(p)
        ax.scatter(
            point_x[p],
            vars_all[p],
            color=color,
            edgecolor="black",
            s=55,
            zorder=5,
            alpha=0.9,
            label=pname,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Variance ratio")
    ax.set_title(f"Axis Variance ({condition.upper()})")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_procrustes_dissimilarity_matrix(
    coords_list, participant_names, condition, out_path, normalize=True
):
    """Plot pairwise Procrustes dissimilarity matrix between all participants."""
    n = len(coords_list)
    D = np.zeros((n, n))

    # Compute pairwise Procrustes disparities
    for i in range(n):
        for j in range(i + 1, n):
            rmse = compute_procrustes_distance(coords_list[i], coords_list[j])
            D[i, j] = rmse
            D[j, i] = rmse

    # compute mean disparity (ignore diagonal zeros)
    mask = ~np.eye(n, dtype=bool)
    mean_disp = np.mean(D[mask])
    max_disp = np.max(D[mask]) if np.any(mask) else 0.0
    if normalize and max_disp > 0:
        matrix_values = D / max_disp
        colorbar_label = "Normalized disparity"
    else:
        matrix_values = D.copy()
        colorbar_label = "Disparity"

    display_max = np.max(matrix_values[mask]) if np.any(mask) else 0.0
    if display_max <= 0:
        display_max = 1.0

    # Plot matrix
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    lower_triangle_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    masked_D = np.ma.array(matrix_values, mask=lower_triangle_mask)
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0)

    im = ax.imshow(masked_D, cmap=cmap, vmin=0, vmax=display_max)

    # Add values inside cells
    for i in range(n):
        for j in range(i + 1):
            ax.text(
                j,
                i,
                f"{matrix_values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix_values[i, j] > display_max * 0.5 else "black",
                fontsize=8,
            )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(participant_names, rotation=45, ha="right")
    ax.set_yticklabels(participant_names)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    plt.colorbar(im, fraction=0.046, pad=0.04, label=colorbar_label)
    plt.title(
        f"Procrustes Dissimilarity Matrix - {condition.upper()}  "
        f"(Mean disparity = {mean_disp:.4f})"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _extract_spearman_rho(spearman_result):
    rho_value = getattr(cast(Any, spearman_result), "statistic", spearman_result[0])
    rho_array = np.asarray(rho_value)
    if rho_array.ndim == 0:
        rho = float(rho_array)
    elif rho_array.shape == (2, 2):
        rho = float(rho_array[0, 1])
    else:
        rho = float(rho_array.reshape(-1)[0])
    if np.isnan(rho):
        rho = 0.0
    return rho


def _condensed_rdm_vector(coords):
    """Return the condensed pairwise-distance vector (one triangle, no diagonal)."""
    return pdist(np.asarray(coords), metric="euclidean")


def plot_spearman_consistency_heatmap(
    coords_list, participant_names, condition, out_path
):
    """Plot a Spearman correlation heatmap between participants based on pairwise distances."""
    n = len(coords_list)
    correlation_matrix = np.ones((n, n))
    distance_vectors = [_condensed_rdm_vector(coords) for coords in coords_list]

    for i in range(n):
        for j in range(i + 1, n):
            spearman_result = spearmanr(
                distance_vectors[i],
                distance_vectors[j],
            )
            rho = _extract_spearman_rho(spearman_result)
            correlation_matrix[i, j] = rho
            correlation_matrix[j, i] = rho

    mean_rho = np.mean(correlation_matrix[~np.eye(n, dtype=bool)])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    lower_triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    masked_correlation = np.ma.array(correlation_matrix, mask=lower_triangle_mask)
    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white", alpha=0)
    im = ax.imshow(masked_correlation, cmap=cmap, vmin=-1, vmax=1)

    for i in range(n):
        for j in range(i + 1):
            value = correlation_matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 else "black",
                fontsize=8,
            )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(participant_names, rotation=45, ha="right")
    ax.set_yticklabels(participant_names)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Spearman rho")
    plt.title(
        f"Spearman Intersubject Consistency - {condition.upper()}  "
        f"(Mean rho = {mean_rho:.4f})"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_crossdimensional_rdm_similarity(embeddings_2d, embeddings_3d, out_path):
    """Compare each selected participant's 2D RDM against their 3D RDM."""
    embeddings_by_participant_2d = {
        embedding[0]: embedding for embedding in embeddings_2d
    }
    embeddings_by_participant_3d = {
        embedding[0]: embedding for embedding in embeddings_3d
    }
    common_participants = sorted(
        set(embeddings_by_participant_2d).intersection(embeddings_by_participant_3d)
    )

    if not common_participants:
        return

    participant_labels = []
    rho_values = []

    for participant_id in common_participants:
        embedding_2d = embeddings_by_participant_2d[participant_id]
        embedding_3d = embeddings_by_participant_3d[participant_id]

        names_2d = embedding_2d[3]
        coords_2d = embedding_2d[4]
        names_3d = embedding_3d[3]
        coords_3d = embedding_3d[4]
        coords_by_name_3d = {name: coords for name, coords in zip(names_3d, coords_3d)}
        common_names = [name for name in names_2d if name in coords_by_name_3d]

        if len(common_names) < 2:
            continue

        ordered_coords_2d = np.asarray(
            [coords_2d[names_2d.index(name)] for name in common_names]
        )
        ordered_coords_3d = np.asarray(
            [coords_by_name_3d[name] for name in common_names]
        )

        rdm_2d = _condensed_rdm_vector(ordered_coords_2d)
        rdm_3d = _condensed_rdm_vector(ordered_coords_3d)
        rho = _extract_spearman_rho(spearmanr(rdm_2d, rdm_3d))

        participant_labels.append(embedding_2d[2])
        rho_values.append(rho)

    if not rho_values:
        return

    mean_rho = float(np.mean(rho_values))
    fig, ax = plt.subplots(figsize=(max(6, len(rho_values) * 1.2), 5))
    x_positions = np.arange(len(rho_values))
    bars = ax.bar(x_positions, rho_values, color="slateblue", edgecolor="black")
    # ax.axhline(mean_rho, color="red", linestyle="--", linewidth=2,
    #            label=f"Mean rho = {mean_rho:.4f}")

    for bar, rho in zip(bars, rho_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            rho + (0.03 if rho >= 0 else -0.05),
            f"{rho:.3f}",
            ha="center",
            va="bottom" if rho >= 0 else "top",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(participant_labels, rotation=45, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-1, 1)
    ax.set_title(f"2D vs 3D RDM Similarity")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_condition_analysis(embeddings, condition, out_dir, title_suffix=""):
    """Run the aggregated analysis suite for one condition."""
    if len(embeddings) < 2:
        return

    print(f"\n--- Procrustes Analysis for {condition.upper()}{title_suffix} ---")

    coords_list = [embedding[4] for embedding in embeddings]
    stimulus_names = embeddings[0][3]

    if ANONYMOUS:
        participant_names = [f"Participant {embedding[0]}" for embedding in embeddings]
    else:
        participant_names = [embedding[1] for embedding in embeddings]

    mean_shape, aligned = generalized_procrustes_analysis(coords_list)

    disparities = []
    for aligned_coords in aligned:
        disp = compute_procrustes_distance(mean_shape, aligned_coords)
        disparities.append(disp)

    print(
        f"Mean disparity {condition.upper()}{title_suffix}: {np.mean(disparities):.4f}"
    )

    analysis_steps = [
        (
            "Intersubject consistency",
            lambda: plot_intersubject_consistency(
                disparities,
                participant_names,
                condition,
                out_dir / f"intersubject_consistency_{condition}.png",
            ),
        ),
        (
            "Procrustes dissimilarity matrix",
            lambda: plot_procrustes_dissimilarity_matrix(
                coords_list,
                participant_names,
                condition,
                out_dir / f"procrustes_dissimilarity_matrix_{condition}.png",
                normalize=NORMALIZE_PROCRUSTES_DISSIMILARITY,
            ),
        ),
        (
            "Spearman consistency heatmap",
            lambda: plot_spearman_consistency_heatmap(
                coords_list,
                participant_names,
                condition,
                out_dir / f"spearman_consistency_heatmap_{condition}.png",
            ),
        ),
        (
            "Shepard diagram",
            lambda: plot_shepard_global(
                coords_list,
                stimulus_names,
                condition,
                out_dir / f"shepard_{condition}.png",
            ),
        ),
        (
            "kNN preservation",
            lambda: plot_knn_curve(
                coords_list,
                condition,
                out_dir / f"knn_preservation_{condition}.png",
            ),
        ),
        (
            "Axis variance",
            lambda: plot_axis_variance(
                coords_list,
                condition,
                out_dir / f"axis_variance_{condition}.png",
                participant_names,
            ),
        ),
        (
            "Procrustes arrangement",
            lambda: (
                plot_procrustes_arrangement_2d(
                    aligned,
                    mean_shape,
                    participant_names,
                    stimulus_names,
                    out_dir / f"procrustes_arrangement_{condition}.png",
                )
                if condition == "2d"
                else plot_procrustes_arrangement_3d(
                    aligned,
                    mean_shape,
                    participant_names,
                    stimulus_names,
                    out_dir / f"procrustes_arrangement_{condition}.png",
                )
            ),
        ),
        (
            "Per-participant kNN",
            lambda: plot_knn_individual(
                coords_list,
                participant_names,
                condition,
                out_dir / f"knn_individual_{condition}.png",
            ),
        ),
    ]

    progress = _progress(
        analysis_steps,
        desc=f"{condition.upper()}{title_suffix or ''} plots",
        unit="plot",
        dynamic_ncols=True,
    )
    for step_name, step_fn in progress:
        _set_progress_label(progress, step_name)
        step_fn()


def main():
    # Collect all embeddings per condition
    embeddings_2d = []  # list of (pid, pname, plabel, names, coords)
    embeddings_3d = []

    # Create output directories
    GENERAL_DIR.mkdir(parents=True, exist_ok=True)
    GENERAL_2D_DIR.mkdir(exist_ok=True)
    GENERAL_3D_DIR.mkdir(exist_ok=True)
    DETAILED_DIR.mkdir(exist_ok=True)

    participant_index = 0

    participant_dirs = [
        participant_dir
        for participant_dir in sorted(FINAL_RESULTS_DIR.iterdir())
        if participant_dir.is_dir() and not participant_dir.name.startswith(".")
    ]

    # ── Phase 1: Load all embeddings ──
    for participant_dir in _progress(
        participant_dirs,
        desc="Loading participants",
        unit="participant",
        dynamic_ncols=True,
    ):
        participant_name = participant_dir.name
        participant_index += 1
        if ANONYMOUS:
            participant_label = f"Participant {participant_index}"
        else:
            participant_label = participant_name

        for cond in ("2d", "3d"):
            cond_dir = participant_dir / cond
            if not cond_dir.exists():
                continue
            csv_files = sorted(cond_dir.glob("*.csv"))
            for csv_file in csv_files:
                names, coords = load_embedding(csv_file)
                if cond == "2d":
                    coords = coords[:, :2]
                if len(coords) >= 2:
                    embedding = (
                        participant_index,
                        participant_name,
                        participant_label,
                        names,
                        coords,
                    )
                    if cond == "2d":
                        embeddings_2d.append(embedding)
                    else:
                        embeddings_3d.append(embedding)

    # ── Phase 2: General aggregated analysis → analysis/general/{2d,3d}/ ──
    run_condition_analysis(embeddings_2d, "2d", GENERAL_2D_DIR)
    run_condition_analysis(embeddings_3d, "3d", GENERAL_3D_DIR)

    # Combined kNN plot (2D vs 3D)
    if len(embeddings_2d) >= 2 and len(embeddings_3d) >= 2:
        coords_list_2d = [e[4] for e in embeddings_2d]
        coords_list_3d = [e[4] for e in embeddings_3d]
        plot_knn_combined(
            coords_list_2d, coords_list_3d,
            GENERAL_DIR / "knn_preservation_2d_vs_3d.png",
        )

    # Cross-dimensional RDM bar chart
    if embeddings_2d and embeddings_3d:
        plot_crossdimensional_rdm_similarity(
            embeddings_2d,
            embeddings_3d,
            GENERAL_DIR / "rdm_similarity_2d_vs_3d.png",
        )

    # ── Phase 3: Detailed per-participant → analysis/detailed/Participant_X/{2d,3d}/ ──
    if PARTICIPANTS:
        selected_indices = set(PARTICIPANTS)
    else:
        selected_indices = (
            set(e[0] for e in embeddings_2d) | set(e[0] for e in embeddings_3d)
        )

    for cond, all_embeddings in [("2d", embeddings_2d), ("3d", embeddings_3d)]:
        if len(all_embeddings) < 2:
            continue

        coords_list = [e[4] for e in all_embeddings]
        D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
        D_consensus_mats = _leave_one_out_consensus_matrices(D_mats)

        for idx, embedding in enumerate(
            _progress(
                all_embeddings,
                desc=f"Detailed {cond.upper()}",
                unit="participant",
                dynamic_ncols=True,
            )
        ):
            pid, pname, plabel, names, coords = embedding
            if pid not in selected_indices:
                continue

            safe_label = plabel.replace(" ", "_")
            p_dir = DETAILED_DIR / safe_label / cond
            p_dir.mkdir(parents=True, exist_ok=True)

            # Dissimilarity matrix
            D = squareform(pdist(coords, metric="euclidean"))
            D_plot = D / D.max() if D.max() > 0 else D.copy()
            plot_dissimilarity_matrix(names, D_plot, f"dissimilarity_{cond}", p_dir)

            # Arrangement
            if cond == "2d":
                plot_participant_arrangement_2d(
                    names, coords, plabel, p_dir / f"arrangement_{cond}.png"
                )
            else:
                plot_participant_arrangement_3d(
                    names, coords, plabel, p_dir / f"arrangement_{cond}.png"
                )

            # Shepard individual
            v_ref = squareform(D_consensus_mats[idx], checks=False)
            v_embed = squareform(D_mats[idx], checks=False)
            _plot_shepard_single(
                v_ref, v_embed, plabel, cond, p_dir / f"shepard_{cond}.png"
            )

            # Axis variance
            v = np.var(coords, axis=0)
            v = v / np.sum(v)
            _plot_axis_variance_single(
                v, plabel, cond, p_dir / f"axis_variance_{cond}.png"
            )

    # Cross-dimensional RDM scatter per participant
    embeddings_by_2d = {e[0]: e for e in embeddings_2d}
    embeddings_by_3d = {e[0]: e for e in embeddings_3d}
    common_pids = sorted(
        selected_indices
        & set(embeddings_by_2d.keys())
        & set(embeddings_by_3d.keys())
    )
    for pid in common_pids:
        e2d, e3d = embeddings_by_2d[pid], embeddings_by_3d[pid]
        plabel = e2d[2]
        names_2d, coords_2d = e2d[3], e2d[4]
        names_3d, coords_3d = e3d[3], e3d[4]

        coords_by_name_3d = dict(zip(names_3d, coords_3d))
        common_names = [n for n in names_2d if n in coords_by_name_3d]
        if len(common_names) < 2:
            continue

        ordered_2d = np.asarray(
            [coords_2d[names_2d.index(n)] for n in common_names]
        )
        ordered_3d = np.asarray([coords_by_name_3d[n] for n in common_names])

        safe_label = plabel.replace(" ", "_")
        p_dir = DETAILED_DIR / safe_label
        _plot_rdm_scatter_single(
            ordered_2d, ordered_3d, plabel, p_dir / "rdm_scatter.png"
        )


if __name__ == "__main__":
    main()
