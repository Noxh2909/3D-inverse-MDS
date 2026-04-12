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


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FINAL_RESULTS_DIR = BASE_DIR / "final_results"
PICTURES_DIR = BASE_DIR / "pictures"
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

ANALYSIS_2D_DIR = ANALYSIS_DIR / "2d"
ANALYSIS_3D_DIR = ANALYSIS_DIR / "3d"
ANALYSIS_2D_DIR.mkdir(exist_ok=True)
ANALYSIS_3D_DIR.mkdir(exist_ok=True)

PROCRUSTES_DIR = ANALYSIS_DIR / "procrustes"
PROCRUSTES_2D_DIR = PROCRUSTES_DIR / "2d"
PROCRUSTES_3D_DIR = PROCRUSTES_DIR / "3d"
PROCRUSTES_DIR.mkdir(exist_ok=True)
PROCRUSTES_2D_DIR.mkdir(exist_ok=True)
PROCRUSTES_3D_DIR.mkdir(exist_ok=True)

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

# Restrict detailed Procrustes-style analysis to specific participant indices.
# Example: [1, 2] will generate analysis/detailed/... for Participant 1 and 2 only.
# Leave empty to skip detailed subset analysis.
PARTICIPANTS = []


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


def _stimulus_size_points(name, zoom, border_width):
    path = PICTURES_DIR / name
    if not path.exists():
        return 42.0 * zoom, 42.0 * zoom

    with Image.open(path) as image:
        width, height = image.size

    width += 2 * border_width
    height += 2 * border_width
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
        penalty += 1.5 * _rect_intersection_area(candidate_rect, label_rect, padding=2.0)

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
            candidate_rect = _candidate_label_rect(point, label_width, label_height, candidate)
            penalty = _candidate_penalty(candidate_rect, index, image_rects, placed_label_rects)

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


def add_projected_stimuli_3d(ax, names, coords, zoom=0.12, border_width=5,
                             label_size=11):
    """Project stimulus thumbnails onto a 3D axes and place labels beside them."""
    fig = ax.figure
    fig.canvas.draw()

    projected_points = []
    for x_coord, y_coord, z_coord in coords:
        x_proj, y_proj, _ = proj3d.proj_transform(x_coord, y_coord, z_coord, ax.get_proj())
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

    for (name, (x_coord, y_coord, z_coord), (x_proj, y_proj), placement) in zip(
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
        ax.plot([x_coord, x_coord], [y_coord, y_coord], [z_min, z_coord],
                linestyle="--", linewidth=0.8, color="black", alpha=0.42,
                zorder=1)


def remap_coords_for_3d_axes(coords):
    """Keep original 3D axis order."""
    return coords[:, [0, 1, 2]]


def add_stimuli_2d(ax, names, x_coords, y_coords, zoom=0.20, border_width=5,
                   label_size=11):
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

    for x_coord, y_coord, name, placement in zip(x_coords, y_coords, names, label_placements):
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
    add_projected_stimuli_3d(ax, names, remapped_coords, zoom=0.40,
                             border_width=8, label_size=12)

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

def plot_procrustes_arrangement_2d(aligned_list, mean_shape, participant_names,
                                   stimulus_names, out_path):
    """2D scatter showing only the Procrustes mean shape with stimulus images."""
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], alpha=0.0)

    pad = 0.22
    rx = mean_shape[:, 0].max() - mean_shape[:, 0].min() or 1
    ry = mean_shape[:, 1].max() - mean_shape[:, 1].min() or 1
    ax.set_xlim(mean_shape[:, 0].min() - pad * rx, mean_shape[:, 0].max() + pad * rx)
    ax.set_ylim(mean_shape[:, 1].min() - pad * ry, mean_shape[:, 1].max() + pad * ry)
    add_stimuli_2d(ax, stimulus_names, mean_shape[:, 0], mean_shape[:, 1],
                   zoom=0.32, border_width=8, label_size=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Procrustes Mean Shape (2D)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_procrustes_arrangement_3d(aligned_list, mean_shape, participant_names,
                                   stimulus_names, out_path):
    """3D cube showing only the Procrustes mean shape with stimulus images."""
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection="3d")
    remapped_mean_shape = remap_coords_for_3d_axes(mean_shape)

    ax.scatter(remapped_mean_shape[:, 0], remapped_mean_shape[:, 1], remapped_mean_shape[:, 2],
               s=0, alpha=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_zlim(-1, 1)
    add_depth_guides_3d(ax, remapped_mean_shape)
    add_projected_stimuli_3d(ax, stimulus_names, remapped_mean_shape, zoom=0.40,
                             border_width=8, label_size=12)

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

    lower_triangle_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    masked_D = np.ma.array(D, mask=lower_triangle_mask)
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0)

    im = ax.imshow(masked_D, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- values inside cells ---
    for i in range(n):
        for j in range(i + 1):
            ax.text(
                j, i,
                f"{D[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if D[i, j] > D.max() * 0.5 else "black",
                fontsize=10
            )

    # --- image ticks ---
    for i, name in enumerate(names):
        img_x = load_image(name)
        img_y = load_image(name)
        if img_x is None:
            continue

        # X axis
        ab_x = AnnotationBbox(
            img_x,
            (i, n - 0.1),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 0)
        )
        ax.add_artist(ab_x)

        # Y axis
        if img_y is not None:
            ab_y = AnnotationBbox(
                img_y,
                (-0.6, i),
                xycoords="data",
                frameon=False,
                box_alignment=(1, 0.5)
            )
            ax.add_artist(ab_y)

    ax.set_xlim(-1, n - 0.5)
    ax.set_ylim(n + 0.1, -1)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Dissimilarity Matrix (Euclidean Distance)")

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
    """Compute Procrustes distance (disparity) between two configurations."""
    _, _, disparity = procrustes(coords1, coords2)
    return disparity


def plot_intersubject_consistency(disparities, participant_names, condition, out_path):
    """Plot bar chart of Procrustes disparities for each participant."""
    fig, ax = plt.subplots(figsize=(max(8, len(disparities) * 0.8), 6))
    
    x = np.arange(len(disparities))
    bars = ax.bar(x, disparities, color='steelblue', edgecolor='black')
    
    # Add mean line
    mean_disp = np.mean(disparities)
    ax.axhline(mean_disp, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_disp:.4f}')
    
    ax.set_xlabel('Participant')
    ax.set_ylabel('Procrustes Disparity')
    ax.set_title(f'Intersubject Consistency - {condition.upper()}\n(lower = more consistent)')
    ax.set_xticks(x)
    ax.set_xticklabels(participant_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# -------------------------------------------------
# SHEPARD DIAGRAM (GLOBAL)
# -------------------------------------------------

def plot_shepard_global(coords_list, stimulus_names, condition, out_path):
    """Aggregate Shepard diagram across participants."""

    # compute consensus distances
    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus = np.mean(D_mats, axis=0)

    v_ref = squareform(D_consensus, checks=False)

    all_embed = []

    for D in D_mats:
        v = squareform(D, checks=False)
        all_embed.extend(v)

    all_embed = np.asarray(all_embed)
    v_ref_rep = np.tile(v_ref, len(coords_list))

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
        label="Perfect reconstruction"
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

    # global R² between reference and embedding distances
    if len(v_ref_rep) > 1:
        r = np.corrcoef(v_ref_rep, all_embed)[0, 1]
        r2 = r ** 2
    else:
        r2 = 0.0

    ax.plot(
        bin_centers,
        bin_means,
        linewidth=3,
        label=f"Mean distortion trend (R²={r2:.3f})"
    )

    ax.set_xlabel("Reference distance")
    ax.set_ylabel("Embedding distance")
    ax.set_title(f"Shepard Diagram ({condition.upper()})")
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
        nn1 = np.argsort(D1[i])[1:k+1]
        nn2 = np.argsort(D2[i])[1:k+1]

        overlap.append(len(set(nn1).intersection(set(nn2))) / k)

    return np.mean(overlap)


def plot_knn_curve(coords_list, condition, out_path):

    D_mats = [squareform(pdist(c, metric="euclidean")) for c in coords_list]
    D_consensus = np.mean(D_mats, axis=0)

    n = D_consensus.shape[0]

    ks = range(1, n)

    scores = []

    for k in ks:
        vals = []
        for D in D_mats:
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
    D_consensus = np.mean(D_mats, axis=0)
    n = D_consensus.shape[0]
    ks = list(range(1, n))
    scores = []
    for k in ks:
        vals = [knn_overlap(D, D_consensus, k) for D in D_mats]
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
    ax.bar(x_pos, mean_vars, width=0.5, color="lightsteelblue",
           edgecolor="black", label="Mean", zorder=2)

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
        ax.scatter(point_x[p],
                   vars_all[p], color=color, edgecolor="black",
                   s=55, zorder=5, alpha=0.9, label=pname)

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


def plot_procrustes_dissimilarity_matrix(coords_list, participant_names, condition, out_path):
    """Plot pairwise Procrustes dissimilarity matrix between all participants."""
    n = len(coords_list)
    D = np.zeros((n, n))
    
    # Compute pairwise Procrustes disparities
    for i in range(n):
        for j in range(i + 1, n):
            _, _, disparity = procrustes(coords_list[i], coords_list[j])
            D[i, j] = disparity
            D[j, i] = disparity

    # compute mean disparity (ignore diagonal zeros)
    mask = ~np.eye(n, dtype=bool)
    mean_disp = np.mean(D[mask])
    max_disp = np.max(D[mask]) if np.any(mask) else 0.0
    if max_disp > 0:
        D_normalized = D / max_disp
    else:
        D_normalized = D.copy()
    
    # Plot matrix
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    lower_triangle_mask = np.triu(np.ones_like(D, dtype=bool), k=1)
    masked_D = np.ma.array(D_normalized, mask=lower_triangle_mask)
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0)

    im = ax.imshow(masked_D, cmap=cmap, vmin=0, vmax=1)
    
    # Add values inside cells
    for i in range(n):
        for j in range(i + 1):
            ax.text(
                j, i,
                f"{D_normalized[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if D_normalized[i, j] > 0.5 else "black",
                fontsize=8
            )
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(participant_names, rotation=45, ha='right')
    ax.set_yticklabels(participant_names)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Normalized disparity")
    plt.title(
        f"Procrustes Dissimilarity Matrix - {condition.upper()}  "
        f"(Mean disparity = {mean_disp:.4f})"
    )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spearman_consistency_heatmap(coords_list, participant_names, condition, out_path):
    """Plot a Spearman correlation heatmap between participants based on pairwise distances."""
    n = len(coords_list)
    correlation_matrix = np.ones((n, n))
    distance_vectors = [squareform(pdist(coords, metric="euclidean"), checks=False)
                        for coords in coords_list]

    for i in range(n):
        for j in range(i + 1, n):
            spearman_result = spearmanr(
                np.ravel(distance_vectors[i]),
                np.ravel(distance_vectors[j]),
            )
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
                j, i,
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

    consistency_path = out_dir / f"intersubject_consistency_{condition}.png"
    plot_intersubject_consistency(disparities, participant_names, condition, consistency_path)

    dissim_path = out_dir / f"procrustes_dissimilarity_matrix_{condition}.png"
    plot_procrustes_dissimilarity_matrix(coords_list, participant_names, condition, dissim_path)

    spearman_path = out_dir / f"spearman_consistency_heatmap_{condition}.png"
    plot_spearman_consistency_heatmap(coords_list, participant_names, condition, spearman_path)

    print(f"Mean disparity {condition.upper()}{title_suffix}: {np.mean(disparities):.4f}")

    shepard_path = out_dir / f"shepard_{condition}.png"
    plot_shepard_global(coords_list, stimulus_names, condition, shepard_path)

    knn_path = out_dir / f"knn_preservation_{condition}.png"
    plot_knn_curve(coords_list, condition, knn_path)

    axis_path = out_dir / f"axis_variance_{condition}.png"
    plot_axis_variance(coords_list, condition, axis_path, participant_names)

    proc_arr_path = out_dir / f"procrustes_arrangement_{condition}.png"
    if condition == "2d":
        plot_procrustes_arrangement_2d(aligned, mean_shape, participant_names,
                                       stimulus_names, proc_arr_path)
    else:
        plot_procrustes_arrangement_3d(aligned, mean_shape, participant_names,
                                       stimulus_names, proc_arr_path)


def main(): 
    # Collect all embeddings per condition
    embeddings_2d = []  # list of (participant_name, names, coords)
    embeddings_3d = []
    
    # Output directories for per-participant arrangement plots
    ARRANGEMENTS_DIR = ANALYSIS_DIR / "arrangements"
    ARR_2D_DIR = ARRANGEMENTS_DIR / "2d"
    ARR_3D_DIR = ARRANGEMENTS_DIR / "3d"
    DETAILED_DIR = ANALYSIS_DIR / "detailed"
    DETAILED_2D_DIR = DETAILED_DIR / "2d"
    DETAILED_3D_DIR = DETAILED_DIR / "3d"
    ARRANGEMENTS_DIR.mkdir(exist_ok=True)
    ARR_2D_DIR.mkdir(exist_ok=True)
    ARR_3D_DIR.mkdir(exist_ok=True)
    DETAILED_DIR.mkdir(exist_ok=True)
    DETAILED_2D_DIR.mkdir(exist_ok=True)
    DETAILED_3D_DIR.mkdir(exist_ok=True)

    participant_index = 0
    
    # Iterate through all participant folders in final_results
    for participant_dir in sorted(FINAL_RESULTS_DIR.iterdir()):
        if not participant_dir.is_dir() or participant_dir.name.startswith('.'):
            continue
        
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
            out_dir = ANALYSIS_2D_DIR if cond == "2d" else ANALYSIS_3D_DIR
            for csv_file in cond_dir.glob("*.csv"):
                # Create dissimilarity matrix
                analyze_csv(csv_file, out_dir)
                
                # Load embedding for Procrustes
                names, coords = load_embedding(csv_file)
                if len(coords) >= 2:
                    if cond == "2d":
                        embeddings_2d.append((participant_index, participant_name,
                                              participant_label, names, coords))
                        # Per-participant 2D scatter
                        arr_path = ARR_2D_DIR / f"{participant_label.replace(' ', '_')}_2d.png"
                        plot_participant_arrangement_2d(names, coords, participant_label, arr_path)
                    else:
                        embeddings_3d.append((participant_index, participant_name,
                                              participant_label, names, coords))
                        # Per-participant 3D cube
                        arr_path = ARR_3D_DIR / f"{participant_label.replace(' ', '_')}_3d.png"
                        plot_participant_arrangement_3d(names, coords, participant_label, arr_path)

    run_condition_analysis(embeddings_2d, "2d", PROCRUSTES_2D_DIR)
    run_condition_analysis(embeddings_3d, "3d", PROCRUSTES_3D_DIR)

    selected_participants = set(PARTICIPANTS)
    if selected_participants:
        detailed_embeddings_2d = [embedding for embedding in embeddings_2d
                                  if embedding[0] in selected_participants]
        detailed_embeddings_3d = [embedding for embedding in embeddings_3d
                                  if embedding[0] in selected_participants]

        run_condition_analysis(detailed_embeddings_2d, "2d", DETAILED_2D_DIR,
                               title_suffix=" (Detailed)")
        run_condition_analysis(detailed_embeddings_3d, "3d", DETAILED_3D_DIR,
                               title_suffix=" (Detailed)")

    # Combined kNN plot (2D vs 3D)
    if len(embeddings_2d) >= 2 and len(embeddings_3d) >= 2:
        coords_list_2d = [embedding[4] for embedding in embeddings_2d]
        coords_list_3d = [embedding[4] for embedding in embeddings_3d]
        knn_combined_path = PROCRUSTES_DIR / "knn_preservation_2d_vs_3d.png"
        plot_knn_combined(coords_list_2d, coords_list_3d, knn_combined_path)


if __name__ == "__main__":
    main()
