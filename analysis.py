import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from PIL import Image


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

def load_image(name, zoom=0.25):
    path = PICTURES_DIR / name
    if not path.exists():
        return None
    img = Image.open(path)
    arr = np.asarray(img)
    return OffsetImage(arr, zoom=zoom)

# -------------------------------------------------
# PLOT MATRIX WITH IMAGES + VALUES
# -------------------------------------------------

def plot_dissimilarity_matrix(names, D, csv_name, out_dir):
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.2 * n, 1.2 * n))

    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- values inside cells ---
    for i in range(n):
        for j in range(n):
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
            (i, -0.9),
            xycoords="data",
            frameon=False,
            box_alignment=(0.5, 1)
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
    ax.set_ylim(n - 0.5, -1)

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
    
    # Plot matrix
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(8, n * 0.8)))
    im = ax.imshow(D, cmap="viridis")
    
    # Add values inside cells
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{D[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if D[i, j] > D.max() * 0.5 else "black",
                fontsize=8
            )
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(participant_names, rotation=45, ha='right')
    ax.set_yticklabels(participant_names)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Procrustes Dissimilarity Matrix - {condition.upper()}")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(): 
    # Collect all embeddings per condition
    embeddings_2d = []  # list of (participant_name, names, coords)
    embeddings_3d = []
    
    # Iterate through all participant folders in final_results
    for participant_dir in FINAL_RESULTS_DIR.iterdir():
        if not participant_dir.is_dir() or participant_dir.name.startswith('.'):
            continue
        
        participant_name = participant_dir.name
        
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
                        embeddings_2d.append((participant_name, names, coords))
                    else:
                        embeddings_3d.append((participant_name, names, coords))
    
    # Procrustes analysis for 2D condition
    if len(embeddings_2d) >= 2:
        print("\n--- Procrustes Analysis for 2D ---")
        coords_list_2d = [e[2] for e in embeddings_2d]
        if ANONYMOUS:
            participant_names_2d = [f"Participant {i+1}" for i in range(len(embeddings_2d))]
        else:
            participant_names_2d = [e[0] for e in embeddings_2d]
        
        mean_shape_2d, aligned_2d = generalized_procrustes_analysis(coords_list_2d)
        
        # Compute disparity of each participant to mean
        disparities_2d = []
        for aligned_coords in aligned_2d:
            disp = compute_procrustes_distance(mean_shape_2d, aligned_coords)
            disparities_2d.append(disp)
        
        # Plot intersubject consistency
        consistency_path_2d = PROCRUSTES_2D_DIR / "intersubject_consistency_2d.png"
        plot_intersubject_consistency(disparities_2d, participant_names_2d, "2d", consistency_path_2d)
        
        # Plot Procrustes dissimilarity matrix
        dissim_path_2d = PROCRUSTES_2D_DIR / "procrustes_dissimilarity_matrix_2d.png"
        plot_procrustes_dissimilarity_matrix(coords_list_2d, participant_names_2d, "2d", dissim_path_2d)
        
        print(f"Mean disparity 2D: {np.mean(disparities_2d):.4f}")
    
    # Procrustes analysis for 3D condition
    if len(embeddings_3d) >= 2:
        print("\n--- Procrustes Analysis for 3D ---")
        coords_list_3d = [e[2] for e in embeddings_3d]
        if ANONYMOUS:
            participant_names_3d = [f"Participant {i+1}" for i in range(len(embeddings_3d))]
        else:
            participant_names_3d = [e[0] for e in embeddings_3d]
        
        mean_shape_3d, aligned_3d = generalized_procrustes_analysis(coords_list_3d)
        
        # Compute disparity of each participant to mean
        disparities_3d = []
        for aligned_coords in aligned_3d:
            disp = compute_procrustes_distance(mean_shape_3d, aligned_coords)
            disparities_3d.append(disp)
        
        # Plot intersubject consistency
        consistency_path_3d = PROCRUSTES_3D_DIR / "intersubject_consistency_3d.png"
        plot_intersubject_consistency(disparities_3d, participant_names_3d, "3d", consistency_path_3d)
        
        # Plot Procrustes dissimilarity matrix
        dissim_path_3d = PROCRUSTES_3D_DIR / "procrustes_dissimilarity_matrix_3d.png"
        plot_procrustes_dissimilarity_matrix(coords_list_3d, participant_names_3d, "3d", dissim_path_3d)
        
        print(f"Mean disparity 3D: {np.mean(disparities_3d):.4f}")


if __name__ == "__main__":
    main()
