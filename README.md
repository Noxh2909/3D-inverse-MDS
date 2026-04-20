# 3D inverse MDS

This repository contains the implementation, analysis pipeline, and build scripts for the bachelor thesis **"Measuring perceived similarity using Inverse Multidimensional Scaling in three dimensions"** by Noah Kogge.

At its core, the thesis investigates how the dimensionality of a spatial arrangement task influences the externalization of perceived similarity. Participants arrange the same set of stimuli in both a 2D and a 3D environment. These arrangements are then transformed into distance structures, consistency measures, and qualitative comparisons between both interaction spaces.

## What the thesis is about

Classical Multidimensional Scaling (MDS) usually reconstructs similarity spaces from many pairwise judgments. Inverse MDS and related spatial arrangement methods reverse that logic: instead of answering many isolated comparisons, participants place multiple stimuli directly in space so that similar objects are close together and dissimilar objects are farther apart.

The bachelor thesis transfers this idea into a controlled comparison between two interaction spaces:

- `2D`: stimuli are arranged on a plane
- `3D`: stimuli are arranged in a volumetric space with an additional depth axis

The central research question can be summarized as follows:

**How does an additional spatial dimension change the variability, reliability, and consistency of the reconstructed similarity structure compared with established 2D procedures?**

## Study design

The implementation follows a single-trial, iMDS-inspired spatial arrangement paradigm.

- `12` abstract body-silhouette stimuli were used.
- All participants completed both conditions (`2D` and `3D`).
- The design is a `within-subject` comparison.
- The thesis analyzes data from `7` participants.
- The task was to arrange the stimuli according to perceived similarity.

Important qualification: the project is not intended as a one-to-one reproduction of a canonical multi-arrangement iMDS protocol. Instead, it provides a custom research application designed to isolate the effect of interaction dimensionality under tightly matched conditions.

## Project structure

The most important files and directories are:

- `experiment.py`: GUI for running the actual experiment
- `analysis.py`: analysis pipeline for the metrics and visualizations used in the thesis
- `main.py`: central launcher that checks dependencies and starts either the experiment or the analysis
- `buildapp.sh`: build script for creating a packaged desktop app on macOS, Linux, or Windows
- `pictures/`: stimulus images
- `final_results/`: structured datasets used by the analysis
- `analysis/`: output directory for generated analysis results

## Installation and requirements

A recent Python setup with `python3` is recommended.

The project uses, among others:

- `numpy`
- `PySide6`
- `pyqtgraph`
- `PyOpenGL`
- `matplotlib`
- `Pillow`
- `scipy`
- `tqdm`

The launcher `main.py` checks these packages automatically on startup and installs missing dependencies when possible in the current Python environment.

## Running the experiment

The experiment is started via the central launcher:

```bash
python3 main.py
```

Alternatively, the GUI can be started directly:

```bash
python3 experiment.py
```

### Experiment flow

The flow follows the procedure described in the thesis:

1. The participant selects or is assigned a condition.
2. Name and age are entered.
3. A tutorial introduces the interaction mechanics.
4. All stimuli are arranged according to perceived similarity.
5. After one condition is completed, the other condition follows.

### Interaction logic

- In `2D`, stimuli are moved on a plane.
- In `3D`, an additional spatial axis is available.
- The scene can be rotated in `3D`.
- Height or depth can be adjusted through the 3D interaction.
- The task is open-ended: participants can iteratively refine their structure.

### Where the data is written

When run locally with Python, the experiment writes its output files to a writable project or app data directory. The analysis pipeline in this repository works with the curated structure under `final_results/`.

If new experimental data should be included in the same analysis workflow, it needs to be organized in a structure like this:

```text
final_results/
  Participant_X/
    2d/
      file.csv
    3d/
      file.csv
    logs/
      logfile.csv
```

## Running the analysis

The analysis can be started via the launcher or directly through `analysis.py`.

Full analysis:

```bash
python3 main.py --analysis
```

or

```bash
python3 analysis.py
```

If **no analysis flags** are provided, the **full analysis suite** is executed automatically.

## Available analysis parameters

The analysis can be restricted to specific metrics or plot groups.

### Core selection flags

- `--procrustes`
  Runs all Procrustes-related outputs.
  This includes in particular:
  - intersubject consistency plot based on Procrustes disparities
  - Procrustes dissimilarity matrix
  - Procrustes arrangement plot

- `--spearman`
  Generates the Spearman consistency heatmaps across participants.

- `--shepard`
  Generates the global Shepard diagrams and the individual Shepard plots per participant.

- `--knn`
  Generates the kNN preservation curves:
  - per condition
  - combined for `2D` vs. `3D`
  - individually per participant

- `--axis-variance`
  Generates the axis-variance analysis:
  - aggregated per condition
  - individually per participant

- `--rdm-similarity`
  Generates the cross-dimensional RDM outputs:
  - aggregated `2D` vs. `3D` RDM similarity
  - individual RDM scatterplots per participant

- `--arrangements`
  Generates individual arrangement plots for participants.

- `--dissimilarity`
  Generates individual dissimilarity matrices for participants.

### Participant selection

- `--participants 1,2,3`
- `--participants "[1,2,3]"`

Both forms are supported. In `zsh` or similar shells, the bracketed version should be quoted. The analysis will then be limited to the selected participants. This affects both aggregated outputs and individual plots.

### Additional options

- `--normalize-procrustes`
  Normalizes the Procrustes dissimilarity matrix to the range `[0, 1]`.

- `--named-participants`
  Uses folder names instead of anonymous labels such as `Participant 1`.

### Tolerant aliases

In addition to the canonical flags, the following aliases are also accepted:

- `--procruster` as an alias for `--procrustes`
- `--spreamean` as an alias for `--spearman`
- `--shaprd` as an alias for `--shepard`

## Analysis examples

Full analysis:

```bash
python3 main.py --analysis
```

Procrustes-only analysis:

```bash
python3 main.py --analysis --procrustes
```

Spearman and Shepard only:

```bash
python3 main.py --analysis --spearman --shepard
```

Analyze only participants `1`, `2`, and `3`:

```bash
python3 main.py --analysis --participants 1,2,3
```

Only kNN and axis-variance outputs for participants `1` to `3`:

```bash
python3 main.py --analysis --knn --axis-variance --participants 1,2,3
```

RDM comparison for a subset:

```bash
python3 main.py --analysis --rdm-similarity --participants 1,2,3,4
```

Direct analysis call without the launcher:

```bash
python3 analysis.py --procrustes --spearman --participants 1,2,3
```

## Metrics used in the analysis

The analysis in `analysis.py` directly follows the evaluation steps described in the thesis.

### 1. Relational dissimilarity

Each final arrangement is converted into a pairwise dissimilarity matrix using Euclidean distances. This matrix represents the relational structure of the arrangement independently of absolute position in space.

### 2. Spearman correlation

Spearman rank correlation compares dissimilarity structures either across participants or between `2D` and `3D` within the same participant. It is used to quantify how strongly relational structure is preserved.

### 3. Procrustes-aligned RMSE

The Procrustes analysis measures geometric similarity between configurations after optimal centering, scaling, and alignment. Lower values indicate stronger geometric agreement.

### 4. Shepard analysis

The Shepard diagrams compare reconstructed distances against a leave-one-out consensus structure. They act as a measure of global structural reliability.

### 5. k-nearest-neighbor preservation

The kNN analysis evaluates how well local neighborhood structure is preserved. This is especially useful for making fine-grained local differences between `2D` and `3D` visible.

### 6. Axis variance

This step analyzes how strongly the available axes are used. In `2D`, it shows how evenly the planar degrees of freedom are used. In `3D`, it reveals whether the additional dimension is actively and evenly used or whether specific axes dominate.

### 7. Cross-dimensional RDM similarity

This step directly compares the relational structures generated by `2D` and `3D` for each participant. It helps quantify how stable a participant's personal similarity structure remains across both interaction spaces.

## Analysis output

Generated figures are written to `analysis/`. Typical output locations are:

- `analysis/general/2d/`
- `analysis/general/3d/`
- `analysis/general/`
- `analysis/detailed/Participant_X/`

Depending on the selected parameters, only the relevant subset of outputs will be written there.

## Building a desktop app

The build script automatically detects the host operating system and creates the appropriate packaged app:

```bash
bash buildapp.sh
```

The build uses `main.py` as its entry point. On macOS it creates a `.app`, on Windows a `.exe`, and on Linux an executable bundle.

## Typical workflows

Start the experiment locally:

```bash
python3 main.py
```

Run the full analysis on the available results:

```bash
python3 main.py --analysis
```

Run the experiment and then analyze directly afterwards:

```bash
python3 main.py --both
```

Run the experiment and then compute only selected analysis modules:

```bash
python3 main.py --both --procrustes --shepard --participants 1,2,3
```

## Reproducibility note

This README is based on the bachelor thesis and the current implementation in the repository. The analysis pipeline reproduces the methodological core of the thesis, especially the steps for:

- relational dissimilarity
- Spearman-based consistency
- Procrustes-based geometric agreement
- Shepard analysis
- kNN preservation
- axis variance
- `2D` vs. `3D` RDM comparisons

This makes it possible to trace and reproduce the methodological core of the thesis directly from the repository.
