# 3D inverse MDS

This repository contains the experiment application and analysis pipeline for the bachelor thesis **"Measuring perceived similarity using Inverse Multidimensional Scaling in three dimensions"** by Noah Kogge.

The project compares spatial similarity arrangements in `2D` and `3D`. Participants arrange the same set of abstract body-silhouette stimuli in both conditions. The resulting layouts are exported as coordinate data and analyzed with dissimilarity matrices, consistency measures, Procrustes alignment, Shepard diagrams, kNN preservation, axis-variance plots, and cross-dimensional RDM comparisons.

## Study design

- `12` abstract body-silhouette stimuli are used.
- Each participant can complete `2D`, `3D`, or both conditions.
- The intended thesis design is a within-subject comparison between `2D` and `3D`.
- The curated dataset in `final_results/` contains `7` participants.
- The task is open-ended: participants place stimuli according to perceived similarity.

This is a custom iMDS-inspired spatial arrangement application. It is not a one-to-one reproduction of a canonical multi-arrangement iMDS protocol.

## Project structure

```text
main.py                    # central launcher with dependency check and CLI modes
experiment.py              # QApplication entry point for the GUI launcher
analysis.py                # analysis pipeline and CLI arguments

ui/
  launcher.py              # launcher window, experiment controls, analysis GUI
  experiment_window.py     # experiment window and condition flow
  widgets.py               # scene widgets, tokens, OpenGL helpers

core/
  paths.py                 # resource and writable data paths
  export.py                # CSV export and participant abbreviation helpers
  logging.py               # experiment interaction logger

pictures/                  # stimulus images
final_results/             # curated participant data used by analysis
analysis/                  # generated analysis output
```

## Requirements

A recent Python 3 setup is recommended. The main dependencies are:

- `numpy`
- `PySide6`
- `pyqtgraph`
- `PyOpenGL`
- `matplotlib`
- `Pillow`
- `scipy`
- `tqdm`

`main.py` checks these packages on startup and installs missing dependencies into the active Python environment when possible.

To skip this check:

```bash
python3 main.py --skip-dependency-check
```

## Starting the application

Start the GUI launcher:

```bash
python3 main.py
```

Direct GUI entry point:

```bash
python3 experiment.py
```

The launcher stays open while experiments or analyses are started from it.

## GUI workflow

The launcher contains two main areas:

- `Experiment`: choose the start condition, select whether `2D`, `3D`, or both conditions should run, open the stimuli folder, and start the experiment.
- `Analysis`: open participant data, select participant numbers, configure analysis parameters, start the analysis, and open the generated analysis folder.

Participant numbers shown in the GUI follow the same folder ordering that `analysis.py` uses. This makes it possible to select participants in the analysis panel by number, for example `1,2,3`.

The analysis terminal at the bottom shows analysis output while keeping progress bars on one updating line instead of printing repeated progress lines.

## Data flow

During a local Python run, experiment data is written below the project directory:

```text
results/
  2d/
    <participant>_<timestamp>.csv
  3d/
    <participant>_<timestamp>.csv
logs/
  <participant>/
    <participant>_log_<timestamp>.csv
```

For analysis, data should be organized in `final_results/`:

```text
final_results/
  N.K/
    2d/
      N.K_20260211_004247.csv
    3d/
      N.K_20260211_004938.csv
    logs/
      N.K_log_20260211_003947.csv
```

Participant abbreviations are derived from the entered name, for example `Noah Kogge` becomes `N.K`.

In packaged builds, writable data is stored in the platform-specific app data folder:

- macOS: `~/Library/Application Support/3D inverse MDS`
- Windows: `%APPDATA%/3D inverse MDS`
- Linux: `$XDG_DATA_HOME/3D inverse MDS` or `~/.local/share/3D inverse MDS`

## Running analysis

Full analysis through the launcher GUI:

```bash
python3 main.py
```

Full analysis from the CLI:

```bash
python3 main.py --analysis
```

Direct analysis call:

```bash
python3 analysis.py
```

If no metric flags are provided, `analysis.py` runs the full analysis suite.

## Launcher CLI modes

```bash
python3 main.py
```

Starts the GUI launcher.

```bash
python3 main.py --analysis
```

Runs the analysis pipeline only. Additional analysis arguments can be appended.

```bash
python3 main.py --both
```

Starts the GUI first and runs analysis after the GUI exits. Additional analysis arguments can be appended.

## Analysis parameters

The GUI exposes these parameters without the leading `--`. The CLI uses the flags listed below.

### Participant and labeling options

- `--participants 1,2,3`: analyze only selected participants.
- `--participants "[1,2,3]"`: equivalent quoted form for shells such as `zsh`.
- `--named-participants`: use participant folder names instead of anonymous labels.
- `--normalize-procrustes`: normalize the Procrustes dissimilarity matrix to `[0, 1]`. This is the default.
- `--raw-procrustes`: use raw Procrustes dissimilarity values.

### Plot and label sizes

- `--font-size`
- `--title-font-size`
- `--scale-number-size`
- `--stimulus-number-size`
- `--rdm-axis-font-size`
- `--rdm-stimulus-number-size` alias for `--rdm-axis-font-size`
- `--rdm-scale-font-size`
- `--rdm-scale-number-size` alias for `--rdm-scale-font-size`
- `--rdm-legend-font-size`
- `--pro-rdm-axis-font-size`
- `--pro-rdm-scale-font-size`
- `--pro-rdm-legend-font-size`
- `--spr-rdm-axis-font-size`
- `--spr-rdm-scale-font-size`
- `--spr-rdm-legend-font-size`
- `--arrangement-axis-font-size`
- `--matrix-number-size`

### Analysis modules

- `--procrustes`: Procrustes-based outputs.
- `--spearman`: Spearman consistency plots.
- `--shepard`: global and individual Shepard diagrams.
- `--knn`: k-nearest-neighbor preservation plots.
- `--axis-variance`: axis-variance plots.
- `--rdm-similarity`: `2D` vs. `3D` RDM comparisons.
- `--arrangements`: participant arrangement plots.
- `--dissimilarity`: participant dissimilarity matrices.

The following tolerant aliases are also accepted:

- `--procruster` for `--procrustes`
- `--spreamean` for `--spearman`
- `--shaprd` for `--shepard`

## Analysis examples

Full analysis:

```bash
python3 main.py --analysis
```

Procrustes only:

```bash
python3 main.py --analysis --procrustes
```

Spearman and Shepard only:

```bash
python3 main.py --analysis --spearman --shepard
```

Analyze participants `1`, `2`, and `3`:

```bash
python3 main.py --analysis --participants 1,2,3
```

Run selected modules with custom plot sizes:

```bash
python3 main.py --analysis --knn --axis-variance --font-size 14 --title-font-size 16
```

Use raw Procrustes values and named participants:

```bash
python3 main.py --analysis --procrustes --raw-procrustes --named-participants
```

## Analysis output

Generated figures and tables are written to `analysis/`. Typical output locations are:

- `analysis/general/2d/`
- `analysis/general/3d/`
- `analysis/general/`
- `analysis/detailed/Participant_X/`

Only the selected analysis modules write output.

## Metrics

The analysis pipeline includes:

- relational dissimilarity from Euclidean distances
- Spearman rank consistency
- Procrustes-aligned geometric agreement
- Shepard diagrams against consensus structures
- k-nearest-neighbor preservation
- axis variance in `2D` and `3D`
- cross-dimensional RDM similarity

## Building a desktop app

The build script uses `main.py` as the entry point:

```bash
bash buildapp.sh
```

On macOS it creates a `.app`, on Windows an `.exe`, and on Linux an executable bundle.

## License

Copyright (c) 2026 Noah Kogge. All rights reserved.

This project is proprietary. No permission is granted to copy, modify, distribute, sublicense, publish, commercially use, or create derivative works without prior written permission from the copyright holder. See `LICENSE` for the full terms.
