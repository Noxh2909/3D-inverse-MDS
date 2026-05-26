"""Persistent launcher hub for experiment and analysis workflows."""

from __future__ import annotations

import sys
import re
import pathlib

from PySide6.QtCore import QPoint, QProcess, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from core.paths import APP_NAME, app_resource_dir, stimuli_dir
from ui.experiment_window import ExperimentWindow

ANALYSIS_METRICS = (
    ("Procrustes", "--procrustes", "Run Procrustes-based outputs."),
    ("Spearman", "--spearman", "Generate Spearman consistency plots."),
    ("Shepard", "--shepard", "Generate global and individual Shepard diagrams."),
    ("kNN", "--knn", "Generate kNN preservation plots."),
    ("Axis variance", "--axis-variance", "Generate axis-variance plots."),
    ("RDM similarity", "--rdm-similarity", "Generate 2D-vs-3D RDM comparisons."),
    ("Arrangements", "--arrangements", "Generate participant arrangement plots."),
    ("Dissimilarity", "--dissimilarity", "Generate participant dissimilarity matrices."),
)

ANALYSIS_NUMBER_PARAMS = (
    ("Font size", "--font-size", "Base font size for plot labels."),
    ("Title font size", "--title-font-size", "Font size for plot titles."),
    ("Scale number size", "--scale-number-size", "Font size for ordinary axis scale numbers."),
    ("Stimulus number size", "--stimulus-number-size", "Font size for ordinary stimulus number labels."),
    (
        "RDM axis font size",
        "--rdm-axis-font-size",
        "Axis/tick/stimulus font size for Spearman, Procrustes and dissimilarity RDM plots.",
    ),
    (
        "RDM scale font size",
        "--rdm-scale-font-size",
        "Colorbar scale-number font size for Spearman, Procrustes and dissimilarity RDM plots.",
    ),
    (
        "RDM legend font size",
        "--rdm-legend-font-size",
        "Legend font size for Spearman and Procrustes RDM mean values.",
    ),
    (
        "Procrustes RDM axis font size",
        "--pro-rdm-axis-font-size",
        "Axis/tick font size for Procrustes dissimilarity RDM plots.",
    ),
    (
        "Procrustes RDM scale font size",
        "--pro-rdm-scale-font-size",
        "Colorbar scale-number font size for Procrustes dissimilarity RDM plots.",
    ),
    (
        "Procrustes RDM legend font size",
        "--pro-rdm-legend-font-size",
        "Legend font size for Procrustes mean disparity.",
    ),
    (
        "Spearman RDM axis font size",
        "--spr-rdm-axis-font-size",
        "Axis/tick font size for Spearman consistency RDM plots.",
    ),
    (
        "Spearman RDM scale font size",
        "--spr-rdm-scale-font-size",
        "Colorbar scale-number font size for Spearman consistency RDM plots.",
    ),
    (
        "Spearman RDM legend font size",
        "--spr-rdm-legend-font-size",
        "Legend font size for Spearman mean rho.",
    ),
    (
        "Arrangement axis font size",
        "--arrangement-axis-font-size",
        "Axis/tick font size for per-participant arrangement plots.",
    ),
    (
        "Matrix number size",
        "--matrix-number-size",
        "Font size for numeric annotations outside RDM heatmaps.",
    ),
)


class InfoLabel(QLabel):
    """Immediate hover info label for analysis parameter descriptions."""

    def __init__(self, description: str):
        super().__init__("i")
        self.description = description
        self.setFixedSize(16, 16)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setToolTip(description)
        self.setStyleSheet(
            "QLabel { color: white; background: #555; border-radius: 8px; "
            "font-size: 11px; font-weight: 700; }"
        )

    def _show_info(self) -> None:
        pos = self.mapToGlobal(QPoint(self.width() + 8, self.height() // 2))
        QToolTip.showText(pos, self.description, self)

    def enterEvent(self, event):
        self._show_info()
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        self._show_info()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)



class LauncherWindow(QMainWindow):
    """Persistent hub for starting experiments and running analysis."""

    def __init__(self):
        super().__init__()
        self.experiment_window: ExperimentWindow | None = None
        self.analysis_process: QProcess | None = None
        self.metric_combos: dict[str, QComboBox] = {}
        self.number_inputs: dict[str, QLineEdit] = {}
        self.plot_params_container: QWidget | None = None
        self.plot_params_toggle: QPushButton | None = None
        self._analysis_log_lines: list[str] = []
        self._analysis_progress_line = ""
        self._analysis_output_buffer = ""

        self.setWindowTitle(APP_NAME)
        self.setFixedSize(1060, 760)
        self._build_ui()
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply a compact macOS-like style to the launcher controls."""
        self.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid #d2d2d7;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 8px;
                font-size: 14px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #d9d9df;
                color: #1d1d1f;
                border: 1px solid #b7b8c0;
                border-radius: 8px;
                padding: 4px 9px;
                min-height: 18px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e1e1e6;
                border-color: #aeb0b8;
            }
            QPushButton:pressed {
                background-color: #cacad1;
            }
            QPushButton:disabled {
                color: #8e8e93;
                background-color: #d7d7dc;
                border-color: #cfcfd6;
            }
            QPushButton#primaryButton {
                background-color: #007aff;
                color: white;
                border: 1px solid #0071eb;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background-color: #0a84ff;
            }
            QPushButton#primaryButton:pressed {
                background-color: #006edb;
            }
            QPushButton#primaryButton:disabled {
                background-color: #9ac7ff;
                border-color: #9ac7ff;
                color: rgba(255, 255, 255, 0.78);
            }
            QLineEdit,
            QComboBox {
                background-color: #dedee4;
                color: #1d1d1f;
                border: 1px solid #b7b8c0;
                border-radius: 7px;
                padding: 2px 6px;
                min-height: 18px;
                font-size: 13px;
                selection-background-color: #007aff;
                selection-color: white;
            }
            QLineEdit:focus,
            QComboBox:focus {
                border: 2px solid #007aff;
                padding: 1px 5px;
                background-color: #e8e8ee;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox QAbstractItemView {
                background-color: #dedee4;
                border: 1px solid #b7b8c0;
                border-radius: 8px;
                padding: 4px;
                selection-background-color: #007aff;
                selection-color: white;
            }
            QPlainTextEdit {
                background-color: #dedee4;
                color: #1d1d1f;
                border: 1px solid #b7b8c0;
                border-radius: 8px;
                padding: 5px;
                font-size: 11px;
                selection-background-color: #007aff;
                selection-color: white;
            }
            QPlainTextEdit#analysisLog {
                background-color: #111111;
                color: #eeeeee;
                border: 1px solid #2c2c2e;
                font-family: Menlo, Consolas, monospace;
            }
            QRadioButton,
            QCheckBox {
                spacing: 6px;
                font-size: 13px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QPushButton#collapsibleHeader {
                background-color: #8e8e93;
                border: 1px solid #85858b;
                border-radius: 7px;
                color: white;
                padding: 3px 7px;
                min-height: 17px;
                text-align: left;
                font-weight: 700;
            }
            QPushButton#collapsibleHeader:hover {
                background-color: #7f7f85;
                color: white;
            }
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("3D inverse MDS")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: white;")
        layout.addWidget(title)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        layout.addLayout(top_row)

        top_row.addWidget(self._build_experiment_panel(), 1)
        top_row.addWidget(self._build_analysis_panel(), 1)

        self.analysis_log = QPlainTextEdit()
        self.analysis_log.setObjectName("analysisLog")
        self.analysis_log.setReadOnly(True)
        self.analysis_log.setPlaceholderText("Analysis output appears here.")
        self.analysis_log.setFixedHeight(125)
        layout.addWidget(self.analysis_log)

    def _build_experiment_panel(self) -> QGroupBox:
        box = QGroupBox("Experiment")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(9)

        condition_label = QLabel("Start experiment condition")
        condition_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(condition_label)

        condition_row = QHBoxLayout()
        condition_row.setSpacing(12)
        self.radio_2d = QRadioButton("2D Condition")
        self.radio_3d = QRadioButton("3D Condition")
        self.radio_2d.setChecked(True)
        condition_row.addWidget(self.radio_2d)
        condition_row.addWidget(self.radio_3d)
        condition_row.addStretch(1)
        layout.addLayout(condition_row)

        run_label = QLabel("Run conditions")
        run_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(run_label)

        run_row = QHBoxLayout()
        run_row.setSpacing(12)
        self.run_2d_cb = QCheckBox("2D")
        self.run_3d_cb = QCheckBox("3D")
        self.run_2d_cb.setChecked(True)
        self.run_3d_cb.setChecked(True)
        run_row.addWidget(self.run_2d_cb)
        run_row.addWidget(self.run_3d_cb)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self.stimuli_btn = QPushButton("Load Stimuli Set")
        self.stimuli_btn.clicked.connect(
            lambda: self._open_folder(stimuli_dir(), self.experiment_status, "Pictures folder opened.")
        )
        folder_row.addWidget(self.stimuli_btn)
        folder_row.addStretch(1)
        layout.addLayout(folder_row)

        self.start_experiment_btn = QPushButton("Start Experiment")
        self.start_experiment_btn.setObjectName("primaryButton")
        self.start_experiment_btn.clicked.connect(self.start_experiment)
        layout.addWidget(self.start_experiment_btn)

        participant_label = QLabel("Participant data numbers")
        participant_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(participant_label)

        self.participant_summary = QPlainTextEdit()
        self.participant_summary.setObjectName("participantSummary")
        self.participant_summary.setReadOnly(True)
        self.participant_summary.setFixedHeight(155)
        layout.addWidget(self.participant_summary)
        self._refresh_participant_summary()

        self.experiment_status = QLabel("")
        self.experiment_status.setWordWrap(True)
        self.experiment_status.setStyleSheet("color: #555;")
        layout.addWidget(self.experiment_status)
        layout.addStretch(1)
        return box

    def _build_analysis_panel(self) -> QGroupBox:
        box = QGroupBox("Analysis")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(9)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        self.participant_data_btn = QPushButton("Participant Data")
        self.participant_data_btn.clicked.connect(
            lambda: self._open_folder(
                app_resource_dir() / "final_results",
                self.analysis_status,
                "Participant data folder opened.",
            )
        )
        self.analysis_folder_btn = QPushButton("Analysis Folder")
        self.analysis_folder_btn.clicked.connect(
            lambda: self._open_folder(
                app_resource_dir() / "analysis",
                self.analysis_status,
                "Analysis folder opened.",
            )
        )
        folder_row.addWidget(self.participant_data_btn)
        folder_row.addWidget(self.analysis_folder_btn)
        layout.addLayout(folder_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        params_widget = QWidget()
        params_layout = QGridLayout(params_widget)
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(6)
        params_layout.setContentsMargins(0, 0, 4, 0)

        row = 0
        params_layout.addWidget(self._section_label("General parameters"), row, 0, 1, 3)
        row += 1

        self.participants_input = QLineEdit()
        self.participants_input.setPlaceholderText("All, e.g. 1,2,3")
        row = self._add_parameter_row(
            params_layout,
            row,
            "Participants",
            self.participants_input,
            "Participant list, e.g. 1,2,3 or [1,2,3]. Empty means all participants.",
        )

        self.procrustes_scale_combo = QComboBox()
        self.procrustes_scale_combo.addItems(("Normalize", "Raw"))
        row = self._add_parameter_row(
            params_layout,
            row,
            "Procrustes scale",
            self.procrustes_scale_combo,
            "Normalize scales the Procrustes dissimilarity matrix to [0, 1]. Raw keeps original values.",
        )

        self.named_participants_combo = QComboBox()
        self.named_participants_combo.addItems(("Disabled", "Enabled"))
        row = self._add_parameter_row(
            params_layout,
            row,
            "Named participants",
            self.named_participants_combo,
            "Uses folder names instead of anonymous labels such as Participant 1.",
        )

        row += 1
        self.plot_params_toggle = QPushButton("> Plot and label sizes")
        self.plot_params_toggle.setObjectName("collapsibleHeader")
        self.plot_params_toggle.clicked.connect(self._toggle_plot_params)
        params_layout.addWidget(self.plot_params_toggle, row, 0, 1, 3)
        row += 1

        self.plot_params_container = QWidget()
        plot_params_layout = QGridLayout(self.plot_params_container)
        plot_params_layout.setHorizontalSpacing(8)
        plot_params_layout.setVerticalSpacing(6)
        plot_params_layout.setContentsMargins(14, 0, 0, 0)
        plot_row = 0
        for label, flag, description in ANALYSIS_NUMBER_PARAMS:
            input_field = QLineEdit()
            input_field.setPlaceholderText("Default")
            input_field.setValidator(QIntValidator(1, 999, input_field))
            input_field.setMaximumWidth(86)
            self.number_inputs[flag] = input_field
            plot_row = self._add_parameter_row(
                plot_params_layout,
                plot_row,
                label,
                input_field,
                description,
            )
        plot_params_layout.setColumnStretch(0, 1)
        plot_params_layout.setColumnStretch(2, 1)
        self.plot_params_container.hide()
        params_layout.addWidget(self.plot_params_container, row, 0, 1, 3)
        row += 1

        row += 1
        params_layout.addWidget(self._section_label("Analysis tools"), row, 0, 1, 3)
        row += 1

        for label, flag, description in ANALYSIS_METRICS:
            combo = QComboBox()
            combo.addItems(("Enabled", "Disabled"))
            combo.setCurrentText("Enabled")
            combo.setMinimumWidth(96)
            self.metric_combos[flag] = combo
            row = self._add_parameter_row(
                params_layout,
                row,
                label,
                combo,
                description,
            )

        params_layout.setColumnStretch(0, 1)
        params_layout.setColumnStretch(2, 1)
        scroll.setWidget(params_widget)
        layout.addWidget(scroll, 1)

        self.start_analysis_btn = QPushButton("Start Analysis")
        self.start_analysis_btn.setObjectName("primaryButton")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.start_analysis_btn)

        self.analysis_status = QLabel("")
        self.analysis_status.setWordWrap(True)
        self.analysis_status.setStyleSheet("color: #555;")
        layout.addWidget(self.analysis_status)
        return box

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700; margin-top: 4px;")
        return label

    def _info_label(self, description: str) -> QLabel:
        return InfoLabel(description)

    def _toggle_plot_params(self) -> None:
        if self.plot_params_container is None or self.plot_params_toggle is None:
            return
        expanded = self.plot_params_container.isHidden()
        self.plot_params_container.setVisible(expanded)
        prefix = "v" if expanded else ">"
        self.plot_params_toggle.setText(f"{prefix} Plot and label sizes")

    def _add_parameter_row(
        self,
        layout: QGridLayout,
        row: int,
        label_text: str,
        widget: QWidget,
        description: str,
    ) -> int:
        label = QLabel(label_text)
        label.setToolTip(description)
        widget.setToolTip(description)
        layout.addWidget(label, row, 0)
        layout.addWidget(self._info_label(description), row, 1)
        layout.addWidget(widget, row, 2)
        return row + 1

    def _participant_rows(self) -> list[tuple[int, str, int, int]]:
        """Return participant folder rows using analysis.py's numbering order."""
        final_results_dir = app_resource_dir() / "final_results"
        if not final_results_dir.is_dir():
            return []

        participant_dirs = sorted(
            path
            for path in final_results_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        rows = []
        for idx, folder in enumerate(participant_dirs, start=1):
            count_2d = len(list((folder / "2d").glob("*.csv")))
            count_3d = len(list((folder / "3d").glob("*.csv")))
            rows.append((idx, folder.name, count_2d, count_3d))
        return rows

    def _refresh_participant_summary(self) -> None:
        rows = self._participant_rows()
        if not rows:
            self.participant_summary.setPlainText(
                "No participant folders found in final_results."
            )
            return

        with_data = sum(1 for _, _, count_2d, count_3d in rows if count_2d or count_3d)
        lines = [
            f"{with_data} participants with data / {len(rows)} folders",
            "Use these numbers in Analysis > Participants:",
        ]
        for idx, name, count_2d, count_3d in rows:
            status = []
            if count_2d:
                status.append(f"2D x{count_2d}")
            if count_3d:
                status.append(f"3D x{count_3d}")
            if not status:
                status.append("no CSV")
            lines.append(f"{idx}: {name} ({', '.join(status)})")
        self.participant_summary.setPlainText("\n".join(lines))

    def _open_folder(self, folder: pathlib.Path, status_label: QLabel, success: str) -> None: #type:ignore
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            status_label.setText(f"Could not create folder: {exc}")
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        status_label.setText(success if opened else f"Could not open folder: {folder}")

    def _selected_experiment_conditions(self) -> list[str]:
        selected = []
        if self.run_2d_cb.isChecked():
            selected.append("2d")
        if self.run_3d_cb.isChecked():
            selected.append("3d")
        return selected

    def start_experiment(self) -> None:
        if self.experiment_window is not None:
            self.experiment_window.raise_()
            self.experiment_window.activateWindow()
            return

        selected = self._selected_experiment_conditions()
        if not selected:
            self.experiment_status.setText("Select at least one condition.")
            return

        requested_start = "2d" if self.radio_2d.isChecked() else "3d"
        first_condition = requested_start if requested_start in selected else selected[0]
        condition_order = [first_condition] + [
            condition for condition in selected if condition != first_condition
        ]

        self.experiment_window = ExperimentWindow(
            condition=first_condition,
            conditions_to_run=condition_order,
            on_finished=self._experiment_finished,
        )
        readable = " then ".join(condition.upper() for condition in condition_order)
        self.experiment_status.setText(f"Running experiment: {readable}")
        self.start_experiment_btn.setEnabled(False)

    def _experiment_finished(self) -> None:
        self.experiment_window = None
        self.start_experiment_btn.setEnabled(True)
        self.experiment_status.setText("Experiment finished.")
        self._refresh_participant_summary()
        self.raise_()
        self.activateWindow()

    def _analysis_args(self) -> list[str]:
        args: list[str] = []

        participants = self.participants_input.text().strip()
        if participants:
            args.extend(["--participants", participants])

        if self.procrustes_scale_combo.currentText() == "Normalize":
            args.append("--normalize-procrustes")
        else:
            args.append("--raw-procrustes")

        if self.named_participants_combo.currentText() == "Enabled":
            args.append("--named-participants")

        for flag, input_field in self.number_inputs.items():
            value = input_field.text().strip()
            if value:
                args.extend([flag, value])

        enabled_metric_flags = [
            flag
            for flag, combo in self.metric_combos.items()
            if combo.currentText() == "Enabled"
        ]
        args.extend(enabled_metric_flags)
        return args

    def start_analysis(self) -> None:
        if self.analysis_process is not None:
            self.analysis_status.setText("Analysis is already running.")
            return

        enabled_count = sum(
            1 for combo in self.metric_combos.values() if combo.currentText() == "Enabled"
        )
        if enabled_count == 0:
            self.analysis_status.setText("Enable at least one analysis tool.")
            return

        script = app_resource_dir() / "analysis.py"
        if not script.exists():
            self.analysis_status.setText(f"analysis.py not found: {script}")
            return

        args = self._analysis_args()
        self._analysis_log_lines = ["Running analysis with selected parameters..."]
        self._analysis_progress_line = ""
        self._analysis_output_buffer = ""
        self._render_analysis_log()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([str(script), *args])
        process.setWorkingDirectory(str(app_resource_dir()))
        process.readyReadStandardOutput.connect(self._read_analysis_stdout)
        process.readyReadStandardError.connect(self._read_analysis_stderr)
        process.finished.connect(self._analysis_finished)

        self.analysis_process = process
        self.start_analysis_btn.setEnabled(False)
        self.analysis_status.setText("Analysis running...")
        process.start()

    def _clean_analysis_text(self, text: str) -> str:
        """Strip ANSI terminal control sequences from subprocess output."""
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)

    def _looks_like_progress(self, line: str) -> bool:
        return any(marker in line for marker in ("%|", "it/s", "plot/s", "Generating plots"))

    def _render_analysis_log(self) -> None:
        lines = list(self._analysis_log_lines)
        if self._analysis_progress_line:
            lines.append(self._analysis_progress_line)
        self.analysis_log.setPlainText("\n".join(lines))
        self.analysis_log.verticalScrollBar().setValue(
            self.analysis_log.verticalScrollBar().maximum()
        )

    def _append_analysis_line(self, line: str) -> None:
        line = self._clean_analysis_text(line).strip()
        if not line:
            return
        self._analysis_log_lines.append(line)
        self._analysis_log_lines = self._analysis_log_lines[-80:]
        self._render_analysis_log()

    def _set_analysis_progress(self, line: str) -> None:
        line = self._clean_analysis_text(line).strip()
        if not line:
            return
        self._analysis_progress_line = line
        self._render_analysis_log()

    def _handle_analysis_output(self, text: str) -> None:
        text = self._clean_analysis_text(text).replace("\r\n", "\n")
        for char in text:
            if char == "\r":
                line = self._analysis_output_buffer.strip()
                if line:
                    self._set_analysis_progress(line)
                self._analysis_output_buffer = ""
                continue

            if char == "\n":
                line = self._analysis_output_buffer.strip()
                if line:
                    if self._looks_like_progress(line):
                        self._set_analysis_progress(line)
                    else:
                        self._append_analysis_line(line)
                self._analysis_output_buffer = ""
                continue

            self._analysis_output_buffer += char

        pending = self._analysis_output_buffer.strip()
        if pending and self._looks_like_progress(pending):
            self._set_analysis_progress(pending)
            self._analysis_output_buffer = ""

    def _read_analysis_stdout(self) -> None:
        if self.analysis_process is None:
            return
        text = bytes(self.analysis_process.readAllStandardOutput()).decode( #type: ignore
            "utf-8", errors="replace"
        )
        if text:
            self._handle_analysis_output(text)

    def _read_analysis_stderr(self) -> None:
        if self.analysis_process is None:
            return
        text = bytes(self.analysis_process.readAllStandardError()).decode( #type: ignore
            "utf-8", errors="replace"
        )
        if text:
            self._handle_analysis_output(text)

    def _analysis_finished(self, exit_code: int, _exit_status) -> None:
        if self._analysis_output_buffer.strip():
            self._append_analysis_line(self._analysis_output_buffer)
            self._analysis_output_buffer = ""
        self.analysis_process = None
        self.start_analysis_btn.setEnabled(True)
        if exit_code == 0:
            self.analysis_status.setText("Analysis finished. Results are in the analysis folder.")
            self._set_analysis_progress("Analysis finished.")
        else:
            self.analysis_status.setText(f"Analysis failed with exit code {exit_code}.")
            self._append_analysis_line(f"Analysis failed with exit code {exit_code}.")
