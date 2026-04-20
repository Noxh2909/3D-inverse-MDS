"""
Central launcher for experiment and analysis.

Default behavior:
    python main.py

Analysis only:
    python main.py --analysis

Experiment followed by analysis:
    python main.py --both
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from typing import Sequence


PROJECT_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("PySide6", "PySide6"),
    ("pyqtgraph", "pyqtgraph"),
    ("OpenGL", "PyOpenGL"),
    ("matplotlib", "matplotlib"),
    ("PIL", "Pillow"),
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
)


def _module_available(module_name: str) -> bool:
    """Return True if a module can be imported in the current interpreter."""
    return importlib.util.find_spec(module_name) is not None


def _ensure_pip_available() -> None:
    """Install pip into the current interpreter if it is missing."""
    try:
        import pip  # noqa: F401
    except ImportError:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)


def _pip_install(packages: list[str], use_break_system_packages: bool = False) -> None:
    """Install packages into the current interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if use_break_system_packages:
        cmd.insert(4, "--break-system-packages")
    subprocess.run(cmd, check=True)


def ensure_dependencies(skip_check: bool = False) -> None:
    """Install missing dependencies for experiment and analysis when needed."""
    if skip_check or getattr(sys, "frozen", False):
        return

    missing_packages = [
        package_name
        for module_name, package_name in PROJECT_DEPENDENCIES
        if not _module_available(module_name)
    ]

    if not missing_packages:
        return

    print("Missing packages found:", ", ".join(missing_packages))
    _ensure_pip_available()

    try:
        _pip_install(missing_packages)
    except subprocess.CalledProcessError:
        if sys.platform.startswith("win"):
            raise
        _pip_install(missing_packages, use_break_system_packages=True)


def run_experiment() -> int:
    """Start the experiment GUI."""
    import experiment

    return experiment.main()


def run_analysis(argv: Sequence[str] | None = None) -> int:
    """Run the analysis pipeline."""
    import analysis

    return analysis.main(argv)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(
        description="Launcher for the 3D inverse MDS project."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--analysis",
        action="store_true",
        help="Run analysis only.",
    )
    mode_group.add_argument(
        "--both",
        action="store_true",
        help="Run experiment first, then analysis.",
    )
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Skip automatic package checking.",
    )
    args, remaining = parser.parse_known_args()
    if remaining and not (args.analysis or args.both):
        parser.error(
            "additional arguments are only supported together with --analysis or --both"
        )
    return args, remaining


def main() -> int:
    """Program entry point."""
    args, analysis_args = parse_args()
    ensure_dependencies(skip_check=args.skip_dependency_check)

    if args.analysis:
        return run_analysis(analysis_args)

    if args.both:
        exit_code = run_experiment()
        if exit_code != 0:
            return exit_code
        return run_analysis(analysis_args)

    return run_experiment()


if __name__ == "__main__":
    raise SystemExit(main())
