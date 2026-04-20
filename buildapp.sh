#!/usr/bin/env bash

set -euo pipefail

APP_NAME="3D inverse MDS"
ENTRY_SCRIPT="main.py"
ASSETS_DIR="assets"
PICTURES_DIR="pictures"
MAC_ICON="app_icon.icns"
WIN_ICON="app_icon.ico"
LINUX_ICON="app_icon.png"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No Python interpreter found."
  exit 1
fi

HOST_OS="$(uname -s)"
DATA_SEPARATOR=":"
ICON_FILE=""
DIST_TARGET="dist/${APP_NAME}"

case "${HOST_OS}" in
  Darwin)
    PLATFORM_NAME="macOS"
    DIST_TARGET="dist/${APP_NAME}.app"
    if [[ -f "${MAC_ICON}" ]]; then
      ICON_FILE="${MAC_ICON}"
    fi
    ;;
  Linux)
    PLATFORM_NAME="Linux"
    if [[ -f "${LINUX_ICON}" ]]; then
      ICON_FILE="${LINUX_ICON}"
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM_NAME="Windows"
    DATA_SEPARATOR=";"
    DIST_TARGET="dist/${APP_NAME}.exe"
    if [[ -f "${WIN_ICON}" ]]; then
      ICON_FILE="${WIN_ICON}"
    fi
    ;;
  *)
    echo "Unsupported operating system: ${HOST_OS}"
    exit 1
    ;;
esac

ensure_pip() {
  if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip is missing. Bootstrapping it via ensurepip..."
  "${PYTHON_BIN}" -m ensurepip --upgrade
}

install_with_pip() {
  if "${PYTHON_BIN}" -m pip install "$@"; then
    return 0
  fi

  if [[ "${PLATFORM_NAME}" != "Windows" ]]; then
    "${PYTHON_BIN}" -m pip install --break-system-packages "$@"
    return 0
  fi

  return 1
}

BROKEN_CONDA_META_FILES=()

restore_conda_meta() {
  if [[ "${#BROKEN_CONDA_META_FILES[@]}" -eq 0 ]]; then
    return 0
  fi

  for disabled_file in "${BROKEN_CONDA_META_FILES[@]}"; do
    if [[ -e "${disabled_file}" ]]; then
      mv "${disabled_file}" "${disabled_file%.codex-disabled}"
    fi
  done
}

disable_broken_conda_meta() {
  local broken_files

  broken_files="$("${PYTHON_BIN}" - <<'PY'
import json
import pathlib
import sys

meta_dir = pathlib.Path(sys.prefix) / "conda-meta"
required_keys = {"name", "version", "files", "depends"}

if meta_dir.is_dir():
    for path in sorted(meta_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(path)
            continue
        if not required_keys.issubset(data):
            print(path)
PY
)"

  if [[ -z "${broken_files}" ]]; then
    return 0
  fi

  while IFS= read -r broken_file; do
    [[ -z "${broken_file}" ]] && continue
    mv "${broken_file}" "${broken_file}.codex-disabled"
    BROKEN_CONDA_META_FILES+=("${broken_file}.codex-disabled")
  done <<< "${broken_files}"

  echo "Temporarily hid invalid conda-meta files for the build."
}

ensure_pip
trap restore_conda_meta EXIT

if ! "${PYTHON_BIN}" -m pip show pyinstaller >/dev/null 2>&1; then
  echo "PyInstaller is missing. Installing it now..."
  install_with_pip pyinstaller
fi

disable_broken_conda_meta

rm -rf build dist
rm -f ./*.spec

PYINSTALLER_ARGS=(
  -m PyInstaller
  --noconfirm
  --clean
  --windowed
  --name "${APP_NAME}"
  --add-data "${PICTURES_DIR}${DATA_SEPARATOR}${PICTURES_DIR}"
)

if [[ -d "${ASSETS_DIR}" ]]; then
  PYINSTALLER_ARGS+=(
    --add-data "${ASSETS_DIR}${DATA_SEPARATOR}${ASSETS_DIR}"
  )
fi

if [[ -n "${ICON_FILE}" ]]; then
  PYINSTALLER_ARGS+=(
    --icon "${ICON_FILE}"
  )
else
  echo "No matching icon found for ${PLATFORM_NAME}. Building without an icon."
fi

PYINSTALLER_ARGS+=("${ENTRY_SCRIPT}")

echo "Starting build for ${PLATFORM_NAME}..."
"${PYTHON_BIN}" "${PYINSTALLER_ARGS[@]}"

echo "Build finished: ${DIST_TARGET}"
