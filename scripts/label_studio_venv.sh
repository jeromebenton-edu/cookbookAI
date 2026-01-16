#!/usr/bin/env bash
# Helper to create and run a dedicated Label Studio venv so it doesn't collide
# with the main OCR/weak-labeling environment (which pins numpy 1.x).
# Usage:
#   ./scripts/label_studio_venv.sh start
#   ./scripts/label_studio_venv.sh shell

set -euo pipefail

VENV_DIR="${HOME}/.venvs/labelstudio"
PY_BIN="${PY_BIN:-python3}"

case "${1:-}" in
  start)
    "${PY_BIN}" -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install label-studio
    label-studio start
    ;;
  shell)
    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
      echo "Venv not found at ${VENV_DIR}. Run: ./scripts/label_studio_venv.sh start" >&2
      exit 1
    fi
    source "${VENV_DIR}/bin/activate"
    exec "$SHELL"
    ;;
  *)
    echo "Usage: $0 {start|shell}" >&2
    exit 1
    ;;
 esac
