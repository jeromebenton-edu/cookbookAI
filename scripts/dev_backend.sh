#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backend"

if [[ -f "../.venv/bin/activate" ]]; then
  source ../.venv/bin/activate
fi

export FRONTEND_ORIGIN="${FRONTEND_ORIGIN:-http://localhost:3000}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "Starting backend on ${HOST}:${PORT}"
echo "FRONTEND_ORIGIN=${FRONTEND_ORIGIN}"
echo "MODEL_DIR=${MODEL_DIR:-auto}"
echo "DATASET_DIR=${DATASET_DIR:-auto}"
echo "WARMUP=${WARMUP:-true}"
echo "CORS origins: ${FRONTEND_ORIGIN}"

uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload
