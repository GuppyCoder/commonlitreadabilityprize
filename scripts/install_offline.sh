#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELHOUSE_DIR="${BUNDLE_DIR}/wheelhouse"
MODEL_ROOT="${BUNDLE_DIR}/model"

if [[ ! -d "$WHEELHOUSE_DIR" ]]; then
  echo "[ERROR] wheelhouse not found in bundle directory: $WHEELHOUSE_DIR"
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] Activate your target virtual environment first."
  echo "Example: python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

echo "[1/3] Installing package and dependencies from local wheelhouse..."
python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" commonlit-readability

echo "[2/3] Configuring local model path for runtime..."
export COMMONLIT_OUTPUT_PATH="$MODEL_ROOT"

echo "[3/3] Sanity test..."
commonlit-predict "hello world" --timestamp latest --device cpu

echo ""
echo "Offline install complete."
echo "Model root: $MODEL_ROOT"
echo "For future sessions set:"
echo "  export COMMONLIT_OUTPUT_PATH=\"$MODEL_ROOT\""
