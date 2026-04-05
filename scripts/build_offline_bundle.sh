#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] Activate your virtual environment first."
  echo "Example: source .venv/bin/activate"
  exit 1
fi

TIMESTAMP="${1:-}"
MODEL_NAME="${2:-roberta-base}"
BUNDLE_ROOT="${3:-offline_bundle}"

if [[ -z "$TIMESTAMP" ]]; then
  echo "Usage: $0 <timestamp> [model_name] [bundle_root]"
  echo "Example: $0 20260404-172000 roberta-base offline_bundle"
  exit 1
fi

MODEL_PATH="output/${TIMESTAMP}/${MODEL_NAME}"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] Model path not found: $MODEL_PATH"
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${BUNDLE_ROOT}/commonlit-offline-${TIMESTAMP}-${STAMP}"
WHEELHOUSE_DIR="${OUT_DIR}/wheelhouse"
MODEL_DIR="${OUT_DIR}/model"

mkdir -p "$WHEELHOUSE_DIR" "$MODEL_DIR"

echo "[1/5] Building package wheel..."
python -m pip wheel . --wheel-dir "$WHEELHOUSE_DIR"

echo "[2/5] Downloading dependency wheels for offline install..."
python -m pip download -r requirements-modern.txt --dest "$WHEELHOUSE_DIR"

echo "[3/5] Downloading package dependencies from pyproject metadata..."
python -m pip download . --dest "$WHEELHOUSE_DIR"

echo "[4/5] Copying model artifacts from ${MODEL_PATH}..."
mkdir -p "$MODEL_DIR/$TIMESTAMP"
cp -R "$MODEL_PATH" "$MODEL_DIR/$TIMESTAMP/"

cp scripts/install_offline.sh "$OUT_DIR/install_offline.sh"
chmod +x "$OUT_DIR/install_offline.sh"

cat > "$OUT_DIR/README_OFFLINE.md" <<EOF
# CommonLit Offline Bundle

This bundle contains:
- wheelhouse/ (all required Python wheels)
- model/${MODEL_NAME}/ (tokenizers/config/checkpoints)
- install_offline.sh (offline install + sanity test)

## On offline target machine
1) Create and activate a Python venv (same major/minor Python recommended):
   python3 -m venv .venv
   source .venv/bin/activate

2) Run installer:
   ./install_offline.sh

3) Run prediction:
   commonlit-predict "hello world" --timestamp ${TIMESTAMP} --model-name ${MODEL_NAME} --device cpu

If needed, set output path to this bundle:
   export COMMONLIT_OUTPUT_PATH="$(pwd)/model"
EOF

echo "[5/5] Creating tarball..."
TARBALL="${OUT_DIR}.tar.gz"
tar -czf "$TARBALL" -C "$BUNDLE_ROOT" "$(basename "$OUT_DIR")"

echo ""
echo "Offline bundle ready: $OUT_DIR"
echo "Tarball ready: $TARBALL"
