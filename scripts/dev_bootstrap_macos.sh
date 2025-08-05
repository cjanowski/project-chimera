#!/usr/bin/env bash
# macOS bootstrap script: set up venv, install deps, and prepare data (AG News).
# Usage:
#   bash scripts/dev_bootstrap_macos.sh
#   bash scripts/dev_bootstrap_macos.sh --subset 2000
#   bash scripts/dev_bootstrap_macos.sh --force --subset 2000
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

SUBSET=""
FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      SUBSET="$2"; shift 2;;
    --force)
      FORCE="--force"; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[1/6] Creating virtual environment ($VENV_DIR) ..."
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[2/6] Upgrading pip and wheel ..."
pip install --upgrade pip wheel

echo "[3/6] Installing project dependencies ..."
# Prefer pyproject, else fallback to individual deps
if [[ -f "pyproject.toml" ]]; then
  pip install -e .
else
  pip install torch datasets pyarrow transformers
fi

echo "[4/6] Installing dev tools (pre-commit, pytest) ..."
pip install pre-commit pytest

echo "[5/6] Installing git hooks ..."
if [[ -f ".pre-commit-config.yaml" ]]; then
  pre-commit install
fi

echo "[6/6] Preparing AG News data ..."
CMD=(python scripts/download_ag_news.py $FORCE)
if [[ -n "$SUBSET" ]]; then
  CMD+=("--subset" "$SUBSET")
fi
echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Bootstrap complete."
echo "Activate venv: source $VENV_DIR/bin/activate"
echo "Run smoke test: python scripts/smoke_test_data_pipeline.py --data_dir data/ag_news --model_name bert-base-uncased --max_len 128 --batch_size 16 --num_workers 2"