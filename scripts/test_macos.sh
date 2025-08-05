#!/usr/bin/env bash
# macOS test helper: run formatting/lint checks and unit/integration tests.
# Usage examples:
#   bash scripts/test_macos.sh                # run all
#   bash scripts/test_macos.sh unit           # unit tests only
#   bash scripts/test_macos.sh smoke          # smoke test data pipeline only
#   bash scripts/test_macos.sh style          # style/format checks only
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN"; exit 1; }

# Ensure venv exists
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtualenv not found ($VENV_DIR). Creating ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Ensure dev dependencies
python - <<'PY'
import sys, importlib, subprocess
pkgs = ["pytest", "pre-commit"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)
if missing:
    print("Installing missing:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

run_style() {
  echo "[style] Running pre-commit on all files..."
  if [[ -f ".pre-commit-config.yaml" ]]; then
    pre-commit run --all-files
  else
    echo "No .pre-commit-config.yaml found; skipping."
  fi
}

run_unit() {
  echo "[unit] Running pytest..."
  if compgen -G "tests/**.py" > /dev/null; then
    pytest -q
  else
    echo "No tests found under tests/; skipping."
  fi
}

run_smoke() {
  echo "[smoke] Running data pipeline smoke test..."
  python scripts/smoke_test_data_pipeline.py --data_dir data/ag_news --model_name bert-base-uncased --max_len 128 --batch_size 16 --num_workers 2
}

TARGET="${1:-all}"

case "$TARGET" in
  all)
    run_style
    run_unit
    run_smoke
    ;;
  style)
    run_style
    ;;
  unit)
    run_unit
    ;;
  smoke)
    run_smoke
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Valid targets: all | style | unit | smoke"
    exit 1
    ;;
esac

echo "Done."