#!/usr/bin/env bash
# macOS run helper: activate venv and run common project entrypoints.
# Usage examples:
#   bash scripts/run_macos.sh smoke           # run data pipeline smoke test
#   bash scripts/run_macos.sh download --subset 2000 --force
#   bash scripts/run_macos.sh train --epochs 1 --batch_size 16  # if scripts/train.py supports these
#   PYTHON_BIN=python3.11 bash scripts/run_macos.sh smoke
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

# Ensure minimal deps
python - <<'PY'
import sys, importlib
pkgs = ["torch", "datasets", "pyarrow", "transformers"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)
if missing:
    print("Installing missing:", " ".join(missing))
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

if [[ $# -lt 1 ]]; then
  cat <<EOF
Usage:
  bash scripts/run_macos.sh <command> [args...]

Commands:
  download [--subset N] [--force]   Download/cached AG News parquet
  smoke [--data_dir DIR] [...]      Run data pipeline smoke test
  train [args...]                   Run training (delegates to scripts/train.py)
EOF
  exit 1
fi

CMD="$1"; shift || true

case "$CMD" in
  download)
    python scripts/download_ag_news.py "$@"
    ;;
  smoke)
    python scripts/smoke_test_data_pipeline.py "$@"
    ;;
  train)
    if [[ -f "scripts/train.py" ]]; then
      python scripts/train.py "$@"
    else
      echo "scripts/train.py not found"; exit 1
    fi
    ;;
  *)
    echo "Unknown command: $CMD"; exit 1;;
esac