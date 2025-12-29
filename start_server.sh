#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
  source "./venv/bin/activate"
else
  echo "[ERROR] .venv not found. Create venv first: python3 -m venv .venv"
  exit 1
fi

python ragkit.py --config config.yaml index
python api.py --host 0.0.0.0 --port 8000