#!/usr/bin/env bash
# Launch the local puzzle playground.
#
#   ./play.sh              -> http://127.0.0.1:7860
#   PUZZLE_PORT=7861 ./play.sh
#
# Uses the `lingbot` conda env (python 3.11 + torch + gradio) and vendors
# cloudpickle from .shim, which that env lacks but torch.load needs.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PUZZLE_PYTHON:-/media/skr/storage/conda_envs/lingbot/bin/python}"

[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }

export PYTHONPATH="$HERE/.shim${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$HERE/play.py"
