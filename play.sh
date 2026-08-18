#!/usr/bin/env bash
# Launch the local puzzle playground.
#
#   ./play.sh              -> http://127.0.0.1:7860
#   PUZZLE_PORT=7861 ./play.sh
#
# Uses the `trex` conda env (python 3.11 + torch + gradio + cloudpickle).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PUZZLE_PYTHON:-/home/satya/anaconda3/envs/trex/bin/python}"

[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 1; }

exec "$PY" "$HERE/play.py"
