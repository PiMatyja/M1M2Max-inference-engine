#!/bin/bash
# start.sh: runs the OpenAI-compatible server on 127.0.0.1:$STRATA_PORT (default 8000) in the foreground.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${STRATA_PORT:-8000}"
CFG="$ROOT/strata-cached.json"
[ -x .venv/bin/python ] || { echo "missing .venv: run scripts/python-env.sh first" >&2; exit 1; }
[ -x build-metal/strata ] || { echo "missing build-metal/strata: run scripts/build.sh first" >&2; exit 1; }
if [ ! -f "$CFG" ]; then
  echo "writing strata-cached.json"
  .venv/bin/python scripts/make-config.py --port "$PORT"
fi
echo "starting the server on http://127.0.0.1:$PORT/v1 (model qwen3.8-flash-next-iq3_xxs); stop it with Ctrl-C"
exec .venv/bin/python serve/server.py --engine strata --config "$CFG" --port "$PORT"
