#!/bin/bash
# check.sh: asks the server on 127.0.0.1:$STRATA_PORT (default 8000) whether the model is loaded (guide section 7).
set -euo pipefail
PORT="${STRATA_PORT:-8000}"
URL="http://127.0.0.1:$PORT/v1/models"
echo "GET $URL"
body="$(curl -sf "$URL")" || { echo "no answer from $URL: the server is not running"; exit 1; }
echo "$body" | python3 -c '
import json, sys
data = json.load(sys.stdin).get("data") or []
if not data:
    print("the server runs but no model is loaded yet (still loading, or unloaded)")
    sys.exit(1)
for m in data:
    status = (m.get("status") or {}).get("value", "?")
    ctx = (m.get("meta") or {}).get("n_ctx")
    model_id = m.get("id")
    print(f"{model_id}: {status}" + (f", context {ctx}" if ctx else ""))
sys.exit(0 if any((m.get("status") or {}).get("value") == "loaded" for m in data) else 1)
'
