#!/bin/bash
# setup-app.sh: runs the existing source setup stages for the macOS launcher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="${PATH:-/usr/bin:/bin}:/opt/homebrew/bin:/usr/local/bin:/usr/sbin:/sbin"
export STRATA_MODELS="${STRATA_MODELS:-$HOME/strata-models}"

preflight() {
  local chip memory major python
  chip="$(sysctl -n machdep.cpu.brand_string)"
  case "$chip" in
    'Apple M1 Max'|'Apple M2 Max') ;;
    *) echo "This setup supports M1 Max / M2 Max only (found $chip)." >&2; return 1 ;;
  esac
  memory="$(sysctl -n hw.memsize)"
  [ "$memory" -ge 68719476736 ] || { echo "This setup requires 64 GB of memory." >&2; return 1; }
  major="$(sw_vers -productVersion)"; major="${major%%.*}"
  [ "$major" -ge 15 ] || { echo "macOS 15 or newer is required." >&2; return 1; }
  xcode-select -p >/dev/null 2>&1 || { echo "Install Apple Command Line Tools: xcode-select --install" >&2; return 1; }
  command -v cmake >/dev/null || { echo "Install CMake: brew install cmake" >&2; return 1; }
  python="$(command -v python3.14 || true)"
  if [ -z "$python" ] && command -v brew >/dev/null; then
    local prefix
    prefix="$(brew --prefix python@3.14 2>/dev/null || true)"
    if [ -n "$prefix" ] && [ -x "$prefix/bin/python3.14" ]; then python="$prefix/bin/python3.14"; fi
  fi
  if [ -z "$python" ]; then python="$(command -v python3 || true)"; fi
  [ -n "$python" ] && "$python" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)' || {
    echo "Install Python 3.14: brew install python@3.14" >&2; return 1;
  }
  echo "Prerequisites OK: M1 Max / M2 Max, 64 GB or more, macOS $major."
  echo "Model folder: $STRATA_MODELS"
  echo "The two model shards alone total about 76 GB; packs and caches need more space."
  echo "Check free space on both the source and model volumes before installing."
  df -h "$ROOT"
  local parent="$STRATA_MODELS"
  while [ ! -d "$parent" ]; do parent="$(dirname "$parent")"; done
  df -h "$parent"
}

case "${1:-check}" in
  check) preflight ;;
  install)
    preflight
    for stage in fetch-deps.sh build.sh python-env.sh download-model.sh prepare-model.sh; do
      echo "=== $stage ==="
      /bin/bash "$ROOT/scripts/$stage" < /dev/null
    done
    if [ -f "$ROOT/strata-cached.json" ]; then
      echo "Keeping existing strata-cached.json (the selected model folder does not replace its paths)."
    else
      "$ROOT/.venv/bin/python" "$ROOT/scripts/make-config.py" --models "$STRATA_MODELS"
    fi
    echo "Setup complete. Press Start to run the server."
    ;;
  start)
    preflight
    limit="$(sysctl -n iogpu.wired_limit_mb)"
    [ "$limit" -ge 58000 ] || {
      echo "GPU memory limit is below 58000 MiB. Use the app's GPU Memory button before starting." >&2; exit 1;
    }
    [ -x "$ROOT/.venv/bin/python" ] || { echo "Run Install from Source first (missing .venv)." >&2; exit 1; }
    exec "$ROOT/.venv/bin/python" "$ROOT/macos/supervise.py" -- /bin/bash "$ROOT/scripts/start.sh"
    ;;
  status) exec /bin/bash "$ROOT/scripts/check.sh" ;;
  *) echo "Usage: scripts/setup-app.sh check|install|start|status" >&2; exit 2 ;;
esac
