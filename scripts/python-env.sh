#!/bin/bash
# python-env.sh: a .venv from Homebrew python 3.14 with the pinned source-only packages of requirements-mac.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REQ="$ROOT/requirements-mac.txt"
[ -f "$REQ" ] || { echo "missing requirements-mac.txt" >&2; exit 1; }

# Prefer Homebrew Python 3.14; the source pins also support Python 3.12 and 3.13.
PY=""
for c in "$(brew --prefix python@3.14 2>/dev/null)/bin/python3.14" /opt/homebrew/bin/python3.14 /usr/local/bin/python3.14; do
  [ -x "$c" ] && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || { echo "no python3 found: brew install python@3.14" >&2; exit 1; }
  "$PY" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)' || \
    { echo "supported Python is 3.12 through 3.14: brew install python@3.14" >&2; exit 1; }
  echo "Homebrew python 3.14 not found, using $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
fi

STAMP="$ROOT/.venv/requirements-mac.sha256"
if [ -x .venv/bin/python ] && [ -f "$STAMP" ] && [ "$(shasum -a 256 "$REQ" | cut -c1-64)" = "$(cat "$STAMP")" ]; then
  echo "done: .venv already matches requirements-mac.txt (delete $STAMP to reinstall)"
  exit 0
fi

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv with $PY"
  "$PY" -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)' || \
  { echo "supported Python is 3.12 through 3.14; recreate .venv with a supported Python" >&2; exit 1; }

# Downloads use the standard library so no package metadata hooks or isolated builds run.
export PATH="$ROOT/.venv/bin:$PATH"
export PIP_NO_INDEX=1 PIP_NO_DEPS=1 PIP_NO_BUILD_ISOLATION=1 PIP_CONFIG_FILE=/dev/null
STAGE="$(mktemp -d "$ROOT/.venv/sources.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
.venv/bin/python - "$REQ" "$STAGE" <<'PYTHON'
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

requirements, stage = Path(sys.argv[1]), Path(sys.argv[2])
entries = []
for line in requirements.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+) --hash=sha256:([0-9a-f]{64})", line)
    if not match:
        raise ValueError("invalid pinned source requirement: " + line)
    name, version, digest = match.groups()
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
        metadata = json.load(response)
    sources = [item for item in metadata["urls"]
               if item["packagetype"] == "sdist" and item["digests"]["sha256"] == digest]
    if len(sources) != 1:
        raise ValueError(f"expected one hash-matched sdist for {name}=={version}")
    source = sources[0]
    url = source["url"]
    if not url.startswith("https://files.pythonhosted.org/") or not source["filename"].endswith(".tar.gz"):
        raise ValueError("expected a PyPI source tarball")
    archive = stage / (name + ".tar.gz")
    print(f"downloading {name}=={version}", flush=True)
    with urllib.request.urlopen(url) as response, archive.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f"sdist hash mismatch: {name}")
    entries.append((name, archive))

# Verify every archive before any backend can execute; no persistent glob selects inputs.
for name, archive in entries:
    destination = stage / name
    destination.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError(f"expected one source directory: {name}")
    for path in roots[0].rglob("*"):
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".dylib", ".whl"}:
            path.unlink()
    with (stage / "install-order").open("a") as order:
        order.write(f"{name}\t{roots[0]}\n")
PYTHON

install_source() {
  .venv/bin/python -m pip --isolated install -q --no-deps --no-build-isolation --no-index --no-cache-dir --no-binary :all: "$source" "$@"
}

# Ninja bootstraps from its bundled C++ sources without another Python build backend.
while IFS=$'\t' read -r name source; do
  if [ "$name" = ninja ]; then
    (cd "$source/ninja-upstream" && "$ROOT/.venv/bin/python" configure.py --bootstrap)
    cp "$source/ninja-upstream/ninja" "$ROOT/.venv/bin/ninja"
  else
    echo "installing $name from verified source"
    if [ "$name" = numpy ]; then
      install_source --config-settings=setup-args=--wrap-mode=nodownload --config-settings=setup-args=-Dblas=accelerate --config-settings=setup-args=-Dlapack=accelerate
    else
      install_source
    fi
  fi
done < "$STAGE/install-order"
shasum -a 256 "$REQ" | cut -c1-64 > "$STAMP"
echo "done: .venv is ready ($(.venv/bin/python -c 'import platform; print(platform.python_version())'))"
