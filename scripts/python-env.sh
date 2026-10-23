#!/bin/bash
# python-env.sh: a .venv from Homebrew python 3.14 with the pinned source-only packages of requirements-mac.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REQ="$ROOT/requirements-mac.txt"
[ -f "$REQ" ] || { echo "missing requirements-mac.txt" >&2; exit 1; }

# Homebrew python 3.14 first; any python3 from 3.12 on otherwise, said out loud.
PY=""
for c in "$(brew --prefix python@3.14 2>/dev/null)/bin/python3.14" /opt/homebrew/bin/python3.14 /usr/local/bin/python3.14; do
  [ -x "$c" ] && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  PY="$(command -v python3 || true)"
  [ -n "$PY" ] || { echo "no python3 found: brew install python@3.14" >&2; exit 1; }
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' || \
    { echo "python3 is older than 3.12: brew install python@3.14" >&2; exit 1; }
  echo "Homebrew python 3.14 not found, using $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
fi

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv with $PY"
  "$PY" -m venv .venv
fi
PIP=".venv/bin/pip"
DL="$ROOT/.venv/sdists"
mkdir -p "$DL"

# Every source archive is fetched with its pinned hash checked; nothing is built yet.
echo "downloading the pinned source archives into .venv/sdists"
"$PIP" download -q --no-binary :all: --no-deps --require-hashes -r "$REQ" -d "$DL"

# The archives install one by one in the file's order, because a backend must be in place before what it builds.
names="$(grep -Eo '^[A-Za-z0-9_.-]+==' "$REQ" | tr -d '=' | tr 'A-Z' 'a-z' | tr '-' '_')"
for name in $names; do
  f="$(find "$DL" -maxdepth 1 \( -iname "${name}-*.tar.gz" -o -iname "${name//_/-}-*.tar.gz" \) | head -n 1)"
  [ -n "$f" ] || { echo "no source archive for $name in $DL" >&2; exit 1; }
  if [ "$name" = "setuptools" ]; then
    # setuptools ships Windows launchers (.exe) inside its archive; they are deleted before it is built.
    echo "unpacking setuptools and deleting its Windows launchers"
    rm -rf "$DL/setuptools-src" && mkdir "$DL/setuptools-src"
    tar -xzf "$f" -C "$DL/setuptools-src" --strip-components 1
    find "$DL/setuptools-src" -type f \( -name '*.exe' -o -name '*.dll' \) -delete
    echo "installing setuptools from the cleaned folder"
    "$PIP" install -q --no-deps --no-build-isolation --no-index "$DL/setuptools-src"
    rm -rf "$DL/setuptools-src"
  else
    echo "installing $name from $(basename "$f")"
    "$PIP" install -q --no-deps --no-build-isolation --no-index "$f"
  fi
done
echo "done: .venv is ready ($(.venv/bin/python -c 'import platform; print(platform.python_version())'))"
