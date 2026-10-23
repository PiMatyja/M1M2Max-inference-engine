#!/bin/bash
# download-model.sh: the two IQ3_XXS shards (hash-checked) and the MTP draft layer into $STRATA_MODELS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${STRATA_MODELS:-$HOME/strata-models}"
REPO="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REV="ed59f92082b1e93c0e96d60a8b11aab089b52f09"
BASE="https://huggingface.co/$REPO/resolve/$REV/IQ3_XXS"
S1="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
S2="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
SHA1="219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"
SHA2="316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
SIZE1=47039860096
SIZE2=28800138432
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
mkdir -p "$M/IQ3_XXS" "$M/mtp"

fetch() {   # file sha256 size
  local f="$M/IQ3_XXS/$1" sha="$2" size="$3"
  if [ -f "$f.sha256-ok" ]; then
    echo "$1: already checked"
    return
  fi
  if [ -f "$f" ] && [ "$(stat -f %z "$f")" -eq "$size" ]; then
    echo "$1: already downloaded"
  else
    echo "$1: downloading (resumes a partial file)"
    curl -fL --retry 5 --retry-all-errors -C - -o "$f" "$BASE/$1"
  fi
  echo "$1: checking sha256 (takes a while)"
  echo "$sha  $f" | shasum -a 256 -c -
  touch "$f.sha256-ok"
}
fetch "$S1" "$SHA1" "$SIZE1"
fetch "$S2" "$SHA2" "$SIZE2"

echo "mtp: fetching the draft layer tensors into $M/mtp"
"$PY" "$ROOT/tools/mtp_fetch.py" fetch --out "$M/mtp"
echo "mtp: verifying the tensors against their sha256"
"$PY" "$ROOT/tools/mtp_fetch.py" verify --out "$M/mtp"
echo "done: model files in $M"
