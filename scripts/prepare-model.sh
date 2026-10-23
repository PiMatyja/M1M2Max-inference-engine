#!/bin/bash
# prepare-model.sh: the expert pack, the packed MTP draft layer and its runtime folder from the downloaded files (guide section 5).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${STRATA_MODELS:-$HOME/strata-models}"
S1="$M/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
PACK="$M/packs/iq3_xxs"
RT="$M/mtp/rt"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "missing .venv: run scripts/python-env.sh first" >&2; exit 1; }
[ -f "$S1" ] || { echo "missing $S1: run scripts/download-model.sh first" >&2; exit 1; }
[ -d "$ROOT/third_party/llama.cpp/gguf-py/gguf" ] || { echo "missing third_party/llama.cpp: run scripts/fetch-deps.sh first" >&2; exit 1; }
"$PY" -c 'import numpy' 2>/dev/null || { echo "numpy is not in .venv: add it first (docs/GUIDE.md section 4)" >&2; exit 1; }
export STRATA_GGUF_PY="$ROOT/third_party/llama.cpp/gguf-py"

if [ -f "$PACK/native_experts.txt" ] && [ -f "$PACK/tokenizer/vocab.json" ]; then
  echo "pack: already in $PACK"
else
  echo "pack: writing the expert pack to $PACK (reads the 47 GB shard once)"
  "$PY" "$ROOT/tools/iq_pack.py" --gguf "$S1" --out "$PACK" < /dev/null
fi

if [ -f "$M/mtp/mtp-q2_0.gguf" ]; then
  echo "mtp: mtp-q2_0.gguf already packed"
else
  [ -f "$M/mtp/mtp-manifest.json" ] || { echo "missing $M/mtp/mtp-manifest.json: run scripts/download-model.sh first" >&2; exit 1; }
  echo "mtp: packing the draft layer to $M/mtp/mtp-q2_0.gguf"
  "$PY" "$ROOT/tools/mtp_pack.py" --src "$M/mtp" --experts q2_0 --out "$M/mtp/mtp-q2_0.gguf"
fi

# the engine reads the draft vocabulary beside the draft layer: <mtp>/rt/draft_vocab.bin
mkdir -p "$RT"
if [ -f "$RT/draft_vocab.bin" ]; then
  echo "mtp: rt/draft_vocab.bin already in place"
else
  echo "mtp: copying data/draft_vocab.bin to $RT"
  cp "$ROOT/data/draft_vocab.bin" "$RT/draft_vocab.bin"
fi
echo "done: pack in $PACK, draft layer in $M/mtp"
