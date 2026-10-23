#!/bin/bash
# fetch-deps.sh: Apple's metal-cpp headers and the pinned llama.cpp checkout into third_party/ (safe to run again).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
METAL_CPP_URL="https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
METAL_CPP_SHA256="4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
LLAMA_URL="https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT="3cf03257f219afbe7334045ff7c6a06ac68c627d"
mkdir -p "$TP"

if [ -f "$TP/metal-cpp/Metal/Metal.hpp" ]; then
  echo "metal-cpp: already in third_party/metal-cpp"
else
  echo "metal-cpp: downloading metal-cpp_26.zip"
  curl -fL --retry 3 -o "$TP/metal-cpp_26.zip" "$METAL_CPP_URL"
  echo "metal-cpp: checking sha256"
  echo "$METAL_CPP_SHA256  $TP/metal-cpp_26.zip" | shasum -a 256 -c -
  rm -rf "$TP/metal-cpp" "$TP/metal-cpp.unpack"
  mkdir "$TP/metal-cpp.unpack"
  unzip -q "$TP/metal-cpp_26.zip" -d "$TP/metal-cpp.unpack"
  # the zip must hold headers and text only, never a program (checked by content; Apple ships one +x helper script)
  if find "$TP/metal-cpp.unpack" -type f -exec file -b --mime-type {} + | grep -v '^text/' | grep -q .; then
    echo "metal-cpp: the zip holds files that are not text, stopping" >&2
    exit 1
  fi
  find "$TP/metal-cpp.unpack" -type f -exec chmod a-x {} +
  mv "$TP/metal-cpp.unpack/metal-cpp" "$TP/metal-cpp"
  rm -rf "$TP/metal-cpp.unpack" "$TP/metal-cpp_26.zip"
  echo "metal-cpp: unpacked $(find "$TP/metal-cpp" -name '*.hpp' | wc -l | tr -d ' ') headers"
fi

if [ -d "$TP/llama.cpp/.git" ] && [ "$(git -C "$TP/llama.cpp" rev-parse HEAD)" = "$LLAMA_COMMIT" ]; then
  echo "llama.cpp: already at $LLAMA_COMMIT"
else
  echo "llama.cpp: fetching commit $LLAMA_COMMIT"
  if [ ! -d "$TP/llama.cpp/.git" ]; then
    rm -rf "$TP/llama.cpp"
    git init -q "$TP/llama.cpp"
    git -C "$TP/llama.cpp" remote add origin "$LLAMA_URL"
  fi
  git -C "$TP/llama.cpp" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$TP/llama.cpp" checkout -q --detach FETCH_HEAD
  if [ "$(git -C "$TP/llama.cpp" rev-parse HEAD)" != "$LLAMA_COMMIT" ]; then
    echo "llama.cpp: HEAD is not $LLAMA_COMMIT, stopping" >&2
    exit 1
  fi
  echo "llama.cpp: checked out $LLAMA_COMMIT"
fi
echo "done: third_party/metal-cpp and third_party/llama.cpp are ready"
