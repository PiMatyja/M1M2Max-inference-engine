#!/bin/bash
# build.sh: configure and build the Metal engine into build-metal/ with Apple clang (guide section 4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for f in third_party/metal-cpp/Metal/Metal.hpp third_party/llama.cpp/ggml/CMakeLists.txt; do
  [ -e "$f" ] || { echo "missing $f: run scripts/fetch-deps.sh first" >&2; exit 1; }
done
command -v cmake >/dev/null || { echo "cmake not found: brew install cmake" >&2; exit 1; }
[ -x /usr/bin/clang ] || { echo "Apple clang not found: xcode-select --install" >&2; exit 1; }
JOBS="$(sysctl -n hw.ncpu)"
echo "configuring build-metal"
cmake -S metal -B build-metal -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++ \
  -DMETAL_CPP_DIR="$ROOT/third_party/metal-cpp" -DSTRATA_GGML_DIR="$ROOT/third_party/llama.cpp"
echo "building strata with $JOBS jobs"
make -C build-metal -j"$JOBS" strata
echo "done: build-metal/strata (the .metal kernels compile at run time from this tree, so keep it here)"
