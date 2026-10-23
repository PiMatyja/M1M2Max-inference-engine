# AGENTS.md

This repository is Strata's Metal engine for the M1 Max / M2 Max (64 GB), built from source, serving
Qwen3.8-Flash-Next at IQ3_XXS. Read README.md first, then docs/GUIDE.md. What each patch changed: CHANGES.md.

## Build

```bash
cmake -S metal -B build-metal -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++ \
  -DMETAL_CPP_DIR=$PWD/third_party/metal-cpp -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
make -C build-metal -j10 strata
```

`scripts/build.sh` does the same. Only `metal/CMakeLists.txt` is used; there is no root CMake project. The GPU
kernels are compiled at run time from this tree's `.metal` files, so a binary always uses its own tree.

## Tests

- Every `metal/tests/*.cpp` is its own target: `make -C build-metal <name>`, then `build-metal/<name>`. The
  `*_parity` programs check a kernel against its CPU reference bit for bit and need the GPU; the `*_bench`
  programs time kernels.
- Without a GPU: `gguf_lite_test` (the GGUF reader on malformed files) and `conversation_ssd_test` (the SSD and
  root-slot cache's file format on synthetic buffers).
- `ctest --test-dir build-metal` runs `runtime_hello`.
- Server: `.venv/bin/python -m unittest serve.test_server -v` (mock engine, no GPU, no pack).

Run the tests that cover what you changed, not the whole set.

## Rules

- Source only. Never add a prebuilt binary, a wheel, a downloader or an updater. Dependencies are fetched at
  build time by `scripts/fetch-deps.sh` and pinned by commit or hash.
- M1 Max and M2 Max are one target. Gate code on Apple GPU families 7-8 and write "M1 Max / M2 Max"; never
  name one chip alone as the target. Measurements were taken on an M2 Max; say so when you quote them.
- Every kernel change is bit-exact: greedy answers stay byte-identical and the parity tests pass. No fast-math,
  no lower precision.
- No invented numbers. Every figure comes from a measurement, with what it was measured on.
- Nothing personal in any file: no names, emails, home paths, machine names or addresses. Placeholders only
  (`~/strata-models`, `$HOME`).
- ASCII punctuation only. Plain English, short sentences. A comment says what a thing does, in one sentence.
