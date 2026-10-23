# Notice

This repository is a derivative of Strata by Niko1221 (https://github.com/Niko1221/Strata), taken at upstream
commit 296b59d (release v0.1.44.1) and licensed under the MIT License. The upstream copyright line is kept in
LICENSE. The changes made here (CHANGES.md) are Copyright (c) 2026 PiMatyja under the same license.

## Third-party code in the tree

- `third_party/ggml/ggml-common.h` is from ggml (the ggml / llama.cpp project), MIT License, Copyright (c)
  2023-2026 The ggml authors. The license text is `third_party/ggml/LICENSE`; the source commit is in
  `third_party/ggml/VERSION.txt` (3cf03257f219afbe7334045ff7c6a06ac68c627d).

## Fetched at build time, never vendored

- Apple's metal-cpp (`metal-cpp_26.zip` from developer.apple.com) into `third_party/metal-cpp`, under the
  license shipped inside that archive.
- llama.cpp (ggml and gguf-py) at commit 3cf03257f219afbe7334045ff7c6a06ac68c627d into
  `third_party/llama.cpp`, MIT License.

`scripts/fetch-deps.sh` downloads both and checks the pins; `.gitignore` keeps them out of the repository.

## Not in the repository

The model, ISTA-DASLab's Qwen3.8-Flash-Next-GSQ-RCO-GGUF on Hugging Face, is downloaded by the user with
`scripts/download-model.sh` and stays under the terms its authors publish with it.
