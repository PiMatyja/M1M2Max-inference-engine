# Strata for M1 Max / M2 Max

This repository holds Strata's inference engine for Apple Silicon, built from
audited source and tuned for the M1 Max and M2 Max with 64 GB (Apple GPU families 7 and 8: one target). It
serves Qwen3.8-Flash-Next at IQ3_XXS as an OpenAI-compatible API on 127.0.0.1:8000.

Nothing else is in here: no CUDA, no AMD, no Intel, no Windows, no Docker, no Nix, no updaters, no prebuilt
binaries. Everything runs offline once the downloads are done.

## Results

Measured on an M2 Max 64 GB

| | Original MLX 3.3-bit | Stock Strata | This setup |
|---|---|---|---|
| Writing code / prose / edits (tok/s) | 39.8 / 32.6 / 60.3 | 34.3 / 35.7 / 32.6 | 44.8 / 44.5 / 47.7 |
| Reading the prompt (tok/s) | - | 270-290 | 280-307, flat to 78K tokens |
| Longest chat | ~74K | 131K | 131K (120K tested) |
| Back to an older chat | - | 60-75 s | ~5 s |
| SQL test (5 tries) | 2/5 | 0/3 | 5/5 |
| Next-word match vs the original | 78.8% | 78.6% | 78.6% |

Greedy answers are bit-identical to stock Strata for a fixed drafting setup.

## What you need

- A Mac with an M1 Max or M2 Max and 64 GB of memory.
- macOS 15 or newer (built and measured on macOS 26) and the Xcode command line tools (Apple clang).
- Homebrew `cmake` and `python@3.14`.
- Disk for the model: the two shards are 47.0 GB and 28.8 GB, plus the expert pack, the MTP draft layer and up
  to 20480 MiB of SSD cache.
- Network only for steps 2, 4 and 5 below.

## Steps

Each step is one script. The full walk-through, with what every command does, is docs/GUIDE.md.

1. Prepare the machine: `scripts/gpu-limit.sh` raises the GPU memory limit
   (`sudo sysctl iogpu.wired_limit_mb=58000`). It resets on every reboot. Stop anything else that holds GPU
   memory first.
2. Fetch the build dependencies: `scripts/fetch-deps.sh` downloads Apple's metal-cpp_26.zip (SHA-256 checked)
   and fetches llama.cpp at commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (a shallow fetch of that one commit)
   into `third_party/`. Both stay out of git.
3. Build: `scripts/build.sh` runs cmake on `metal/` and makes the `strata` binary in `build-metal/`. The GPU
   kernels are compiled at run time from this tree's `.metal` files, so keep the tree where it was built.
4. Python: `scripts/python-env.sh` makes `.venv` from Homebrew python 3.14 and installs the server, pack tools'
   dependencies (including numpy) and their build backends from source archives in `requirements-mac.txt`.
   Each archive is pinned by its SHA-256. Build isolation and automatic dependency installation are disabled.
5. Download the model: `scripts/download-model.sh` fetches the two IQ3_XXS shards of ISTA-DASLab's
   Qwen3.8-Flash-Next-GSQ-RCO-GGUF at the pinned revision into `~/strata-models/IQ3_XXS/` and checks their
   SHA-256, then fetches and verifies the MTP draft layer.
6. Prepare the model: `scripts/prepare-model.sh` builds the expert pack (`tools/iq_pack.py`) and the MTP pack
   (`tools/mtp_pack.py`). The Python setup includes numpy. The pack is tied to the exact GGUF: rebuild it if the
   model file changes.
7. Make the run config: `scripts/make-config.py` writes `strata-cached.json` with the tuned settings.
8. Run: `scripts/start.sh` starts the server on port 8000. Model id: `qwen3.8-flash-next-iq3_xxs`.
9. Verify: `scripts/check.sh` asks the server whether the model is `loaded` (the same as
   `curl -s 127.0.0.1:8000/v1/models`), and the engine log shows `wrapped (47.0 GB)`, `root slot: ... reserved`
   and `ssd cache: ...`.

`chat.py` is a small terminal chat for the running server: `.venv/bin/python chat.py`.

## The settings

`STRATA_METAL_STREAM=0` keeps the model resident. `--spec 8 --mtp-max-t 2`: the MTP draft layer guesses 2 words
ahead (3 or more was 13-28% slower) and prompt lookup drafts up to 8 for edits. `--max-context 131072 --kv int8`:
131K of context. `--root-slot-tokens 16384` and `--ssd-cache-mib 20480`: the system prompt is read once and
kept, also across restarts, and a side request or a chat switch no longer re-reads the whole chat. Thinking at
`reasoning_effort` medium with an 8000-token budget: the default overthinks and often never answers. Why each
setting, and what was measured: docs/TUNING.md.

## Rules

- No prebuilt executables. Upstream's installer downloads a ready-made engine and a ready-made Python; its
  updaters pull new code. None of that is here: the installer, the updaters and the PC build files are removed,
  and the engine is compiled from source that was read first.
- Python packages only as source archives, pinned by SHA-256 in `requirements-mac.txt` and verified before
  any build runs. No automatic build-dependency downloads.
- The server listens on 127.0.0.1 only, never fetches a URL and starts no MCP servers. See SECURITY.md.
- One engine on the GPU at a time: the model is 47 GB resident.

## Tested and dropped

8-bit dense weights (5% slower, same quality); wide GEMV decoders (no gain); prompt QSA flash and experts v3
(no gain); PROMPT_CHUNK 4096 (slower); PROMPT_WIDE, MOE_STAGE 1/2, MOE_GU/DN tiles (no gain or changed
answers); `--spec 3+` (slower). Prompt reading is bound by the expert matmuls, so the only lever left is new
prompt MoE kernels for these chips.

## Credit

Based on Strata by Niko1221 (MIT), upstream commit 296b59d (v0.1.44.1): https://github.com/Niko1221/Strata.
The changes on top of it are by PiMatyja and are listed in CHANGES.md. Three of them were offered upstream and
closed there; their ideas live on here: #2475 (gguf hardening), #2609 (faster decode GEMVs on M1 / M2) and
#2631 (conversation cache across a side request). License: MIT, see LICENSE and NOTICE.md.
