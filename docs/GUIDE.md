# Fast local inference on an M1 Max / M2 Max (64 GB): the full guide

Qwen3.8-Flash-Next at IQ3_XXS on Strata's Metal engine, built from audited source, tuned for Apple GPU families
7-8 (M1 Max and M2 Max are one target). Everything runs offline after the downloads in step 2.

Result on an M2 Max 64 GB (2026-10-15):

| | Original MLX 3.3-bit | Stock Strata | This setup |
|---|---|---|---|
| Writing code / prose / edits (tok/s) | 39.8 / 32.6 / 60.3 | 34.3 / 35.7 / 32.6 | 44.8 / 44.5 / 47.7 |
| Reading the prompt (tok/s) | - | 270-290 | 280-307, flat to 78K tokens |
| Longest chat | ~74K | 131K | 131K (120K tested) |
| Back to an older chat | - | 60-75 s | ~5 s |
| SQL test (5 tries) | 2/5 | 0/3 | 5/5 |
| Next-word match vs the original | 78.8% | 78.6% | 78.6% |

Every step below names the script that does it, then the commands the script runs, so you can do each step by
hand. `STRATA_MODELS` is where the model files go; it defaults to `~/strata-models`. Run everything from the
repository root.

## 0. Rules

- No prebuilt executables. Upstream Strata's setup downloads a ready-made engine from GitHub releases and a
  ready-made Python; its updaters pull new code. None of that is in this repository: the engine is compiled here
  from source that was read first. The setup scripts, the updaters and the release downloads are gone.
- Python packages only as source archives, pinned by SHA-256 (`--no-binary :all: --require-hashes`).
- One engine on the GPU at a time (the model is 47 GB resident).
- The server listens on 127.0.0.1 only, whatever a flag or config says. MCP is off. Image URLs are never
  fetched. The browser is not opened for you.

## 1. Machine prep

- An M1 Max or M2 Max with 64 GB. The Metal engine needs Apple Silicon; an Intel Mac cannot run it.
- macOS 15 or newer (built and measured on macOS 26), the Xcode Command Line Tools (`xcode-select --install`, that is the Apple clang compiler; Xcode
  itself is not needed), Homebrew `cmake` and `python@3.14`.
- GPU memory limit: `scripts/gpu-limit.sh`. It runs the sysctl and prints a LaunchDaemon plist that runs the
  same sysctl at boot (RunAtLoad), because the setting resets on every reboot. By hand:

  ```bash
  sudo sysctl iogpu.wired_limit_mb=58000
  ```

- Stop anything else that holds GPU memory (other model servers, Docker or Colima VMs).

## 2. Source and downloads (the only network step)

1. **The build dependencies:** `scripts/fetch-deps.sh`. It puts Apple's metal-cpp headers (the `metal-cpp_26.zip`,
   checked against its SHA-256) in `third_party/metal-cpp` and a llama.cpp checkout at the pinned commit (a
   shallow fetch of that one commit, for `ggml` and `gguf-py`) in `third_party/llama.cpp`. The zip also holds
   `LICENSE.txt`, `README.md` and a text Python helper (`SingleHeader/MakeSingleHeader.py`); the script checks
   that every file in it is text and clears the executable bit. By hand:

   ```bash
   curl -L -o metal-cpp_26.zip https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip
   shasum -a 256 metal-cpp_26.zip   # 4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4
   unzip metal-cpp_26.zip           # headers and text; move the unpacked folder to third_party/metal-cpp
   git init third_party/llama.cpp
   git -C third_party/llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d
   git -C third_party/llama.cpp checkout --detach FETCH_HEAD
   ```

   Check that the zip unpacks to headers and text only (`Metal/Metal.hpp` and friends, no binaries).

2. **The model:** `scripts/download-model.sh`. It downloads the two IQ3_XXS shards of ISTA-DASLab's
   `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` into
   `$STRATA_MODELS/IQ3_XXS/` (resumable), checks each SHA-256 and writes a `<shard>.sha256-ok` marker next to it
   so a second run does not hash the 76 GB again, then fetches the MTP draft layer (step 3). By hand:

   ```bash
   R=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS
   mkdir -p ~/strata-models/IQ3_XXS && cd ~/strata-models/IQ3_XXS
   curl -L -C - -O "$R/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
   curl -L -C - -O "$R/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
   shasum -a 256 *.gguf
   ```

   Shard 1 (47.0 GB) must hash to `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`, shard 2
   (28.8 GB) to `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`. Those are the Hugging Face
   LFS oids of that revision (`https://huggingface.co/api/models/<repo>/tree/<revision>/IQ3_XXS`). The files,
   sizes and hashes are also in [MODEL.md](MODEL.md).

3. **The MTP draft layer:** the GGUF ships no draft head, so its 31 tensors are read from the original BF16
   checkpoint (range requests, nothing else is downloaded) and checked against pinned SHA-256 sums. The download
   script does this; by hand (`tools/mtp_fetch.py` is standard library only, so any python3 works, the venv is
   not needed):

   ```bash
   .venv/bin/python tools/mtp_fetch.py fetch --out ~/strata-models/mtp
   .venv/bin/python tools/mtp_fetch.py verify --out ~/strata-models/mtp
   ```

## 3. Patches

This repository is upstream Strata at commit `296b59d` (v0.1.44.1) with these changes already applied, in this
order (the ideas also went upstream as PRs #2475, #2609 and #2631):

1. **Security:** the updaters deleted, setup cannot download or pip, the server never fetches image URLs, MCP
   off, host forced to 127.0.0.1, browser-open off, bounds checks in `gguf_lite.cpp`.
2. **Chunked wrap:** the 47 GB file is wrapped as overlapping Metal buffers (one buffer may be at most 38.88 GiB
   on these chips), so resident mode works.
3. **M1 / M2 tuning:** bit-identical GEMV tiles tuned per shape, SPLIT=64, expert CTAB off on M1 / M2, the prompt
   warm-up (no 14 s first request), the prompt gate/up tile.
4. **Spec knobs, Metal parking, review fixes.**
5. **Root slot + SSD tier** (pinned system-prompt slot and an SSD conversation tier) and **progress** (per-request
   "reading X of Y").

What each one does and why: [TUNING.md](TUNING.md).

## 4. Build

`scripts/build.sh`. By hand:

```bash
cmake -S metal -B build-metal -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++ \
  -DMETAL_CPP_DIR=$PWD/third_party/metal-cpp -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
make -C build-metal -j10 strata
```

The engine is `build-metal/strata`. GPU kernels are compiled at run time from this tree's `.metal` files, so the
tree must stay where it was built.

**Python:** `scripts/python-env.sh`. It makes `.venv` from Homebrew's python 3.14 and installs
`requirements-mac.txt`: setuptools, flit-core, markupsafe, jinja2, pyyaml and regex, each from its source archive,
each pinned by SHA-256. By hand:

```bash
$(brew --prefix)/bin/python3.14 -m venv .venv
.venv/bin/pip download --no-binary :all: --no-deps --require-hashes -r requirements-mac.txt -d .venv/sdists
# setuptools first: unpack it, delete its Windows launchers, install the folder
mkdir .venv/sdists/setuptools-src
tar -xzf .venv/sdists/setuptools-*.tar.gz -C .venv/sdists/setuptools-src --strip-components 1
find .venv/sdists/setuptools-src -type f \( -name '*.exe' -o -name '*.dll' \) -delete
.venv/bin/pip install --no-deps --no-build-isolation --no-index .venv/sdists/setuptools-src
# then the rest, one archive at a time, in the file's order
for n in flit_core markupsafe jinja2 pyyaml regex; do
  .venv/bin/pip install --no-deps --no-build-isolation --no-index .venv/sdists/$n-*.tar.gz
done
```

One `pip install -r` line does not work here: `--no-build-isolation` needs the build backend already in the venv,
and setuptools must be installed before markupsafe, so the script downloads every archive first (hash-checked)
and installs them one by one in the file's order. The versions are jinja2 3.1.6, markupsafe 3.0.3, pyyaml 6.0.3,
regex 2026.9.10; setuptools and flit-core are current releases. The hashes are in `requirements-mac.txt`.

**numpy:** the pack tools of step 5 (`iq_pack.py`, `mtp_pack.py` and llama.cpp's `gguf-py`) import numpy, which
is not in `requirements-mac.txt`. Add it to the venv before packing, the same way (a source archive, pinned by
its hash), or from a wheel if you accept that.

## 5. Model files Strata needs

`scripts/prepare-model.sh`. It writes the expert pack to `$STRATA_MODELS/packs/iq3_xxs`, packs the MTP draft
layer to `$STRATA_MODELS/mtp/mtp-q2_0.gguf` and copies `data/draft_vocab.bin` into `$STRATA_MODELS/mtp/rt`. By
hand:

```bash
STRATA_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf ~/strata-models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out ~/strata-models/packs/iq3_xxs
.venv/bin/python tools/mtp_pack.py --src ~/strata-models/mtp --experts q2_0 --out ~/strata-models/mtp/mtp-q2_0.gguf
mkdir -p ~/strata-models/mtp/rt && cp data/draft_vocab.bin ~/strata-models/mtp/rt/
```

The pack is tied to the exact GGUF: rebuild it if the model file changes. The engine takes `--mtp <mtp>/rt` and
reads `mtp-q2_0.gguf` from the folder above `rt`.

## 6. Run

**The config:** `scripts/make-config.py` writes `strata-cached.json` (options: `--context`, `--port`, `--out`
for another file name, `--models` for another model folder). What it holds: `exe` is the built `strata`, `env`
has `STRATA_METAL_STREAM=0` (resident), and `args` are:

```
--pack <packs/iq3_xxs> --native <shard 1> --ple-gguf <shard 2> --expert-profile data/expert-profile.bin
--expert-cache 64 --prefill auto --spec 8 --mtp <mtp/rt> --mtp-max-t 2 --max-context 131072 --kv int8 --mmap-experts
--root-slot-tokens 16384 --ssd-cache-mib 20480 --ssd-cache-dir ~/.strata-cache
--conversation-cache-mib 0 --conversation-cache-min-tokens 4096 --conversation-cache-min-free-mib 1024
```

plus `"reasoning_effort": "medium"`, `"reasoning_budget_tokens": 8000`, `"effort_position": "end"`, the
sampling defaults (`temperature` 1.0, `top_p` 0.95, `top_k` 20), and the bookkeeping keys the server reads: `cwd`,
`tokenizer` (the pack's `tokenizer/`), `model_name`, `log`, `lib_dirs`, `port`, `backend`.

**Start:** `scripts/start.sh` runs the server in the foreground; Ctrl-C stops it. By hand:

```bash
.venv/bin/python serve/server.py --engine strata --config strata-cached.json --port 8000
```

Model id: `qwen3.8-flash-next-iq3_xxs`, OpenAI-compatible on `http://127.0.0.1:8000/v1`, Anthropic-compatible on
`/v1/messages`, the chat page on `http://127.0.0.1:8000/`. How to use the API: [DETAILS.md](DETAILS.md).

What each setting buys:
- `--spec 8 --mtp-max-t 2`: MTP drafts 2 words (3+ was 13-28% slower); prompt lookup up to 8 for edits.
- Medium thinking with an 8K budget: the default (xhigh) overthinks and often never answers.
- Root slot + SSD tier: the 14K-token agent system prompt is read once and kept (also across restarts); a side
  request (a chat title) or a chat switch no longer re-reads the whole chat.

Start at login: a user LaunchAgent (RunAtLoad, no KeepAlive) that waits for the GPU limit from step 1, then runs
the start command. With FileVault on, it starts only after someone logs in.

The first prompt after a reboot is slower while macOS reads the model file into its cache.

## 7. Verify

`scripts/check.sh` asks `GET /v1/models` and says whether the model is loaded. By hand:

- `curl -s 127.0.0.1:8000/v1/models` shows the model `loaded`.
- The engine log shows `wrapped (47.0 GB)`, `root slot: ... reserved`, `ssd cache: ...`.
- Greedy answers are bit-identical to stock Strata for a fixed drafting setup (drafting history changes near-tie
  answers, so cache tests use `--suffix-draft 0`).

The benchmark scripts are in `bench/`: `sbench.py` (the full-model benchmark: decode per prompt, draft acceptance,
prompt speed, a hash of each answer), `sbspec.py`, `sqltask.py` and `sqltest.py` (the SQL test). `bench.py` and
`compare.py` drive the MLX comparison, not Strata.

## 8. Tested and dropped

8-bit dense weights (5% slower, same quality); wide GEMV decoders (no gain); prompt QSA flash and experts v3 (no
gain); PROMPT_CHUNK 4096 (slower); PROMPT_WIDE, MOE_STAGE 1/2, MOE_GU/DN tiles (no gain or changed answers);
`--spec 3+` (slower); an expert precision map (not worth it). Prompt reading is bound by the expert matmuls, so
the only lever left is new prompt MoE kernels for these chips.

## 9. Switches (for testing; the defaults are above)

Set them in the config's `env`. All keep the same tokens unless the table says otherwise.

| Variable | Effect |
|---|---|
| `STRATA_METAL_STREAM=0` / `=1` | Keep the model resident (this setup) / stream the experts from the SSD |
| `STRATA_METAL_GEMV=0` | The untuned dense GEMV tile instead of the M1 / M2 one (same bits, slower) |
| `STRATA_METAL_CTAB=1` | The routed gate/up grids from constant memory (slower on M1 / M2, the default from Apple GPU family 9 on) |
| `STRATA_METAL_SPLIT=n` | Dispatches committed before the rest of a decode window is encoded (default 64; 0 = one command buffer) |
| `STRATA_METAL_PROMPT_WARM=0` | Compile the prompt kernels on first use instead of at load |
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of the Q8_0 copy |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (the slow, exact fallback) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_MAX_BUFFER_BYTES=N` | A shorter buffer view (a test hook for the chunked wrap; the answers must stay bitwise the same) |
| `STRATA_METAL_IGNORE_REPACK=1` | Start although a pack or model preparation runs beside the engine (refused by default) |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |
| `STRATA_SPEC_COSTS=1` | Print each request's drafting rounds by kind and size |

## 10. Limits

- One request at a time. Batch slots (`--parallel`) are not ported to Metal; several clients queue.
- Not ported: the control vector (`--control-vector-scaled`) and the PC tuning (`--calibrate`).
- Pictures: this build does not set up the image encoder, and image URLs are never fetched.
- Context: 131K configured; 120K tested.
