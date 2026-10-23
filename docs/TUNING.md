# What was tuned for the M1 Max / M2 Max, and why

This page explains the changes this repository makes to upstream Strata's Metal engine. The measurements were
made on an M2 Max (64 GB). The M1 Max is the same Apple GPU family, and every change is gated on
that family (Apple GPU families 7-8), so both chips take the same code paths.

## The chip

- Apple GPU family 8 on the M2 Max (it reports `apple7` and `apple8`, with `metal3` and `metal4`); family 7 on
  the M1 Max. Both answer yes to the same family checks, so they are one target.
- 400 GB/s of unified memory bandwidth; 64 GB in the tuned configuration.
- `maxBufferLength` 38.88 GiB: one Metal buffer cannot cover the 47 GB model file.
- 32 KB of threadgroup memory, 16 KB pages.
- No native bf16 math. No tensor units of the kind newer chips have.

The model: ISTA-DASLab's Qwen3.8-Flash-Next GSQ-RCO at IQ3_XXS. 48 layers, hidden 2560, 512 routed experts per
layer with 10 used per token, MoE intermediate 640, 36 GDN and 12 QSA attention layers, 4-stream
hyper-connections, an MTP draft head. The tensor formats are in [MODEL.md](MODEL.md).

## The rule: every change is bit-exact

Greedy (temperature 0) answers must be byte-identical to stock Strata for the same drafting setup, and the
engine's parity tests must pass. No fast-math, no lower precision, no lossy change. The tests live in `metal/tests`
(`gemv_parity`, `expert_parity`, `fuse_parity`, `gr_parity`, `gr2_parity`, `gdn_parity`, `qsa_parity`,
`route_parity`, `pair_parity`, `moe_mm_parity`, `prompt_parity`, `dense_mm_parity`, `mma_parity`, `sample_parity`,
`token_chain`, `rt_bench` and more); `metal/CMakeLists.txt` builds each `tests/*.cpp` as its own program. The SSD
cache's file format and copies have their own test (`conversation_ssd_test`, no GPU needed), and the GGUF reader's
bounds checks have `gguf_lite_test`.

One caveat: **drafting history changes near-tie answers.** Which tokens a verify window holds depends on the
drafts, and two answers that agree on every token but one near-tie can still differ from there on. So a
bit-exact comparison fixes the drafting setup, and the cache tests run with `--suffix-draft 0`.

## Where the time went

Stock Strata on this chip, resident, `--spec 2` with MTP, 131072 context: 32.5 tok/s on a code prompt and 36.4 on
prose, with MTP accepting about 74-83% of its drafts. One verify window took about 50 ms, against a memory floor
of about 9-18 ms for the roughly 3.7 GB a token reads. The time was overhead, not bandwidth. That is what the
changes below go after.

## The changes

### Chunked wrap (resident mode on a 47 GB file)

The model's main file is mapped and handed to the GPU without a copy. On these chips one Metal buffer may be at
most 38.88 GiB, so the file is wrapped as several overlapping buffers over the same mapped pages: consecutive
views overlap by more than the longest range one kernel binding reads (a tensor, or a layer's routed gate and up
weights), so every range lies whole inside the view the engine picks. No copy, no extra memory. Without this,
resident mode did not work on a 64 GB Mac and the engine fell back to streaming. `STRATA_METAL_MAX_BUFFER_BYTES=N`
shortens the views as a test; the answers must stay bitwise the same.

### GEMV tiles (dense decode)

The dense projections run as GEMVs on windows of 1-3 tokens. A tile has L lanes per row and K row splits, which
fix each row's summation order and are never changed, and R rows per lane, S simdgroups per threadgroup and
whether x is staged in threadgroup memory, which only move work. So every tile gives bitwise the same result. The
tiles were picked per shape with `gemv_parity --sweep` on the M2 Max: two rows per lane (one x load feeds both),
except the K-quants' 3-token windows, which spill with two rows and read x from threadgroup memory instead. The
tuned tiles are used on Apple GPU families 7-8 only; newer families keep the untuned tile. `STRATA_METAL_GEMV=0`
restores the untuned tile anywhere.

### SPLIT=64 (the GPU starts while the host still encodes)

A decode window is one graph of GPU dispatches. With `STRATA_METAL_SPLIT=64` (the default) the first 64 dispatches
are committed as their own command buffer before the rest is encoded, so the GPU starts while the host encodes.
The next command buffer's first encoder waits for the last one's, as a barrier would. Measured: -0.7 ms per
decode window on the M2 Max. `0` goes back to one command buffer.

### Expert CTAB off on M1 / M2

The routed gate/up i-quant grids can be read from constant memory (`QUANT_CTAB`) instead of a threadgroup copy.
That is bitwise the same and 3-7% faster on an M5 Max, but slower on the M2 Max, whose divergent constant reads
made `moe_gu` 8-20% slower at 3 tokens. So it is off on Apple GPU families 7-8 and on from family 9 (M3) on.
`STRATA_METAL_CTAB=0` / `1` overrides.

### Prompt warm-up (no 14 s first request)

The prompt chain's kernel libraries are compiled at load, on a thread beside the decode window's, and every
pipeline a chunk uses is built by encoding one chunk without running it. Before, the first long prompt paid
that: 14 s on the M2 Max. `STRATA_METAL_PROMPT_WARM=0` compiles on first use, as before.

### The prompt gate/up tile

For the prompt path's expert matmuls, the 16-value i-quants use a TK 32 tile on M1 / M2: gate/up 14-16% faster
than TK 16 there in `moe_mm_parity --bench`, with bitwise TK 16's output.

### Spec knobs

The drafting is set from the config: `--spec 8 --mtp-max-t 2`. The MTP draft layer drafts 2 tokens per round;
3 or more was 13-28% slower on this chip. Prompt lookup (a draft taken from an earlier copy of the text, for edits
and quotes) may draft up to 8. A request can lower its own windows without a restart (`mtp_t`, `sfx_t`, `lchain` and
`sfx_gate` in a `strata_tune` object in the request body, each capped by the start-up values) for an A/B of
drafting; greedy output is unchanged by the knobs themselves. `STRATA_SPEC_COSTS=1` prints each request's rounds by kind and size.

### Metal parking

Upstream's conversation cache parks a conversation's K/V and running state in host memory when another chat
takes the engine, and restores it when the first one comes back. This change gives the Metal window the hooks
that needs (its per-layer K/V parts and state sizes), so parking works on the Mac engine. The Mac's free-memory
figure counts what macOS can give without paging anything out: free and purgeable pages and the clean file cache.

### Root slot + SSD tier

Two tiers on top of that, both holding the exact bytes of the decode window's K/V and running state, so a
restore gives the state the conversation had:

- **The root slot** (`--root-slot-tokens 16384`): a pinned slot for the shared system prompt and tool list that an
  agent sends with every request. It is reserved at start-up and never evicted by other requests, so the 14K-token
  agent system prompt is read once.
- **The SSD tier** (`--ssd-cache-mib 20480 --ssd-cache-dir ~/.strata-cache`): conversations displaced by another
  chat or a side request are parked on the SSD and restored on the next request that continues them, also across
  restarts. A chat switch or a side request (a chat title) no longer re-reads the whole chat. The cache directory
  must be a plain directory owned by the user; the engine refuses a symlink or another user's folder.

The host-memory tier is turned off in this setup (`--conversation-cache-mib 0`), and conversations under 4096
tokens are not parked (`--conversation-cache-min-tokens 4096`).

### Progress

The server reports per-request prompt progress ("reading X of Y") on the stream's keep-alives for the request
that is being read, and never another request's stale progress line.

## Measured (M2 Max, 64 GB, 2026-10-15)

| | Stock Strata | This setup |
|---|---|---|
| Writing code / prose / edits (tok/s) | 34.3 / 35.7 / 32.6 | 44.8 / 44.5 / 47.7 |
| Reading the prompt (tok/s) | 270-290 | 280-307, flat to 78K tokens |
| Longest chat | 131K | 131K (120K tested) |
| Back to an older chat | 60-75 s | ~5 s |
| SQL test (5 tries) | 0/3 | 5/5 |
| Next-word match vs the MLX original | 78.6% | 78.6% |

Answers are bit-identical to stock Strata per drafting setup. The quality equals the original MLX 3.3-bit model
on the next-word test (top-1 78.6%).

## Tested and dropped

8-bit dense weights (5% slower, same quality); wide GEMV decoders (no gain); prompt QSA flash and experts v3 (no
gain); PROMPT_CHUNK 4096 (slower); PROMPT_WIDE, MOE_STAGE 1/2, MOE_GU/DN tiles (no gain or changed answers);
`--spec 3+` (slower); an expert precision map (not worth it). Prompt reading is bound by the expert matmuls, so
the only lever left is new prompt MoE kernels for these chips.

## Measuring it yourself

- `STRATA_METAL_PROFILE=1`: a time breakdown per window. `STRATA_METAL_KPROF=T`: per-kernel GPU times.
  `STRATA_METAL_TRACE=1`: per-window traces. All in the engine log.
- `bench/sbench.py` runs fixed greedy prompts against a build and writes decode tok/s per prompt, draft
  acceptance, prompt speed and a hash of each answer; compare the hashes between two builds to prove they are
  bit-exact.
