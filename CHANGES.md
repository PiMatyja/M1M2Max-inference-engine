# Changes on top of upstream Strata 296b59d (v0.1.44.1)

One section per patch, in the order they were applied. Together they are the difference between upstream and
this repository, apart from the files removed to make the tree Mac-only.

## 1. strata-local-v2 (security)

Removes everything that fetches or runs code from outside. The Windows and Linux installer launchers
(START-HERE.bat, SETUP.bat, setup.sh) and the updaters (UPDATE.bat, update.sh) are deleted, and `setup.py`
refuses to download or pip-install anything and says what is missing instead (this repository has since removed
`setup.py` altogether; `scripts/` replaces it). The server never fetches an image
URL (a picture is sent as a `data:` URL), starts no MCP servers even when the config lists them, listens on
127.0.0.1 whatever the flags say, runs no `before_load` command and never opens a browser. The Metal GGUF reader
checks every length, alignment, tensor type and offset against the file before it trusts them, so a malformed
file is refused with a message instead of being read past its end. Files: `serve/server.py`,
`serve/mcp.py`, `metal/src/gguf_lite.cpp`, `setup.py` (since removed); deleted: `SETUP.bat`, `START-HERE.bat`,
`UPDATE.bat`, `setup.sh`, `update.sh` (and a FetchContent guard in the PC `CMakeLists.txt`, since removed too).

## 2. metal-chunked-wrap

The 47 GB model file is longer than one Metal buffer may be on these chips (38.88 GiB), so resident mode could
not wrap it. The file is now mapped once and wrapped as overlapping no-copy Metal buffer views, each within the
limit, with an overlap larger than the longest range one kernel binding reads (a tensor, or a layer's routed
gate + up). Every binding picks the view that holds its whole range; the views share the mapping's pages, so
nothing is copied and no extra memory is used. A wrap that fails returns an error naming the view and byte range
instead of exiting. `STRATA_METAL_MAX_BUFFER_BYTES` lowers the view length so the several-view path can be tested
on a small Mac. Files: `metal/src/engine/window.cpp`, `metal/src/engine/prompt_chain.inc`, `metal/src/mtl_rt.h`,
`metal/src/mtl_rt.mm`.

## 3. m2-combined

Decode and prompt tuning for Apple GPU families 7-8, all bit-exact. The dense GEMV tile is picked per weight shape
from a sweep on the M2 Max (rows per lane, simdgroups per threadgroup, x staged in threadgroup memory); the lanes
per row and the row splits that fix each row's summation order never change. The decode window's command buffer
is split every 64 dispatches (`STRATA_METAL_SPLIT`, default 64) so the GPU starts while the host still encodes.
The expert table copy to threadgroup memory (CTAB) is off on M1 / M2, where it was slower, and on from family 9
up. The prompt chain's kernels are compiled at load on a thread beside the decode window's (prompt_warm), which
removes the 14 s first request. The prompt gate/up tile for the 16-value i-quants gets its own width on M1 / M2.
The parity tests cover the new tiles and gained a `--sweep`. Files: `metal/kernels/gemv.metal`,
`metal/kernels/moe.metal`, `metal/kernels/moe_mm.metal`, `metal/kernels/quant.metal`, `metal/kernels/units.metal`,
`metal/src/engine/window.cpp`, `metal/src/engine/prompt_chain.inc`, `metal/src/mtl_rt.h`, `metal/src/mtl_rt.mm`,
`metal/tests/expert_parity.cpp`, `metal/tests/gemv_parity.cpp`, `metal/tests/moe_mm_parity.cpp`.

## 4. m2-spec-knobs

Per-request drafting knobs, so drafting setups can be compared without a restart: `mtp_t` (the MTP window),
`sfx_t` (the lookup window, 0 = none), `lchain` (lookup-chain tokens) and `sfx_gate` (whether a lookup window
needs the MTP's first draft to agree), each capped by the engine's start-up values. The server passes them
through from the request's tune keys. `STRATA_SPEC_COSTS` prints each request's rounds by kind and size next to
the policy's cost view. Greedy output is unchanged. Files: `include/strata/spec/draft_policy.hpp`,
`serve/server.py`, `src/program/generate.cpp`.

## 5. m2-cache-metal-park

Lets the conversation cache park and restore conversations on Metal. The decode window keeps its own positional
K/V (the session's QSA pools are not used there), so a parked image now holds the window's exact bytes for every
attention layer, and for the draft layer when it is loaded, plus the running state; a restore puts back exactly
what the conversation left. The RAM budget follows the Metal arena's rule: free and purgeable pages and the
clean file cache, never anything paged out. Session files that read the QSA pools are refused on Metal instead of
restored as garbage. Files: `metal/src/engine/window.cpp`, `src/core/conversation_state.cpp`,
`src/core/conversation_memory.cpp`.

## 6. m2-review-fixes

Fixes from a review of the patches above. The GGUF reader refuses unknown tensor types through one `known_type()`
check in `ref_quant`, and a new test builds malformed GGUF headers field by field and checks that each is refused
with a message, no crash or exit. The tuned GEMV tiles are used only on M1 / M2 (Apple GPU family 8 and older);
newer families keep the untuned tile. The patch also turned `wrap_model`'s `wrap` into `try_wrap` with a clear
error; the newer metal-chunked-wrap patch already carries those `wrap_model` / `try_wrap` fixes, so that part
changes nothing here. Files: `metal/src/engine/window.cpp`, `metal/src/gguf_lite.cpp`, `metal/src/ref_quant.cpp`,
`metal/src/ref_quant.h`, `metal/tests/gguf_lite_test.cpp` (new).

## 7. m2-cache-ssd-v3_2

Two more cache tiers for `--serve` on Metal, both opt-in. `--root-slot-tokens N` reserves a pinned slot for the
first N tokens every chat starts with (the system prompt and tools); it is filled once and never evicted, so a
new chat, or one switched back to, reads only what follows. `--ssd-cache-mib N` parks displaced conversations on
the SSD under `--ssd-cache-dir` (default `~/.strata-cache`) as files that outlive a restart, tied to the model
and draft files, this build, the kernel sources it compiles and every engine setting. Both hold the decode
window's exact K/V and running-state bytes, and the copies and the file format are tested on synthetic buffers
without a GPU. This is what takes a return to an older chat from 60-75 s to about 5 s. Files:
`include/strata/core/conversation_ssd.hpp` (new), `src/core/conversation_ssd.cpp` (new),
`src/core/conversation_state.cpp`, `src/program/generate.cpp`, `metal/src/engine/window.cpp`,
`metal/CMakeLists.txt`, `metal/tests/conversation_ssd_test.cpp` (new).

## 8. m2-progress-fix

The server's "reading X of Y" prompt progress is now per request. A heartbeat carries progress only when the
engine's progress line changed since this pass started, so a request that is queued, inside a tool call or
waiting behind another request no longer shows that other request's numbers. Covered by new cases in the server
tests. Files: `serve/server.py`, `serve/test_server.py`.
