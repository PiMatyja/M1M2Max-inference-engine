# Something went wrong?

The common problems on an M1 Max / M2 Max and what to do. Back to the [README](../README.md). The full setup is
in [GUIDE.md](GUIDE.md).

## While building or starting

**`cmake` says the Metal runtime and the engine are not built (`METAL_CPP_DIR is not set`).**
`third_party/metal-cpp` is missing or does not hold `Metal/Metal.hpp`. Run `scripts/fetch-deps.sh`, or unpack
Apple's `metal-cpp_26.zip` there by hand (GUIDE.md, step 2), then configure again.

**The model does not fit, or the engine streams from the SSD instead of staying resident.**
The GPU memory limit is too low. It resets on every reboot: run `scripts/gpu-limit.sh` again (or
`sudo sysctl iogpu.wired_limit_mb=58000`), and stop anything else that holds GPU memory (other model servers,
Docker or Colima VMs). The engine log says `wrapped (47.0 GB)` when the model is resident.

**The engine refuses to start because a pack or model preparation is running.**
It refuses on purpose: running the engine beside a repack froze a Mac. Wait for the preparation to finish.
`STRATA_METAL_IGNORE_REPACK=1` starts it anyway, if you insist.

**The pack tools fail with `No module named numpy`.**
`requirements-mac.txt` does not install numpy, and `iq_pack.py`, `mtp_pack.py` and llama.cpp's `gguf-py` import it.
Add numpy to the venv, then pack again.

**`port 8000 is already in use`.**
Strata is already running (check `curl -s 127.0.0.1:8000/v1/models`), or another program uses the port: start with
another `--port` (`scripts/make-config.py --port N`, or `--port N` on the server command). Only an address that is
really taken says "already in use"; any other reason is printed as what the OS said.

**The first start, or the first prompt after a reboot, is slow.**
macOS is reading the 47 GB model file into its cache. The next starts and prompts are faster while the file is
in the cache.

**The engine was built, but the GPU kernels fail to compile at start.**
The kernels are compiled at run time from this tree's `.metal` files, and the paths are baked into the binary.
If the tree was moved after the build, build it again where it is now.

## While it answers

**An answer stopped with "the engine stopped unexpectedly".**
The engine process ended mid-answer. Send your message again: the server starts the engine by itself. If it
keeps happening, check the GPU limit and what else is using memory.

**It says the prompt exceeds the context.**
The conversation is longer than the configured context (131072 tokens). Start a new chat, or raise `--context`
in `scripts/make-config.py` if your memory allows it.

**A request never finishes, and the model "thinks" forever.**
The model's default thinking level (xhigh) often never answers on this setup. The config sets
`"reasoning_effort": "medium"` and `"reasoning_budget_tokens": 8000`; a request that asks for `high` overrides the
level. Ask for `medium` or `low`, or set the budget lower.

**Going back to an older chat re-reads the whole chat.**
The SSD tier parks conversations in `--ssd-cache-dir` (`~/.strata-cache`). That folder must be a plain directory
owned by you (the engine refuses a symlink or another user's folder) and have room for `--ssd-cache-mib`. The
engine log says `ssd cache: ...` at start when it is on. Conversations under 4096 tokens are not parked
(`--conversation-cache-min-tokens`).

**A side request (a chat title) made the next turn re-read the chat.**
That is what the root slot and the SSD tier prevent; check that the engine log shows `root slot: ... reserved` and
`ssd cache: ...`. The root slot holds a shared system prompt of up to `--root-slot-tokens` (16384) tokens.

**Pictures are refused.**
This build does not set up the image encoder, and image URLs are never fetched.

## Still stuck?

The engine log is the file the config's `log` key names, in the folder you started from. The server window
prints the request errors.
