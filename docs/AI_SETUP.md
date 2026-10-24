# Setting it up for a user (instructions for AI agents)

This page is written for an AI coding assistant that a user asked to set up this repository on their Mac. It
runs Qwen3.8-Flash-Next locally on an M1 Max or M2 Max with 64 GB and serves an OpenAI- and Anthropic-compatible
API on `http://127.0.0.1:8000`. The human-oriented version of everything below is [GUIDE.md](GUIDE.md); follow its
steps in order and tell the user what you are doing in plain words.

## 0. Ground rules

- **The server listens on this Mac only.** The host is forced to 127.0.0.1 whatever a flag or config says. Do not
  put a tunnel or a port forward in front of it.
- **Nothing is downloaded but what the guide names:** Apple's metal-cpp headers, a llama.cpp checkout at a pinned
  commit, the Python packages as source archives pinned by hash, the two model shards and the MTP tensors. No
  prebuilt engine, no updater, no pip wheels. If a step wants anything else, stop and ask.
- **Do not change the user's system beyond what the guide says.** The one system change is the GPU memory limit
  (`sudo sysctl iogpu.wired_limit_mb=58000`), which needs sudo: ask the user to run it, or to install the
  LaunchDaemon `scripts/gpu-limit.sh` prints. You cannot type their password.
- The model download is about 76 GB (47.0 GB + 28.8 GB) plus the draft layer's tensors. Confirm the user is fine
  with that before you start, especially on a metered connection.
- The downloads are long-running. Run them in the background or with a long timeout and poll; they resume if
  interrupted.
- One engine on the GPU at a time. Do not start a second server.

## 1. Check the Mac

| What | Command |
| --- | --- |
| Chip and memory | `sysctl -n machdep.cpu.brand_string hw.memsize` |
| macOS version | `sw_vers` |
| Command Line Tools | `xcode-select -p` (install with `xcode-select --install`) |
| Homebrew cmake and python | `brew list cmake python@3.14` |
| Free disk | `df -h ~` |

Requirements: an M1 Max or M2 Max with 64 GB, macOS 15 or newer (built and measured on macOS 26), the Command
Line Tools, Homebrew `cmake` and
`python@3.14`, and room for the model files (about 76 GB) plus the pack and the draft layer. If the Mac does
not meet them, say which part is missing and stop.

## 2. The steps

Run from the repository root, in this order. Each script's manual commands are in GUIDE.md.

1. The GPU limit: the user runs `scripts/gpu-limit.sh` (sudo). It resets on every reboot.
2. `scripts/fetch-deps.sh` (metal-cpp headers and llama.cpp at the pinned commit, both checked).
3. `scripts/build.sh` (the engine, `build-metal/strata`).
4. `scripts/python-env.sh` (`.venv` with the pinned packages, numpy included; a few minutes).
5. `scripts/download-model.sh` (both shards, hash-checked, then the MTP tensors, hash-checked).
6. `scripts/prepare-model.sh` (the expert pack and the MTP draft layer; numpy comes from step 4).
7. `scripts/make-config.py` (writes `strata-cached.json`; `--context`, `--port`, `--out`, `--models`).
8. `scripts/start.sh` (the server in the foreground; Ctrl-C stops it). Start it in its own terminal or in the
   background, so it does not block your shell.
9. `scripts/check.sh` (asks the server whether the model is loaded).

## 3. Verify

```
scripts/check.sh
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "qwen3.8-flash-next-iq3_xxs", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 64, "reasoning_effort": "none"}'
```

The engine log shows `wrapped (47.0 GB)`, `root slot: ... reserved` and `ssd cache: ...` when everything is on.

## 4. Connect the user's apps

- **Browser:** `http://127.0.0.1:8000` (chat and monitor).
- **Any OpenAI-compatible app or agent:** base URL `http://127.0.0.1:8000/v1`, any API key, model
  `qwen3.8-flash-next-iq3_xxs` (any name is served).
- **Anthropic-compatible apps:** `http://127.0.0.1:8000/v1/messages`.
- **Codex CLI and other Responses API apps:** `http://127.0.0.1:8000/v1/responses` (stateless).
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_MODEL` set to a Claude model name it
  knows (the name is ignored), and any `ANTHROPIC_AUTH_TOKEN`.
- **Thinking level:** `"reasoning_effort": "none" | "low" | "medium" | "high"`; the config's default is `medium`
  with an 8000-token budget.
- One request at a time; several clients queue. API details: [DETAILS.md](DETAILS.md).

## 5. When something fails

[TROUBLESHOOTING.md](TROUBLESHOOTING.md). The engine log is the file the config's `log` key names.
