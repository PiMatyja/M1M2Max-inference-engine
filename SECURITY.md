# Security

## What this build does not do

The security patch (strata-local-v2 in CHANGES.md) removed every path that fetches or runs code from outside:

- The updaters (UPDATE.bat, update.sh), the installer launchers and the installer itself (`setup.py`) are gone.
  The scripts in `scripts/` replace them and download only what the README lists, pinned.
- No prebuilt engine and no prebuilt Python. The engine is built from this tree by `scripts/build.sh`; the
  Python packages come from source archives pinned by SHA-256 (`scripts/python-env.sh`).
- The server never fetches an image URL. A picture is sent inline as a `data:` URL or it is refused.
- MCP servers are never started, even when a config lists them.
- The server listens on 127.0.0.1 only, whatever `--host` or the config says. There is no remote host binding.
- No `before_load` command runs, and the server never opens a browser.
- The GGUF reader checks every length, alignment, tensor type and offset before it trusts a file.

The engine itself (the `strata` binary) has no network code. The only network steps are the downloads you run
yourself: `scripts/fetch-deps.sh` (metal-cpp and llama.cpp, pinned), `scripts/python-env.sh` (PyPI source
archives, hash-checked) and `scripts/download-model.sh` (the model, SHA-256 checked).

## What stays yours to keep

The server has no API key by default because it answers this machine only. If you put a tunnel or a proxy in
front of it, set `"api_key"` in the run config first. This code has not had an outside security audit.

## Reporting

Open an issue in this repository on GitHub. Say what you found, how to reproduce it (the request, the config
keys, the commit) and what an attacker gains. There is no email address and no other reporting channel.
