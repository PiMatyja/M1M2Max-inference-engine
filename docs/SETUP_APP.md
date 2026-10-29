# macOS setup app

A small native setup window for M1 Max / M2 Max with 64 GB or more.
It builds the same engine and uses the same settings as the command-line guide.
It does not change kernels, precision or sampling.

## Open it

Install Apple's Command Line Tools first:

```sh
xcode-select --install
```

Finish Apple's installer, then double-click `Setup.command` in this checkout.
It compiles the window with Apple's Swift compiler and opens
`build-launcher/Strata Setup.app`. No Swift packages are fetched.
If Finder refuses to open the command file, run it from Terminal:

```sh
/bin/bash Setup.command
```

The source checkout must stay on disk. The engine compiles Metal kernels from it
at runtime. If the app is moved elsewhere, choose the checkout with Source Folder.
The app remembers the selected source and model folders in macOS preferences.

## Prepare the machine

- macOS 15 or newer.
- M1 Max / M2 Max and at least 64 GB RAM; other chips are refused.
- Apple Command Line Tools, Homebrew CMake and Python 3.14.
- Space for the two model shards (about 76 GB), expert pack, draft weights,
  source builds and the configured SSD conversation cache.

If Homebrew is already installed:

```sh
brew install cmake python@3.14
```

Homebrew installation is manual. The app does not install package managers or
silently run an external installer. Python 3.12 and 3.13 are also supported by
the existing source-package setup.

## Use the window

1. **Source Folder:** select this checkout if it was not found automatically.
2. **Model Folder:** select a folder on a volume with room for the model and pack.
   The default is `~/strata-models`.
3. **Check Setup:** checks hardware, macOS and tools, then prints free disk space
   for the source and model volumes. It makes no changes and downloads nothing.
   The disk-space display is not a guarantee that all later builds will fit.
4. **Install from Source:** after confirmation, runs `fetch-deps.sh`, `build.sh`,
   `python-env.sh`, `download-model.sh` and `prepare-model.sh` in order. The first
   failed stage stops setup and leaves its error in the log. Re-run Install to
   reuse completed downloads and prepared files as those scripts permit.
   An existing `strata-cached.json` is preserved. The selected model folder does
   not rewrite paths in an existing config; edit that config deliberately if
   moving an existing installation. Without a config, the app creates one for
   the selected folder.
5. **GPU Memory:** separately asks for administrator permission to set
   `iogpu.wired_limit_mb=58000`. It resets after reboot. No LaunchDaemon or login
   service is installed. Close other GPU model servers first.
6. **Start:** runs the server in the foreground under the app on
   `http://127.0.0.1:8000`. Start refuses a GPU limit below 58000 MiB.
   Starting the process is not the same as the model being ready.
7. **Status:** requests `/v1/models` with a five-second timeout and displays the
   response. It may also report a server started outside this app.
8. **Stop:** the server runs under `macos/supervise.py`, which owns its whole
   process group. Stop asks that group to end and waits until the server and the
   engine have both exited, with a 10-second cleanup grace before the group is
   ended by force. This also covers a stop while the model is still loading,
   before the server has installed its own signal handling. Stop and quit never
   signal another server by name or port, and the window stays unusable until the
   group is gone, so a new Start cannot race the old engine's GPU memory.

The log is selectable and bounded in size. Setup stages run without Terminal
input. The app prevents closing or quitting during setup rather than leave a
partially cancelled build running. While serving, closing or quitting asks the
owned server to stop and waits for cleanup. There is no force-kill button.

Only one setup app or command-line installation should use a checkout at a time.
Do not start another server while the model already occupies GPU memory.

## What this is not

This is a source-built setup interface, not a self-contained consumer installer.
First use still needs Apple's tools, CMake and a supported Python. Model downloads
and preparation are not instantaneous. The app uses existing download scripts;
it adds no downloader, updater, telemetry or network service.

The local app has an ad-hoc signature, not a Developer ID signature or Apple
notarization. No prebuilt app, engine, Python or wheel is committed or distributed
by this repository. A prebuilt quick-install edition would require a separate,
explicit distribution-policy decision, signing credentials and release testing.

## Build and test

Build without opening the window:

```sh
/bin/bash scripts/build-setup-app.sh
python3 -m unittest discover -s macos/tests -v
```

The tests use fake setup stages and tool responses. They cover supported and
unsupported machines, stage order, failure handling, paths with spaces, config
preservation and the memory-limit guard. The supervisor tests cover the delayed-engine
cancellation a review found: a stop during startup still ends and reaps the owned
process group. They do not download weights, change
system settings or start the real engine. A successful compile is not a full
clean-machine or interactive UI test.
