#!/bin/bash
# Setup.command: builds and opens the source-only setup app on this Mac.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Apple Command Line Tools are required to build this app."
  echo "Run xcode-select --install, finish Apple's installer, then open Setup.command again."
  read -r -p "Press Return to close. " _
  exit 1
fi
if ! /bin/bash "$ROOT/scripts/build-setup-app.sh" --open; then
  echo "Could not build the setup app. See the error above."
  read -r -p "Press Return to close. " _
  exit 1
fi
