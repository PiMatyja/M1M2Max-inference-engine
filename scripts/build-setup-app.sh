#!/bin/bash
# build-setup-app.sh: compiles the local macOS setup app without fetching dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/build-launcher/Strata Setup.app"
[ "$(uname -s)" = Darwin ] || { echo "This app requires macOS." >&2; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Install Apple Command Line Tools: xcode-select --install" >&2; exit 1; }
mkdir -p "$APP/Contents/MacOS"
xcrun swiftc -swift-version 5 -target arm64-apple-macosx15.0 -O -framework AppKit -framework Foundation \
  "$ROOT/macos/SetupApp/main.swift" -o "$APP/Contents/MacOS/StrataSetup"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>StrataSetup</string>
<key>CFBundleIdentifier</key><string>local.strata.source-setup</string>
<key>CFBundleName</key><string>Strata Setup</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1.0</string>
<key>LSMinimumSystemVersion</key><string>15.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
echo "Built $APP"
echo "Local ad-hoc signature only; this is not a notarized distribution."
if [ "${1:-}" = --open ]; then open "$APP"; fi
