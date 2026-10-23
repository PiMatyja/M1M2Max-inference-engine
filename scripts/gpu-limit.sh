#!/bin/bash
# gpu-limit.sh: lets the GPU wire 58000 MiB (resets on reboot), and prints a LaunchDaemon that would make it stick.
set -euo pipefail
echo "running: sudo sysctl iogpu.wired_limit_mb=58000"
sudo sysctl iogpu.wired_limit_mb=58000
cat <<'PLIST'

To keep the limit across reboots, save this as /Library/LaunchDaemons/local.iogpu-wired-limit.plist
(owner root:wheel, mode 644) and run: sudo launchctl load -w /Library/LaunchDaemons/local.iogpu-wired-limit.plist

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.iogpu-wired-limit</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/sbin/sysctl</string>
    <string>iogpu.wired_limit_mb=58000</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
PLIST
