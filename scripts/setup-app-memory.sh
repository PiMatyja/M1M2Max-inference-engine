#!/bin/bash
# setup-app-memory.sh: requests permission for the temporary GPU memory limit after preflight.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/bin/bash "$ROOT/scripts/setup-app.sh" check
exec /usr/bin/osascript -e 'do shell script "/usr/sbin/sysctl iogpu.wired_limit_mb=58000" with administrator privileges'
