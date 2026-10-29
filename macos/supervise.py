#!/usr/bin/env python3
"""Keep ownership of the server process group until its engine has exited."""
import argparse
import math
import os
import signal
import subprocess
import time


def supervise(command, grace_seconds=10.0):
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def exists(group):
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False

    def send(group, sig):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    child = None
    try:
        # A stop during Popen only sets a flag, so ownership is recorded before cancellation.
        child = subprocess.Popen(command, start_new_session=True)
        group = child.pid
        while child.poll() is None and not stopping:
            time.sleep(0.05)
        if exists(group):
            print("[setup] stopping the owned server and engine process group ...", flush=True)
            send(group, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            killed = False
            while True:
                child.poll()
                if not exists(group):
                    break
                if not killed and time.monotonic() >= deadline:
                    print("[setup] cleanup grace expired; ending the owned process group ...", flush=True)
                    send(group, signal.SIGKILL)
                    killed = True
                time.sleep(0.05)
        result = child.wait()
        print("[setup] owned server and engine have exited", flush=True)
        return 0 if stopping else (result if result >= 0 else 128 - result)
    finally:
        # Unexpected supervisor errors still end and reap the direct child.
        if child is not None and child.poll() is None:
            send(child.pid, signal.SIGKILL)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.grace_seconds) or args.grace_seconds <= 0:
        parser.error("provide a command and a positive finite cleanup grace")
    if os.name != "posix":
        parser.error("this launcher requires POSIX process groups")
    return supervise(command, args.grace_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
