"""Stop-during-startup regression tests for macos/supervise.py (no GPU, no model)."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]
SUPERVISE = REPO / "macos" / "supervise.py"


class SuperviseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="strata-supervise-test-")
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve()

    def start_supervisor(self, script, grace=None):
        args = [sys.executable, str(SUPERVISE)] + ([] if grace is None else ["--grace-seconds", str(grace)]) + \
               ["--", "/bin/sh", "-c", script]
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.work)

    def start_and_wait_ready(self, script, grace=None):
        proc = self.start_supervisor(script, grace)
        deadline = time.monotonic() + 20.0
        while not (self.work / "ready").exists():
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise AssertionError("the supervised command never became ready")
            time.sleep(0.02)
        return proc

    def finish(self, proc, wait_s=20.0):
        try:
            out, _ = proc.communicate(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            raise AssertionError("the supervisor did not exit in time:\n" + out)
        return out

    def assert_group_gone(self):
        pgid = int((self.work / "pgid").read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.killpg(pgid, 0)

    def test_passes_exit_code_when_left_alone(self):
        proc = self.start_supervisor('echo $$ > pgid; exit 3')
        out = self.finish(proc)
        self.assertEqual(proc.returncode, 3, out)
        self.assert_group_gone()

    def test_stop_forwards_to_group_and_waits_for_delayed_engine(self):
        marker = self.work / "engine-done"
        script = (f'echo $$ > pgid\n'
                  f'sh -c \'trap "" TERM; sleep 1.5; touch "{marker}"\' &\n'
                  f'touch ready\n'
                  f'sleep 300')
        proc = self.start_and_wait_ready(script)
        proc.send_signal(signal.SIGTERM)
        started = time.monotonic()
        out = self.finish(proc)
        waited = time.monotonic() - started
        # The engine ignores SIGTERM for 1.5 s; the supervisor must outlive it, not just the server.
        self.assertGreaterEqual(waited, 1.4)
        self.assertTrue(marker.exists())
        self.assertIn("owned server and engine have exited", out)
        self.assert_group_gone()

    def test_grace_expiry_ends_stubborn_group(self):
        marker = self.work / "never"
        script = (f'echo $$ > pgid\n'
                  f'sh -c \'trap "" TERM; sleep 300; touch "{marker}"\' &\n'
                  f'touch ready\n'
                  f'sleep 300')
        proc = self.start_and_wait_ready(script, grace=0.8)
        proc.send_signal(signal.SIGTERM)
        started = time.monotonic()
        out = self.finish(proc)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 15)
        self.assertFalse(marker.exists())
        self.assertIn("ending the owned process group", out)
        self.assert_group_gone()

    def test_cancellation_during_startup_is_kept(self):
        # SIGTERM can arrive before the supervised command exists: the supervisor must still
        # adopt the group and end it, never leak the server it was about to own.
        marker = self.work / "engine-done-early"
        script = (f'echo $$ > pgid\n'
                  f'sh -c \'trap "" TERM; sleep 0.8; touch "{marker}"\' &\n'
                  f'touch ready\n'
                  f'sleep 300')
        proc = self.start_supervisor(script, grace=5)
        proc.send_signal(signal.SIGTERM)   # possibly before the child is even spawned
        out = self.finish(proc)
        pgid_file = self.work / "pgid"
        deadline = time.monotonic() + 4.0
        while not pgid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)   # the race may end the supervisor before anything was spawned at all
        if pgid_file.exists():
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists(), out)   # the trapped engine ran to the end of its sleep
            self.assert_group_gone()
        # Either outcome leaks nothing: no adopted group outlives the supervisor.

    def test_python_syntax(self):
        self.assertEqual(subprocess.run([sys.executable, "-m", "py_compile", str(SUPERVISE)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
