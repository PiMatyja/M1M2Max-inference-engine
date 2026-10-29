"""Test setup orchestration with fake tools and no model, network or GPU changes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-setup-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source with spaces"
        self.scripts = self.root / "scripts"
        self.scripts.mkdir(parents=True)
        shutil.copyfile(SOURCE / "scripts/setup-app.sh", self.scripts / "setup-app.sh")
        (self.root / "macos").mkdir()
        shutil.copyfile(SOURCE / "macos/supervise.py", self.root / "macos/supervise.py")
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        self.env = {**os.environ, "PATH": str(self.bin) + ":/usr/bin:/bin:/usr/sbin:/sbin",
                    "STRATA_MODELS": str(Path(self.temp.name) / "models with spaces"),
                    "STAGE_LOG": str(self.root / "stages"), "TEST_CHIP": "Apple M2 Max",
                    "TEST_MEMORY": "68719476736", "TEST_OS": "15.0", "TEST_LIMIT": "58000"}
        self.tool("sysctl", '''case "$2" in
machdep.cpu.brand_string) echo "$TEST_CHIP" ;;
hw.memsize) echo "$TEST_MEMORY" ;;
iogpu.wired_limit_mb) echo "$TEST_LIMIT" ;;
*) exit 2 ;;
esac''')
        self.tool("sw_vers", 'echo "$TEST_OS"')
        self.tool("xcode-select", 'exit 0')
        self.tool("cmake", 'exit 0')
        self.tool("python3.14", 'exit "${TEST_PYTHON_EXIT:-0}"')
        for stage in ["fetch-deps.sh", "build.sh", "python-env.sh", "download-model.sh",
                      "prepare-model.sh", "start.sh", "check.sh"]:
            script = self.scripts / stage
            script.write_text('#!/bin/bash\nset -eu\n'
                              f'echo {stage} >> "$STAGE_LOG"\n'
                              f'[ "${{FAIL_STAGE:-}}" != {stage} ] || exit 7\n')
        venv = self.root / ".venv/bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        python.write_text('#!/bin/bash\nset -eu\n'
                          'if [ "${1##*/}" = supervise.py ]; then exec /usr/bin/python3 "$@"; fi\n'
                          'echo config >> "$STAGE_LOG"\n'
                          'printf "%s" "$STRATA_MODELS" > strata-cached.json\n')
        python.chmod(0o755)

    def tool(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_setup(self, action):
        return subprocess.run(["/bin/bash", str(self.scripts / "setup-app.sh"), action],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def stages(self):
        path = self.root / "stages"
        return path.read_text().splitlines() if path.exists() else []

    def test_supported_chips(self):
        for chip in ["Apple M1 Max", "Apple M2 Max"]:
            with self.subTest(chip=chip):
                self.env["TEST_CHIP"] = chip
                result = self.run_setup("check")
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.stages(), [])

    def test_unsupported_machine_does_not_install(self):
        for key, value in [("TEST_CHIP", "Apple M4 Max"), ("TEST_MEMORY", "34359738368"),
                           ("TEST_OS", "14.7"), ("TEST_PYTHON_EXIT", "1")]:
            with self.subTest(key=key):
                old = self.env.get(key)
                self.env[key] = value
                result = self.run_setup("install")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.stages(), [])
                if old is None:
                    del self.env[key]
                else:
                    self.env[key] = old

    def test_install_orders_stages_and_preserves_folder(self):
        result = self.run_setup("install")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.stages(), ["fetch-deps.sh", "build.sh", "python-env.sh",
                                        "download-model.sh", "prepare-model.sh", "config"])
        self.assertEqual((self.root / "strata-cached.json").read_text(), self.env["STRATA_MODELS"])

    def test_existing_config_is_not_overwritten(self):
        config = self.root / "strata-cached.json"
        config.write_text("existing settings")
        result = self.run_setup("install")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config.read_text(), "existing settings")
        self.assertNotIn("config", self.stages())

    def test_failed_stage_stops_later_stages(self):
        self.env["FAIL_STAGE"] = "build.sh"
        result = self.run_setup("install")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.stages(), ["fetch-deps.sh", "build.sh"])
        self.assertFalse((self.root / "strata-cached.json").exists())

    def test_start_requires_memory_limit(self):
        self.env["TEST_LIMIT"] = "40000"
        result = self.run_setup("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU memory limit", result.stderr)
        self.assertEqual(self.stages(), [])
        self.env["TEST_LIMIT"] = "58000"
        result = self.run_setup("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        # Start goes through macos/supervise.py, so the real supervisor runs the fake start stage here.
        self.assertEqual(self.stages(), ["start.sh"])

    def test_start_supervises_the_process_group(self):
        # The server and any engine it spawns must live in one group the supervisor owns and reaps.
        self.script_stage_marker = self.root / "start-pgid"
        (self.scripts / "start.sh").write_text(
            '#!/bin/bash\nset -eu\n'
            'echo start.sh >> "$STAGE_LOG"\n'
            f'echo $PPID > "{self.script_stage_marker}"\n')
        result = self.run_setup("start")
        self.assertEqual(result.returncode, 0, result.stderr)
        supervisor_pid = int(self.script_stage_marker.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(supervisor_pid, 0)   # the supervisor is gone only after the whole group ended

    def test_status_delegates_to_existing_check(self):
        self.assertEqual(self.run_setup("status").returncode, 0)
        self.assertEqual(self.stages(), ["check.sh"])

    def test_unknown_action_is_rejected(self):
        self.assertEqual(self.run_setup("update").returncode, 2)
        self.assertEqual(self.stages(), [])

    def test_shell_syntax(self):
        for name in ["setup-app.sh", "setup-app-memory.sh", "build-setup-app.sh"]:
            result = subprocess.run(["/bin/bash", "-n", str(SOURCE / "scripts" / name)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(subprocess.run(["/bin/bash", "-n", str(SOURCE / "Setup.command")]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
