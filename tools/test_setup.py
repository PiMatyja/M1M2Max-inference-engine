"""Setup regressions with synthetic sdists and no package installation."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def download_program():
    return (ROOT / "scripts/python-env.sh").read_text().split("<<'PYTHON'\n", 1)[1].split("\nPYTHON", 1)[0]


class SourceSetupTests(unittest.TestCase):
    def archive(self, name="demo-1.0"):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for file, data in [("pyproject.toml", b"[build-system]\nrequires=[]\n"),
                               ("launcher.exe", b"not source")]:
                info = tarfile.TarInfo(name + "/" + file)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return output.getvalue()

    def run_download(self, root, archive, expected=None):
        digest = expected or hashlib.sha256(archive).hexdigest()
        requirements = root / "requirements.txt"
        requirements.write_text(f"demo==1.0 --hash=sha256:{digest}\n")
        metadata = {"urls": [
            {"packagetype": "bdist_wheel", "digests": {"sha256": digest}, "filename": "demo.whl", "url": "bad"},
            {"packagetype": "sdist", "digests": {"sha256": digest}, "filename": "demo-1.0.tar.gz",
             "url": "https://files.pythonhosted.org/demo-1.0.tar.gz"}]}
        responses = [io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(archive)]
        with patch("sys.argv", ["setup", str(requirements), str(root)]), \
                patch("urllib.request.urlopen", side_effect=responses) as network:
            exec(compile(download_program(), "python-env.sh", "exec"), {})
        self.assertEqual(network.call_count, 2)

    def test_exact_verified_source_ignores_stale_sdists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo-0.1.tar.gz").write_bytes(b"unverified stale data")
            self.run_download(root, self.archive())
            order = (root / "install-order").read_text()
            self.assertEqual(order, f"demo\t{root}/demo/demo-1.0\n")
            self.assertFalse((root / "demo/demo-1.0/launcher.exe").exists())

    def test_hash_mismatch_never_reaches_build_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.run_download(root, self.archive(), "0" * 64)
            self.assertFalse((root / "install-order").exists())
            self.assertFalse((root / "demo").exists())

    def test_all_sources_verified_before_unpacking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive()
            digest = hashlib.sha256(archive).hexdigest()
            requirements = root / "requirements.txt"
            requirements.write_text(f"demo==1.0 --hash=sha256:{digest}\nother==1.0 --hash=sha256:{'0' * 64}\n")
            def metadata(expected):
                return io.BytesIO(json.dumps({"urls": [{"packagetype": "sdist",
                    "digests": {"sha256": expected}, "filename": "source.tar.gz",
                    "url": "https://files.pythonhosted.org/source.tar.gz"}]}).encode())
            responses = [metadata(digest), io.BytesIO(archive), metadata("0" * 64), io.BytesIO(archive)]
            with patch("sys.argv", ["setup", str(requirements), str(root)]), \
                    patch("urllib.request.urlopen", side_effect=responses):
                with self.assertRaisesRegex(ValueError, "hash mismatch: other"):
                    exec(compile(download_program(), "python-env.sh", "exec"), {})
            self.assertFalse((root / "demo").exists())
            self.assertFalse((root / "install-order").exists())

    def test_build_tools_precede_their_consumers(self):
        names = [line.split("==", 1)[0] for line in (ROOT / "requirements-mac.txt").read_text().splitlines()
                 if line and not line.startswith("#")]
        for package, dependencies in (("meson-python", ("meson", "ninja", "packaging", "pyproject-metadata")),
                                      ("numpy", ("meson-python", "cython", "ninja")),
                                      ("pyyaml", ("setuptools", "cython"))):
            for dependency in dependencies:
                with self.subTest(package=package, dependency=dependency):
                    self.assertLess(names.index(dependency), names.index(package))

    def test_no_pip_download_or_isolated_install(self):
        script = (ROOT / "scripts/python-env.sh").read_text()
        self.assertNotIn('"$PIP" download', script)
        self.assertIn("--no-build-isolation --no-index --no-cache-dir --no-binary :all:", script)
        self.assertIn("--wrap-mode=nodownload", script)
        self.assertNotIn('find "$DL"', script)


class CheckScriptTests(unittest.TestCase):
    def test_statuses(self):
        cases = [([], 1, "no model"),
                 ([{"id": "test", "status": {"value": "loading"}}], 1, "test: loading"),
                 ([{"id": "test", "status": {"value": "loaded"}, "meta": {"n_ctx": 42}}],
                  0, "test: loaded, context 42")]
        with tempfile.TemporaryDirectory() as temporary:
            curl = Path(temporary) / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s" "$TEST_BODY"\n')
            curl.chmod(0o755)
            for data, status, text in cases:
                with self.subTest(data=data):
                    env = dict(os.environ, PATH=temporary + os.pathsep + os.environ["PATH"],
                               TEST_BODY=json.dumps({"data": data}))
                    result = subprocess.run(["bash", str(ROOT / "scripts/check.sh")], env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, status, result.stderr)
                    self.assertIn(text, result.stdout)
                    self.assertNotIn("SyntaxError", result.stderr)


if __name__ == "__main__":
    unittest.main()
