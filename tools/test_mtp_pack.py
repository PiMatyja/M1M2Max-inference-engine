"""MTP publication regressions with a synthetic writer and no NumPy or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_pack_module():
    gguf = types.ModuleType("gguf")
    gguf.GGMLQuantizationType = types.SimpleNamespace(Q2_0=42, Q4_0=2, Q8_0=8)
    paths = types.ModuleType("_paths")
    paths.add_gguf_py = lambda: None
    spec = importlib.util.spec_from_file_location("tested_mtp_pack", ROOT / "tools/mtp_pack.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"numpy": types.ModuleType("numpy"), "gguf": gguf, "_paths": paths}), \
            patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


class MtpPackTests(unittest.TestCase):
    def setUp(self):
        self.pack = load_pack_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "mtp-manifest.json").write_text("[]")
        self.out = self.root / "pack.gguf"
        self.receipt = Path(str(self.out) + ".report.json")

    def run_pack(self, writer, reuse=False):
        args = ["mtp_pack", "--src", str(self.root), "--out", str(self.out)]
        if reuse:
            args.append("--reuse")
        with patch.object(sys, "argv", args), patch.object(self.pack, "write_pack", writer):
            return self.pack.main()

    @staticmethod
    def writer(args, src, manifest):
        Path(args.out).write_bytes(b"complete synthetic pack")
        return {"total_bytes": 23, "tensors": []}

    def test_success_receipt_and_verified_reuse(self):
        self.assertEqual(self.run_pack(self.writer), 0)
        report = json.loads(self.receipt.read_text())
        self.assertEqual(report["output_sha256"], hashlib.sha256(self.out.read_bytes()).hexdigest())
        with patch.object(self.pack, "write_pack", side_effect=AssertionError("unexpected repack")) as writer:
            self.assertEqual(self.run_pack(writer, reuse=True), 0)
            writer.assert_not_called()
        self.assertEqual(list(self.root.glob(".mtp-pack-*")), [])

    def test_failures_preserve_old_output_and_remove_temporary_files(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.out.write_bytes(b"previous valid output")
                def failing(args, src, manifest):
                    Path(args.out).write_bytes(b"partial")
                    raise RuntimeError("interrupted build")
                with self.assertRaisesRegex(RuntimeError, "interrupted build"):
                    self.run_pack(failing)
                self.assertEqual(self.out.exists(), existing)
                if existing:
                    self.assertEqual(self.out.read_bytes(), b"previous valid output")
                self.assertFalse(self.receipt.exists())
                self.assertEqual(list(self.root.glob(".mtp-pack-*")), [])

    def test_partial_legacy_changed_or_missing_pack_is_rebuilt(self):
        for damage in ("no receipt", "changed", "truncated", "missing", "source", "format", "bad receipt"):
            with self.subTest(damage=damage):
                self.run_pack(self.writer)
                if damage == "no receipt":
                    self.receipt.unlink()
                elif damage == "changed":
                    self.out.write_bytes(b"x" * self.out.stat().st_size)
                elif damage == "truncated":
                    self.out.write_bytes(b"partial")
                elif damage == "missing":
                    self.out.unlink()
                elif damage == "source":
                    (self.root / "mtp-manifest.json").write_text("[ ]")
                elif damage == "format":
                    report = json.loads(self.receipt.read_text())
                    report["expert_format"] = "q4_0"
                    self.receipt.write_text(json.dumps(report))
                else:
                    self.receipt.write_text("{")
                with patch.object(self.pack, "write_pack", side_effect=self.writer) as writer:
                    self.assertEqual(self.run_pack(writer, reuse=True), 0)
                    writer.assert_called_once()

    def test_prepare_always_invokes_receipt_check(self):
        script = (ROOT / "scripts/prepare-model.sh").read_text()
        self.assertIn('--out "$M/mtp/mtp-q2_0.gguf" --reuse', script)
        self.assertNotIn('if [ -f "$M/mtp/mtp-q2_0.gguf" ]', script)

    def test_real_write_path_closes_writer_on_failure(self):
        writer = types.SimpleNamespace(add_string=lambda *args: None,
                                       write_header_to_file=lambda: None,
                                       write_kv_data_to_file=lambda: None,
                                       write_tensors_to_file=lambda: (_ for _ in ()).throw(RuntimeError("write failed")))
        from unittest.mock import Mock
        writer.close = Mock()
        with patch.object(self.pack.gguf, "GGUFWriter", return_value=writer, create=True):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.pack.write_pack(types.SimpleNamespace(out=str(self.out), experts="q2_0"), self.root, [])
        writer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
