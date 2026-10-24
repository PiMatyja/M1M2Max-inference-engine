"""#879: a Stop that arrives while the engine reads the prompt (PP lines, no tokens yet) is sent at once, not after the
first token."""
import contextlib
import json
import os
import socket
import time
from pathlib import Path
from unittest import mock
import queue
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import server  # noqa: E402


class ControlCancelTest(unittest.TestCase):
    def _engine(self, lines):
        eng = object.__new__(server.StrataEngine)
        eng.lines = queue.Queue()
        for l in lines:
            eng.lines.put(l)
        eng.sent = []
        eng._send = eng.sent.append
        eng._ctl_mode = "solo"
        eng._ctl_result = None
        eng.progress = None
        eng.prefill_tok_s_mean = None
        eng._last_done = None
        eng._parse_done = lambda line: None
        return eng

    def test_stop_during_prompt_read(self):
        cancel = threading.Event()
        eng = self._engine(["PP 256 4096 1 100.0", "PP 512 4096 2 100.0", "PP 768 4096 3 100.0", "DONE 0"])
        gen = eng._control(cancel, lambda t: None)
        self.assertIsNone(next(gen))               # first PP line, nothing cancelled yet
        self.assertEqual(eng.sent, [])
        cancel.set()
        self.assertIsNone(next(gen))               # the next PP line: STOP goes out now
        self.assertEqual(eng.sent, ["STOP"])
        self.assertIsNone(next(gen))
        self.assertEqual(eng.sent, ["STOP"])       # once

    def test_no_stop_without_cancel(self):
        eng = self._engine(["PP 256 4096 1 100.0", "DONE 0"])
        for _ in eng._control(threading.Event(), lambda t: None):
            pass
        self.assertEqual(eng.sent, [])


class LoadCancelTest(unittest.TestCase):
    def setUp(self):
        tok = server.ByteTokenizer()
        self.svc = server.Service(server.MockEngine(tok, "ok"), tok,
                                  server.ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_disconnected_generation_does_not_load_after_waiting(self):
        httpd = server.serve(self.svc, port=0)
        try:
            for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/responses", {"input": "hi", "store": False})):
                with self.subTest(path=path), mock.patch.object(self.svc, "loaded", return_value=False), \
                        mock.patch.object(self.svc, "ensure_loaded") as load:
                    self.svc.fifo.acquire()
                    try:
                        with socket.create_connection(httpd.server_address, timeout=3) as sock:
                            data = json.dumps(body).encode()
                            sock.sendall(f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
                            self.wait_for(lambda: self.svc.status["queued"] == 1)
                        self.wait_for(lambda: self.svc.status["queued"] == 0)
                    finally:
                        self.svc.fifo.release()
                    load.assert_not_called()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cancel_is_checked_after_acquiring_turn(self):
        cancel = threading.Event()
        @contextlib.contextmanager
        def turn(event):
            self.assertIs(event, cancel)
            cancel.set()
            yield
        with mock.patch.object(self.svc, "loaded", return_value=False), \
                mock.patch.object(self.svc, "_turn", turn), mock.patch.object(self.svc, "ensure_loaded") as load:
            with self.assertRaises(server.ClientGone):
                self.svc.load(cancel)
            load.assert_not_called()
        self.assertEqual(self.svc.status["queued"], 0)

    def test_explicit_load_still_works_without_cancel(self):
        with mock.patch.object(self.svc, "loaded", return_value=False), \
                mock.patch.object(self.svc, "ensure_loaded") as load:
            self.svc.load()
            load.assert_called_once_with()
        self.assertEqual(self.svc.status["queued"], 0)


if __name__ == "__main__":
    unittest.main()
