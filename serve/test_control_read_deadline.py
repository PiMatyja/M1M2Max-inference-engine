"""Control requests have one body-read budget, independent of model loading time."""
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class ControlReadDeadline(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.handler = self.httpd.RequestHandlerClass
        budget = mock.patch.object(self.handler, "CONTROL_BODY_SECONDS", 0.2)
        budget.start()
        self.addCleanup(budget.stop)

    def test_trickling_control_bodies_stop_without_running_controls(self):
        cases = ((b"Content-Length: 100", b"{"),
                 (b"Transfer-Encoding: chunked", b"40\r\n{"),
                 (b"Transfer-Encoding: chunked", b"1"),
                 (b"Transfer-Encoding: chunked", b"0\r\nX-Slow: "))
        for path in ("/load", "/unload"):
            for framing, body in cases:
                with self.subTest(path=path, framing=framing, body=body), \
                        mock.patch.object(self.svc, "load") as load, \
                        mock.patch.object(self.svc, "unload") as unload:
                    with socket.create_connection(self.httpd.server_address, timeout=3) as sock:
                        sock.sendall(f"POST {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n".encode() +
                                     b"Content-Type: application/json\r\n" + framing + b"\r\n\r\n" + body)
                        stop = threading.Event()

                        def trickle():
                            while not stop.wait(0.03):
                                try:
                                    sock.sendall(b" ")
                                except OSError:
                                    return

                        writer = threading.Thread(target=trickle)
                        start = time.monotonic()
                        writer.start()
                        try:
                            response = http.client.HTTPResponse(sock)
                            response.begin()
                            self.assertEqual(response.status, 400)
                            self.assertIn("incomplete control request body", json.loads(response.read())["error"]["message"])
                            self.assertLess(time.monotonic() - start, 1.5)
                            load.assert_not_called()
                            unload.assert_not_called()
                        finally:
                            stop.set()
                            writer.join(2)
                        self.assertFalse(writer.is_alive())

    def test_completed_control_body_clears_deadline_before_slow_operation(self):
        original = self.handler._control_body
        observed = []

        def checked(handler):
            result = original(handler)
            observed.append((handler.rfile.deadline, handler.connection.gettimeout()))
            return result

        def slow_load():
            time.sleep(0.3)

        with mock.patch.object(self.handler, "_control_body", checked), \
                mock.patch.object(self.svc, "load", side_effect=slow_load) as load:
            for framing, body in ((b"Content-Length: 2", b"{}"),
                                  (b"Transfer-Encoding: chunked", b"2\r\n{}\r\n0\r\n\r\n")):
                with self.subTest(framing=framing):
                    with socket.create_connection(self.httpd.server_address, timeout=3) as sock:
                        sock.sendall(b"POST /load HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n" +
                                     framing + b"\r\n\r\n" + body)
                        response = http.client.HTTPResponse(sock)
                        response.begin()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read())["status"], "loaded")
            self.assertEqual(load.call_count, 2)
        self.assertEqual(observed, [(None, None), (None, None)])


if __name__ == "__main__":
    unittest.main()
