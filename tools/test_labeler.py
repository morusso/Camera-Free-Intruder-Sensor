"""Tests for the labeling server. Run from the repo root: python -m unittest discover -s tools"""
import csv
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "labeler"))
import server as lab


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class LogTest(unittest.TestCase):
    def test_press_ends_previous_label(self):
        with tempfile.TemporaryDirectory() as d:
            log = lab.LabelLog(Path(d) / "labels.csv", FakeClock())
            log.press("empty")
            log.press("walking", "route A, fast")
            log.stop()
            log.close()
            rows = read_rows(Path(d) / "labels.csv")
        self.assertEqual([(r["label"], r["event"]) for r in rows],
                         [("empty", "start"), ("empty", "end"), ("walking", "start"), ("walking", "end")])
        self.assertEqual(rows[1]["unix_time"], rows[2]["unix_time"])  # end and next start share one instant
        self.assertEqual(rows[2]["note"], "route A, fast")

    def test_note_with_comma_and_quote_survives(self):
        with tempfile.TemporaryDirectory() as d:
            log = lab.LabelLog(Path(d) / "labels.csv", FakeClock())
            log.press("interference", 'fan, "speed 2"')
            log.close()
            rows = read_rows(Path(d) / "labels.csv")
        self.assertEqual(rows[0]["note"], 'fan, "speed 2"')

    def test_unknown_label_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            log = lab.LabelLog(Path(d) / "labels.csv", FakeClock())
            with self.assertRaises(ValueError):
                log.press("dancing")
            log.close()
            self.assertEqual(read_rows(Path(d) / "labels.csv"), [])

    def test_close_ends_active_label_and_append_keeps_one_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            log = lab.LabelLog(p, FakeClock())
            log.press("standing")
            log.close()
            lab.LabelLog(p, FakeClock()).close()
            text = p.read_text()
            last_event = read_rows(p)[-1]["event"]
        self.assertEqual(text.count("unix_time"), 1)
        self.assertEqual(last_event, "end")


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = lab.LabelLog(Path(self.dir.name) / "labels.csv")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), lab.make_handler(self.log, "tok"))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.log.close()
        self.dir.cleanup()

    def request(self, path, body=None, token="tok"):
        req = urllib.request.Request(self.base + path, headers={"X-Token": token},
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page_is_served(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"CSI labeler", r.read())

    def test_event_flow(self):
        code, st = self.request("/event", {"label": "walking", "note": "x"})
        self.assertEqual((code, st["current"]), (200, "walking"))
        code, st = self.request("/stop", {})
        self.assertEqual((code, st["current"]), (200, None))
        self.assertEqual(len(read_rows(Path(self.dir.name) / "labels.csv")), 2)

    def test_bad_token_and_bad_label(self):
        self.assertEqual(self.request("/state", token="wrong")[0], 403)
        self.assertEqual(self.request("/event", {"label": "walking"}, token="wrong")[0], 403)
        self.assertEqual(self.request("/event", {"label": "nope"})[0], 400)
        self.assertEqual(len(read_rows(Path(self.dir.name) / "labels.csv")), 0)


if __name__ == "__main__":
    unittest.main()
