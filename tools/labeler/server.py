#!/usr/bin/env python3
"""Labeling server: open the page on a phone, press buttons while you record (step 2.2).

Event times are taken from THIS computer's clock when the click arrives, the same
clock csi_reader.py uses for pc_time. So phone and CSI never need to be synced
with each other, only this computer needs a correct clock (NTP).

Example:
    python tools/labeler/server.py --session data/raw/2026-11-12_office-walk

Then open the printed URL on the phone (same Wi-Fi as the computer).
Output: <session>/labels.csv with columns unix_time,iso_time,label,event,note
"""
from __future__ import annotations

import json
import secrets
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

import typer

LABELS = ["empty", "standing", "walking", "entering", "leaving", "interference"]
HEADER = "unix_time,iso_time,label,event,note\n"
PAGE = Path(__file__).with_name("index.html")


def csv_field(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    return '"' + text.replace('"', '""') + '"' if any(c in text for c in ',"') else text


class LabelLog:
    """Thread-safe append-only log. Pressing a label ends the previous one and starts the new one."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        self.lock = threading.Lock()
        self.current: Optional[str] = None
        self.since: Optional[float] = None
        self.recent: List[dict] = []
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", encoding="utf-8")
        if new_file:
            self._fh.write(HEADER)
            self._fh.flush()

    def _write(self, t: float, label: str, event: str, note: str) -> None:
        iso = datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")
        self._fh.write(f"{t:.3f},{iso},{label},{event},{csv_field(note)}\n")
        self._fh.flush()
        self.recent = (self.recent + [{"t": t, "label": label, "event": event, "note": note}])[-6:]

    def press(self, label: str, note: str = "") -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
        with self.lock:
            t = self.clock()
            if self.current:
                self._write(t, self.current, "end", "")
            self._write(t, label, "start", note.strip())
            self.current, self.since = label, t

    def stop(self) -> None:
        with self.lock:
            if self.current:
                self._write(self.clock(), self.current, "end", "")
            self.current, self.since = None, None

    def state(self) -> dict:
        with self.lock:
            return {"current": self.current, "since": self.since, "now": self.clock(),
                    "recent": list(self.recent), "labels": LABELS}

    def close(self) -> None:
        self.stop()
        self._fh.close()


def make_handler(log: LabelLog, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode())

        def _authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Token", ""), token)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                self._json(200, log.state()) if self._authorized() else self._json(403, {"error": "bad token"})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorized():
                return self._json(403, {"error": "bad token"})
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 4096)
                data = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/event":
                    log.press(str(data.get("label", "")), str(data.get("note", ""))[:200])
                elif self.path == "/stop":
                    log.stop()
                else:
                    return self._json(404, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            self._json(200, log.state())

    return Handler


def lan_ip() -> str:
    """Address the phone should use. A UDP 'connect' sends nothing; it only picks a route."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


app = typer.Typer(add_completion=False, help="Phone labeling page for CSI recording sessions.")


@app.command()
def main(
    session: Path = typer.Option(..., help="Session folder, e.g. data/raw/2026-11-12_office-walk."),
    port: int = typer.Option(8000, min=1, max=65535),
    host: str = typer.Option("0.0.0.0", help="Interface to listen on (0.0.0.0 = whole local network)."),
) -> None:
    log = LabelLog(session / "labels.csv")
    token = secrets.token_urlsafe(6)
    server = ThreadingHTTPServer((host, port), make_handler(log, token))
    typer.echo(f"Writing labels to {log.path}")
    typer.echo(f"Open on the phone:  http://{lan_ip()}:{port}/?t={token}")
    typer.echo("Make sure this computer's clock is synced (NTP). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        log.close()
        typer.echo("Stopped.")


if __name__ == "__main__":
    app()
