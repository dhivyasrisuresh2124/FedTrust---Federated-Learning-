#!/usr/bin/env python3
"""Start a local federation: N campus services + the coordinator, each in its own process.

    python scripts/start_cluster.py --campuses 4

Every campus is a real, independent Flask server on its own port; the coordinator talks to
them over HTTP exactly as it would across a network. Press Ctrl+C to stop everything.
"""
from __future__ import annotations

import argparse
import base64
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["Campus A", "Campus B", "Campus C", "Campus D", "Campus E", "Campus F", "Campus G", "Campus H"]


def wait_http(url: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


def post_json(url: str, body: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campuses", type=int, default=4, choices=range(2, 9), metavar="2-8")
    ap.add_argument("--base-port", type=int, default=5001)
    ap.add_argument("--coordinator-port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-register", action="store_true", help="do not auto-register the campuses")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONUNBUFFERED": "1"}
    procs: list[subprocess.Popen] = []

    def stop(*_):
        print("\nstopping…")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    campus_urls = []
    for i in range(args.campuses):
        port = args.base_port + i
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "campus.app", "--name", NAMES[i], "--host", args.host, "--port", str(port)],
            cwd=ROOT, env=env))
        campus_urls.append((NAMES[i], f"http://{args.host}:{port}"))
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "coordinator.app", "--host", args.host, "--port", str(args.coordinator_port)],
        cwd=ROOT, env=env))

    coord = f"http://{args.host}:{args.coordinator_port}"
    if not wait_http(coord + "/api/health"):
        print("coordinator did not start", file=sys.stderr)
        stop()
    headers = {}
    if os.environ.get("COORDINATOR_ADMIN_PASSWORD"):
        cred = base64.b64encode(f"admin:{os.environ['COORDINATOR_ADMIN_PASSWORD']}".encode()).decode()
        headers["Authorization"] = f"Basic {cred}"
    for name, url in campus_urls:
        if not wait_http(url + "/api/health"):
            print(f"{name} did not start", file=sys.stderr)
            stop()
        if not args.no_register:
            post_json(coord + "/api/nodes", {"name": name, "url": url}, headers)

    print("\nFedTrust federation is up\n")
    for name, url in campus_urls:
        print(f"  {name:<10} {url}")
    print(f"\n  Coordinator dashboard  ->  {coord}\n\nCtrl+C to stop.")
    while True:
        for p in procs:
            if p.poll() is not None:
                print("a process exited unexpectedly; shutting down", file=sys.stderr)
                stop()
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
