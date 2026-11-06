"""Campus node HTTP service.

Run one per campus:

    python -m campus.app --name "Campus A" --port 5001
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import requests
from flask import Flask, jsonify, render_template, request

from common.security import TOKEN_HEADER, get_token, token_ok

from .node import CampusError, CampusNode

log = logging.getLogger("campus")


def _need(body: dict, *keys):
    missing = [k for k in keys if k not in body]
    if missing:
        raise CampusError(f"missing field(s): {', '.join(missing)}")


def create_app(name: str = "Campus", data_dir: str = "data") -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    node = CampusNode(name, data_dir)
    app.extensions["node"] = node

    @app.before_request
    def _auth():
        if request.path.startswith("/api/") and request.path != "/api/health":
            if not token_ok(request.headers.get(TOKEN_HEADER)):
                return jsonify(error="invalid or missing token"), 401

    @app.errorhandler(CampusError)
    def _campus_error(e: CampusError):
        return jsonify(error=str(e)), e.status

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        log.exception("unhandled error")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def home():
        return render_template("status.html", name=node.name, token_hint=get_token() == "fedtrust-dev-token")

    @app.get("/status.json")
    def status_public():
        i = node.info()
        i.pop("activity", None)
        i.pop("class_counts", None)
        return jsonify(i)

    # -------------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        return jsonify(ok=True, name=node.name, state=node.state)

    @app.get("/api/info")
    def info():
        return jsonify(node.info())

    @app.post("/api/init")
    def init():
        body = request.get_json(force=True)
        _need(body, "dataset", "seed", "campus_index", "n_campuses", "alpha", "test_frac", "anchor_frac",
              "model", "lr", "batch_size")
        return jsonify(node.init(body))

    @app.post("/api/train")
    def train():
        body = request.get_json(force=True)
        _need(body, "epochs")
        return jsonify(node.train_local(int(body["epochs"])))

    @app.post("/api/predict")
    def predict():
        body = request.get_json(force=True)
        _need(body, "indices", "sigma")
        return jsonify(node.predict_anchor(body["indices"], float(body["sigma"]),
                                           body.get("wire_dtype", "float16"), int(body.get("round", 0))))

    @app.post("/api/distill")
    def distill():
        body = request.get_json(force=True)
        _need(body, "indices", "teacher", "lambda", "temperature", "epochs")
        return jsonify(node.distill(body["indices"], body["teacher"], body.get("confidence"),
                                    float(body["lambda"]), float(body["temperature"]), int(body["epochs"])))

    @app.post("/api/evaluate")
    def evaluate():
        return jsonify(node.evaluate())

    return app


def _self_register(node_name: str, public_url: str, coordinator_url: str) -> None:
    """Optional: announce this node to the coordinator until it accepts us."""
    while True:
        try:
            r = requests.post(
                coordinator_url.rstrip("/") + "/api/nodes/register",
                json={"name": node_name, "url": public_url},
                headers={TOKEN_HEADER: get_token()}, timeout=5,
            )
            if r.ok:
                log.info("registered with coordinator %s", coordinator_url)
                return
        except requests.RequestException:
            pass
        time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description="FedTrust campus node")
    ap.add_argument("--name", default=os.environ.get("CAMPUS_NAME", "Campus"))
    ap.add_argument("--host", default=os.environ.get("CAMPUS_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CAMPUS_PORT", "5001")))
    ap.add_argument("--data-dir", default=os.environ.get("FEDTRUST_DATA_DIR", "data"))
    ap.add_argument("--public-url", default=os.environ.get("CAMPUS_PUBLIC_URL"))
    ap.add_argument("--coordinator", default=os.environ.get("COORDINATOR_URL"),
                    help="if set, the node registers itself with this coordinator")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if get_token() == "fedtrust-dev-token" and args.host not in ("127.0.0.1", "localhost"):
        log.warning("running on a non-local interface with the default token; set FEDTRUST_TOKEN!")

    app = create_app(args.name, args.data_dir)
    if args.coordinator:
        url = args.public_url or f"http://{args.host}:{args.port}"
        threading.Thread(target=_self_register, args=(args.name, url, args.coordinator), daemon=True).start()
    log.info("campus '%s' listening on http://%s:%d", args.name, args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
