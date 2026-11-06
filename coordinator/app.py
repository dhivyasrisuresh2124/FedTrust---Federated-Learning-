"""Coordinator: web UI + REST API that orchestrates the campus nodes.

    python -m coordinator.app --port 5000
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, Response, abort, jsonify, render_template, request

from common.config import DATASETS, DEFAULT_CONFIG, MODEL_CHOICES, MODES, validate_config
from common.security import TOKEN_HEADER, get_token, token_ok

from .db import Database
from .studies import STUDY_KINDS, Scheduler, aggregate_study, expand_study

log = logging.getLogger("coordinator")
MAX_STUDY_RUNS = 80


def create_app(db_path: str | None = None, data_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    db_path = db_path or os.environ.get("FEDTRUST_DB", "instance/fedtrust.db")
    data_dir = data_dir or os.environ.get("FEDTRUST_DATA_DIR", "data")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    db = Database(db_path)
    interrupted = db.mark_interrupted()
    if interrupted:
        log.warning("marked %d unfinished experiment(s) from a previous run as failed", interrupted)
    sched = Scheduler(db, get_token(), data_dir)
    app.extensions.update(db=db, scheduler=sched)

    admin_pw = os.environ.get("COORDINATOR_ADMIN_PASSWORD")

    # ------------------------------------------------------------------ security
    @app.before_request
    def _guard():
        if request.path.startswith("/static/") or request.path == "/api/health":
            return None
        if request.path == "/api/nodes/register":          # nodes authenticate with the shared token
            if not token_ok(request.headers.get(TOKEN_HEADER)):
                return jsonify(error="invalid or missing token"), 401
            return None
        if admin_pw:
            h = request.headers.get("Authorization", "")
            ok = False
            if h.startswith("Basic "):
                try:
                    ok = base64.b64decode(h[6:]).decode().split(":", 1)[1] == admin_pw
                except Exception:  # noqa: BLE001
                    ok = False
            if not ok:
                return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="FedTrust"'})
        return None

    @app.errorhandler(ValueError)
    def _bad_request(e):
        return jsonify(error=str(e)), 400

    # --------------------------------------------------------------------- pages
    @app.get("/")
    def page_index():
        return render_template("index.html", page="home")

    @app.get("/nodes")
    def page_nodes():
        return render_template("nodes.html", page="nodes")

    @app.get("/experiments")
    def page_experiments():
        return render_template("experiments.html", page="experiments")

    @app.get("/experiments/new")
    def page_new():
        return render_template("new.html", page="experiments")

    @app.get("/experiments/<eid>")
    def page_experiment(eid):
        if not db.get_experiment(eid):
            abort(404)
        return render_template("experiment.html", page="experiments", eid=eid)

    @app.get("/compare")
    def page_compare():
        return render_template("compare.html", page="experiments")

    @app.get("/studies")
    def page_studies():
        return render_template("studies.html", page="studies")

    @app.get("/studies/<sid>")
    def page_study(sid):
        if not db.get_study(sid):
            abort(404)
        return render_template("study.html", page="studies", sid=sid)

    @app.get("/method")
    def page_method():
        return render_template("method.html", page="method")

    # ----------------------------------------------------------------------- API
    @app.get("/api/health")
    def api_health():
        return jsonify(ok=True, time=time.time())

    @app.get("/api/meta")
    def api_meta():
        return jsonify(defaults=DEFAULT_CONFIG, modes=MODES, models=MODEL_CHOICES, datasets=DATASETS,
                       study_kinds=STUDY_KINDS, max_study_runs=MAX_STUDY_RUNS,
                       running=sched.current, stage=sched.stage, queued=sched.queue_size())

    # ---- nodes
    def _probe(node: dict) -> dict:
        out = {**node, "online": False, "info": None, "error": None}
        try:
            r = requests.get(node["url"] + "/api/info", headers={TOKEN_HEADER: get_token()}, timeout=3)
            if r.status_code == 401:
                out["error"] = "token rejected"
            elif r.ok:
                out["online"], out["info"] = True, r.json()
            else:
                out["error"] = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            out["error"] = e.__class__.__name__
        return out

    @app.get("/api/nodes")
    def api_nodes():
        nodes = db.list_nodes()
        with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as ex:
            return jsonify(list(ex.map(_probe, nodes)))

    @app.post("/api/nodes")
    def api_add_node():
        body = request.get_json(force=True)
        url = (body.get("url") or "").strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        name = (body.get("name") or "").strip()
        try:
            r = requests.get(url + "/api/health", timeout=4)
            name = name or r.json().get("name", url)
        except (requests.RequestException, ValueError):
            if not body.get("force"):
                return jsonify(error=f"cannot reach a campus node at {url}"), 400
            name = name or url
        return jsonify(db.add_node(name, url)), 201

    @app.post("/api/nodes/register")
    def api_register():
        body = request.get_json(force=True)
        if not body.get("url"):
            raise ValueError("url required")
        return jsonify(db.add_node(body.get("name") or body["url"], body["url"])), 201

    @app.delete("/api/nodes/<nid>")
    def api_del_node(nid):
        return jsonify(deleted=db.delete_node(nid))

    # ---- experiments
    def _create_experiment(cfg: dict, study_id=None, tags=None, name=None) -> str:
        cfg = validate_config(cfg)
        known = {n["id"] for n in db.list_nodes()}
        for c in cfg["campuses"]:
            if c["node_id"] not in known:
                raise ValueError("unknown campus node selected")
        eid = db.create_experiment(name or cfg["name"], cfg, study_id, tags)
        sched.enqueue(eid)
        return eid

    @app.post("/api/experiments")
    def api_create_experiment():
        eid = _create_experiment(request.get_json(force=True))
        return jsonify(id=eid), 201

    @app.get("/api/experiments")
    def api_list_experiments():
        out = []
        for e in db.list_experiments(limit=int(request.args.get("limit", 300))):
            e.pop("config", None)
            out.append(e)
        return jsonify(out)

    @app.get("/api/experiments/<eid>")
    def api_get_experiment(eid):
        e = db.get_experiment(eid)
        if not e:
            abort(404)
        e["rounds"] = db.get_rounds(eid)
        if sched.current == eid:
            e["stage"] = sched.stage
        return jsonify(e)

    @app.post("/api/experiments/<eid>/cancel")
    def api_cancel(eid):
        return jsonify(cancelled=sched.cancel(eid))

    @app.delete("/api/experiments/<eid>")
    def api_delete_experiment(eid):
        e = db.get_experiment(eid)
        if not e:
            abort(404)
        if e["status"] in ("queued", "running"):
            raise ValueError("cancel the experiment before deleting it")
        db.delete_experiment(eid)
        return jsonify(deleted=True)

    @app.get("/api/experiments/<eid>/export.csv")
    def api_export(eid):
        e = db.get_experiment(eid)
        if not e:
            abort(404)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["experiment", "mode", "round", "campus", "model", "accuracy", "loss", "trust", "lambda",
                    "entropy", "released_accuracy", "bytes_up", "bytes_down", "sigma", "epsilon_total",
                    "teacher_acc", "teacher_uniform_acc"])
        for r in db.get_rounds(eid):
            for c in r["campuses"]:
                t = r.get("teacher") or {}
                w.writerow([eid, e["config"]["mode"], r["round"], c["name"], c["model"], c["acc"], c["loss"],
                            c["trust"], c["lambda"], c["entropy"], c["released_acc"], c["bytes_up"],
                            c["bytes_down"], r["privacy"]["sigma"], r["privacy"]["epsilon_total"],
                            t.get("acc"), t.get("uniform_acc")])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={eid}.csv"})

    @app.get("/api/compare")
    def api_compare():
        ids = [i for i in request.args.get("ids", "").split(",") if i]
        out = []
        for i in ids[:12]:
            e = db.get_experiment(i)
            if e:
                e["rounds"] = db.get_rounds(i)
                out.append(e)
        return jsonify(out)

    # ---- studies
    @app.post("/api/studies")
    def api_create_study():
        spec = request.get_json(force=True)
        if not spec.get("name"):
            raise ValueError("study name required")
        kind = spec.get("kind", "custom")
        if kind not in STUDY_KINDS:
            raise ValueError("unknown study kind")
        spec["base"] = validate_config(spec.get("base") or {})
        jobs = expand_study(spec)
        if len(jobs) > MAX_STUDY_RUNS:
            raise ValueError(f"study expands to {len(jobs)} runs; the limit is {MAX_STUDY_RUNS}")
        sid = db.create_study(spec["name"], kind, spec)
        for name, cfg, tags in jobs:
            _create_experiment(cfg, study_id=sid, tags=tags, name=name)
        return jsonify(id=sid, runs=len(jobs)), 201

    @app.get("/api/studies")
    def api_list_studies():
        out = []
        for s in db.list_studies():
            exps = db.list_experiments(study_id=s["id"])
            s["runs"] = len(exps)
            s["done"] = sum(e["status"] in ("completed", "failed", "cancelled") for e in exps)
            s.pop("config", None)
            out.append(s)
        return jsonify(out)

    @app.get("/api/studies/<sid>")
    def api_get_study(sid):
        s = db.get_study(sid)
        if not s:
            abort(404)
        exps = db.list_experiments(study_id=sid)
        for e in exps:
            e.pop("config", None)
        s["experiments"] = exps
        s["aggregate"] = aggregate_study(exps)
        return jsonify(s)

    @app.post("/api/studies/<sid>/cancel")
    def api_cancel_study(sid):
        n = 0
        for e in db.list_experiments(study_id=sid):
            if e["status"] in ("queued", "running"):
                n += int(sched.cancel(e["id"]))
        return jsonify(cancelled=n)

    @app.delete("/api/studies/<sid>")
    def api_delete_study(sid):
        if any(e["status"] in ("queued", "running") for e in db.list_experiments(study_id=sid)):
            raise ValueError("cancel the study before deleting it")
        db.delete_study(sid)
        return jsonify(deleted=True)

    @app.get("/api/studies/<sid>/export.csv")
    def api_study_export(sid):
        s = db.get_study(sid)
        if not s:
            abort(404)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["study", "experiment", "status", "mode", "seed", "vary_key", "vary_value", "final_mean_acc",
                    "final_min_acc", "convergence_round", "total_comm_bytes", "sigma", "epsilon_total",
                    "teacher_acc_mean", "teacher_uniform_acc_mean"])
        for e in db.list_experiments(study_id=sid):
            t, sm = e["tags"], e["summary"] or {}
            w.writerow([sid, e["id"], e["status"], t.get("mode"), t.get("seed"), t.get("vary_key"),
                        t.get("vary_value"), sm.get("final_mean_acc"), sm.get("final_min_acc"),
                        sm.get("convergence_round"), sm.get("total_comm_bytes"), sm.get("sigma"),
                        sm.get("epsilon_total"), sm.get("teacher_acc_mean"), sm.get("teacher_uniform_acc_mean")])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={sid}.csv"})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="FedTrust coordinator")
    ap.add_argument("--host", default=os.environ.get("COORDINATOR_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("COORDINATOR_PORT", "5000")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    if args.host not in ("127.0.0.1", "localhost") and get_token() == "fedtrust-dev-token":
        log.warning("non-local interface with the default token; set FEDTRUST_TOKEN and COORDINATOR_ADMIN_PASSWORD")
    app = create_app()
    log.info("coordinator listening on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
