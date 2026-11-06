"""Studies (grids of experiments) and the single-worker experiment scheduler."""
from __future__ import annotations

import copy
import logging
import queue
import threading
import time
import traceback
from collections import defaultdict

import numpy as np

from common.config import MODES, validate_config

from .orchestrator import Cancelled, ExperimentRunner

log = logging.getLogger("coordinator.scheduler")

STUDY_KINDS = {
    "modes": "Method comparison",
    "privacy": "Privacy level sweep",
    "noniid": "Non-IID severity sweep",
    "custom": "Custom grid",
}


def _set_path(cfg: dict, path: str, value) -> None:
    cur = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def expand_study(spec: dict) -> list[tuple[str, dict, dict]]:
    """Turn a study spec into [(name, config, tags)]; every config is validated."""
    base = spec["base"]
    modes = spec.get("modes") or [base.get("mode", "trust_adaptive")]
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown mode {m!r}")
    seeds = spec.get("seeds") or [base.get("seed", 42)]
    vary = spec.get("vary") or {}
    if len(vary) > 1:
        raise ValueError("vary supports one parameter per study")
    vkey, vvals = next(iter(vary.items())) if vary else (None, [None])
    if vary and not vvals:
        raise ValueError("vary needs at least one value")
    out = []
    for v in vvals:
        for mode in modes:
            for seed in seeds:
                cfg = copy.deepcopy(base)
                cfg["mode"], cfg["seed"] = mode, int(seed)
                label = ""
                if vkey:
                    if vkey == "dp.epsilon" and (v is None or v == "off"):
                        _set_path(cfg, "dp.enabled", False)
                        label = "no DP"
                    else:
                        if vkey == "dp.epsilon":
                            _set_path(cfg, "dp.enabled", True)
                            _set_path(cfg, "dp.sigma", None)
                        _set_path(cfg, vkey, v)
                        label = f"{vkey}={v}"
                cfg = validate_config(cfg)
                name = f"{spec['name']} | {mode}" + (f" | {label}" if label else "") + f" | seed {seed}"
                cfg["name"] = name
                out.append((name, cfg, {"mode": mode, "seed": int(seed), "vary_key": vkey,
                                        "vary_value": "off" if (vkey and label == "no DP") else v}))
    return out


def aggregate_study(experiments: list[dict]) -> dict:
    """Group finished experiments by (vary value, mode) and average over seeds."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in experiments:
        if e["status"] == "completed" and e["summary"]:
            t = e["tags"]
            groups[(str(t.get("vary_value")), t.get("mode"))].append(e)
    rows = []
    for (vv, mode), items in groups.items():
        def stat(key):
            vals = [i["summary"][key] for i in items if i["summary"].get(key) is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
        acc_m, acc_s = stat("final_mean_acc")
        min_m, _ = stat("final_min_acc")
        conv_m, _ = stat("convergence_round")
        comm_m, _ = stat("total_comm_bytes")
        eps_m, _ = stat("epsilon_total")
        gain_m, _ = stat("gain_over_warmup")
        t_acc, _ = stat("teacher_acc_mean")
        t_uni, _ = stat("teacher_uniform_acc_mean")
        rows.append({"vary_value": vv, "mode": mode, "runs": len(items), "acc_mean": acc_m, "acc_std": acc_s,
                     "min_acc_mean": min_m, "convergence_round": conv_m, "comm_bytes": comm_m,
                     "epsilon_total": eps_m, "gain": gain_m, "teacher_acc": t_acc, "teacher_uniform_acc": t_uni,
                     "experiment_ids": [i["id"] for i in items]})
    rows.sort(key=lambda r: (_sort_key(r["vary_value"]), r["mode"]))
    return {"rows": rows}


def _sort_key(v: str):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, 0.0) if v != "off" else (2, 0.0)


class Scheduler:
    """Runs queued experiments one at a time (campus nodes are shared resources)."""

    def __init__(self, db, token: str, data_dir: str):
        self.db, self.token, self.data_dir = db, token, data_dir
        self.q: queue.Queue = queue.Queue()
        self.cancel_flags: dict[str, threading.Event] = {}
        self.current: str | None = None
        self.stage: str = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()

    def enqueue(self, eid: str) -> None:
        with self._lock:
            self.cancel_flags[eid] = threading.Event()
        self.q.put(eid)

    def cancel(self, eid: str) -> bool:
        with self._lock:
            ev = self.cancel_flags.get(eid)
        if ev is None:
            return False
        ev.set()
        exp = self.db.get_experiment(eid)
        if exp and exp["status"] == "queued":
            self.db.update_experiment(eid, status="cancelled", finished_at=time.time())
        return True

    def queue_size(self) -> int:
        return self.q.qsize()

    def _loop(self) -> None:
        while True:
            eid = self.q.get()
            try:
                self._run_one(eid)
            except Exception:  # noqa: BLE001
                log.exception("scheduler crashed on %s", eid)
            finally:
                self.current, self.stage = None, ""
                self._refresh_study(eid)

    def _run_one(self, eid: str) -> None:
        exp = self.db.get_experiment(eid)
        flag = self.cancel_flags.get(eid) or threading.Event()
        if exp is None or exp["status"] != "queued" or flag.is_set():
            return
        self.current = eid
        self.db.update_experiment(eid, status="running", started_at=time.time())
        self._refresh_study(eid)
        nodes = {n["id"]: n for n in self.db.list_nodes()}

        def set_status(stage: str = "") -> None:
            self.stage = stage

        try:
            runner = ExperimentRunner(eid, exp["config"], nodes, self.db.save_round, set_status, flag,
                                      self.token, self.data_dir)
            summary = runner.run()
            self.db.update_experiment(eid, status="completed", finished_at=time.time(), summary=summary)
        except Cancelled:
            self.db.update_experiment(eid, status="cancelled", finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            log.error("experiment %s failed: %s", eid, e)
            log.debug(traceback.format_exc())
            self.db.update_experiment(eid, status="failed", finished_at=time.time(), error=str(e))

    def _refresh_study(self, eid: str) -> None:
        exp = self.db.get_experiment(eid)
        if not exp or not exp.get("study_id"):
            return
        exps = self.db.list_experiments(study_id=exp["study_id"])
        st = {e["status"] for e in exps}
        if st & {"queued", "running"}:
            status = "running"
        elif "failed" in st:
            status = "failed"
        elif st == {"cancelled"}:
            status = "cancelled"
        else:
            status = "completed"
        self.db.update_study(exp["study_id"], status=status)
