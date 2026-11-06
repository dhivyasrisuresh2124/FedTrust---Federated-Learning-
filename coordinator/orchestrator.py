"""Runs one federated experiment by driving the campus services over HTTP.

One round (proposed method)
---------------------------
 1. every campus trains locally on its private data            POST /api/train
 2. every campus releases noise-protected anchor predictions    POST /api/predict
 3. coordinator: entropy -> trust -> trust-weighted teacher
 4. every campus distils the teacher into its local model       POST /api/distill
 5. every campus is evaluated on the common held-out test set   POST /api/evaluate
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from common import trust as T
from common.data import anchor_labels, load_dataset
from common.privacy import PrivacyAccountant, calibrate_sigma, epsilon_for_sigma
from common.security import TOKEN_HEADER
from common.serde import decode_array, encode_array


class Cancelled(Exception):
    pass


class NodeError(Exception):
    pass


class NodeClient:
    """Thin HTTP client that also meters the bytes sent and received."""

    def __init__(self, name: str, base_url: str, token: str, timeout: float = 900.0):
        self.name = name
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers[TOKEN_HEADER] = token
        self.timeout = timeout
        self.up = 0
        self.down = 0

    def call(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        url = self.base + path
        last: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.request(method, url, json=payload, timeout=timeout or self.timeout)
                body = r.request.body or b""
                self.up += len(body) if isinstance(body, (bytes, str)) else 0
                self.down += len(r.content)
                if r.status_code >= 400:
                    try:
                        msg = r.json().get("error", r.text)
                    except ValueError:
                        msg = r.text[:200]
                    raise NodeError(f"{self.name}: HTTP {r.status_code} on {path}: {msg}")
                return r.json()
            except requests.ConnectionError as e:     # safe to retry: request never reached the node
                last = e
                time.sleep(0.5 * (attempt + 1))
            except requests.Timeout as e:
                raise NodeError(f"{self.name}: timeout on {path}") from e
        raise NodeError(f"{self.name}: cannot connect to {self.base} ({last.__class__.__name__})")

    def take_bytes(self) -> tuple[int, int]:
        u, d = self.up, self.down
        self.up = self.down = 0
        return u, d


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if len(y) else float("nan")


class ExperimentRunner:
    def __init__(self, exp_id: str, cfg: dict, nodes: dict[str, dict], save_round, set_status,
                 cancel: threading.Event, token: str, data_dir: str = "data"):
        self.id = exp_id
        self.cfg = cfg
        self.save_round = save_round
        self.set_status = set_status
        self.cancel = cancel
        self.data_dir = data_dir
        self.clients: list[dict] = []
        for i, c in enumerate(cfg["campuses"]):
            node = nodes.get(c["node_id"])
            if node is None:
                raise NodeError(f"unknown node {c['node_id']}")
            self.clients.append({**c, "idx": i, "name": node["name"],
                                 "client": NodeClient(node["name"], node["url"], token)})
        self.K = len(self.clients)
        self.trust_state: dict[int, float] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def _parallel(self, fn, targets: list[dict]) -> tuple[dict[int, object], dict[int, str]]:
        """Run fn(campus) on every target in parallel. Returns (results, errors) keyed by campus idx."""
        ok: dict[int, object] = {}
        bad: dict[int, str] = {}
        if not targets:
            return ok, bad
        with ThreadPoolExecutor(max_workers=len(targets)) as ex:
            futs = {c["idx"]: ex.submit(fn, c) for c in targets}
            for idx, f in futs.items():
                try:
                    ok[idx] = f.result()
                except Exception as e:  # noqa: BLE001 - one failing campus must not kill the round
                    bad[idx] = str(e)
        return ok, bad

    def _privacy_setup(self) -> tuple[float, float, float]:
        dp = self.cfg["dp"]
        if not dp["enabled"]:
            return 0.0, math.inf, dp.get("delta", 1e-5)
        delta = float(dp["delta"])
        if dp.get("sigma") is not None:
            sigma = float(dp["sigma"])
            return sigma, epsilon_for_sigma(sigma, delta), delta
        eps = float(dp["epsilon"])
        return calibrate_sigma(eps, delta), eps, delta

    # --------------------------------------------------------------------- main
    def run(self) -> dict:
        cfg, K = self.cfg, self.K
        mode = cfg["mode"]
        self.set_status(stage="connecting to campuses")

        # 0. verify every node is reachable and the token is accepted
        _, bad = self._parallel(lambda c: c["client"].call("GET", "/api/info", timeout=10), self.clients)
        if bad:
            raise NodeError("; ".join(bad.values()))

        # 1. initialise campuses (each derives its own private shard from the shared seed)
        self.set_status(stage="partitioning data")

        def _init(c):
            payload = {
                "experiment_id": self.id, "dataset": cfg["dataset"], "max_samples": cfg["max_samples"],
                "seed": cfg["seed"], "campus_index": c["idx"], "n_campuses": K, "alpha": cfg["alpha"],
                "test_frac": cfg["test_frac"], "anchor_frac": cfg["anchor_frac"], "model": c["model"],
                "label_noise": c["label_noise"], "data_fraction": c["data_fraction"],
                "lr": cfg["lr"], "batch_size": cfg["batch_size"],
            }
            return c["client"].call("POST", "/api/init", payload)

        inits, bad = self._parallel(_init, self.clients)
        if bad:
            raise NodeError("initialisation failed: " + "; ".join(bad.values()))
        n_anchor = min(v["n_anchor"] for v in inits.values())
        n_classes = next(iter(inits.values()))["n_classes"]
        for c in self.clients:
            c["n_private"], c["n_params"] = inits[c["idx"]]["n_private"], inits[c["idx"]]["n_params"]
            c["class_counts"] = inits[c["idx"]]["class_counts"]

        # evaluation oracle: ground-truth anchor labels, used only to *score* the teacher
        bundle = load_dataset(cfg["dataset"], self.data_dir, cfg["max_samples"], cfg["seed"])
        y_anchor = anchor_labels(bundle, cfg["seed"], cfg["test_frac"], cfg["anchor_frac"])

        sigma, eps_step, delta = self._privacy_setup()
        accountant = PrivacyAccountant(eps_step, delta)
        collaborative = mode != "local_only"
        fedavg_bytes = int(sum(c["n_params"] for c in self.clients) * 4 * 2)  # up+down, fp32, per round

        # 2. warm-up local training, then record round 0 (the starting point)
        if cfg["warmup_epochs"] > 0:
            self.set_status(stage="warm-up training")
            _, bad = self._parallel(lambda c: c["client"].call("POST", "/api/train", {"epochs": cfg["warmup_epochs"]}),
                                    self.clients)
            if bad:
                raise NodeError("warm-up failed: " + "; ".join(bad.values()))
        self._check()
        self._record_round(0, None, None, None, None, None, accountant, sigma, eps_step, {}, K, final=False)

        # 3. federated rounds
        for r in range(1, cfg["n_rounds"] + 1):
            self._check()
            t_round = time.time()
            self.set_status(stage=f"round {r}/{cfg['n_rounds']}")
            errors: dict[int, str] = {}
            if not collaborative:
                ep = cfg["local_epochs"] + cfg["distill_epochs"]   # equal compute budget as collaborative modes
                _, bad = self._parallel(lambda c: c["client"].call("POST", "/api/train", {"epochs": ep}), self.clients)
                errors.update(bad)
                self._record_round(r, None, None, None, None, None, accountant, sigma, eps_step, errors,
                                   K, t_round=t_round, final=(r == cfg["n_rounds"]))
                continue

            rng = np.random.default_rng(cfg["seed"] * 7919 + r)
            m = min(cfg["anchor_per_round"], n_anchor)
            idx = np.sort(rng.choice(n_anchor, size=m, replace=False))

            # 3a. local training
            ok, bad = self._parallel(lambda c: c["client"].call("POST", "/api/train", {"epochs": cfg["local_epochs"]}),
                                     self.clients)
            errors.update(bad)
            self._check()

            # 3b. privacy-protected prediction release
            alive = [c for c in self.clients if c["idx"] not in errors]
            rel, bad = self._parallel(lambda c: c["client"].call("POST", "/api/predict", {
                "indices": idx.tolist(), "sigma": sigma, "wire_dtype": cfg["wire_dtype"], "round": r}), alive)
            errors.update(bad)
            if len(rel) < 2:
                raise NodeError(f"round {r}: fewer than 2 campuses answered ({'; '.join(errors.values())})")
            accountant.step()
            contrib = [c for c in self.clients if c["idx"] in rel]
            P = np.stack([decode_array(rel[c["idx"]]["probs"]) for c in contrib])      # (K', m, C)
            self._check()

            # 3c. entropy -> trust -> teacher
            floor1 = T.noise_floor_entropy(sigma, n_classes, 1)
            floor_t = T.noise_floor_entropy(sigma, n_classes, len(contrib))
            H = T.calibrate_entropy(T.normalized_entropy(P), floor1)                     # (K', m)
            mean_H = H.mean(axis=1)
            if mode == "uniform_fd":
                trust_vec = np.full(len(contrib), 1.0 / len(contrib))
                W = T.uniform_weights(len(contrib), m)
            else:
                prev = None
                if all(c["idx"] in self.trust_state for c in contrib):
                    prev = np.array([self.trust_state[c["idx"]] for c in contrib])
                    prev = prev / prev.sum()
                trust_vec = T.smooth_trust(prev, T.campus_trust(mean_H, cfg["trust"]["tau"]), cfg["trust"]["ema"])
                W = T.sample_weights(H, trust_vec, cfg["trust"]["tau_sample"])
            for c, t in zip(contrib, trust_vec):
                self.trust_state[c["idx"]] = float(t)
            q = T.aggregate_teacher(P, W)
            conf = T.consensus_confidence(P, W, q)
            quality = float(conf.mean())
            if mode == "trust_adaptive":
                dcfg = cfg["distill"]
                lam = T.adaptive_lambda(trust_vec, quality, dcfg["lambda_max"], dcfg["lambda_floor"])
            else:
                lam = np.full(len(contrib), cfg["distill"]["lambda_fixed"])

            ya = y_anchor[idx]
            teacher_diag = {
                "acc": _acc(q.argmax(1), ya),
                "uniform_acc": _acc(T.aggregate_teacher(P, T.uniform_weights(len(contrib), m)).argmax(1), ya),
                "mean_entropy": float(T.calibrate_entropy(T.normalized_entropy(q), floor_t).mean()),
                "quality": quality, "n_anchor": int(m),
            }
            released_acc = {c["idx"]: _acc(P[i].argmax(1), ya) for i, c in enumerate(contrib)}
            diag = {c["idx"]: {"trust": float(trust_vec[i]), "lambda": float(lam[i]), "entropy": float(mean_H[i]),
                               "released_acc": released_acc[c["idx"]], "weight": float(W[i].mean())}
                    for i, c in enumerate(contrib)}

            # 3d. distillation
            lam_by_idx = {c["idx"]: float(lam[i]) for i, c in enumerate(contrib)}
            wire = cfg["wire_dtype"]
            q_enc, conf_enc = encode_array(q, wire), encode_array(conf, wire)
            dis, bad = self._parallel(lambda c: c["client"].call("POST", "/api/distill", {
                "indices": idx.tolist(), "teacher": q_enc, "confidence": conf_enc, "lambda": lam_by_idx[c["idx"]],
                "temperature": cfg["distill"]["temperature"], "epochs": cfg["distill_epochs"]}), contrib)
            errors.update(bad)
            for k, v in dis.items():
                diag[k]["kd_loss"] = v["kd"]

            self._record_round(r, diag, teacher_diag, trust_vec, contrib, fedavg_bytes, accountant, sigma,
                               eps_step, errors, K, t_round=t_round, final=(r == cfg["n_rounds"]))

        summary = self._summarise(accountant, sigma, eps_step, fedavg_bytes)
        return summary

    # ---------------------------------------------------------------- recording
    def _record_round(self, r, diag, teacher, trust_vec, contrib, fedavg_bytes, accountant, sigma, eps_step,
                      errors, K, t_round=None, final=False):
        """Evaluate every campus, meter communication and persist the round."""
        ev, bad = self._parallel(lambda c: c["client"].call("POST", "/api/evaluate", {}), self.clients)
        errors = {**errors, **bad}
        campuses = []
        for c in self.clients:
            e = ev.get(c["idx"])
            d = (diag or {}).get(c["idx"], {})
            up, down = c["client"].take_bytes()
            entry = {
                "idx": c["idx"], "node_id": c["node_id"], "name": c["name"], "model": c["model"],
                "n_private": c["n_private"], "n_params": c["n_params"],
                "acc": e["accuracy"] if e else None, "loss": e["loss"] if e else None,
                "trust": d.get("trust"), "lambda": d.get("lambda"), "entropy": d.get("entropy"),
                "released_acc": d.get("released_acc"), "weight": d.get("weight"), "kd_loss": d.get("kd_loss"),
                "bytes_up": up, "bytes_down": down, "error": errors.get(c["idx"]),
            }
            if final and e:
                entry["per_class"] = e["per_class"]
            campuses.append(entry)
        accs = [c["acc"] for c in campuses if c["acc"] is not None]
        spent = accountant.spent()
        total_bytes = sum(c["bytes_up"] + c["bytes_down"] for c in campuses)
        rec = {
            "round": r,
            "mean_acc": float(np.mean(accs)) if accs else None,
            "min_acc": float(np.min(accs)) if accs else None,
            "max_acc": float(np.max(accs)) if accs else None,
            "std_acc": float(np.std(accs)) if accs else None,
            "mean_loss": float(np.mean([c["loss"] for c in campuses if c["loss"] is not None])) if accs else None,
            "campuses": campuses,
            "teacher": teacher,
            "privacy": {"sigma": sigma, "epsilon_step": None if math.isinf(eps_step) else eps_step,
                        "epsilon_total": None if math.isinf(spent["epsilon"]) else spent["epsilon"],
                        "delta_total": spent["delta"], "method": spent["method"]},
            "comm": {"bytes": total_bytes, "fedavg_equiv_bytes": fedavg_bytes},
            "seconds": (time.time() - t_round) if t_round else None,
            "errors": list(errors.values()),
        }
        self.history.append(rec)
        self.save_round(self.id, r, rec)

    def _summarise(self, accountant, sigma, eps_step, fedavg_bytes) -> dict:
        h = self.history
        final = h[-1]
        means = [x["mean_acc"] for x in h if x["mean_acc"] is not None]
        start = h[0]["mean_acc"]
        target = start + 0.95 * (final["mean_acc"] - start)
        conv = next((x["round"] for x in h if x["mean_acc"] is not None and x["mean_acc"] >= target), final["round"])
        K = self.K
        trust_means = []
        for k in range(K):
            vals = [x["campuses"][k]["trust"] for x in h if x["campuses"][k]["trust"] is not None]
            trust_means.append(float(np.mean(vals)) if vals else None)
        pairs = [(c["trust"], c["released_acc"]) for x in h for c in x["campuses"]
                 if c.get("trust") is not None and c.get("released_acc") is not None]
        corr = None
        if len(pairs) >= 4:
            a, b = np.array(pairs).T
            if a.std() > 1e-9 and b.std() > 1e-9:
                corr = float(np.corrcoef(a, b)[0, 1])
        teach = [x["teacher"] for x in h if x.get("teacher")]
        spent = accountant.spent()
        return {
            "mode": self.cfg["mode"],
            "final_mean_acc": final["mean_acc"],
            "final_min_acc": final["min_acc"],
            "final_std_acc": final["std_acc"],
            "best_mean_acc": float(max(means)) if means else None,
            "start_mean_acc": start,
            "gain_over_warmup": (final["mean_acc"] - start) if start is not None else None,
            "convergence_round": conv,
            "total_comm_bytes": int(sum(x["comm"]["bytes"] for x in h[1:])),
            "fedavg_equiv_bytes": int(fedavg_bytes * self.cfg["n_rounds"]),
            "sigma": sigma,
            "epsilon_step": None if math.isinf(eps_step) else eps_step,
            "epsilon_total": None if math.isinf(spent["epsilon"]) else spent["epsilon"],
            "mean_trust": trust_means,
            "trust_vs_released_acc_corr": corr,
            "teacher_acc_final": teach[-1]["acc"] if teach else None,
            "teacher_uniform_acc_final": teach[-1]["uniform_acc"] if teach else None,
            "teacher_acc_mean": float(np.mean([t["acc"] for t in teach])) if teach else None,
            "teacher_uniform_acc_mean": float(np.mean([t["uniform_acc"] for t in teach])) if teach else None,
        }
