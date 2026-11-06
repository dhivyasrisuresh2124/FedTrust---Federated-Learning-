"""A campus node: owns private data, trains a local model, shares only
privacy-protected predictions on the public anchor set."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from common.data import build_campus_data, load_dataset
from common.models import build_model, count_params
from common.privacy import privatize
from common.serde import decode_array, encode_array

torch.set_num_threads(int(__import__("os").environ.get("FEDTRUST_TORCH_THREADS", "1")))


class CampusError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class CampusNode:
    def __init__(self, name: str, data_dir: str = "data"):
        self.name = name
        self.data_dir = data_dir
        self.lock = threading.RLock()
        self.state = "uninitialized"          # uninitialized | ready | busy
        self.activity: deque = deque(maxlen=200)
        self.cfg: dict = {}
        self.model = None
        self.opt = None
        self.data = None
        self.gen = torch.Generator()
        self.n_params = 0
        self.epochs_done = 0
        self.releases = 0
        self.started = time.time()
        self.current_op: str | None = None

    # ------------------------------------------------------------------ helpers
    def _log(self, op: str, detail: str = "", **extra) -> None:
        self.activity.appendleft({"t": time.time(), "op": op, "detail": detail, **extra})

    def _idle(self) -> None:
        self.state = "ready" if (self.model is not None and self.data is not None) else "uninitialized"
        self.current_op = None

    def _require_ready(self) -> None:
        if self.model is None or self.data is None:
            raise CampusError("node is not initialised; the coordinator must call /init first", 409)

    def _batches(self, n: int, bs: int):
        perm = torch.randperm(n, generator=self.gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) >= 2:           # BatchNorm needs >1 sample
                yield b

    def info(self) -> dict:
        """Lock-free status snapshot (safe to call while training)."""
        d = self.data
        return {
            "name": self.name,
            "state": self.state,
            "op": self.current_op,
            "uptime_s": round(time.time() - self.started, 1),
            "experiment_id": self.cfg.get("experiment_id"),
            "model": self.cfg.get("model"),
            "n_params": self.n_params,
            "n_private": int(len(d.y_train)) if d is not None else None,
            "class_counts": d.class_counts if d is not None else None,
            "n_anchor": int(len(d.X_anchor)) if d is not None else None,
            "epochs_done": self.epochs_done,
            "releases": self.releases,
            "label_noise": self.cfg.get("label_noise"),
            "activity": list(self.activity)[:30],
        }

    # --------------------------------------------------------------------- init
    def init(self, cfg: dict) -> dict:
        with self.lock:
            self.state, self.current_op = "busy", "init"
            try:
                bundle = load_dataset(cfg["dataset"], self.data_dir, cfg.get("max_samples"), cfg["seed"])
                self.data = build_campus_data(
                    bundle, seed=cfg["seed"], campus_index=cfg["campus_index"], n_campuses=cfg["n_campuses"],
                    alpha=cfg["alpha"], test_frac=cfg["test_frac"], anchor_frac=cfg["anchor_frac"],
                    label_noise=cfg.get("label_noise", 0.0), data_fraction=cfg.get("data_fraction", 1.0),
                )
                seed = cfg["seed"] * 1000 + cfg["campus_index"]
                torch.manual_seed(seed)
                self.gen.manual_seed(seed)
                self.model = build_model(cfg["model"], bundle.in_shape, bundle.n_classes)
                self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
                self.n_params = count_params(self.model)
                self.cfg = dict(cfg)
                self.epochs_done = 0
                self.releases = 0
                self._Xtr = torch.from_numpy(self.data.X_train)
                self._ytr = torch.from_numpy(self.data.y_train)
                self._Xan = torch.from_numpy(self.data.X_anchor)
                self._Xte = torch.from_numpy(self.data.X_test)
                self._yte = torch.from_numpy(self.data.y_test)
                self._log("init", f"{cfg['model']} | {len(self._ytr)} private samples | "
                                  f"label noise {cfg.get('label_noise', 0):.0%}")
                return {"n_private": len(self._ytr), "n_anchor": len(self._Xan),
                        "n_test": len(self._Xte), "n_params": self.n_params,
                        "class_counts": self.data.class_counts, "n_classes": bundle.n_classes}
            finally:
                self._idle()

    # ----------------------------------------------------------- local training
    def train_local(self, epochs: int) -> dict:
        with self.lock:
            self._require_ready()
            self.state, self.current_op = "busy", "train"
            t0 = time.time()
            try:
                self.model.train()
                total, steps = 0.0, 0
                bs = self.cfg["batch_size"]
                for _ in range(epochs):
                    for b in self._batches(len(self._ytr), bs):
                        self.opt.zero_grad()
                        loss = F.cross_entropy(self.model(self._Xtr[b]), self._ytr[b])
                        loss.backward()
                        self.opt.step()
                        total += loss.item()
                        steps += 1
                self.epochs_done += epochs
                res = {"loss": total / max(steps, 1), "steps": steps, "seconds": time.time() - t0}
                self._log("train", f"{epochs} epoch(s), loss {res['loss']:.3f}")
                return res
            finally:
                self._idle()

    # ------------------------------------------------- privacy-protected release
    def predict_anchor(self, indices: list[int], sigma: float, wire_dtype: str, round_no: int) -> dict:
        """Predict on the anchor subset and release ONLY the noised probabilities."""
        with self.lock:
            self._require_ready()
            self.state, self.current_op = "busy", "predict"
            try:
                idx = torch.as_tensor(indices, dtype=torch.long)
                if idx.numel() == 0 or int(idx.max()) >= len(self._Xan) or int(idx.min()) < 0:
                    raise CampusError("anchor indices out of range")
                self.model.eval()
                with torch.no_grad():
                    probs = F.softmax(self.model(self._Xan[idx]), dim=1).numpy()
                rng = np.random.default_rng([self.cfg["seed"], self.cfg["campus_index"], round_no, 17])
                released = privatize(probs, float(sigma), rng)
                self.releases += 1
                self._log("release", f"{len(indices)} anchor predictions, sigma={sigma:.3f}")
                return {"probs": encode_array(released, wire_dtype)}
            finally:
                self._idle()

    # ------------------------------------------------------- knowledge distillation
    def distill(self, indices: list[int], teacher: dict, confidence: dict | None, lam: float,
                temperature: float, epochs: int) -> dict:
        """Joint objective per step:  CE(local batch) + lam * T^2 * KL(teacher_T || student_T)(anchor batch)."""
        with self.lock:
            self._require_ready()
            self.state, self.current_op = "busy", "distill"
            t0 = time.time()
            try:
                idx = torch.as_tensor(indices, dtype=torch.long)
                q = torch.from_numpy(decode_array(teacher)).float()
                if q.shape[0] != idx.numel():
                    raise CampusError("teacher size does not match the anchor indices")
                if confidence is not None:
                    w = torch.from_numpy(decode_array(confidence)).float().flatten()
                    w = w / w.mean().clamp_min(1e-6)
                else:
                    w = torch.ones(idx.numel())
                T = float(temperature)
                qt = q.clamp_min(1e-8).pow(1.0 / T)
                qt = qt / qt.sum(dim=1, keepdim=True)
                Xan = self._Xan[idx]
                bs = self.cfg["batch_size"]
                abs_ = min(2 * bs, len(idx))
                self.model.train()
                tot_ce = tot_kd = 0.0
                steps = 0
                for _ in range(epochs):
                    for b in self._batches(len(self._ytr), bs):
                        self.opt.zero_grad()
                        ce = F.cross_entropy(self.model(self._Xtr[b]), self._ytr[b])
                        a = torch.randperm(len(idx), generator=self.gen)[:abs_]
                        if len(a) >= 2 and lam > 0:
                            logp = F.log_softmax(self.model(Xan[a]) / T, dim=1)
                            kl = (qt[a] * (qt[a].clamp_min(1e-8).log() - logp)).sum(dim=1)
                            kd = (w[a] * kl).mean() * (T * T)
                        else:
                            kd = torch.zeros(())
                        (ce + lam * kd).backward()
                        self.opt.step()
                        tot_ce += ce.item()
                        tot_kd += kd.item()
                        steps += 1
                self.epochs_done += epochs
                res = {"ce": tot_ce / max(steps, 1), "kd": tot_kd / max(steps, 1), "steps": steps,
                       "lambda": lam, "seconds": time.time() - t0}
                self._log("distill", f"lambda={lam:.2f}, T={T:g}, kd={res['kd']:.3f}")
                return res
            finally:
                self._idle()

    # ---------------------------------------------------------------- evaluation
    def evaluate(self) -> dict:
        with self.lock:
            self._require_ready()
            self.model.eval()
            with torch.no_grad():
                logits = torch.cat([self.model(self._Xte[i:i + 512]) for i in range(0, len(self._Xte), 512)])
                loss = float(F.cross_entropy(logits, self._yte))
                pred = logits.argmax(dim=1)
            acc = float((pred == self._yte).float().mean())
            per_class = []
            for c in range(self.data.n_classes):
                m = self._yte == c
                per_class.append(float((pred[m] == c).float().mean()) if int(m.sum()) else None)
            self._log("evaluate", f"test acc {acc:.3f}")
            return {"accuracy": acc, "loss": loss, "per_class": per_class, "n_test": int(len(self._yte))}
