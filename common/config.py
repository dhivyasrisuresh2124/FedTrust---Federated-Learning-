"""Experiment configuration: defaults, merging and validation."""
from __future__ import annotations

import copy
from typing import Any

MODES = {
    "trust_adaptive": "Proposed: trust-weighted teacher + adaptive distillation",
    "trust_fixed": "Trust-weighted teacher + fixed distillation weight",
    "uniform_fd": "Baseline: uniform-average federated distillation (FedMD-style)",
    "local_only": "Baseline: no collaboration (local training only)",
}

MODEL_CHOICES = ["mlp_small", "mlp_wide", "cnn_small", "cnn_deep", "linear"]
DATASETS = ["digits", "mnist", "fashion_mnist"]

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "Untitled experiment",
    "seed": 42,
    "dataset": "digits",
    "max_samples": None,          # subsample large datasets (e.g. 12000 for MNIST)
    "test_frac": 0.2,
    "anchor_frac": 0.2,
    "alpha": 0.5,                 # Dirichlet concentration (smaller = more non-IID)
    "mode": "trust_adaptive",
    "n_rounds": 10,
    "warmup_epochs": 5,
    "local_epochs": 2,
    "distill_epochs": 2,
    "batch_size": 32,
    "lr": 0.002,
    "anchor_per_round": 200,
    "wire_dtype": "float16",
    "dp": {"enabled": True, "epsilon": 50.0, "delta": 1e-5, "sigma": None},
    "distill": {
        "temperature": 2.0,
        "lambda_max": 1.0,
        "lambda_floor": 0.4,
        "lambda_fixed": 0.7,
    },
    "trust": {"tau": 0.15, "tau_sample": 0.5, "ema": 0.5},
    # one entry per participating campus
    "campuses": [],
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_config(user_cfg: dict | None) -> dict:
    return _deep_merge(DEFAULT_CONFIG, user_cfg or {})


def _num(cfg, key, lo, hi, integer=False):
    v = cfg.get(key)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number")
    if integer and int(v) != v:
        raise ValueError(f"{key} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}")


def validate_config(cfg: dict) -> dict:
    """Return a cleaned config or raise ValueError with a human-readable reason."""
    cfg = merge_config(cfg)
    if cfg["mode"] not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if cfg["dataset"] not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    _num(cfg, "seed", 0, 2**31 - 1, True)
    _num(cfg, "alpha", 0.01, 1000)
    _num(cfg, "n_rounds", 1, 200, True)
    _num(cfg, "warmup_epochs", 0, 100, True)
    _num(cfg, "local_epochs", 1, 50, True)
    _num(cfg, "distill_epochs", 1, 50, True)
    _num(cfg, "batch_size", 4, 1024, True)
    _num(cfg, "lr", 1e-5, 1.0)
    _num(cfg, "anchor_per_round", 10, 100000, True)
    _num(cfg, "test_frac", 0.05, 0.5)
    _num(cfg, "anchor_frac", 0.05, 0.5)
    if cfg["wire_dtype"] not in ("float16", "float32"):
        raise ValueError("wire_dtype must be float16 or float32")

    dp = cfg["dp"]
    if dp["enabled"]:
        if dp.get("sigma") is None:
            _num(dp, "epsilon", 0.05, 10000)
        else:
            _num(dp, "sigma", 0.0, 100)
        _num(dp, "delta", 1e-12, 0.5)
    d = cfg["distill"]
    _num(d, "temperature", 0.5, 20)
    _num(d, "lambda_max", 0.0, 10)
    _num(d, "lambda_floor", 0.0, 1)
    _num(d, "lambda_fixed", 0.0, 10)
    t = cfg["trust"]
    _num(t, "tau", 0.01, 10)
    _num(t, "tau_sample", 0.01, 10)
    _num(t, "ema", 0.0, 0.99)

    camps = cfg["campuses"]
    if not isinstance(camps, list) or len(camps) < 2:
        raise ValueError("select at least 2 campuses")
    seen = set()
    for i, c in enumerate(camps):
        if not c.get("node_id"):
            raise ValueError(f"campus #{i + 1} has no node_id")
        if c["node_id"] in seen:
            raise ValueError("a campus node can only be used once per experiment")
        seen.add(c["node_id"])
        c.setdefault("model", MODEL_CHOICES[i % len(MODEL_CHOICES)])
        if c["model"] not in MODEL_CHOICES:
            raise ValueError(f"unknown model {c['model']!r}")
        c["label_noise"] = float(c.get("label_noise", 0.0))
        c["data_fraction"] = float(c.get("data_fraction", 1.0))
        if not 0 <= c["label_noise"] <= 0.9:
            raise ValueError("label_noise must be in [0, 0.9]")
        if not 0.1 <= c["data_fraction"] <= 1.0:
            raise ValueError("data_fraction must be in [0.1, 1]")
    return cfg
