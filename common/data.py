"""Datasets, deterministic splits and non-IID partitioning.

Every node (campus services and the coordinator) derives the *same* splits from
(dataset, seed, fractions), so no raw data ever has to be transferred:

    all samples -> test set | anchor set (public, unlabeled for learning) | private pool
    private pool -> Dirichlet(alpha) label-skew partition -> one shard per campus
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

_CACHE: dict[tuple, "Bundle"] = {}
_LOCK = threading.Lock()


@dataclass
class Bundle:
    X: np.ndarray          # float32, shape (N, C, H, W), standardised
    y: np.ndarray          # int64, shape (N,)
    n_classes: int
    in_shape: tuple[int, int, int]
    name: str


def _standardise(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mu, sd = float(X.mean()), float(X.std() + 1e-6)
    return (X - mu) / sd


def load_dataset(name: str = "digits", data_dir: str = "data", max_samples: int | None = None,
                 seed: int = 0) -> Bundle:
    key = (name, data_dir, max_samples, seed if max_samples else 0)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if name == "digits":
            from sklearn.datasets import load_digits

            d = load_digits()
            X = (d.images / 16.0)[:, None, :, :]
            y = d.target
        elif name in ("mnist", "fashion_mnist"):
            from sklearn.datasets import fetch_openml

            openml_name = "mnist_784" if name == "mnist" else "Fashion-MNIST"
            os.makedirs(data_dir, exist_ok=True)
            d = fetch_openml(openml_name, version=1, as_frame=False, data_home=data_dir)
            X = (d.data.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
            y = d.target.astype(np.int64)
        else:
            raise ValueError(f"unknown dataset {name!r}")
        if max_samples and max_samples < len(y):
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(y), size=max_samples, replace=False))
            X, y = X[keep], y[keep]
        X = _standardise(X)
        y = y.astype(np.int64)
        bundle = Bundle(X=X, y=y, n_classes=int(y.max()) + 1, in_shape=tuple(X.shape[1:]), name=name)
        _CACHE[key] = bundle
        return bundle


@dataclass
class Splits:
    test: np.ndarray
    anchor: np.ndarray
    private: np.ndarray


def make_splits(y: np.ndarray, seed: int, test_frac: float = 0.2, anchor_frac: float = 0.2) -> Splits:
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    anchor_rel = anchor_frac / (1.0 - test_frac)
    private, anchor = train_test_split(rest, test_size=anchor_rel, stratify=y[rest], random_state=seed + 1)
    return Splits(test=np.sort(test), anchor=np.sort(anchor), private=np.sort(private))


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 25, max_tries: int = 200) -> list[np.ndarray]:
    """Label-skew partition: for every class draw client proportions ~ Dir(alpha).

    Returns, per client, positions into ``y``. Small alpha => strongly non-IID.
    Retries until every client has at least ``min_size`` samples.
    """
    n_classes = int(y.max()) + 1
    min_size = min(min_size, max(1, len(y) // (n_clients * 2)))
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        shards: list[list[int]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            pos = np.where(y == c)[0]
            rng.shuffle(pos)
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(pos)).astype(int)[:-1]
            for k, part in enumerate(np.split(pos, cuts)):
                shards[k].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            break
    return [np.array(sorted(s), dtype=np.int64) for s in shards]


def apply_label_noise(y: np.ndarray, noise: float, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    if noise <= 0:
        return y.copy()
    y = y.copy()
    flip = rng.random(len(y)) < noise
    offsets = rng.integers(1, n_classes, size=int(flip.sum()))
    y[flip] = (y[flip] + offsets) % n_classes
    return y


@dataclass
class CampusData:
    X_train: np.ndarray
    y_train: np.ndarray       # possibly noisy labels (campus data quality)
    X_anchor: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_counts: list[int]
    n_classes: int
    in_shape: tuple[int, int, int]


def build_campus_data(bundle: Bundle, seed: int, campus_index: int, n_campuses: int, alpha: float,
                      test_frac: float, anchor_frac: float, label_noise: float = 0.0,
                      data_fraction: float = 1.0) -> CampusData:
    sp = make_splits(bundle.y, seed, test_frac, anchor_frac)
    shards = dirichlet_partition(bundle.y[sp.private], n_campuses, alpha, seed + 7)
    idx = sp.private[shards[campus_index]]
    rng = np.random.default_rng(seed * 131 + campus_index)
    if data_fraction < 1.0:
        keep = rng.choice(len(idx), size=max(8, int(len(idx) * data_fraction)), replace=False)
        idx = np.sort(idx[keep])
    y_local = apply_label_noise(bundle.y[idx], label_noise, bundle.n_classes, rng)
    counts = np.bincount(y_local, minlength=bundle.n_classes).tolist()
    return CampusData(
        X_train=bundle.X[idx], y_train=y_local,
        X_anchor=bundle.X[sp.anchor], X_test=bundle.X[sp.test], y_test=bundle.y[sp.test],
        class_counts=counts, n_classes=bundle.n_classes, in_shape=bundle.in_shape,
    )


def anchor_labels(bundle: Bundle, seed: int, test_frac: float, anchor_frac: float) -> np.ndarray:
    """Ground-truth anchor labels. Used ONLY by the coordinator for evaluation
    metrics (teacher accuracy); they are never sent to campuses."""
    sp = make_splits(bundle.y, seed, test_frac, anchor_frac)
    return bundle.y[sp.anchor]
