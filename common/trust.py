"""Entropy-based trust, trust-weighted global teacher and adaptive distillation.

Notation: K campuses, n anchor samples, C classes.
P has shape (K, n, C): the privacy-protected prediction of every campus.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy of the last axis scaled to [0, 1] (1 = fully uncertain)."""
    c = p.shape[-1]
    p = np.clip(p, EPS, 1.0)
    h = -(p * np.log(p)).sum(axis=-1)
    return h / np.log(c)


def noise_floor_entropy(sigma: float, n_classes: int, k: int = 1, n: int = 1500, seed: int = 0) -> float:
    """Normalised entropy that pure privacy noise produces (the 'nothing but noise' reference).

    Estimated by releasing uniform vectors through the same Gaussian mechanism and, for k > 1,
    averaging k independent releases (what the teacher does). The coordinator knows sigma, and
    this uses no private data, so it is pure post-processing and costs no privacy budget.
    Dividing entropies by this reference keeps trust and teacher confidence meaningful at
    strong privacy levels, where raw entropy would otherwise saturate near 1 for everyone.
    """
    if sigma <= 0:
        return 1.0
    from common.privacy import privatize
    rng = np.random.default_rng(seed)
    u = np.full((n, n_classes), 1.0 / n_classes, dtype=np.float32)
    avg = np.mean([privatize(u, sigma, rng) for _ in range(max(1, k))], axis=0)
    return float(max(normalized_entropy(avg).mean(), 1e-3))


def calibrate_entropy(h: np.ndarray, floor: float) -> np.ndarray:
    return np.clip(h / floor, 0.0, 1.0)


def campus_trust(mean_entropy: np.ndarray, tau: float) -> np.ndarray:
    """Softmax over negative mean entropy: confident campuses earn more trust.

    Returns a vector that sums to 1. ``tau`` controls how sharply trust reacts to
    differences in uncertainty (small tau = sharper).
    """
    z = -np.asarray(mean_entropy, dtype=np.float64) / tau
    z -= z.max()
    w = np.exp(z)
    return w / w.sum()


def smooth_trust(prev: np.ndarray | None, new: np.ndarray, ema: float) -> np.ndarray:
    """Exponential moving average across rounds so trust is stable, then renormalise."""
    if prev is None or ema <= 0:
        return new
    t = ema * prev + (1.0 - ema) * new
    return t / t.sum()


def sample_weights(H: np.ndarray, trust: np.ndarray, tau_sample: float) -> np.ndarray:
    """Per-sample, per-campus weights w_k(x) ~ T_k * exp(-H_k(x)/tau_sample).

    H has shape (K, n). The result sums to 1 over campuses for every sample.
    """
    logw = np.log(np.clip(trust, EPS, None))[:, None] - H / tau_sample
    logw -= logw.max(axis=0, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=0, keepdims=True)


def uniform_weights(k: int, n: int) -> np.ndarray:
    return np.full((k, n), 1.0 / k)


def aggregate_teacher(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Global teacher q(x) = sum_k w_k(x) * p_k(x), shape (n, C), rows sum to 1."""
    q = (W[:, :, None] * P).sum(axis=0)
    q = np.clip(q, EPS, None)
    return (q / q.sum(axis=1, keepdims=True)).astype(np.float32)


def teacher_confidence(q: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Per-sample confidence in [0, 1]: 1 - (noise-calibrated) normalised entropy of the teacher."""
    return 1.0 - calibrate_entropy(normalized_entropy(q), floor)


def consensus_confidence(P: np.ndarray, W: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-sample teacher confidence from agreement between campuses, in [0, 1].

    For every anchor sample: the trust-weighted share of campuses whose top class equals the
    teacher's top class, rescaled so chance level (1/C) maps to 0 and full agreement to 1.
    Unlike entropy this is self-calibrating under privacy noise: pure noise gives chance-level
    agreement, so it cannot be fooled into reporting a confident teacher.
    """
    c = P.shape[-1]
    agree = (P.argmax(axis=-1) == q.argmax(axis=-1)[None, :]).astype(np.float64)   # (K, n)
    share = (W * agree).sum(axis=0)
    return np.clip((share - 1.0 / c) / (1.0 - 1.0 / c), 0.0, 1.0)


def adaptive_lambda(trust: np.ndarray, teacher_quality: float, lambda_max: float,
                    lambda_floor: float) -> np.ndarray:
    """Per-campus distillation strength for this round.

    * campuses the federation trusts *less* lean more on the teacher
      (their own knowledge is the less reliable of the two);
    * the whole strength is scaled by the teacher's mean confidence, so a noisy,
      uncertain teacher is automatically listened to less.

        lambda_k = Q * lambda_max * (floor + (1 - floor) * (1 - T_k / max_j T_j))
    """
    rel = trust / max(float(trust.max()), EPS)
    lam = teacher_quality * lambda_max * (lambda_floor + (1.0 - lambda_floor) * (1.0 - rel))
    return np.clip(lam, 0.0, None)
