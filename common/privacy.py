"""Differential privacy for shared prediction vectors.

Mechanism
---------
Before a campus releases its softmax predictions on the anchor set, it perturbs
every probability vector with Gaussian noise (the *Gaussian mechanism*).

* The L2 sensitivity of a probability vector is sqrt(2): two different softmax
  outputs differ by at most sqrt(2) in L2 norm (one-hot vs. a different one-hot).
* sigma is calibrated with the **analytic Gaussian mechanism** (Balle & Wang,
  ICML 2018), which is exact for every epsilon > 0 (the classical
  sigma = Delta*sqrt(2 ln(1.25/delta))/epsilon bound only holds for epsilon < 1).
* Clipping to non-negative values and re-normalising to the simplex is
  post-processing, so it does not weaken the guarantee.

Accounting
----------
The guarantee is *per released vector per round*. Across ``R`` rounds the
coordinator reports the tighter of basic composition (R*eps, R*delta) and
advanced composition (Dwork-Rothblum-Vadhan). See README for the exact claim.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr

SENSITIVITY = math.sqrt(2.0)


def gaussian_delta(sigma: float, epsilon: float, sensitivity: float = SENSITIVITY) -> float:
    """Exact delta of the Gaussian mechanism N(0, sigma^2) at a given epsilon."""
    if sigma <= 0:
        return 1.0
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    t1 = math.exp(log_ndtr(a - b))
    t2 = math.exp(epsilon + log_ndtr(-a - b))
    return max(0.0, min(1.0, t1 - t2))


def calibrate_sigma(epsilon: float, delta: float, sensitivity: float = SENSITIVITY) -> float:
    """Smallest sigma such that the Gaussian mechanism is (epsilon, delta)-DP."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    lo, hi = 1e-6, 1e5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gaussian_delta(mid, epsilon, sensitivity) > delta:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-9:
            break
    return hi


def epsilon_for_sigma(sigma: float, delta: float, sensitivity: float = SENSITIVITY) -> float:
    """Smallest epsilon achieved by noise level sigma at a given delta."""
    if sigma <= 0:
        return math.inf
    lo, hi = 1e-6, 1e4
    if gaussian_delta(sigma, hi, sensitivity) > delta:
        return math.inf
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if gaussian_delta(sigma, mid, sensitivity) > delta:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-9:
            break
    return hi


def to_simplex(v: np.ndarray) -> np.ndarray:
    """Clip negatives and re-normalise rows (post-processing; DP-safe)."""
    v = np.clip(v, 0.0, None)
    s = v.sum(axis=-1, keepdims=True)
    n = v.shape[-1]
    return np.where(s > 1e-12, v / np.maximum(s, 1e-12), 1.0 / n).astype(np.float32)


def privatize(probs: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-mechanism release of probability vectors."""
    if sigma <= 0:
        return probs.astype(np.float32)
    noisy = probs + rng.normal(0.0, sigma, size=probs.shape)
    return to_simplex(noisy)


class PrivacyAccountant:
    """Tracks the cumulative (epsilon, delta) after repeated releases."""

    def __init__(self, epsilon_step: float, delta_step: float):
        self.eps = float(epsilon_step)
        self.delta = float(delta_step)
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def spent(self) -> dict:
        r = self.steps
        if r == 0:
            return {"epsilon": 0.0, "delta": 0.0, "method": "none", "rounds": 0}
        if math.isinf(self.eps):
            return {"epsilon": math.inf, "delta": r * self.delta, "method": "none", "rounds": r}
        basic = (r * self.eps, r * self.delta)
        d_slack = max(self.delta, 1e-12)
        try:
            adv_eps = self.eps * math.sqrt(2 * r * math.log(1 / d_slack)) + r * self.eps * math.expm1(self.eps)
        except OverflowError:
            adv_eps = math.inf
        adv = (adv_eps, r * self.delta + d_slack)
        best, method = (adv, "advanced") if adv[0] < basic[0] else (basic, "basic")
        return {"epsilon": best[0], "delta": best[1], "method": method, "rounds": r}
