"""Unit tests for the pure-numpy parts of the algorithm (no network, no torch)."""
import math

import numpy as np
import pytest

from common import trust as T
from common.config import validate_config
from common.data import (apply_label_noise, build_campus_data, dirichlet_partition, load_dataset,
                         make_splits)
from common.privacy import (SENSITIVITY, PrivacyAccountant, calibrate_sigma, epsilon_for_sigma,
                            gaussian_delta, privatize, to_simplex)
from common.serde import decode_array, encode_array


# ------------------------------------------------------------------- privacy
@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 5.0, 20.0, 100.0])
def test_calibrated_sigma_meets_delta(eps):
    delta = 1e-5
    s = calibrate_sigma(eps, delta)
    assert gaussian_delta(s, eps) <= delta * 1.001
    assert gaussian_delta(s * 0.9, eps) > delta          # and it is (nearly) tight


def test_sigma_decreases_with_epsilon():
    sig = [calibrate_sigma(e, 1e-5) for e in (0.5, 1, 5, 20, 100)]
    assert all(a > b for a, b in zip(sig, sig[1:]))


def test_analytic_sigma_is_tighter_than_classical_bound_for_small_eps():
    eps, delta = 0.5, 1e-5
    classical = SENSITIVITY * math.sqrt(2 * math.log(1.25 / delta)) / eps
    s = calibrate_sigma(eps, delta)
    assert 0.5 * classical < s <= classical          # valid, and less wasteful than the classical formula


def test_epsilon_sigma_roundtrip():
    s = calibrate_sigma(7.0, 1e-5)
    assert epsilon_for_sigma(s, 1e-5) == pytest.approx(7.0, rel=1e-3)
    assert math.isinf(epsilon_for_sigma(0.0, 1e-5))


def test_privatize_outputs_valid_distributions_and_noise_scale():
    rng = np.random.default_rng(0)
    p = np.tile(np.array([0.9, 0.05, 0.05], dtype=np.float32), (4000, 1))
    out = privatize(p, 0.3, rng)
    assert out.shape == p.shape and np.all(out >= 0)
    assert np.allclose(out.sum(1), 1.0, atol=1e-5)
    assert out.mean(0).argmax() == 0                      # signal survives averaging
    assert np.abs(out - p).mean() > 0.05                  # but individual vectors are perturbed
    assert np.array_equal(privatize(p, 0.0, rng), p)      # sigma=0 is a no-op


def test_to_simplex_handles_all_negative_rows():
    v = np.array([[-1.0, -2.0, -3.0]])
    assert np.allclose(to_simplex(v), 1 / 3)


def test_accountant_composition():
    a = PrivacyAccountant(1.0, 1e-5)
    assert a.spent()["epsilon"] == 0
    for _ in range(10):
        a.step()
    s = a.spent()
    assert s["rounds"] == 10 and 0 < s["epsilon"] <= 10.0 + 1e-9    # never worse than basic composition
    b = PrivacyAccountant(0.1, 1e-5)
    for _ in range(100):
        b.step()
    assert b.spent()["method"] == "advanced" and b.spent()["epsilon"] < 10.0


# --------------------------------------------------------------------- trust
def test_entropy_bounds():
    uniform = np.full((1, 10), 0.1)
    onehot = np.eye(10)[:1]
    assert T.normalized_entropy(uniform)[0] == pytest.approx(1.0, abs=1e-6)
    assert T.normalized_entropy(onehot)[0] == pytest.approx(0.0, abs=1e-6)


def test_trust_prefers_confident_campuses_and_sums_to_one():
    t = T.campus_trust(np.array([0.1, 0.5, 0.9]), tau=0.25)
    assert t.sum() == pytest.approx(1.0)
    assert t[0] > t[1] > t[2]
    flat = T.campus_trust(np.array([0.4, 0.4, 0.4]), tau=0.25)
    assert np.allclose(flat, 1 / 3)


def test_trust_ema_smoothing():
    prev = np.array([0.8, 0.2])
    new = np.array([0.2, 0.8])
    out = T.smooth_trust(prev, new, 0.5)
    assert out.sum() == pytest.approx(1.0) and out[0] == pytest.approx(0.5)
    assert np.array_equal(T.smooth_trust(None, new, 0.5), new)


def test_sample_weights_normalised_and_entropy_aware():
    H = np.array([[0.1, 0.9], [0.9, 0.1]])               # campus 0 sure on x0, campus 1 sure on x1
    W = T.sample_weights(H, np.array([0.5, 0.5]), 0.3)
    assert np.allclose(W.sum(0), 1.0)
    assert W[0, 0] > W[1, 0] and W[1, 1] > W[0, 1]


def test_teacher_follows_trusted_campus():
    good = np.tile([0.9, 0.05, 0.05], (5, 1))
    bad = np.tile([0.05, 0.05, 0.9], (5, 1))
    P = np.stack([good, bad]).astype(np.float32)
    q_trust = T.aggregate_teacher(P, np.stack([np.full(5, 0.9), np.full(5, 0.1)]))
    q_uni = T.aggregate_teacher(P, T.uniform_weights(2, 5))
    assert np.all(q_trust.argmax(1) == 0)
    assert np.allclose(q_trust.sum(1), 1.0) and np.allclose(q_uni.sum(1), 1.0)
    assert q_trust[0, 0] > q_uni[0, 0]


def test_adaptive_lambda_behaviour():
    trust = np.array([0.6, 0.3, 0.1])
    lam = T.adaptive_lambda(trust, teacher_quality=1.0, lambda_max=1.0, lambda_floor=0.4)
    assert lam[0] == pytest.approx(0.4) and lam[0] < lam[1] < lam[2] <= 1.0   # less trusted -> listens more
    weak = T.adaptive_lambda(trust, teacher_quality=0.5, lambda_max=1.0, lambda_floor=0.4)
    assert np.allclose(weak, lam * 0.5)                                       # unsure teacher -> listen less


# ---------------------------------------------------------------------- data
def test_splits_are_disjoint_and_deterministic():
    d = load_dataset("digits")
    a = make_splits(d.y, 1)
    b = make_splits(d.y, 1)
    assert np.array_equal(a.private, b.private)
    assert not (set(a.test) & set(a.anchor)) and not (set(a.test) & set(a.private)) \
        and not (set(a.anchor) & set(a.private))
    assert len(a.test) + len(a.anchor) + len(a.private) == len(d.y)


def test_dirichlet_alpha_controls_skew():
    d = load_dataset("digits")
    y = d.y[make_splits(d.y, 0).private]

    def skew(alpha):
        parts = dirichlet_partition(y, 4, alpha, seed=3)
        assert sorted(np.concatenate(parts).tolist()) == list(range(len(y)))   # exact partition
        fr = np.array([np.bincount(y[p], minlength=10) / len(p) for p in parts])
        return fr.std(axis=0).mean()

    assert skew(0.1) > skew(100.0)


def test_label_noise_rate():
    rng = np.random.default_rng(0)
    y = np.zeros(20000, dtype=np.int64)
    n = apply_label_noise(y, 0.3, 10, rng)
    assert (n != y).mean() == pytest.approx(0.3, abs=0.02)
    assert np.array_equal(apply_label_noise(y, 0.0, 10, rng), y)


def test_all_campuses_see_identical_public_sets_but_distinct_private_shards():
    d = load_dataset("digits")
    c = [build_campus_data(d, 5, i, 3, 0.5, 0.2, 0.2) for i in range(3)]
    assert np.array_equal(c[0].X_anchor, c[1].X_anchor) and np.array_equal(c[0].X_test, c[2].X_test)
    total = sum(len(x.y_train) for x in c)
    assert total == len(make_splits(d.y, 5).private)


# -------------------------------------------------------------- serde/config
def test_serde_roundtrip():
    a = np.random.default_rng(0).random((7, 10)).astype(np.float32)
    assert np.allclose(decode_array(encode_array(a, "float32")), a)
    assert np.allclose(decode_array(encode_array(a, "float16")), a, atol=1e-3)
    with pytest.raises(ValueError):
        encode_array(a, "object")


def test_config_validation():
    ok = validate_config({"campuses": [{"node_id": "a"}, {"node_id": "b"}]})
    assert ok["campuses"][0]["model"] and ok["dp"]["enabled"]
    with pytest.raises(ValueError):
        validate_config({"campuses": [{"node_id": "a"}]})
    with pytest.raises(ValueError):
        validate_config({"campuses": [{"node_id": "a"}, {"node_id": "a"}]})
    with pytest.raises(ValueError):
        validate_config({"mode": "magic", "campuses": [{"node_id": "a"}, {"node_id": "b"}]})
    with pytest.raises(ValueError):
        validate_config({"alpha": -1, "campuses": [{"node_id": "a"}, {"node_id": "b"}]})


def test_noise_floor_entropy():
    assert T.noise_floor_entropy(0.0, 10) == 1.0
    lo, hi = T.noise_floor_entropy(0.1, 10), T.noise_floor_entropy(1.0, 10)
    assert 0 < lo < 1 and 0 < hi <= 1
    assert T.noise_floor_entropy(0.5, 10, k=4) > T.noise_floor_entropy(0.5, 10, k=1)   # averaging noise-only vectors flattens them
    h = np.array([0.1, 0.5, 2.0])
    assert np.allclose(T.calibrate_entropy(h, 0.5), [0.2, 1.0, 1.0])


def test_consensus_confidence():
    good = np.tile([0.9, 0.05, 0.05], (6, 1))
    other = np.tile([0.05, 0.9, 0.05], (6, 1))
    W = T.uniform_weights(3, 6)
    q = T.aggregate_teacher(np.stack([good, good, good]).astype(np.float32), W)
    assert np.allclose(T.consensus_confidence(np.stack([good, good, good]), W, q), 1.0)
    P = np.stack([good, good, other]).astype(np.float32)
    q2 = T.aggregate_teacher(P, W)
    c = T.consensus_confidence(P, W, q2)
    assert np.allclose(c, (2 / 3 - 1 / 3) / (2 / 3))               # 2 of 3 agree
    rng = np.random.default_rng(0)
    noise = rng.dirichlet(np.ones(10), size=(4, 4000)).astype(np.float32)
    qn = T.aggregate_teacher(noise, T.uniform_weights(4, 4000))
    assert T.consensus_confidence(noise, T.uniform_weights(4, 4000), qn).mean() < 0.35   # pure noise ~ low
