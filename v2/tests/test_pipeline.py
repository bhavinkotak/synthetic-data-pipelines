"""Unit tests for generators, metrics, gate, orchestrator, and the preregistered verdict logic.
They use small in-memory tables, so they need no network access."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sdp import config as C
from sdp.analysis import ci, verdict
from sdp.gate import QualityGate, orchestrate, validate_schema
from sdp.generators import REGISTRY, FaultyGenerator, Marginal, gower_knn, make
from sdp.metrics import Encoder, correlation_distance, dcr_share, marginal_distance, mia_auc, privacy_refs


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    a = rng.normal(y * 1.5, 1.0)
    b = 0.8 * a + rng.normal(0, 0.5, n)
    c = rng.integers(0, 4, n).astype(float)
    cat = np.where(rng.random(n) < 0.3 + 0.4 * y, "red", np.where(rng.random(n) < 0.5, "blue", "green"))
    X = pd.DataFrame({"a": a, "b": b, "c": c, "k": cat})
    return X, y, ["k"]


@pytest.fixture(scope="module")
def data():
    X, y, cat = toy()
    return X[:300].reset_index(drop=True), y[:300], X[300:450].reset_index(drop=True), y[300:450], \
        X[450:].reset_index(drop=True), y[450:], cat


def test_preregistration_matches_config():
    p = json.loads(C.PREREG.read_text())
    assert p["gate"]["kappa"] == C.KAPPA
    assert p["gate"]["tau_privacy_dcr_share"] == C.TAU_PRIVACY
    assert p["gate"]["batches_per_request"] == C.BATCHES
    assert tuple(p["seeds"]) == C.SEEDS


def test_marginal_uniform_roundtrip():
    rng = np.random.default_rng(0)
    s = pd.Series(["x", "y", "y", "z", "z", "z"])
    m = Marginal(s, categorical=True)
    u = m.to_uniform(s, rng)
    assert np.all((u > 0) & (u < 1))
    assert set(m.from_uniform(rng.random(500))) == {"x", "y", "z"}
    v = pd.Series(np.linspace(0, 1, 100))
    mv = Marginal(v, categorical=False)
    assert mv.continuous
    out = mv.from_uniform(rng.random(1000))
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_generators_produce_valid_schema(data, name):
    X_tr, y_tr, *_, cat = data
    g = make(name).fit(X_tr, y_tr, cat, np.random.default_rng(1))
    X, y = g.sample(200, np.random.default_rng(2))
    assert validate_schema(X, y, list(X_tr.columns), cat, 200)
    assert set(np.unique(y)) <= {0, 1}
    assert abs(y.mean() - y_tr.mean()) < 0.15
    assert set(X["k"]) <= set(X_tr["k"])


def test_copula_preserves_correlation_better_than_independent(data):
    X_tr, y_tr, X_ho, _, _, _, cat = data
    enc = Encoder(X_tr, cat)
    rng = np.random.default_rng(3)
    cop = make("copula").fit(X_tr, y_tr, cat, rng).sample(300, rng)[0]
    ind = make("independent").fit(X_tr, y_tr, cat, rng).sample(300, rng)[0]
    assert correlation_distance(cop, X_ho, enc) < correlation_distance(ind, X_ho, enc)
    assert marginal_distance(cop, X_ho, cat) < 0.3


def test_gower_knn_matches_brute_force():
    rng = np.random.default_rng(4)
    A = np.c_[rng.random((40, 3)), rng.integers(0, 3, (40, 2))].astype(np.float32)
    B = np.c_[rng.random((30, 3)), rng.integers(0, 3, (30, 2))].astype(np.float32)
    mask = np.array([False, False, False, True, True])
    full = np.abs(A[:, None, :3] - B[None, :, :3]).sum(-1) + (A[:, None, 3:] != B[None, :, 3:]).sum(-1)
    d, i = gower_knn(A, B, mask, k=3, chunk=7)
    np.testing.assert_allclose(d, np.sort(full, axis=1)[:, :3], rtol=1e-5)
    np.testing.assert_allclose(full[np.arange(40)[:, None], i], d, rtol=1e-5)


def test_privacy_metrics_detect_copies(data):
    X_tr, y_tr, X_ho, y_ho, X_te, y_te, cat = data
    enc = Encoder(X_tr, cat)
    T, H = privacy_refs(enc, X_tr, y_tr, X_ho, y_ho, np.random.default_rng(5))
    assert dcr_share(T, T, H, enc.mask) == 1.0
    assert dcr_share(H, T, H, enc.mask) == 0.0
    assert mia_auc(T, T, H, enc.mask) > 0.95
    fresh = enc.gower(X_te, y_te)
    assert 0.35 < dcr_share(fresh, T, H, enc.mask) < 0.65


def test_gate_rejects_memorisation_and_accepts_fresh_real(data):
    X_tr, y_tr, X_ho, y_ho, X_te, y_te, cat = data
    enc = Encoder(X_tr, cat)
    gate = QualityGate(enc, X_tr, y_tr, X_ho, y_ho, 100, np.random.default_rng(6))
    verdict_copy, m = gate.evaluate(X_tr.iloc[:100], y_tr[:100])
    assert verdict_copy == "rejected_privacy" and m["dcr_share"] > C.TAU_PRIVACY
    assert gate.evaluate(X_te.iloc[:100], y_te[:100])[0] == "accepted"


def test_gate_fails_closed_on_undefined_metric(data, monkeypatch):
    X_tr, y_tr, X_ho, y_ho, X_te, y_te, cat = data
    gate = QualityGate(Encoder(X_tr, cat), X_tr, y_tr, X_ho, y_ho, 100, np.random.default_rng(7))
    monkeypatch.setattr("sdp.gate.marginal_distance", lambda *a: float("nan"))
    assert gate.evaluate(X_te.iloc[:100], y_te[:100])[0] == "unknown"


def test_orchestrator_falls_back_and_abstains(data):
    X_tr, y_tr, X_ho, y_ho, _, _, cat = data
    enc = Encoder(X_tr, cat)
    rng = np.random.default_rng(8)
    gate = QualityGate(enc, X_tr, y_tr, X_ho, y_ho, 30, rng)
    copy = make("copy").fit(X_tr, y_tr, cat, rng)
    cop = make("copula").fit(X_tr, y_tr, cat, rng)
    cols = list(X_tr.columns)
    rel = orchestrate([copy, cop], 300, rng, gate, cols, cat)
    assert not rel.abstained and rel.batches_accepted_by.get("copula", 0) >= 8
    assert rel.outcomes["rejected_privacy"] >= 8
    assert sum(rel.outcomes.values()) == rel.attempts
    dead = FaultyGenerator(cop, 1.0, np.random.default_rng(9))
    rel = orchestrate([dead], 300, rng, gate, cols, cat)
    assert rel.abstained and rel.X is None and rel.delivered == 0
    assert rel.outcomes["error"] + rel.outcomes["timeout"] + rel.outcomes["invalid"] == rel.attempts


def test_schema_validation_rejects_nan(data):
    X_tr, y_tr, *_, cat = data
    bad = X_tr.iloc[:10].copy()
    bad.loc[0, "a"] = np.nan
    assert not validate_schema(bad, y_tr[:10], list(X_tr.columns), cat, 10)


def test_ci_and_verdict_rules():
    r = ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert r["mean"] == 1.0 and r["ci_lo"] == r["ci_hi"] == 1.0
    rows = []
    for ds in ("breast_cancer", "credit_g", "adult", "bank_marketing"):
        for s in range(5):
            rows.append({"dataset": ds, "seed": s, "condition": "orchestrated_gated", "ratio_hgb": 0.97 + 0.001 * s,
                         "dcr_share": 0.5, "mia_auc": 0.5, "abstained": 0})
    df = pd.DataFrame(rows)
    assert verdict(df)["verdict"] == "SUPPORTED"
    df.loc[df.dataset == "adult", "ratio_hgb"] = 0.8 + 0.001 * np.arange(5)
    assert verdict(df)["verdict"] == "REFUTED"
    df.loc[df.dataset == "adult", "ratio_hgb"] = [0.85, 0.95, 0.9, 0.92, 0.88]
    assert verdict(df)["verdict"] == "INCONCLUSIVE"
