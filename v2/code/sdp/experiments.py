"""Experiment scenarios. Each returns a list of flat result rows (one per dataset x seed x condition)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from .data import Table, split, subsample
from .gate import QualityGate, orchestrate
from .generators import FaultyGenerator, make
from .metrics import (Encoder, auc, cap_rows, correlation_distance, dcr_share, exact_match_rate, marginal_distance,
                      mia_auc, privacy_refs)


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng(list(keys))


class Context:
    """Per (dataset, seed) evaluation state: real splits, encoder, privacy references, TRTR baselines."""

    def __init__(self, t: Table, X_tr, y_tr, X_ho, y_ho, X_te, y_te, seed: int):
        self.t, self.seed, self.cat = t, seed, t.categorical
        self.X_tr, self.y_tr, self.X_ho, self.y_ho, self.X_te, self.y_te = X_tr, y_tr, X_ho, y_ho, X_te, y_te
        self.enc = Encoder(X_tr, t.categorical)
        self.T_emb, self.H_emb = privacy_refs(self.enc, X_tr, y_tr, X_ho, y_ho, _rng(seed, 7))
        self.trtr = {k: auc(k, X_tr, y_tr, X_te, y_te, t.categorical, seed) for k in ("lr", "hgb")}

    def utility(self, X, y) -> dict:
        row = {}
        for k in ("lr", "hgb"):
            a = auc(k, X, y, self.X_te, self.y_te, self.cat, self.seed)
            row[f"auc_{k}"], row[f"ratio_{k}"] = a, a / self.trtr[k]
        return row

    def quality(self, X, y) -> dict:
        S, mk = cap_rows(self.enc.gower(X, y), _rng(self.seed, 5)), self.enc.mask
        return {"marginal": marginal_distance(X, self.X_ho, self.cat),
                "correlation": correlation_distance(X, self.X_ho, self.enc),
                "dcr_share": dcr_share(S, self.T_emb, self.H_emb, mk),
                "mia_auc": mia_auc(S, self.T_emb, self.H_emb, mk),
                "exact_match": exact_match_rate(S, self.T_emb, mk)}

    def evaluate(self, X, y, with_utility=True) -> dict:
        row = self.quality(X, y)
        if with_utility:
            row.update(self.utility(X, y))
        return row

    def gate(self, n: int, rng) -> QualityGate:
        return QualityGate(self.enc, self.X_tr, self.y_tr, self.X_ho, self.y_ho, int(np.ceil(n / C.BATCHES)), rng)


def _release_row(rel, ctx: Context, with_utility=True) -> dict:
    row = {"requested": rel.requested, "delivered": rel.delivered, "delivered_frac": rel.delivered / rel.requested,
           "abstained": int(rel.abstained), "attempts": rel.attempts,
           **{f"n_{k}": v for k, v in rel.outcomes.items()},
           **{f"batches_{k}": v for k, v in rel.batches_accepted_by.items()}}
    if not rel.abstained:
        row.update(ctx.evaluate(rel.X, rel.y, with_utility))
    return row


def fit_all(ctx: Context, names, seed: int) -> dict:
    return {g: make(g).fit(ctx.X_tr, ctx.y_tr, ctx.cat, _rng(seed, 11, i)) for i, g in enumerate(names)}


def run_main(t: Table, seed: int) -> tuple[list[dict], list[dict]]:
    """E1 (generators x gate on/off, orchestrated and adversarial pipelines) and E4 (fault injection)."""
    s = split(t, seed)
    ctx = Context(t, s.X_tr, s.y_tr, s.X_ho, s.y_ho, s.X_te, s.y_te, seed)
    n = len(s.X_tr)
    base = {"dataset": t.name, "seed": seed, "n_train": n}
    rows = [{**base, "condition": "trtr", "auc_lr": ctx.trtr["lr"], "auc_hgb": ctx.trtr["hgb"],
             "ratio_lr": 1.0, "ratio_hgb": 1.0}]
    rows.append({**base, "condition": "fresh_real", **ctx.quality(s.X_te, s.y_te)})
    names = (*C.CANDIDATES, "copy")
    t0 = time.perf_counter()
    gens = fit_all(ctx, names, seed)
    fit_s = time.perf_counter() - t0
    gate = ctx.gate(n, _rng(seed, 13))
    cols = list(ctx.X_tr.columns)
    for i, g in enumerate(names):
        X, y = gens[g].sample(n, _rng(seed, 17, i))
        rows.append({**base, "condition": f"{g}_ungated", "generator": g, **ctx.evaluate(X, y)})
        rel = orchestrate([gens[g]], n, _rng(seed, 19, i), gate, cols, ctx.cat)
        rows.append({**base, "condition": f"{g}_gated", "generator": g, **_release_row(rel, ctx)})
    rel = orchestrate([gens[g] for g in C.CANDIDATES], n, _rng(seed, 23), gate, cols, ctx.cat)
    rows.append({**base, "condition": "orchestrated_gated", **_release_row(rel, ctx)})
    rel = orchestrate([gens[g] for g in ("copy", *C.CANDIDATES)], n, _rng(seed, 29), gate, cols, ctx.cat)
    rows.append({**base, "condition": "adversarial_gated", **_release_row(rel, ctx)})
    rows[0]["fit_seconds_all_generators"] = fit_s
    gate_row = {**base, "condition": "gate_thresholds", "tau_marginal": gate.thresholds.marginal,
                "tau_correlation": gate.thresholds.correlation, "tau_privacy": gate.thresholds.privacy,
                "batch_size": int(np.ceil(n / C.BATCHES))}
    rows.append(gate_row)

    faults = []
    for j, p in enumerate(C.FAULT_RATES):
        frng = _rng(seed, 31, j)
        cands = [FaultyGenerator(gens[g], p, frng) for g in C.CANDIDATES]
        for policy, cs, attempts in (("retry_fallback", cands, C.MAX_ATTEMPTS), ("single_attempt", cands[:1], 1)):
            rel = orchestrate(cs, n, _rng(seed, 37, j), gate, cols, ctx.cat, max_attempts=attempts)
            r = _release_row(rel, ctx, with_utility=False)
            if not rel.abstained:
                r.update(ctx.utility(rel.X, rel.y))
            faults.append({**base, "p_fail": p, "policy": policy, **r})
    return rows, faults


def run_scarcity(t: Table, seed: int) -> list[dict]:
    """E2: generators fitted on n_real rows; real-only vs synthetic-only vs real+synthetic training."""
    s = split(t, seed)
    out = []
    for n_real in C.SCARCITY_N:
        if n_real > len(s.X_tr):
            continue
        X_r, y_r = subsample(s.X_tr, s.y_tr, n_real, seed)
        X_h, y_h = subsample(s.X_ho, s.y_ho, min(n_real, len(s.X_ho)), seed)
        ctx = Context(t, X_r, y_r, X_h, y_h, s.X_te, s.y_te, seed)
        base = {"dataset": t.name, "seed": seed, "n_real": n_real}
        out.append({**base, "condition": "real_only", "auc_lr": ctx.trtr["lr"], "auc_hgb": ctx.trtr["hgb"]})
        gens = fit_all(ctx, C.CANDIDATES, seed)
        n_syn = C.SYN_MULT * n_real
        Xu, yu = gens["copula"].sample(n_syn, _rng(seed, 41))
        rel = orchestrate([gens[g] for g in C.CANDIDATES], n_syn, _rng(seed, 43), ctx.gate(n_syn, _rng(seed, 47)),
                          list(X_r.columns), t.categorical)
        variants = {"synthetic_only_ungated": (Xu, yu),
                    "hybrid_ungated": (pd.concat([X_r, Xu], ignore_index=True), np.r_[y_r, yu])}
        if not rel.abstained:
            variants["synthetic_only_gated"] = (rel.X, rel.y)
            variants["hybrid_gated"] = (pd.concat([X_r, rel.X], ignore_index=True), np.r_[y_r, rel.y])
        for name, (X, y) in variants.items():
            out.append({**base, "condition": name, **ctx.utility(X, y)})
        out.append({**base, "condition": "gated_release", "abstained": int(rel.abstained),
                    "delivered_frac": rel.delivered / n_syn})
    return out


def run_missing(t: Table, seed: int) -> list[dict]:
    """E3: MCAR missingness injected into numeric cells of the train and holdout splits."""
    s = split(t, seed)
    num = [c for c in s.X_tr.columns if c not in t.categorical]
    out = []
    for j, rate in enumerate(C.MISSING_RATES):
        rng = _rng(seed, 53, j)
        X_tr, X_ho = s.X_tr.copy(), s.X_ho.copy()
        for X in (X_tr, X_ho):
            for c in num:
                X.loc[rng.random(len(X)) < rate, c] = np.nan
        ctx = Context(t, X_tr, s.y_tr, X_ho, s.y_ho, s.X_te, s.y_te, seed)
        base = {"dataset": t.name, "seed": seed, "missing_rate": rate}
        out.append({**base, "condition": "trtr", "auc_lr": ctx.trtr["lr"], "auc_hgb": ctx.trtr["hgb"]})
        gens = fit_all(ctx, C.CANDIDATES, seed)
        n = len(X_tr)
        X, y = gens["copula"].sample(n, _rng(seed, 59, j))
        out.append({**base, "condition": "copula_ungated", **ctx.utility(X, y)})
        rel = orchestrate([gens[g] for g in C.CANDIDATES], n, _rng(seed, 61, j), ctx.gate(n, _rng(seed, 67, j)),
                          list(X_tr.columns), t.categorical)
        r = {"abstained": int(rel.abstained), "delivered_frac": rel.delivered / n}
        if not rel.abstained:
            r.update(ctx.utility(rel.X, rel.y))
        out.append({**base, "condition": "orchestrated_gated", **r})
    return out


def run_scaling(t: Table, seed: int = 0, reps: int = 3) -> list[dict]:
    """E5: wall-clock cost of fitting, sampling, gating, and privacy evaluation on one machine."""
    s = split(t, seed)
    ctx = Context(t, s.X_tr, s.y_tr, s.X_ho, s.y_ho, s.X_te, s.y_te, seed)
    out = []
    gate = ctx.gate(len(s.X_tr), _rng(seed, 71))
    for i, g in enumerate(C.CANDIDATES):
        fits, gen = [], None
        for _ in range(reps):
            t0 = time.perf_counter()
            gen = make(g).fit(s.X_tr, s.y_tr, t.categorical, _rng(seed, 73, i))
            fits.append(time.perf_counter() - t0)
        out.append({"dataset": t.name, "generator": g, "stage": "fit", "n": len(s.X_tr),
                    "seconds": float(np.median(fits))})
        for n in C.SCALING_N:
            ts, X, y = [], None, None
            for _ in range(reps):
                t0 = time.perf_counter()
                X, y = gen.sample(n, _rng(seed, 79, n))
                ts.append(time.perf_counter() - t0)
            out.append({"dataset": t.name, "generator": g, "stage": "sample", "n": n, "seconds": float(np.median(ts))})
            if n in C.SCALING_DCR_N:
                t0 = time.perf_counter()
                dcr_share(ctx.enc.gower(X, y), ctx.T_emb, ctx.H_emb, ctx.enc.mask)
                out.append({"dataset": t.name, "generator": g, "stage": "dcr_share", "n": n,
                            "seconds": time.perf_counter() - t0})
        Xb, yb = gen.sample(int(np.ceil(len(s.X_tr) / C.BATCHES)), _rng(seed, 83))
        t0 = time.perf_counter()
        gate.evaluate(Xb, yb)
        out.append({"dataset": t.name, "generator": g, "stage": "gate_batch", "n": len(Xb),
                    "seconds": time.perf_counter() - t0})
    return out
