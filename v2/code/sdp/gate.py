"""Quality gate and rule-based orchestrator (the measurable stand-in for an LLM planner)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .generators import GeneratorError
from .metrics import Encoder, correlation_distance, dcr_share, marginal_distance, privacy_refs

OUTCOMES = ("accepted", "rejected_fidelity", "rejected_privacy", "error", "timeout", "invalid", "unknown")


@dataclass
class Thresholds:
    marginal: float
    correlation: float
    privacy: float


class QualityGate:
    """Accepts a synthetic batch only if marginal and correlation distances to the holdout stay within
    kappa times the 95th percentile of real-vs-real batches of equal size, and its DCR share <= tau."""

    def __init__(self, enc: Encoder, X_tr, y_tr, X_ho, y_ho, batch_size: int, rng: np.random.Generator,
                 kappa: float = C.KAPPA, tau: float = C.TAU_PRIVACY):
        self.enc, self.X_ho, self.cat = enc, X_ho, enc.categorical
        self.T_emb, self.H_emb = privacy_refs(enc, X_tr, y_tr, X_ho, y_ho, rng)
        b = min(batch_size, len(X_tr))
        ref_m, ref_c = [], []
        for _ in range(C.REF_DRAWS):
            idx = rng.choice(len(X_tr), size=b, replace=False)
            B = X_tr.iloc[idx]
            ref_m.append(marginal_distance(B, X_ho, self.cat))
            ref_c.append(correlation_distance(B, X_ho, enc))
        self.thresholds = Thresholds(kappa * float(np.quantile(ref_m, C.REF_QUANTILE)),
                                     kappa * float(np.quantile(ref_c, C.REF_QUANTILE)), tau)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> tuple[str, dict]:
        m = marginal_distance(X, self.X_ho, self.cat)
        c = correlation_distance(X, self.X_ho, self.enc)
        p = dcr_share(self.enc.gower(X, y), self.T_emb, self.H_emb, self.enc.mask)
        metrics = {"marginal": m, "correlation": c, "dcr_share": p}
        if not all(np.isfinite(v) for v in metrics.values()):
            return "unknown", metrics  # fail closed on undefined metrics
        t = self.thresholds
        if m > t.marginal or c > t.correlation:
            return "rejected_fidelity", metrics
        if p > t.privacy:
            return "rejected_privacy", metrics
        return "accepted", metrics


def validate_schema(X: pd.DataFrame, y: np.ndarray, columns: list[str], categorical: list[str], n: int) -> bool:
    if list(X.columns) != list(columns) or len(X) != n or len(y) != n:
        return False
    num = [c for c in columns if c not in categorical]
    if num and X[num].isna().to_numpy().any():
        return False
    return not X[categorical].isna().to_numpy().any() if categorical else True


@dataclass
class Release:
    X: pd.DataFrame | None
    y: np.ndarray | None
    requested: int
    delivered: int
    abstained: bool
    outcomes: dict = field(default_factory=dict)
    batches_accepted_by: dict = field(default_factory=dict)
    attempts: int = 0


def orchestrate(candidates: list, n: int, rng: np.random.Generator, gate: QualityGate | None,
                columns: list[str], categorical: list[str], max_attempts: int = C.MAX_ATTEMPTS,
                batches: int = C.BATCHES, time_budget_s: float = C.TIME_BUDGET_S,
                min_release: float = C.MIN_RELEASE) -> Release:
    """For each of `batches` slots, try candidates in order (up to `max_attempts` each) until one batch
    passes validation and the gate. Unfilled slots are dropped; release is withheld below `min_release`."""
    b = math.ceil(n / batches)
    outcomes = dict.fromkeys(OUTCOMES, 0)
    by = {}
    acc_X, acc_y, attempts = [], [], 0
    for slot in range(batches):
        size = min(b, n - slot * b)
        if size <= 0:
            break
        done = False
        for gen in candidates:
            for _ in range(max_attempts):
                attempts += 1
                t0 = time.perf_counter()
                try:
                    X, y = gen.sample(size, rng)
                except TimeoutError:
                    outcomes["timeout"] += 1
                    continue
                except (GeneratorError, ValueError, np.linalg.LinAlgError):
                    outcomes["error"] += 1
                    continue
                if time.perf_counter() - t0 > time_budget_s:
                    outcomes["timeout"] += 1
                    continue
                if not validate_schema(X, y, columns, categorical, size):
                    outcomes["invalid"] += 1
                    continue
                verdict = gate.evaluate(X, y)[0] if gate is not None else "accepted"
                outcomes[verdict] += 1
                if verdict == "accepted":
                    acc_X.append(X)
                    acc_y.append(y)
                    by[gen.name] = by.get(gen.name, 0) + 1
                    done = True
                    break
            if done:
                break
    delivered = int(sum(len(y) for y in acc_y))
    if delivered < min_release * n or delivered == 0:
        return Release(None, None, n, delivered, True, outcomes, by, attempts)
    return Release(pd.concat(acc_X, ignore_index=True), np.concatenate(acc_y), n, delivered, False, outcomes, by,
                   attempts)
