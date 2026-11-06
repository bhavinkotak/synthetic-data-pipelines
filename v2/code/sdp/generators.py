"""Locally fitted tabular generators. All are class-conditional and preserve the training class prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

from .config import CONTINUOUS_MIN_UNIQUE, GMM_COMPONENTS, GMM_MIN_ROWS_PER_COMPONENT, SMOTE_K

EPS = 1e-6


class GeneratorError(RuntimeError):
    """Raised by a generator that cannot produce a batch."""


class Marginal:
    """Empirical marginal of one column on a numeric code scale (categories are frequency-ordered codes)."""

    def __init__(self, values: pd.Series, categorical: bool, median: float | None = None):
        self.categorical = categorical
        if categorical:
            counts = values.value_counts()
            self.levels = list(counts.index)
            codes = pd.Categorical(values, categories=self.levels).codes.astype(float)
        else:
            self.median = float(np.nanmedian(values)) if median is None else median
            codes = values.fillna(self.median).to_numpy(float)
            if np.isnan(self.median):
                self.median, codes = 0.0, np.zeros(len(values))
        self.sorted = np.sort(codes)
        self.continuous = (not categorical) and len(np.unique(codes)) >= CONTINUOUS_MIN_UNIQUE
        self.integer = (not categorical) and bool(np.all(np.mod(codes, 1) == 0))

    def to_codes(self, values: pd.Series) -> np.ndarray:
        if self.categorical:
            c = pd.Categorical(values, categories=self.levels).codes.astype(float)
            c[c < 0] = len(self.levels)
            return c
        return values.fillna(self.median).to_numpy(float)

    def from_codes(self, codes: np.ndarray) -> np.ndarray:
        if self.categorical:
            idx = np.clip(np.rint(codes), 0, len(self.levels) - 1).astype(int)
            return np.asarray(self.levels, dtype=object)[idx]
        return codes

    def to_uniform(self, values: pd.Series, rng: np.random.Generator) -> np.ndarray:
        """Randomised probability integral transform; ties are spread uniformly inside their CDF jump."""
        c, n = self.to_codes(values), len(self.sorted)
        lo = np.searchsorted(self.sorted, c, "left") / n
        hi = np.searchsorted(self.sorted, c, "right") / n
        return np.clip(lo + rng.random(len(c)) * (hi - lo), EPS, 1 - EPS)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0, 1)
        if self.continuous:
            return self.from_codes(np.quantile(self.sorted, u))
        n = len(self.sorted)
        return self.from_codes(self.sorted[np.minimum((u * n).astype(int), n - 1)])


class Generator:
    name = "base"

    def fit(self, X: pd.DataFrame, y: np.ndarray, categorical: list[str], rng: np.random.Generator):
        self.columns, self.categorical = list(X.columns), list(categorical)
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / counts.sum()
        self.parts = {}
        for c in self.classes:
            Xc = X[y == c].reset_index(drop=True)
            marg = {col: Marginal(Xc[col], col in self.categorical,
                                  None if col in self.categorical else float(np.nanmedian(X[col])))
                    for col in self.columns}
            self.parts[c] = self._fit_class(Xc, marg, rng)
        return self

    def _fit_class(self, Xc, marg, rng):
        raise NotImplementedError

    def _sample_class(self, part, n, rng) -> pd.DataFrame:
        raise NotImplementedError

    def class_counts(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multinomial(n, self.prior)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
        frames, labels = [], []
        for c, k in zip(self.classes, self.class_counts(n, rng), strict=True):
            if k:
                frames.append(self._sample_class(self.parts[c], int(k), rng))
                labels.append(np.full(int(k), c))
        X = pd.concat(frames, ignore_index=True)[self.columns]
        for col in self.columns:
            if col not in self.categorical:
                X[col] = X[col].astype(float)
        y = np.concatenate(labels).astype(int)
        perm = rng.permutation(len(y))
        return X.iloc[perm].reset_index(drop=True), y[perm]


def _latent(Xc, marg, rng) -> np.ndarray:
    return np.column_stack([norm.ppf(marg[col].to_uniform(Xc[col], rng)) for col in Xc.columns])


def _decode(z: np.ndarray, marg, columns) -> pd.DataFrame:
    u = norm.cdf(z)
    return pd.DataFrame({col: marg[col].from_uniform(u[:, j]) for j, col in enumerate(columns)})


class GaussianCopula(Generator):
    """Gaussian copula over randomised normal scores with a Ledoit-Wolf shrunk correlation matrix."""
    name = "copula"

    def _fit_class(self, Xc, marg, rng):
        z = _latent(Xc, marg, rng)
        if len(z) < 2:
            return {"marg": marg, "R": np.eye(z.shape[1])}
        cov = LedoitWolf().fit(z).covariance_
        sd = np.sqrt(np.clip(np.diag(cov), EPS, None))
        return {"marg": marg, "R": cov / np.outer(sd, sd)}

    def _sample_class(self, part, n, rng):
        d = part["R"].shape[0]
        z = rng.multivariate_normal(np.zeros(d), part["R"], size=n, method="eigh")
        return _decode(z, part["marg"], self.columns)


class IndependentMarginals(GaussianCopula):
    """Baseline: class-conditional marginals sampled independently (identity copula)."""
    name = "independent"

    def _fit_class(self, Xc, marg, rng):
        return {"marg": marg, "R": np.eye(len(self.columns))}


class ClassConditionalGMM(Generator):
    """Gaussian mixture in the normal-score space; component count chosen by BIC."""
    name = "gmm"

    def _fit_class(self, Xc, marg, rng):
        z = _latent(Xc, marg, rng)
        best, best_bic = None, np.inf
        for k in GMM_COMPONENTS:
            if k > 1 and len(z) < GMM_MIN_ROWS_PER_COMPONENT * k:
                break
            if len(z) < 2:
                break
            g = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=int(rng.integers(2**31)))
            g.fit(z)
            bic = g.bic(z)
            if bic < best_bic:
                best, best_bic = g, bic
        return {"marg": marg, "gmm": best, "z": z}

    def _sample_class(self, part, n, rng):
        if part["gmm"] is None:
            z = part["z"][rng.integers(len(part["z"]), size=n)]
        else:
            part["gmm"].random_state = int(rng.integers(2**31))
            z, _ = part["gmm"].sample(n)
            z = z[rng.permutation(n)]
        return _decode(z, part["marg"], self.columns)


def gower_matrix(X: pd.DataFrame, marg: dict, categorical: list[str], ranges: dict) -> tuple[np.ndarray, np.ndarray]:
    """Range-scaled numeric columns and categorical codes, plus a mask marking categorical columns."""
    cols, mask = [], []
    for col in X.columns:
        if col in categorical:
            cols.append(pd.Categorical(X[col], categories=marg[col].levels).codes.astype(float))
        else:
            cols.append(X[col].fillna(marg[col].median).to_numpy(float) / ranges[col])
        mask.append(col in categorical)
    return np.column_stack(cols).astype(np.float32), np.asarray(mask)


def gower_knn(A: np.ndarray, B: np.ndarray, mask: np.ndarray, k: int = 1, chunk: int = 512):
    """k nearest rows of B for each row of A under the unnormalised Gower distance
    (sum of absolute range-scaled numeric differences plus the number of categorical mismatches)."""
    num, cat = np.flatnonzero(~mask), np.flatnonzero(mask)
    Bt = np.ascontiguousarray(B.T, dtype=np.float32)
    dist = np.empty((len(A), k), np.float32)
    idx = np.empty((len(A), k), np.int64)
    for s in range(0, len(A), chunk):
        at = np.ascontiguousarray(A[s:s + chunk].T, dtype=np.float32)
        d = np.zeros((at.shape[1], len(B)), np.float32)
        tmp = np.empty_like(d)
        for j in num:
            np.subtract(at[j][:, None], Bt[j][None, :], out=tmp)
            np.abs(tmp, out=tmp)
            d += tmp
        for j in cat:
            d += at[j][:, None] != Bt[j][None, :]
        part = np.argpartition(d, k - 1, axis=1)[:, :k] if k < len(B) else np.argsort(d, axis=1)[:, :k]
        pd_ = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd_, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        dist[s:s + chunk] = np.take_along_axis(pd_, order, axis=1)
    return dist, idx


class SmoteInterpolation(Generator):
    """SMOTE-style sampling: convex combination of a record and one of its k same-class neighbours."""
    name = "smote"

    def _fit_class(self, Xc, marg, rng):
        ranges = {c: max(np.ptp(marg[c].sorted), EPS) for c in self.columns if c not in self.categorical}
        emb, mask = gower_matrix(Xc, marg, self.categorical, ranges)
        k = min(SMOTE_K, len(Xc) - 1)
        nbrs = None
        if k >= 1:
            _, idx = gower_knn(emb, emb, mask, k + 1)
            # drop each row itself (duplicates may tie with it, so remove by identity, not position)
            rows = np.arange(len(emb))[:, None]
            keep = idx != rows
            nbrs = np.array([r[m][:k] for r, m in zip(idx, keep, strict=True)])
        return {"X": Xc, "nbrs": nbrs, "marg": marg}

    def _sample_class(self, part, n, rng):
        X = part["X"]
        base = rng.integers(len(X), size=n)
        if part["nbrs"] is None:
            return X.iloc[base].reset_index(drop=True)
        mate = part["nbrs"][base, rng.integers(part["nbrs"].shape[1], size=n)]
        lam = rng.random(n)
        out = {}
        for col in self.columns:
            a, b = X[col].to_numpy()[base], X[col].to_numpy()[mate]
            if col in self.categorical:
                out[col] = np.where(lam < 0.5, a, b)
            else:
                a = np.where(np.isnan(a.astype(float)), part["marg"][col].median, a).astype(float)
                b = np.where(np.isnan(b.astype(float)), part["marg"][col].median, b).astype(float)
                v = a + lam * (b - a)
                out[col] = np.rint(v) if part["marg"][col].integer else v
        return pd.DataFrame(out)


class LeakyCopy(Generator):
    """Negative control (memorising generator): resamples training records verbatim."""
    name = "copy"

    def _fit_class(self, Xc, marg, rng):
        return {"X": Xc, "marg": marg}

    def _sample_class(self, part, n, rng):
        X = part["X"].iloc[rng.integers(len(part["X"]), size=n)].reset_index(drop=True).copy()
        for col in self.columns:
            m = part["marg"][col]
            if not m.categorical:
                X[col] = X[col].fillna(m.median).to_numpy(float)
        return X


class FaultyGenerator:
    """Simulated unreliable generator: with probability p a call raises, times out, or returns a corrupt batch."""

    def __init__(self, base: Generator, p_fail: float, rng: np.random.Generator, timeout_s: float = 1e9):
        self.base, self.p_fail, self.rng, self.timeout_s = base, p_fail, rng, timeout_s
        self.name = base.name
        self.columns, self.categorical = base.columns, base.categorical

    def sample(self, n: int, rng: np.random.Generator):
        r = self.rng.random()
        if r < self.p_fail / 3:
            raise GeneratorError("simulated generator exception")
        if r < 2 * self.p_fail / 3:
            raise TimeoutError("simulated generator timeout")
        X, y = self.base.sample(n, rng)
        if r < self.p_fail:
            X = X.copy()
            X.iloc[:, 0] = np.nan  # corrupt batch: schema validation must reject it
        return X, y


REGISTRY = {g.name: g for g in (GaussianCopula, ClassConditionalGMM, SmoteInterpolation, IndependentMarginals,
                                 LeakyCopy)}


def make(name: str) -> Generator:
    return REGISTRY[name]()
