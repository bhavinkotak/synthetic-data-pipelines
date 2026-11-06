"""Utility, fidelity, and privacy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import PRIVACY_SYN_N
from .generators import EPS, Marginal, gower_knn, gower_matrix


class Encoder:
    """Fitted on the real train split; shared by every metric so all conditions are measured on one scale."""

    def __init__(self, X: pd.DataFrame, categorical: list[str]):
        self.columns, self.categorical = list(X.columns), list(categorical)
        self.marg = {c: Marginal(X[c], c in categorical) for c in self.columns}
        self.ranges = {c: max(np.ptp(self.marg[c].sorted), EPS) for c in self.columns if c not in categorical}
        self.mask = np.array([c in categorical for c in self.columns] + [True])

    def codes(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.marg[c].to_codes(X[c]) for c in self.columns])

    def gower(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Gower embedding with the label appended as one more categorical column."""
        emb, _ = gower_matrix(X[self.columns], self.marg, self.categorical, self.ranges)
        return np.column_stack([emb, np.asarray(y, np.float32)])


def marginal_distance(S: pd.DataFrame, R: pd.DataFrame, categorical: list[str]) -> float:
    """Mean over columns of the KS statistic (numeric) or total-variation distance (categorical)."""
    ds = []
    for c in R.columns:
        if c in categorical:
            p = S[c].value_counts(normalize=True)
            q = R[c].value_counts(normalize=True)
            ds.append(0.5 * p.subtract(q, fill_value=0).abs().sum())
        else:
            a, b = S[c].dropna().to_numpy(), R[c].dropna().to_numpy()
            ds.append(ks_2samp(a, b).statistic if len(a) and len(b) else np.nan)
    return float(np.mean(ds))


def _spearman(M: np.ndarray) -> np.ndarray:
    rho = np.atleast_2d(spearmanr(M).statistic)
    return np.nan_to_num(rho, nan=0.0)


def correlation_distance(S: pd.DataFrame, R: pd.DataFrame, enc: Encoder) -> float:
    """Mean absolute difference of pairwise Spearman correlations over the upper triangle."""
    a, b = _spearman(enc.codes(S)), _spearman(enc.codes(R))
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.mean(np.abs(a[iu] - b[iu])))


def nn_distance(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return gower_knn(A, B, mask, 1)[0][:, 0]


def privacy_refs(enc: Encoder, X_tr, y_tr, X_ho, y_ho, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Train reference T' (random train subsample) and holdout H of equal size m = min(|train|, |holdout|)."""
    m = min(len(X_tr), len(X_ho))
    it, ih = rng.choice(len(X_tr), m, replace=False), rng.choice(len(X_ho), m, replace=False)
    return enc.gower(X_tr.iloc[it], y_tr[it]), enc.gower(X_ho.iloc[ih], y_ho[ih])


def cap_rows(S_emb: np.ndarray, rng: np.random.Generator, n: int = PRIVACY_SYN_N) -> np.ndarray:
    return S_emb if len(S_emb) <= n else S_emb[rng.choice(len(S_emb), n, replace=False)]


def dcr_share(S_emb: np.ndarray, T_emb: np.ndarray, H_emb: np.ndarray, mask: np.ndarray) -> float:
    """Share of synthetic records closer to the train reference than to the equally sized holdout."""
    dt, dh = nn_distance(S_emb, T_emb, mask), nn_distance(S_emb, H_emb, mask)
    return float(np.mean(dt < dh) + 0.5 * np.mean(dt == dh))


def exact_match_rate(S_emb: np.ndarray, T_emb: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(nn_distance(S_emb, T_emb, mask) <= 1e-6))


def mia_auc(S_emb: np.ndarray, T_emb: np.ndarray, H_emb: np.ndarray, mask: np.ndarray) -> float:
    """Distance-to-synthetic membership attack: members = train reference, non-members = holdout."""
    scores = -np.concatenate([nn_distance(T_emb, S_emb, mask), nn_distance(H_emb, S_emb, mask)])
    labels = np.r_[np.ones(len(T_emb)), np.zeros(len(H_emb))]
    return float(roc_auc_score(labels, scores))


def make_model(kind: str, X: pd.DataFrame, categorical: list[str], seed: int):
    num = [c for c in X.columns if c not in categorical]
    if kind == "lr":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])
        return make_pipeline(pre, LogisticRegression(max_iter=2000))
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical)])
    return make_pipeline(pre, HistGradientBoostingClassifier(max_iter=200, random_state=seed))


def auc(kind: str, X_tr, y_tr, X_te, y_te, categorical, seed) -> float:
    """Test ROC-AUC; NaN (recorded, never imputed) when the training labels contain a single class."""
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    m = make_model(kind, X_tr, categorical, seed).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
