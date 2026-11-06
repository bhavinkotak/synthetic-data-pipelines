"""Download, cache, and split the public benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import HOLDOUT_FRAC, N_CAP, RAW, TRAIN_FRAC

MISSING_CAT = "__missing__"

DATASETS = {
    "breast_cancer": {"source": "sklearn", "target": "target", "positive": "0",
                      "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                      "license": "CC BY 4.0 (UCI); bundled with scikit-learn (BSD-3-Clause)"},
    "credit_g": {"source": "openml", "openml_id": 31, "target": "class", "positive": "bad",
                 "url": "https://www.openml.org/d/31", "license": "CC BY 4.0 (UCI Statlog German Credit)"},
    "adult": {"source": "openml", "openml_id": 1590, "target": "class", "positive": ">50K",
              "url": "https://www.openml.org/d/1590", "license": "CC BY 4.0 (UCI Adult)"},
    "bank_marketing": {"source": "openml", "openml_id": 1461, "target": "Class", "positive": "2",
                       "url": "https://www.openml.org/d/1461", "license": "CC BY 4.0 (UCI Bank Marketing)"},
}


@dataclass
class Table:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical: list[str]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(name: str) -> tuple[pd.DataFrame, list[str]]:
    spec = DATASETS[name]
    if spec["source"] == "sklearn":
        from sklearn.datasets import load_breast_cancer
        b = load_breast_cancer(as_frame=True)
        df = b.frame.copy()
        return df, []
    from sklearn.datasets import fetch_openml
    try:
        b = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="pandas",
                         data_home=str(RAW / "_openml_cache"))
    except TypeError:  # sklearn's ARFF parsers fail on integer-coded nominals in some sklearn/pandas versions
        return _download_arff(spec["openml_id"], spec["target"])
    df = b.frame.copy()
    cats = [c for c in df.columns if c != spec["target"] and not pd.api.types.is_numeric_dtype(df[c])]
    return df, cats


def _download_arff(data_id: int, target: str) -> tuple[pd.DataFrame, list[str]]:
    import urllib.request

    from sklearn.externals import _arff

    api = f"https://api.openml.org/api/v1/json/data/{data_id}"
    with urllib.request.urlopen(api, timeout=60) as r:  # noqa: S310 - fixed https host
        url = json.loads(r.read())["data_set_description"]["url"]
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
        obj = _arff.loads(r.read().decode())
    names = [a[0] for a in obj["attributes"]]
    df = pd.DataFrame(obj["data"], columns=names)
    cats = [a[0] for a in obj["attributes"] if isinstance(a[1], list) and a[0] != target]
    return df, cats


def ensure_cached(name: str) -> dict:
    """Write data/raw/<name>.csv and .schema.json once; return provenance metadata."""
    RAW.mkdir(parents=True, exist_ok=True)
    csv, schema = RAW / f"{name}.csv", RAW / f"{name}.schema.json"
    if not csv.exists() or not schema.exists():
        df, cats = _download(name)
        df.to_csv(csv, index=False)
        schema.write_text(json.dumps({
            "categorical": cats, "target": DATASETS[name]["target"],
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1))
    meta = json.loads(schema.read_text())
    return {"name": name, "file": csv.name, "sha256": sha256(csv), "url": DATASETS[name]["url"],
            "license": DATASETS[name]["license"], "retrieved_utc": meta["retrieved_utc"]}


def load(name: str) -> Table:
    ensure_cached(name)
    spec = DATASETS[name]
    meta = json.loads((RAW / f"{name}.schema.json").read_text())
    cats = meta["categorical"]
    df = pd.read_csv(RAW / f"{name}.csv", dtype={c: str for c in cats + [spec["target"]]},
                     keep_default_na=False, na_values=["", "?", "NaN", "nan"])
    y = (df.pop(spec["target"]).astype(str).str.strip() == spec["positive"]).astype(int).to_numpy()
    for c in cats:
        df[c] = df[c].fillna(MISSING_CAT).astype(str).str.strip()
    for c in df.columns:
        if c not in cats:
            df[c] = df[c].astype(float)
    return Table(name, df.reset_index(drop=True), y, cats)


@dataclass
class Split:
    X_tr: pd.DataFrame
    y_tr: np.ndarray
    X_ho: pd.DataFrame
    y_ho: np.ndarray
    X_te: pd.DataFrame
    y_te: np.ndarray


def split(t: Table, seed: int) -> Split:
    X, y = t.X, t.y
    if len(X) > N_CAP:
        idx, _ = train_test_split(np.arange(len(X)), train_size=N_CAP, stratify=y, random_state=seed)
        X, y = X.iloc[np.sort(idx)].reset_index(drop=True), y[np.sort(idx)]
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=TRAIN_FRAC, stratify=y, random_state=seed)
    ho = HOLDOUT_FRAC / (1 - TRAIN_FRAC)
    X_ho, X_te, y_ho, y_te = train_test_split(X_rest, y_rest, train_size=ho, stratify=y_rest, random_state=seed)
    r = lambda d: d.reset_index(drop=True)  # noqa: E731
    return Split(r(X_tr), y_tr, r(X_ho), y_ho, r(X_te), y_te)


def subsample(X: pd.DataFrame, y: np.ndarray, n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if n >= len(X):
        return X.reset_index(drop=True), y
    idx, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=seed)
    return X.iloc[idx].reset_index(drop=True), y[idx]
