"""Aggregation, preregistered verdict, LaTeX tables, and figures, all computed from results/*.csv."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from . import config as C

GEN_LABEL = {"copula": "Gaussian copula", "gmm": "Class-cond.\\ GMM", "smote": "SMOTE-style",
             "independent": "Indep.\\ marginals", "copy": "Leaky copy (control)"}
DS_LABEL = {"breast_cancer": "Breast cancer", "credit_g": "Credit-g", "adult": "Adult",
            "bank_marketing": "Bank marketing"}
DATASET_ORDER = ("breast_cancer", "credit_g", "adult", "bank_marketing")


def ci(x) -> dict:
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    h = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return {"n": n, "mean": m, "std": sd, "ci_lo": m - h, "ci_hi": m + h}


METRICS = ("ratio_hgb", "ratio_lr", "auc_hgb", "auc_lr", "marginal", "correlation", "dcr_share", "mia_auc",
           "exact_match", "delivered_frac")


def summarize(df: pd.DataFrame, keys: list[str], metrics=METRICS) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        base = dict(zip(keys, k, strict=True))
        base["runs"] = len(g)
        if "abstained" in g:
            base["abstained"] = int(g["abstained"].fillna(0).sum())
        for m in metrics:
            if m in g and g[m].notna().any():
                for s, v in ci(g[m]).items():
                    base[f"{m}_{s}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def verdict(main: pd.DataFrame) -> dict:
    per = {}
    status = "SUPPORTED"
    any_refute = False
    orch = main[main.condition == "orchestrated_gated"]
    for ds in DATASET_ORDER:
        g = orch[orch.dataset == ds]
        u, p = ci(g["ratio_hgb"]), ci(g["dcr_share"])
        abst = int(g["abstained"].sum())
        sup = (u["n"] > 0 and u["ci_lo"] >= C.UTILITY_TARGET and p["mean"] <= C.TAU_PRIVACY and abst == 0)
        ref = ((u["n"] > 0 and u["ci_hi"] < C.UTILITY_TARGET) or (p["n"] > 0 and p["ci_lo"] > C.TAU_PRIVACY)
               or abst >= C.ABSTAIN_REFUTE_SEEDS)
        per[ds] = {"utility_ratio_hgb": u, "dcr_share": p, "abstained_seeds": abst, "seeds": len(g),
                   "meets_support": bool(sup), "meets_refute": bool(ref)}
        any_refute |= ref
        if not sup:
            status = "INCONCLUSIVE"
    if any_refute:
        status = "REFUTED"
    # Secondary endpoints (reported, not part of the verdict)
    s1 = {}
    for ds in DATASET_ORDER:
        adv = main[(main.dataset == ds) & (main.condition == "adversarial_gated")]["mia_auc"]
        leak = main[(main.dataset == ds) & (main.condition == "copy_ungated")]["mia_auc"]
        s1[ds] = {"adversarial_gated_mia": ci(adv), "copy_ungated_mia": ci(leak),
                  "met": bool(ci(adv)["mean"] <= C.MIA_TARGET_GATED and ci(leak)["mean"] > C.MIA_LEAKY_FLOOR)}
    return {"verdict": status, "primary_endpoint": "utility_ratio_hgb (orchestrated_gated)",
            "target": C.UTILITY_TARGET, "tau_privacy": C.TAU_PRIVACY, "per_dataset": per,
            "secondary_S1_gate_attribution": {"per_dataset": s1, "met_all": all(v["met"] for v in s1.values())}}


def fmt(m, s=None, d=3) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    return f"{m:.{d}f}" if s is None or not np.isfinite(s) else f"${m:.{d}f}\\pm{s:.{d}f}$"


MAIN_CONDITIONS = ("copula_ungated", "gmm_ungated", "smote_ungated", "independent_ungated", "copy_ungated",
                   "copula_gated", "gmm_gated", "smote_gated", "independent_gated", "copy_gated",
                   "orchestrated_gated", "adversarial_gated")


def cond_label(c: str) -> str:
    if c == "orchestrated_gated":
        return "Orchestrated, gated"
    if c == "adversarial_gated":
        return "Adversarial order, gated"
    if c == "fresh_real":
        return "Fresh real (reference)"
    g, mode = c.rsplit("_", 1)
    return f"{GEN_LABEL[g]}, {mode}"


def table_metric(summ: pd.DataFrame, metrics: list[tuple[str, int]], conds, label, caption,
                 colw: str | None = None) -> str:
    """table*: rows = conditions; per dataset, the listed metrics as mean+-std over seeds."""
    ncol = len(metrics)
    head = " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASET_ORDER)
    sub = " & ".join(" & ".join(name for name, _ in metrics) for _ in DATASET_ORDER)
    spec = f"w{{c}}{{{colw}}}" if colw else "c"
    lines = [f"\\begin{{table*}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n\\scriptsize",
             "\\setlength{\\tabcolsep}{2.5pt}",
             f"\\begin{{tabular}}{{l{spec * ncol * len(DATASET_ORDER)}}}\n\\toprule",
             f"Condition & {head} \\\\", f" & {sub} \\\\\n\\midrule"]
    for c in conds:
        cells = []
        for d in DATASET_ORDER:
            r = summ[(summ.dataset == d) & (summ.condition == c)]
            for _, (m, dec) in metrics:
                if r.empty or f"{m}_mean" not in r or not np.isfinite(r[f"{m}_mean"].iloc[0]):
                    cells.append("n/a")
                else:
                    cells.append(f"{r[f'{m}_mean'].iloc[0]:.{dec}f}")
        lines.append(f"{cond_label(c)} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    return "\n".join(lines)


def inject(tex: str, name: str, body: str) -> str:
    pat = re.compile(rf"(% AUTO:BEGIN {name}\n).*?(% AUTO:END {name})", re.DOTALL)
    if not pat.search(tex):
        return tex
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), tex)
