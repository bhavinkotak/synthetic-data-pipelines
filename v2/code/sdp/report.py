"""Build every derived artefact from results/*_runs.csv: summaries, verdict.json, figures, LaTeX tables."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .analysis import (DATASET_ORDER, DS_LABEL, GEN_LABEL, ci, cond_label, inject, summarize,  # noqa: E402
                       table_metric, verdict)

GENS = (*C.CANDIDATES, "copy")
OUTCOME_COLS = ("n_accepted", "n_rejected_fidelity", "n_rejected_privacy", "n_error", "n_timeout", "n_invalid",
                "n_unknown")
COLORS = {"copula": "#1b6ca8", "gmm": "#4aa564", "smote": "#e08e0b", "independent": "#8c8c8c", "copy": "#c0392b",
          "orchestrated": "#000000"}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "font.family": "serif"})
COL_W = 3.49  # IEEE column width in inches (8.86 cm)
TEXT_W = 7.16


def _paired(df: pd.DataFrame, a: str, b: str, metrics, keys=("dataset",)) -> pd.DataFrame:
    """Per-seed difference a - b for each metric, summarised with a 95% t-interval."""
    rows = []
    for k, g in df.groupby(list(keys), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        A = g[g.condition == a].set_index("seed")
        B = g[g.condition == b].set_index("seed")
        row = {**dict(zip(keys, k, strict=True)), "a": a, "b": b}
        for m in metrics:
            if m in A and m in B:
                d = (A[m] - B[m]).dropna()
                for s, v in ci(d).items():
                    row[f"diff_{m}_{s}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def gate_stats(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gated = [c for c in main.condition.unique() if c.endswith("_gated")]
    for (ds, cond), g in main[main.condition.isin(gated)].groupby(["dataset", "condition"], sort=False):
        r = {"dataset": ds, "condition": cond, "seeds": len(g), "attempts": int(g.attempts.sum()),
             "abstained_seeds": int(g.abstained.sum())}
        for c in OUTCOME_COLS:
            r[c] = int(g[c].sum())
        r["acceptance_rate"] = r["n_accepted"] / r["attempts"]
        for gen in GENS:
            col = f"batches_{gen}"
            r[col] = int(g[col].fillna(0).sum()) if col in g else 0
        rows.append(r)
    return pd.DataFrame(rows)


def fig_utility(summ: pd.DataFrame) -> None:
    conds = [f"{g}_ungated" for g in GENS] + ["orchestrated_gated"]
    keys = [*GENS, "orchestrated"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    w = 0.13
    x = np.arange(len(DATASET_ORDER))
    for i, (c, key) in enumerate(zip(conds, keys, strict=True)):
        r = summ[summ.condition == c].set_index("dataset").reindex(DATASET_ORDER)
        err = (r["ratio_hgb_ci_hi"] - r["ratio_hgb_mean"]).to_numpy()
        label = "Orchestrated, gated" if key == "orchestrated" else GEN_LABEL[key].replace("\\ ", " ")
        ax.bar(x + (i - 2.5) * w, r["ratio_hgb_mean"], w, yerr=err, color=COLORS[key], label=label,
               error_kw={"elinewidth": 0.6, "capsize": 1.2})
    ax.axhline(C.UTILITY_TARGET, ls="--", lw=0.7, color="k")
    ax.set_xticks(x, [DS_LABEL[d] for d in DATASET_ORDER])
    ax.set_ylim(0.8, 1.12)
    ax.set_ylabel("TSTR/TRTR AUC ratio (HGB)")
    ax.legend(ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGURES / "utility_ratio.pdf")
    plt.close(fig)


def fig_privacy(summ: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    marks = dict(zip(DATASET_ORDER, ("o", "s", "^", "D"), strict=True))
    for g in GENS:
        r = summ[summ.condition == f"{g}_ungated"]
        for _, row in r.iterrows():
            ax.scatter(row["dcr_share_mean"], row["mia_auc_mean"], c=COLORS[g], marker=marks[row["dataset"]], s=14,
                       edgecolors="none")
    r = summ[summ.condition == "fresh_real"]
    for _, row in r.iterrows():
        ax.scatter(row["dcr_share_mean"], row["mia_auc_mean"], facecolors="none", edgecolors="k",
                   marker=marks[row["dataset"]], s=18, linewidths=0.6)
    ax.axvline(C.TAU_PRIVACY, ls="--", lw=0.7, color="k")
    ax.set_xlabel("DCR share (0.5 = fresh real data)")
    ax.set_ylabel("Membership-inference AUC")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=COLORS[g], label=GEN_LABEL[g].replace("\\ ", " ")) for g in GENS]
    h += [Line2D([], [], ls="", marker=marks[d], color="0.4", mfc="none", label=DS_LABEL[d]) for d in DATASET_ORDER]
    h += [Line2D([], [], ls="", marker="o", mfc="none", color="k", label="Fresh real (ref.)")]
    ax.legend(handles=h, ncol=2, frameon=False, loc="upper left", handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGURES / "privacy.pdf")
    plt.close(fig)


def fig_scarcity(sc: pd.DataFrame, main_summ: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 1.95), sharey=False)
    styles = {"real_only": ("Real only", "k", "-"), "synthetic_only_gated": ("Synthetic only (gated)", COLORS["copula"], "--"),
              "hybrid_gated": ("Real + synthetic (gated)", COLORS["gmm"], "-")}
    for ax, ds in zip(axes, DATASET_ORDER, strict=True):
        for cond, (lab, col, ls) in styles.items():
            r = sc[(sc.dataset == ds) & (sc.condition == cond)].sort_values("n_real")
            if r.empty:
                continue
            ax.errorbar(r["n_real"], r["auc_lr_mean"], yerr=r["auc_lr_ci_hi"] - r["auc_lr_mean"], label=lab, color=col,
                        ls=ls, lw=0.9, marker="o", ms=2.5, capsize=1.5, elinewidth=0.6)
        full = main_summ[(main_summ.dataset == ds) & (main_summ.condition == "trtr")]["auc_lr_mean"]
        if len(full):
            ax.axhline(full.iloc[0], color="0.5", lw=0.6, ls=":", label="Real, full train split")
        ax.set_xscale("log")
        ax.set_xticks(list(C.SCARCITY_N), [str(n) for n in C.SCARCITY_N])
        ax.minorticks_off()
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Real training rows")
    axes[0].set_ylabel("Test AUC (logistic regr.)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86))
    fig.savefig(C.FIGURES / "scarcity.pdf")
    plt.close(fig)


def fig_faults(fs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    for pol, col, lab in (("retry_fallback", COLORS["copula"], "Retry + fallback (4 candidates)"),
                          ("single_attempt", COLORS["copy"], "Single attempt, single candidate")):
        r = fs[fs.policy == pol].sort_values("p_fail")
        ax.errorbar(r["p_fail"], r["delivered_frac_mean"], yerr=r["delivered_frac_ci_hi"] - r["delivered_frac_mean"],
                    color=col, marker="o", ms=3, lw=0.9, capsize=1.5, label=lab)
    ax.axhline(C.MIN_RELEASE, ls=":", color="0.4", lw=0.7)
    ax.set_xlabel("Simulated fault probability per generator call")
    ax.set_ylabel("Delivered fraction of rows")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGURES / "faults.pdf")
    plt.close(fig)


def _tab_datasets(info: list[dict]) -> str:
    lines = ["\\begin{table}[!t]", "\\caption{Benchmark datasets (public, binary classification). Rows used per seed "
             f"after the stratified cap of {C.N_CAP} rows.}}", "\\label{tab:datasets}", "\\centering",
             "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Dataset & Rows & Num. & Cat. & Pos. rate & Rows used \\\\", "\\midrule"]
    for d in info:
        lines.append(f"{DS_LABEL[d['dataset']]} & {d['rows']} & {d['numeric']} & {d['categorical']} & "
                     f"{d['positive_rate']:.3f} & {d['rows_used']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def gate_pooled(gs: pd.DataFrame) -> pd.DataFrame:
    """Gate outcomes summed over datasets and seeds (20 runs per condition)."""
    cols = ["seeds", "attempts", "abstained_seeds", *OUTCOME_COLS, *[f"batches_{g}" for g in GENS]]
    p = gs.groupby("condition", sort=False)[cols].sum().reset_index()
    p["acceptance_rate"] = p["n_accepted"] / p["attempts"]
    return p


def _tab_gate(gp: pd.DataFrame) -> str:
    conds = [f"{g}_gated" for g in GENS] + ["orchestrated_gated", "adversarial_gated"]
    lines = ["\\begin{table}[!t]", "\\caption{Quality-gate outcomes pooled over four datasets and five seeds. "
             "Accept = accepted batches / generator calls. Fid./Priv. = rejections by the fidelity or privacy rule. "
             "Abst. = runs (of 20) in which the pipeline withheld release.}", "\\label{tab:gate}", "\\centering",
             "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Condition & Calls & Accept & Fid. & Priv. & Abst. \\\\", "\\midrule"]
    for c in conds:
        g = gp[gp.condition == c].iloc[0]
        lines.append(f"{cond_label(c)} & {int(g.attempts)} & {g.acceptance_rate:.3f} & {int(g.n_rejected_fidelity)} & "
                     f"{int(g.n_rejected_privacy)} & {int(g.abstained_seeds)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _tab_missing(ms: pd.DataFrame) -> str:
    lines = ["\\begin{table}[!t]", "\\caption{Utility ratio (HGB, mean over five seeds) when a fraction of numeric "
             "cells in the train and holdout splits is deleted completely at random. The breast-cancer set has no "
             "categorical columns; all others keep their categorical columns intact.}", "\\label{tab:missing}",
             "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llrrr}", "\\toprule",
             "Dataset & Condition & " + " & ".join(f"{r:.1f}" for r in C.MISSING_RATES) + " \\\\", "\\midrule"]
    for ds in DATASET_ORDER:
        for cond, lab in (("copula_ungated", "Copula, ungated"), ("orchestrated_gated", "Orchestrated, gated")):
            cells = []
            for rate in C.MISSING_RATES:
                r = ms[(ms.dataset == ds) & (ms.condition == cond) & (np.isclose(ms.missing_rate, rate))]
                cells.append(f"{r['ratio_hgb_mean'].iloc[0]:.3f}" if len(r) else "n/a")
            lines.append(f"{DS_LABEL[ds] if cond == 'copula_ungated' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _tab_scaling(sc: pd.DataFrame) -> str:
    lines = ["\\begin{table}[!t]", "\\caption{Single-machine wall-clock seconds (median of three repetitions) on the "
             "Adult train split, seed 0. Fit uses the full train split; the other columns give the number of "
             "synthetic rows sampled or checked.}", "\\label{tab:scaling}", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{2pt}"]
    cols = [("fit", None)] + [("sample", n) for n in C.SCALING_N] + [("dcr_share", n) for n in C.SCALING_DCR_N]
    lines += [f"\\begin{{tabular}}{{l{'r' * (len(cols) - len(C.SCALING_DCR_N))}"
              f"{'w{c}{27pt}' * len(C.SCALING_DCR_N)}}}", "\\toprule",
              " & & \\multicolumn{3}{c}{Sample (rows)} & \\multicolumn{2}{c}{DCR share (rows)} \\\\",
              "Generator & Fit & " + " & ".join(str(n) for n in C.SCALING_N) + " & "
              + " & ".join(str(n) for n in C.SCALING_DCR_N) + " \\\\", "\\midrule"]
    for g in C.CANDIDATES:
        cells = []
        for stage, n in cols:
            r = sc[(sc.generator == g) & (sc.stage == stage)]
            if n is not None:
                r = r[r.n == n]
            cells.append(f"{r.seconds.iloc[0]:.3f}")
        lines.append(f"{GEN_LABEL[g].replace('Class-cond.', 'Cl.-cond.').replace('marginals', 'marg.')} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def build() -> None:
    R = C.RESULTS
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(R / "main_runs.csv")
    faults = pd.read_csv(R / "faults_runs.csv")
    scar = pd.read_csv(R / "scarcity_runs.csv")
    miss = pd.read_csv(R / "missing_runs.csv")
    scal = pd.read_csv(R / "scaling_runs.csv")
    info = json.loads((R / "datasets.json").read_text())["datasets"]

    ms = summarize(main, ["dataset", "condition"])
    ms.to_csv(R / "summary_main.csv", index=False)
    pooled = summarize(main, ["condition"])
    pooled.to_csv(R / "summary_pooled.csv", index=False)
    gs = gate_stats(main)
    gs.to_csv(R / "gate_stats.csv", index=False)
    gp = gate_pooled(gs)
    gp.to_csv(R / "gate_stats_pooled.csv", index=False)
    faults.groupby(["policy", "p_fail"])[["attempts", *OUTCOME_COLS, "abstained", "delivered", "requested"]].sum() \
        .reset_index().to_csv(R / "faults_outcomes.csv", index=False)
    eff = pd.concat([_paired(main, f"{g}_gated", f"{g}_ungated", ("ratio_hgb", "dcr_share", "mia_auc"))
                     for g in GENS] +
                    [_paired(main, "orchestrated_gated", "copula_ungated", ("ratio_hgb", "ratio_lr", "dcr_share")),
                     _paired(main, "adversarial_gated", "copy_ungated", ("ratio_hgb", "dcr_share", "mia_auc"))])
    eff.to_csv(R / "gate_effect.csv", index=False)
    fs = summarize(faults, ["policy", "p_fail"], ("delivered_frac", "ratio_hgb"))
    fs.to_csv(R / "summary_faults.csv", index=False)
    summarize(faults, ["dataset", "policy", "p_fail"], ("delivered_frac", "ratio_hgb")).to_csv(
        R / "summary_faults_by_dataset.csv", index=False)
    ss = summarize(scar, ["dataset", "n_real", "condition"], ("auc_lr", "auc_hgb", "delivered_frac"))
    ss.to_csv(R / "summary_scarcity.csv", index=False)
    se = pd.concat([_paired(scar, c, "real_only", ("auc_lr", "auc_hgb"), keys=("dataset", "n_real"))
                    for c in ("hybrid_gated", "hybrid_ungated", "synthetic_only_gated")])
    se.to_csv(R / "scarcity_effect.csv", index=False)
    mm = summarize(miss, ["dataset", "missing_rate", "condition"], ("ratio_hgb", "ratio_lr", "auc_hgb"))
    mm.to_csv(R / "summary_missing.csv", index=False)
    me = pd.concat([_paired(miss[miss.missing_rate.isin([0.0, r])].assign(
        condition=lambda d: d.condition + np.where(d.missing_rate == 0.0, "@0", "@r")),
        f"{c}@r", f"{c}@0", ("ratio_hgb",)).assign(missing_rate=r, condition=c)
        for r in C.MISSING_RATES[1:] for c in ("copula_ungated", "orchestrated_gated")])
    me.to_csv(R / "missing_effect.csv", index=False)

    v = verdict(main)
    v["secondary_S3_fault_tolerance"] = {
        "p_fail": 0.3, "target_delivered": C.DELIVERY_TARGET,
        "retry_fallback_min_dataset_mean": float(summarize(faults[(faults.policy == "retry_fallback") &
                                                                  np.isclose(faults.p_fail, 0.3)], ["dataset"],
                                                           ("delivered_frac",))["delivered_frac_mean"].min())}
    v["secondary_S3_fault_tolerance"]["met"] = v["secondary_S3_fault_tolerance"]["retry_fallback_min_dataset_mean"] \
        >= C.DELIVERY_TARGET
    (R / "verdict.json").write_text(json.dumps(v, indent=1, default=float))

    fig_utility(ms)
    fig_privacy(ms)
    fig_scarcity(ss, ms)
    fig_faults(fs)

    if C.PAPER.exists():
        tex = C.PAPER.read_text()
        tex = inject(tex, "datasets", _tab_datasets(info))
        tex = inject(tex, "utility", table_metric(
            ms, [("HGB", ("ratio_hgb", 3)), ("LR", ("ratio_lr", 3))],
            [c for c in ("copula_ungated", "gmm_ungated", "smote_ungated", "independent_ungated", "copy_ungated",
                         "copula_gated", "gmm_gated", "independent_gated", "orchestrated_gated",
                         "adversarial_gated")],
            "tab:utility", "Utility ratio TSTR/TRTR (ROC-AUC on the real test split) for the HistGradientBoosting "
            "(HGB) and logistic-regression (LR) classifiers, mean over five seeds. Gated rows average only the runs "
            "in which data were released (see Table~\\ref{tab:gate}); SMOTE-style and leaky-copy gated runs never "
            "released data on any dataset and are omitted.", colw="20pt"))
        tex = inject(tex, "privacy", table_metric(
            ms, [("Marg.", ("marginal", 3)), ("Corr.", ("correlation", 3)), ("DCR", ("dcr_share", 3)),
                 ("MIA", ("mia_auc", 3))],
            ("fresh_real", "copula_ungated", "gmm_ungated", "smote_ungated", "independent_ungated", "copy_ungated",
             "orchestrated_gated", "adversarial_gated"),
            "tab:privacy", "Fidelity and privacy, mean over five seeds. Marg.: mean per-column KS/TV distance to the "
            "holdout. Corr.: mean absolute Spearman-correlation difference. DCR: share of synthetic records nearer "
            "to training records than to holdout records (0.5 is the fresh-real reference). MIA: AUC of the "
            "distance-based membership attack (0.5 is chance)."))
        tex = inject(tex, "gate", _tab_gate(gp))
        tex = inject(tex, "missing", _tab_missing(mm))
        tex = inject(tex, "scaling", _tab_scaling(scal))
        C.PAPER.write_text(tex)
    print(json.dumps({"verdict": v["verdict"], **{k: {"ratio": round(x["utility_ratio_hgb"]["mean"], 4),
                                                     "lo": round(x["utility_ratio_hgb"]["ci_lo"], 4),
                                                     "dcr": round(x["dcr_share"]["mean"], 4),
                                                     "abst": x["abstained_seeds"]}
                                                  for k, x in v["per_dataset"].items()}}, indent=1))
