"""Single entry point: download/cache data, run all scenarios with fixed seeds, write results/, figures, tables.

Usage:  python code/run_all.py            (full run)
        python code/run_all.py --analyze  (recompute summaries/figures/tables from results/*.csv)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from sdp import config as C  # noqa: E402
from sdp.data import DATASETS, ensure_cached, load  # noqa: E402
from sdp.experiments import run_main, run_missing, run_scaling, run_scarcity  # noqa: E402


def check_prereg() -> None:
    p = json.loads(C.PREREG.read_text())
    g = p["gate"]
    assert g["kappa"] == C.KAPPA and g["tau_privacy_dcr_share"] == C.TAU_PRIVACY
    assert g["batches_per_request"] == C.BATCHES and g["max_attempts_per_candidate"] == C.MAX_ATTEMPTS
    assert g["min_release_fraction"] == C.MIN_RELEASE and g["reference_draws"] == C.REF_DRAWS
    assert tuple(p["seeds"]) == C.SEEDS


def _job(kind: str, name: str, seed: int):
    t = load(name)
    if kind == "main":
        return kind, run_main(t, seed)
    if kind == "scarcity":
        return kind, run_scarcity(t, seed)
    return kind, run_missing(t, seed)


def run() -> None:
    check_prereg()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    provenance = [ensure_cached(n) for n in DATASETS]
    info = []
    for n in DATASETS:
        t = load(n)
        info.append({"dataset": n, "rows": len(t.X), "features": t.X.shape[1], "categorical": len(t.categorical),
                     "numeric": t.X.shape[1] - len(t.categorical), "positive_rate": float(t.y.mean()),
                     "rows_used": min(len(t.X), C.N_CAP)})
    (C.RESULTS / "datasets.json").write_text(json.dumps({"datasets": info, "provenance": provenance}, indent=1))

    jobs = [(k, n, s) for k in ("main", "scarcity", "missing") for n in DATASETS for s in C.SEEDS]
    collected = {"main": [], "faults": [], "scarcity": [], "missing": []}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for kind, res in ex.map(_job, *zip(*jobs, strict=True)):
            if kind == "main":
                collected["main"] += res[0]
                collected["faults"] += res[1]
            else:
                collected[kind] += res
    for k, rows in collected.items():
        pd.DataFrame(rows).to_csv(C.RESULTS / f"{k}_runs.csv", index=False)
    pd.DataFrame(run_scaling(load("adult"))).to_csv(C.RESULTS / "scaling_runs.csv", index=False)
    (C.RESULTS / "run_info.json").write_text(json.dumps({
        "wall_clock_seconds": time.time() - started, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true", help="skip experiments; rebuild outputs from results/")
    a = ap.parse_args()
    if not a.analyze:
        run()
    from sdp.report import build
    build()


if __name__ == "__main__":
    main()
