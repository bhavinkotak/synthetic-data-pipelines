"""Experiment constants. Values mirror preregistration.json; run_all.py asserts they agree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
PAPER = ROOT / "paper" / "main.tex"
PREREG = ROOT / "preregistration.json"

SEEDS = (0, 1, 2, 3, 4)
N_CAP = 10000
TRAIN_FRAC = 0.5
HOLDOUT_FRAC = 0.25

# Quality gate (preregistered)
KAPPA = 2.0
REF_QUANTILE = 0.95
REF_DRAWS = 20
TAU_PRIVACY = 0.6
BATCHES = 10
MAX_ATTEMPTS = 2
MIN_RELEASE = 0.5
TIME_BUDGET_S = 30.0

# Decision rule (preregistered)
UTILITY_TARGET = 0.9
MIA_TARGET_GATED = 0.55
MIA_LEAKY_FLOOR = 0.6
DELIVERY_TARGET = 0.95
ABSTAIN_REFUTE_SEEDS = 3

# Generators
CANDIDATES = ("copula", "gmm", "smote", "independent")
GMM_COMPONENTS = (1, 2, 4, 8)
GMM_MIN_ROWS_PER_COMPONENT = 10
SMOTE_K = 5
CONTINUOUS_MIN_UNIQUE = 20

# Privacy evaluation: synthetic records subsampled for distance metrics (reference sets are not capped)
PRIVACY_SYN_N = 2000

# Breakpoint scenarios
SCARCITY_N = (50, 100, 200, 400)
SYN_MULT = 4
MISSING_RATES = (0.0, 0.1, 0.3)
FAULT_RATES = (0.0, 0.1, 0.3, 0.5)
SCALING_N = (1000, 10000, 100000)
SCALING_DCR_N = (1000, 10000)
