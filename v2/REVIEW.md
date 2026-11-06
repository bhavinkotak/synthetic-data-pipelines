# REVIEW.md: hostile review of v1 and plan for v2

Paper: "From Data Scarcity to Abundance: Scaling Model Training with LLM-Orchestrated Synthetic Data Pipelines" (v1, Zenodo 10.5281/zenodo.18445584).

## Round 1: Frontier (assumptions, prerequisites, baselines)

- Hidden assumption: an LLM orchestrator improves synthetic-data quality ("40-60% quality improvement"). No LLM, no prompt, no metric, no data. No LLM API is available for v2, so this cannot be measured. Action: the orchestrator is reframed as an architecture; its measurable part (generator selection order, retry, fallback, fail-closed gate) is implemented with a deterministic rule-based planner and evaluated; the LLM planner is a stated limitation.
- Hidden assumption: synthetic data is private by construction ("eliminating privacy risks"). Known false for memorising or interpolating generators. Must be measured (DCR vs holdout, membership inference).
- Hidden assumption: one "quality score" transfers across domains. v1 thresholds (Wasserstein < 0.15, KS p > 0.05, PSI < 0.15) are unit-dependent or sample-size-dependent. v2 calibrates gate thresholds against real-vs-real batches of the same size.
- Prerequisites: public data (sklearn breast cancer, OpenML credit-g, adult, bank-marketing); CPU-only; numpy/scipy/pandas/scikit-learn. No GPU, so no diffusion/GAN/LLM generators.
- Baselines a reviewer demands: TRTR (real-data model), independent-marginals generator (floor), SMOTE interpolation (classic oversampler), a memorising "copy" generator (negative control for privacy metrics), ungated vs gated (ablation). Deep generators (CTGAN, TVAE, TabDDPM, GReaT) are unavailable in this environment and are listed as threats.

## Round 2: Breakpoints

| Breakpoint | v2 handling |
|---|---|
| Data scarcity (the title claim) | Tested: E2, n_real in {50, 100, 200, 400}; real-only vs synthetic-only vs real+synthetic |
| Missing data | Tested: E3, MCAR deletion of 10% and 30% of numeric cells; adult also has native categorical missingness |
| Generator exceptions, timeouts, corrupt batches | Tested (simulated): E4 fault injection p in {0, 0.1, 0.3, 0.5}, retry+fallback vs single attempt |
| Undefined gate metric | Tested: unit test, gate fails closed ("unknown" = reject) |
| Adversarial/memorising generator | Tested: leaky copy generator alone and placed first in the candidate order |
| Scale x100 | Partially tested: E5 sampling to 100000 rows (20x train); fitting at 100x rows not tested -> threat |
| Distribution shift (temporal/subgroup) | Not tested -> threat |
| High-cardinality / text / image / time-series data | Not tested -> threat (v1 case studies were imaging, transactions, demand) |
| Small classes in gate batches | Observed: breast cancer batches of 29 rows give noisy DCR shares (false rejections); reported |

## Round 3: Defensibility

- Primary metric: TSTR/TRTR ROC-AUC ratio with a HistGradientBoosting classifier on a real test split never used for fitting or gating. It is a proxy for "model performance"; it ignores calibration, subgroup error, and decision cost. LR ratio is reported as a second model.
- Falsifier (preregistered): REFUTED if any dataset's 95% CI for the mean ratio lies below 0.90, if mean DCR share is credibly above 0.60, or if the pipeline abstains in >= 3 of 5 seeds.
- Measurement bias: (a) ratio > 1 is possible because AUC on a small test split is noisy; (b) DCR/MIA are distance-based attacks, a lower bound on leakage, not a privacy guarantee; (c) gate thresholds are calibrated on the same holdout used for privacy evaluation, so gated privacy numbers are not fully independent of the gate; (d) 5 seeds give wide intervals on the small datasets.

## v1 claims: status and action

| v1 claim | Status | Action |
|---|---|---|
| "90-95% of the performance of models trained on real data" | No data, no datasets named | Measured (E1): replaced by per-dataset TSTR/TRTR ratios with 95% CIs |
| "reducing data acquisition costs by 60-80%" | Fabricated, no source | Removed (cost not measured) |
| "compressed model development timelines from 6-18 months to 4-8 weeks" | Fabricated | Removed |
| "eliminating privacy risks" | False in general | Reframed: measured DCR share and MIA AUC; SMOTE and copy generators leak |
| "40-60% quality improvement" from LLM orchestration | Fabricated | Removed; LLM layer reframed as architecture + limitation |
| Case studies: 94.2% vs 96.1% X-ray accuracy, $2.1M savings, 78%/89% fraud detection, 2.3% FPR, $4.2M ROI, 18% MAPE, $12M, 34% overstock | No source, presented as production results | Removed |
| "approximately 60% of enterprise AI projects are abandoned" | Uncited survey | Removed |
| Mixing table (90:10 ... 10:90 real:synthetic) | Unsupported recommendation | Reframed: E2 measures real+synthetic vs real-only at small n |
| Thresholds (Wasserstein < 0.15, KS p > 0.05, PSI < 0.15) | Arbitrary, scale-dependent | Replaced with calibrated real-vs-real thresholds (eq:gate) |
| HIPAA: synthetic data "falls outside PHI", no IRB needed | Legal claim, unsupported | Removed (out of scope) |
| GDPR/EU AI Act mapping table | Unsupported legal mapping | Removed; regulatory context mentioned only as motivation |
| Reference [1] PATE-GAN in "ICLR" | Not verifiable via DOI/arXiv | Dropped |
| Reference [11] Dwork and Roth, pages 211-407 | Wrong page range (211-487) | Corrected |
| GitHub data-availability claim of evaluation notebooks | Not verifiable | Replaced by the Zenodo package with run command |

## Scores

### v1 (desk criteria)

| Criterion | v1 |
|---|---|
| C1 no filler/hype | FAIL (transformative, comprehensive, production-ready, seamless; em-dashes) |
| C2 citation rigor | FAIL (no inline citations for numbers; PATE-GAN unverifiable; wrong page range) |
| C3 equations | FAIL (no system model, objective, constraints) |
| C4 outcome handling | FAIL (LLM/API failure not specified) |
| C5 substantiated measurements | FAIL (no seeds, no CIs, no splits) |
| C6 internal consistency | FAIL (code thresholds vs text thresholds differ; "eliminating privacy risk" vs pitfalls section) |
| C7 labels/refs | FAIL (Markdown, no labeled floats) |
| C8 claims within evidence | FAIL (case studies presented as production results) |
| C9 numeric grounding | FAIL |
| C10 threats | FAIL (absent) |
| C11 preregistration | FAIL (absent) |

v1 review dimensions: Soundness 2/10, Novelty 4/10, Clarity 5/10, Reproducibility 1/10. Weighted 0.6*2 + 0.4*4 = 2.8 -> reject.

### v2

Process notes (disclosed):
- preregistration.json was written before the full run; `run_all.py` asserts that code constants match it.
- One full run (470 s, 8 cores) produced results/*_runs.csv. Afterwards only report.py (tables, figures, summaries) changed; `run_all.py --analyze` rebuilt summaries and verdict.json from the same run files.
- Reproduction check: a fresh `run_all.py` in a scratch copy (124 s) gave byte-identical verdict.json and summary/gate/fault/scarcity/missing CSVs; only wall-clock timing columns (fit seconds, Table scaling) differ.
- Title adjusted: "From Data Scarcity to Abundance? A Reproducible Benchmark of Quality-Gated Synthetic Data Pipelines for Tabular Model Training". "LLM-Orchestrated" dropped because no LLM was run.
- Stadler et al. (arXiv:2011.07018, real; title checked on arxiv.org) was cut from Related Work: the arXiv API returned HTTP 429 to the desk check for over 40 minutes, so it could not be machine-verified. No result depends on it.

Outcomes:
- H1 (primary, HGB utility ratio, orchestrated gated): SUPPORTED. Means 0.996 / 0.961 / 0.990 / 0.962 (breast cancer / credit-g / adult / bank); lower 95% bounds 0.982 / 0.900 / 0.986 / 0.931; DCR share 0.527 / 0.502 / 0.497 / 0.508; 0 abstentions in 20 runs. Credit-g lower bound 0.9001 sits on the threshold; reported as borderline.
- S1 (gate attribution): met. Adversarial-order MIA AUC 0.550 / 0.521 / 0.502 / 0.501 vs ungated copy 0.799 / 0.820 / 0.672 / 0.664. Narrow on breast cancer; 9 copied batches passed on the two small datasets.
- S2 (scarcity, exploratory): no consistent gain. One positive LR cell (adult, n=50), positive HGB on adult at 200/400, negative HGB on bank at 50.
- S3 (faults, simulated): met. Retry+fallback at p=0.3 delivered 53420 of 53920 rows; lowest dataset mean 0.980. Single attempt: 35757 of 53920, 4 of 20 runs abstained.
- Weakness found by v2 itself: utility ratio is lenient (independent marginals and the leaky copy keep >= 0.949); privacy metrics, not utility, separate the generators.

| Criterion | v2 | Evidence |
|---|---|---|
| C1 no filler/hype | Pass | desk check; 0 em-dashes in tex, bib, REVIEW, metadata |
| C2 citation rigor | Pass | 31 bib entries, each with DOI or arXiv id, verified by the desk check against Crossref/arXiv; dataset DOIs from UCI; v1 PATE-GAN dropped |
| C3 equations | Pass | 9 labelled Methodology equations (copula, interpolation, utility, marginal, correlation, DCR/Gower, MIA, gate, objective) + Algorithm 1 |
| C4 outcome handling | Pass | Outcome Handling subsection; error/timeout/invalid/unknown fail closed; abstention rule; missing values measured (E3) |
| C5 substantiated measurements | Pass | 5 seeds, disjoint train/holdout/test, t-intervals, paired differences, counts with denominators |
| C6 internal consistency | Pass | all tables auto-injected from results/ by report.py; prose numbers grounded by desk check |
| C7 labels/refs | Pass | every fig/tab/eq/alg labelled and referenced |
| C8 claims within evidence | Pass | faults labelled simulated; LLM layer stated untested; no cost/time/compliance claims |
| C9 numeric grounding | Pass | desk check numbers_checked=453 |
| C10 threats | Pass | untested LLM layer, unavailable baselines, construct/statistical validity, untested breakpoints, open questions |
| C11 preregistration | Pass | preregistration.json + verdict.json computed by code |

v2 review dimensions (self-assessed): Soundness 7/10, Novelty 5/10, Clarity 8/10, Reproducibility 9/10. Weighted 0.6*7 + 0.4*5 = 6.2 -> below the 7.0 accept line.
- Soundness capped by: simple generators only, five seeds, credit-g on the threshold, distance-based attack only, gate and privacy metrics share the holdout.
- Novelty capped by: the gate combines known pieces (holdout DCR, real-vs-real calibration); the contribution is the measured attribution and breakpoints, not a new method.

Final desk check:

```
Desk check: .../synthetic-data-pipelines/paper/main.tex  sections=['Introduction', 'Related Work', 'Methodology', 'Evaluation', 'Threats to Validity', 'Conclusion', 'Acknowledgment']  abstract_words=229  method_eqs=9  numbers_checked=465
PASS
```

Build: pdflatex/bibtex clean (no undefined references or citations, no overfull boxes), 8 pages. pytest: 15 passed.
