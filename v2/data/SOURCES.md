# Data sources

All four datasets are public. `code/run_all.py` downloads each one once and caches it here as a CSV plus a
`<name>.schema.json` (categorical columns, target, retrieval time). Later runs reuse the cached CSVs, so
results do not depend on the network. The checksums below are of the cached CSV files in this directory.
`_openml_cache/` is a download cache written by scikit-learn and is not part of the package.

| Dataset | File | Source URL | License | Retrieved (UTC) | SHA-256 of CSV |
|---|---|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | `breast_cancer.csv` | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded via `sklearn.datasets.load_breast_cancer`) | CC BY 4.0 (UCI); bundled with scikit-learn (BSD-3-Clause) | 2026-11-05T03:49:58 | `75a1d74a59df9cb0a78ecc0a36085bf8a85239557f178d3b8d758e88b26416e1` |
| Statlog German Credit (OpenML 31, `credit-g`) | `credit_g.csv` | https://www.openml.org/d/31 | CC BY 4.0 (UCI) | 2026-11-05T03:50:19 | `38b6dbf6fb4b0311a3ffc005730f42623128591fb36473ab3c22d270c0467632` |
| Adult (OpenML 1590) | `adult.csv` | https://www.openml.org/d/1590 | CC BY 4.0 (UCI) | 2026-11-05T03:50:26 | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` |
| Bank Marketing (OpenML 1461) | `bank_marketing.csv` | https://www.openml.org/d/1461 (ARFF parsed with scikit-learn's bundled liac-arff because the pandas ARFF parser failed on integer-coded nominals) | CC BY 4.0 (UCI) | 2026-11-05T03:51:41 | `17d229e152b7789eb835fd785a9f3d14ba7d9c90bf2a4bb83630ef6c7fcad969` |

Target and positive class (minority class) used by the code:

- breast_cancer: `target`, positive = `0` (malignant)
- credit_g: `class`, positive = `bad`
- adult: `class`, positive = `>50K`; `?` entries are kept as the category `__missing__`
- bank_marketing: `Class`, positive = `2` (subscribed)

Citations: Street, Wolberg, Mangasarian (1993) doi:10.1117/12.148698; Hofmann (1994) doi:10.24432/C5NC77;
Becker and Kohavi (1996) doi:10.24432/C5XW20; Moro, Cortez, Rita (2014) doi:10.1016/j.dss.2014.03.001.
