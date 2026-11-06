"""Build zenodo/: copy the compiled paper and zip paper source, code, data, results, tests, and review."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "zenodo"
SLUG = "synthetic-data-pipelines"
SKIP_DIRS = {"__pycache__", ".venv", "_openml_cache", ".pytest_cache"}
SKIP_SUFFIX = {".aux", ".log", ".blg", ".bbl", ".out", ".toc", ".synctex.gz", ".pyc", ".fls", ".fdb_latexmk"}
SKIP_NAMES = {".ref_cache.json", ".DS_Store", "main.pdf"}


def _files():
    yield from (ROOT / n for n in ("REVIEW.md", "preregistration.json", "run.log"))
    for top in ("paper", "code", "data", "results", "tests"):
        for p in sorted((ROOT / top).rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir() or SKIP_DIRS & set(rel.parts) or p.name in SKIP_NAMES:
                continue
            if any(p.name.endswith(s) for s in SKIP_SUFFIX):
                continue
            yield p


def main() -> None:
    OUT.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "paper" / "main.pdf", OUT / f"{SLUG}-v2.pdf")
    zpath = OUT / f"{SLUG}-v2-code-data.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in _files():
            z.write(p, f"{SLUG}-v2/{p.relative_to(ROOT)}")
    size_mb = zpath.stat().st_size / 1e6
    assert size_mb < 50, size_mb
    print(f"{zpath.name}: {size_mb:.2f} MB, {len(z.namelist())} files")


if __name__ == "__main__":
    main()
