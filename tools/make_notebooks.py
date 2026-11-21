"""Generate the thin analysis notebooks in ``notebooks/``.

Each notebook only imports functions from ``src``, runs one stage and
displays the results. Running this script overwrites the notebooks with a
clean, unexecuted version.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

# Shared first cell: make the repository root importable and set display options.
SETUP = """import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)"""


def cell(kind: str, text: str) -> dict:
    """Return a notebook cell of the given kind ('markdown' or 'code')."""
    src = text.strip("\n").splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def write(name: str, cells: list[dict]) -> None:
    """Write a notebook file with the standard Python 3 kernel.

    Cells get fixed ids (notebook number and cell index) so that regenerated
    notebooks are identical and valid under nbformat 4.5.
    """
    for i, c in enumerate(cells):
        c["id"] = f"nb{name[:2]}-{i:02d}"
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    NB_DIR.mkdir(exist_ok=True)
    (NB_DIR / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", NB_DIR / name)


def nb01() -> None:
    """Notebook 01: data cleaning and screening."""
    write("01_data_cleaning.ipynb", [
        cell("markdown", """# 01 Data cleaning

Builds the analysis table `data/processed/matchups.csv` from the Sentinel-1,
Sentinel-2, ERA5-Land, static and ISMN inputs, and reports the data-screening
summary. All logic is in `src/data.py`."""),
        cell("code", SETUP),
        cell("code", "from src.data import run_stage1\n\ndf, scr = run_stage1()"),
        cell("markdown", "## Transformation log (rows and nulls after each step)"),
        cell("code", "scr[\"steps\"]"),
        cell("markdown", "## Removed rows and excluded stations"),
        cell("code", "display(scr[\"removed\"])\nscr[\"excluded_stations\"]"),
        cell("markdown", "## Counts per network"),
        cell("code", "scr[\"per_network\"]"),
        cell("markdown", "## Counts per station"),
        cell("code", "scr[\"per_station\"]"),
        cell("markdown", """## Outliers (1.5 IQR rule)

Values outside the fences are flagged only; none are removed."""),
        cell("code", "scr[\"outliers\"]"),
        cell("markdown", """## Normality checks

The Shapiro-Wilk test uses a reproducible random subsample of at most 5,000
values."""),
        cell("code", "scr[\"normality\"]"),
    ])


def nb02() -> None:
    """Notebook 02: clean (base) model."""
    write("02_base_model.ipynb", [
        cell("markdown", """# 02 Clean (base) model

Random forest, XGBoost and a change-detection baseline, evaluated with
station-wise 5-fold cross-validation (test stations are never seen in
training). The tree models are fitted on both feature configurations
declared in `config.yaml`: F1 (primary, all features) and F2
(radar-reliant, without `p7` and `p30`). The change-detection baseline uses
VV only and is reported with per-station scaling (`cd`) and with the
original pooled scaling (`cd_pooled`). USCRN is the main sample. OZNET is the second sample, used as a
transfer test (trained on all USCRN stations) and with within-OZNET
cross-validation, which rests on only 12 stations and therefore has wide
intervals. Intervals are 95% cluster-bootstrap intervals over stations
(5,000 resamples). All logic is in `src/models.py` and `src/stats.py`."""),
        cell("code", SETUP),
        cell("code", "from src.models import run_stage2, summary_table\n\nout = run_stage2()"),
        cell("markdown", """## Metrics with 95% intervals

Pooled metrics use all test rows. `r_station_median` and
`ubrmsd_station_median` are medians over stations of the per-station
(temporal) R and ubRMSD."""),
        cell("code", "summary_table(out[\"stage2_metrics\"])"),
        cell("markdown", "## Metrics per fold (stability across folds)"),
        cell("code", "out[\"stage2_fold_metrics\"].round(3)"),
        cell("markdown", "## Mean feature importance (impurity or gain based)"),
        cell("code", "out[\"stage2_feature_importance\"].round(3)"),
    ])


if __name__ == "__main__":
    nb01()
    nb02()
