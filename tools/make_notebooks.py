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
declared in `config.yaml`: F1 (primary, all features), F2
(radar-reliant, without `p7` and `p30`) and F3 (secondary, relative
backscatter `rVV`, `rVH` with `VHVV`, `angle`, `ndvi`, `doy`). The change-detection baseline uses
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


def nb03() -> None:
    """Notebook 03: RFI statistics from OpenSAR (placeholder)."""
    write("03_rfi_statistics.ipynb", [
        cell("markdown", """# 03 RFI statistics from OpenSAR

The RFI statistics are derived from the OpenSAR data in a separate analysis
(Google Colab), which is not part of this repository. Its output used here is
`data/processed/opensar_rfi_stats.csv` (not tracked): one row per patch with
`patch_id, split, extent, severity_db_vv, severity_db_vh`. `severity_db_*` is
the measured excess power (dB) of the masked RFI pixels against the unmasked
pixels, from SLC power per polarisation. The `sir_db_*` columns of the file
are not used (they hold a capped value for contrasts of 0.1 dB or less);
SIR is recomputed as SIR = -10 log10(10^(C/10) - 1) for C > 0.1 dB.

Notes: the statistics come from 125 of 223 OpenSAR patches (98 had no
measurable RFI in the mask). They are measured on SLC pixels, while the
injection acts on station-level GRD backscatter. Some patches show no power
excess in one polarisation. If the file is absent, the provisional grid is
used with a warning."""),
        cell("code", SETUP),
        cell("code", """from src.rfi import grid_table, injection_grid, load_opensar_stats

stats = load_opensar_stats()
print("OpenSAR statistics:", "not available (provisional grid)" if stats is None
      else f"{len(stats)} patches")
grid = injection_grid()
grid_table(grid).round(4)"""),
        cell("markdown", "## Distribution summary (aggregated)"),
        cell("code", """if stats is not None:
    display(stats[["extent", "severity_db_vv", "severity_db_vh"]].describe().round(3))
    for p in ("vv", "vh"):
        c = stats[f"severity_db_{p}"]
        print(p.upper(), "patches with contrast <= 0.1 dB:", int((c <= 0.1).sum()),
              "(of which <= 0:", int((c <= 0).sum()), ")")
    print("patches with extent > 0.85:", int((stats["extent"] > 0.85).sum()))"""),
    ])


def nb04() -> None:
    """Notebook 04: extended RFI-aware pipeline (Stage 3)."""
    write("04_extended_model.ipynb", [
        cell("markdown", """# 04 Extended model: RFI injection and mitigation

Injects RFI into the Sentinel-1 test inputs and evaluates the variants
M0 (clean), M1 (contaminated), M2 (published suppression, not available),
M3 (detect-and-fill), M3b (detect-and-discard) and M4 (detect-and-fill with
RFI-aware training) for the feature configurations F1, F2 and F3 and the
change-detection baseline. The injection grid comes from the
OpenSAR statistics (severity and extent levels from 125 patches). All logic is in `src/rfi.py` and
`src/experiment.py`. A full run takes about 70 minutes on four workers."""),
        cell("code", SETUP),
        cell("code", "from src.experiment import run_stage3\n\nout = run_stage3()"),
        cell("markdown", """## Degradation and recovery by variant (all settings)

D = RMSE_M1 - RMSE_M0, G = RMSE_M1 - RMSE_variant, RR = G / D, in m3/m3,
averaged over settings and seeds. M2 rows are empty (not available)."""),
        cell("code", """cols = ["evaluation", "feature_set", "model", "variant", "rmse_M0", "rmse_M1",
        "rmse_variant", "D_mean", "G_mean", "RR_of_means", "RR_median"]
out["stage3_summary_overall"][cols].round(4)"""),
        cell("markdown", "## Degradation by severity level and by extent, variant M3 rows"),
        cell("code", """s = out["stage3_summary_by_severity"]
s[s.variant == "M3"].pivot_table(index=["evaluation", "feature_set", "model"],
                                 columns="severity_q", values="D_mean").round(4)"""),
        cell("code", """s = out["stage3_summary_by_extent"]
s[s.variant == "M3"].pivot_table(index=["evaluation", "feature_set", "model"],
                                 columns="extent", values="D_mean").round(4)"""),
        cell("markdown", "## Detection precision, recall and F1 by severity level"),
        cell("code", "out[\"stage3_detection_by_severity\"].round(3)"),
    ])


def nb05() -> None:
    """Notebook 05: statistics and robustness (Stage 4)."""
    write("05_robustness_statistics.ipynb", [
        cell("markdown", """# 05 Statistics and robustness

Recomputes the Stage 4 scenarios (reference, severity and extent sweeps,
polarisation and threshold checks) with per-station errors, then runs the
cluster bootstrap (5,000 resamples over stations), Wilcoxon signed-rank tests
with Holm correction, mediation, moderation, the nested robustness and
alternative checks, the station-count learning curve, the power analysis and
the reliability and validity analogues. All logic is in `src/experiment.py`
and `src/stats.py`. The scenarios are PROVISIONAL (OpenSAR statistics
pending)."""),
        cell("code", SETUP),
        cell("code", """from src.experiment import run_stage4_recompute
from src.stats import run_stage4, SECONDARY_WARNING

run_stage4_recompute()
out = run_stage4()"""),
        cell("markdown", "## Hypothesis verdicts (Holm-corrected, alpha = 0.05)"),
        cell("code", "out[\"stage4_hypotheses\"]"),
        cell("markdown", "## Tests with uncorrected and Holm-corrected p-values"),
        cell("code", """cols = ["evaluation", "feature_set", "model", "test", "n", "median_diff", "estimate",
        "z", "r", "p", "p_holm"]
out["stage4_tests"][[c for c in cols if c in out["stage4_tests"]]].round(5)"""),
        cell("markdown", "## Mediation (primary and secondary)"),
        cell("code", "print(SECONDARY_WARNING)\nout[\"stage4_mediation\"].drop(columns=\"note\").round(5)"),
        cell("markdown", "## Moderation (NDVI terciles and polarisation)"),
        cell("code", "out[\"stage4_moderation\"].round(5)"),
        cell("markdown", "## Learning curve and power"),
        cell("code", "display(out[\"stage4_learning_curve_summary\"].round(4))\nout[\"stage4_power\"].round(3)"),
    ])


def nb06() -> None:
    """Notebook 06: figures, tables and diagrams (Stages 5 and 6)."""
    write("06_plots_tables.ipynb", [
        cell("markdown", """# 06 Figures, tables and diagrams

Draws the result figures 5 to 14 (PNG at 400 dpi and vector PDF in
`figures/`), writes the tables T1 to T11 (CSV in `results/aggregate/`) and
draws the four diagrams (PNG in `figures/diagrams/`). Everything is read from
saved result files; no experiment is rerun. The station map is written to
`figures/private/` because it shows station locations."""),
        cell("code", SETUP),
        cell("code", """from src import diagrams, plots, tables

figs = plots.make_all()
tabs = tables.make_all()
dgms = diagrams.make_all()
print(len(figs), "figure files;", len(tabs), "tables;", len(dgms), "diagrams")"""),
        cell("code", "tabs[\"t08_hypotheses\"]"),
        cell("code", "tabs[\"t09_nested\"]"),
    ])


def nb07() -> None:
    """Notebook 07: exploratory follow-up experiments."""
    write("07_followup.ipynb", [
        cell("markdown", """# 07 Follow-up experiments (exploratory)

Declared in `config.yaml` (section `followup`) after the main analysis and run
separately; no earlier result is changed. (1) Oracle mitigation: fill and
discard with the true injection flags, an upper bound on recovery.
(2) Persistent RFI: one contiguous block of each station's series, or every
acquisition of a share of the stations. (3) Stress test beyond the OpenSAR
P90, up to the largest observed values, with a degradation breakpoint. The
tests form their own Holm family and are reported as exploratory. Logic in
`src/followup.py`; figures 15 to 17 in `src/plots.py`."""),
        cell("code", SETUP),
        cell("code", """from src.followup import run_followup
from src import plots

out = run_followup()
figs = plots.make_followup()"""),
        cell("markdown", "## Exploratory tests (Holm within the follow-up family)"),
        cell("code", """cols = ["evaluation", "feature_set", "model", "test", "n", "median_diff", "z", "r",
        "p", "p_holm", "significant_holm"]
out["followup_tests"][cols].round(5)"""),
        cell("markdown", "## Breakpoints: weakest setting where D reaches 5% or 10% of the clean RMSE"),
        cell("code", "out[\"followup_breakpoints\"].round(4)"),
        cell("markdown", "## Oracle against detector recovery (USCRN, random forest)"),
        cell("code", """s = out["followup_summary"]
s[(s.experiment == "oracle") & (s.evaluation == "USCRN_cv") & (s.model == "rf")].pivot_table(
    index=["feature_set", "label"], columns="variant", values="G_mean").round(5)"""),
    ])


def nb08() -> None:
    """Notebook 08: v2 model comparison with F4 and nested selection."""
    write("08_v2_models.ipynb", [
        cell("markdown", """# 08 v2 models: F4 and nested model selection

Radar-reliant feature sets F2, F3 and F4 (F3 plus temporal context from past
overpasses) with random forest, XGBoost, histogram gradient boosting, ridge
regression and a random forest grid; change detection for comparison.
Hyperparameters and the best model per feature set are chosen by nested
station-grouped cross-validation on the training stations only; every model
is reported on the outer test folds. Clean (M0) and contaminated (M1)
predictions over the full injection grid with the Stage 3 seeds; H1 dose
response test; Stage 7 conditional and drought-alert metrics on the per-row
predictions of the 7 row-level scenarios. Declared in `config.yaml`
(`v2_study`). Logic in `src/stage7_models.py` and `src/stage7.py`."""),
        cell("code", SETUP),
        cell("code", """from src.stage7_models import run_models, aggregate_models, merge_rows
from src.stage7 import run_stage7_metrics
from src.config import path

for fset in ["F4", "F2", "F3"]:
    run_models([fset])
out = aggregate_models()
rows_file = merge_rows("models")
metrics = run_stage7_metrics(rows_file, path("results_aggregate") / "stage4_scenarios.csv",
                             tag="_models")"""),
        cell("markdown", "## Outer RMSE (M0, M1) and damage D, with the inner selection share"),
        cell("code", """o = out["stage7_models_outer"]
o.pivot_table(index=["evaluation", "feature_set", "model", "selected_share"],
              columns="quantity", values="estimate").round(4)"""),
        cell("markdown", "## H1 dose response (v2 Holm family on USCRN)"),
        cell("code", "out[\"stage7_models_h1\"].round(5)"),
    ])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:          # generate selected notebooks only, e.g. "nb08"
        for name in sys.argv[1:]:
            globals()[name]()
        raise SystemExit
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    nb06()
    nb07()
