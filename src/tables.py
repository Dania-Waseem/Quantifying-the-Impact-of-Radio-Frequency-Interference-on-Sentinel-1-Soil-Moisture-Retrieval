"""Result tables T1 to T11.

Each table is written as a plain CSV file to ``results/aggregate/``
(aggregated values only). Where a table compares variants, the best value
of each row group is marked with an asterisk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, package_versions, path

TODO = "[to be filled]"


def agg(name: str) -> pd.DataFrame:
    """Read an aggregated result table."""
    return pd.read_csv(path("results_aggregate") / f"{name}.csv")


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Write a table to results/aggregate/<name>.csv and return it."""
    df.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    return df


def fmt_ci(est: float, lo: float, hi: float, d: int = 3) -> str:
    """Format an estimate with its interval as 'x [lo, hi]'."""
    if not np.isfinite(est):
        return "n/a"
    return f"{est:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"



# --------------------------------------------------------------------------
# T1 to T11
# --------------------------------------------------------------------------

def t01_gap_table() -> pd.DataFrame:
    """Gap table skeleton: 12 studies to be filled, last row this study."""
    rows = [{"No.": i, "Study": TODO, "Year": TODO, "SAR data": TODO, "RFI handling": TODO,
             "Product evaluated": TODO, "Gap (G1 to G3)": TODO} for i in range(1, 13)]
    rows.append({"No.": 13, "Study": "This study", "Year": "2026", "SAR data": "Sentinel-1 GRD",
                 "RFI handling": "controlled injection; detect-and-fill, detect-and-discard, "
                                 "RFI-aware training",
                 "Product evaluated": "soil moisture (ISMN USCRN, OZNET)",
                 "Gap (G1 to G3)": "G1, G2, G3"})
    return save(pd.DataFrame(rows), "T01_gap_table")


def t02_constructs() -> pd.DataFrame:
    """Construct definitions."""
    rows = [
        ("RFI severity", "Strength of interference relative to the signal", "SIR (dB); lower is stronger", "Injection setting"),
        ("RFI extent", "Affected fraction of the footprint", "ρ (0 to 1)", "Injection setting"),
        ("Contamination probability", "Probability that an acquisition is contaminated", "p_c (0 to 1)", "Injection setting"),
        ("Backscatter error (injected)", "Increase of sigma0 caused by interference", "ε = 10 log10(1 + ρ 10^(-SIR/10)) (dB)", "Derived"),
        ("Residual backscatter error", "Mean over VV and VH of |input - clean| after mitigation", "dB", "Derived"),
        ("Soil moisture error", "Absolute and root mean square difference to ISMN", "m3/m3", "Model output vs ISMN"),
        ("Degradation", "Loss of accuracy caused by RFI", "D = RMSE_M1 - RMSE_M0 (m3/m3)", "Derived"),
        ("Recovery", "Accuracy regained by a mitigation variant", "G = RMSE_M1 - RMSE_variant (m3/m3)", "Derived"),
        ("Recovery ratio", "Share of the degradation recovered", "RR = G / D (-)", "Derived"),
        ("Mitigation (treatment)", "Detect-and-fill (M3), detect-and-discard (M3b), with RFI-aware training (M4)", "variant", "Design"),
        ("Vegetation (moderator)", "Sentinel-2 NDVI terciles", "NDVI (-)", "Sentinel-2"),
        ("Polarisation (moderator)", "Polarisation affected by interference", "VV or VH", "Injection setting"),
    ]
    return save(pd.DataFrame(rows, columns=["Construct", "Definition", "Measure", "Source"]),
                "T02_constructs")


def t03_base_vs_extended() -> pd.DataFrame:
    """Base model against extended model: constructs, logic, research design."""
    rows = [
        ("Constructs", "backscatter, vegetation, precipitation and site features; soil moisture",
         "adds RFI severity, extent, contamination, backscatter error (mediator), mitigation (treatment), NDVI and polarisation (moderators)"),
        ("Input assumption", "radar input is clean", "radar input may be contaminated by RFI"),
        ("Logic", "features -> soil moisture", "RFI -> backscatter error -> soil moisture error; mitigation reduces the backscatter error"),
        ("Models", "random forest, XGBoost, change detection (F1, F2, F3)", "same models under M0, M1, M3, M3b, M4 (M2 reserved)"),
        ("Research design", "station-wise 5-fold cross-validation; transfer to OZNET", "controlled injection on test inputs, 10 seeds per setting, nested variants"),
        ("Outcome measures", "RMSE, MAE, bias, ubRMSD, R2, R", "plus D, G, RR, detection precision, recall and F1"),
        ("Inference", "cluster bootstrap intervals over stations", "plus Wilcoxon tests with Holm correction, mediation, moderation, power, learning curve"),
    ]
    return save(pd.DataFrame(rows, columns=["Aspect", "Base model", "Extended model"]),
                "T03_base_vs_extended")


def t04_sample_profile() -> pd.DataFrame:
    """Sample profile per network (aggregated)."""
    net = agg("screening_per_network")
    lc = agg("stage5_landcover_counts")
    names = {10: "tree", 20: "shrub", 30: "grassland", 40: "cropland", 60: "bare", 90: "wetland"}
    lc_txt = (lc.assign(t=lc["lc"].map(names).fillna(lc["lc"].astype(str)) + " "
                        + lc["stations"].astype(str))
              .groupby("network")["t"].apply(", ".join))
    t = net.assign(**{"land cover (stations)": net["network"].map(lc_txt)})
    t["matchups per station"] = (t["matchups"] / t["stations"]).round(1)
    t = t.rename(columns={"sm_mean": "mean SM (m3/m3)", "sm_sd": "SD SM (m3/m3)",
                          "vh_low": "rows flagged VH < -30 dB"})
    t["mean SM (m3/m3)"] = t["mean SM (m3/m3)"].round(3)
    t["SD SM (m3/m3)"] = t["SD SM (m3/m3)"].round(3)
    t.insert(1, "role", t["network"].map({"USCRN": "main sample",
                                          "OZNET": "second sample (transfer test)"}))
    return save(t, "T04_sample_profile")


def _injection_text() -> str:
    """Description of the injection grid for T5."""
    from .rfi import extent_value, injection_grid

    g = injection_grid(verbose=False)
    if g["source"] != "opensar":
        return "PROVISIONAL grid (OpenSAR statistics absent)"
    sev = "; ".join(f"P{q}: VV {g['contrast_db'][q]['VV']:.2f} dB (SIR {g['sir'][q]['VV']:.1f}), "
                    f"VH {g['contrast_db'][q]['VH']:.2f} dB (SIR {g['sir'][q]['VH']:.1f})"
                    for q in g["severity_q"])
    ext = ", ".join(f"{v:.3f}" for v in g["extent"])
    return (f"OpenSAR ({g['n_patches']} patches): severity levels {sev}; extent levels {ext}; "
            f"p_c {g['p_c']} (assumed); polarisation {g['polarisation']}")


def t05_configuration() -> pd.DataFrame:
    """Analysis configuration."""
    cfg = load_config()
    net = agg("screening_per_network")
    v = package_versions()
    rows = [
        ("Stations", f"{int(net['stations'].sum())} ({', '.join(f'{a} {b}' for a, b in zip(net['network'], net['stations']))})"),
        ("Matchups", f"{int(net['matchups'].sum()):,}"),
        ("Study period", "2019 to 2023"),
        ("Matchup rules", f"one relative orbit per station; ISMN within {cfg['matchup']['ismn_tolerance_minutes']} min; NDVI within {cfg['matchup']['ndvi_tolerance_days']} days; MIN_N = {cfg['matchup']['min_n']}"),
        ("Missing-data treatment", "rows with any missing feature removed (no imputation); quality rules logged"),
        ("Estimation", "random forest (300 trees, min leaf 3), XGBoost, change detection; station-wise 5-fold GroupKFold"),
        ("Feature configurations", "F1 (primary), F2, F3"),
        ("Injection", _injection_text()),
        ("Seeds", f"master seed {cfg['seed']}; {cfg['n_seeds']} injection seeds per setting"),
        ("Bootstrap", f"{cfg['stats']['n_bootstrap']:,} cluster resamples over stations"),
        ("Significance level", f"alpha = {cfg['stats']['alpha']}; Holm correction per family"),
        ("Software", f"Python {v['python']}; pandas {v['pandas']}; numpy {v['numpy']}; scikit-learn {v['scikit-learn']}; xgboost {v['xgboost']}; scipy {v['scipy']}; statsmodels {v['statsmodels']}; matplotlib {v['matplotlib']}"),
    ]
    return save(pd.DataFrame(rows, columns=["Item", "Setting"]), "T05_configuration")


def t06_reliability() -> pd.DataFrame:
    """Reliability and validity analogues (primary evaluation, random forest)."""
    r = agg("stage4_reliability")
    ev = load_config()["stage4"]["primary_evaluation"]
    r = r[(r.evaluation == ev) & r.variant.isin(["M0", "M1", "M3", "M3b", "M4"])]
    t = r.pivot_table(index=["feature_set", "model"], columns="variant",
                      values="rmse_sd_over_seeds").add_prefix("SD of RMSE over seeds, ")
    a = r.groupby(["feature_set", "model"])[["clean_R", "clean_ubRMSD",
                                             "clean_station_R"]].first()
    t = t.join(a).reset_index().round(4)
    t["validation design"] = "station-wise 5-fold cross-validation"
    t["note"] = ("Cronbach's alpha, AVE and HTMT do not apply to sensor data; satellite and "
                 "in situ data come from independent sources.")
    return save(t, "T06_reliability_validity")


def t07_structural() -> pd.DataFrame:
    """Structural model: mediation and moderation paths."""
    ev, mdl = load_config()["stage4"]["primary_evaluation"], load_config()["stage4"]["primary_model"]
    med = agg("stage4_mediation")
    mod = agg("stage4_moderation")
    rows = []
    for _, r in med[(med.evaluation == ev) & (med.model == mdl)].iterrows():
        rows.append({"feature_set": r.feature_set, "analysis": f"mediation ({r.mediation})",
                     "path": r.term, "estimate [95% CI]": fmt_ci(r.estimate, r.ci_low, r.ci_high, 4),
                     "p (bootstrap)": r.p_boot})
    for _, r in mod[(mod.evaluation == ev) & (mod.model == mdl)
                    & mod.term.isin(["E", "E:T2", "E:T3", "VH", "E:VH"])].iterrows():
        rows.append({"feature_set": r.feature_set, "analysis": f"moderation ({r.moderator})",
                     "path": r.term, "estimate [95% CI]": fmt_ci(r.estimate, r.ci_low, r.ci_high, 4),
                     "p (bootstrap)": r.p_boot})
    t = pd.DataFrame(rows).sort_values(["feature_set", "analysis"], kind="stable")
    return save(t, "T07_structural_model")


def t08_hypotheses() -> pd.DataFrame:
    """Hypothesis results with uncorrected and Holm-corrected p-values."""
    tests = agg("stage4_tests")
    verd = agg("stage4_hypotheses")
    ev, mdl = load_config()["stage4"]["primary_evaluation"], load_config()["stage4"]["primary_model"]
    t = tests[(tests.evaluation == ev) & (tests.model == mdl)].copy()
    v = verd[(verd.evaluation == ev) & (verd.model == mdl)].set_index("feature_set")
    t["hypothesis"] = t["test"].str.slice(0, 2)
    t["verdict"] = [v.loc[f, h] if h in v.columns else "" for f, h in zip(t.feature_set,
                                                                         t.hypothesis)]
    cols = ["feature_set", "test", "description", "n", "median_diff", "estimate", "z", "r",
            "p", "p_holm", "verdict"]
    return save(t[[c for c in cols if c in t.columns]].round(5), "T08_hypotheses")


def t09_nested() -> pd.DataFrame:
    """Nested robustness: variants and one-at-a-time ablation of M4."""
    h = agg("stage4_headline")
    ev = load_config()["stage4"]["primary_evaluation"]
    h = h[(h.analysis == "reference") & (h.evaluation == ev)]
    order = ["M0", "M1", "M2", "M3", "M3b", "M4", "M4_nofill"]
    label = {"M0": "M0 clean", "M1": "M1 RFI", "M2": "M2 published (n/a)", "M3": "M3 fill",
             "M3b": "M3b discard", "M4": "M4 fill + RFI-aware",
             "M4_nofill": "M4 without fill (ablation)"}
    rows = []
    for (f, m), g in h.groupby(["feature_set", "model"]):
        rm = g[g.quantity == "rmse"].set_index("variant")
        gg = g[g.quantity == "G"].set_index("variant")
        rr = g[g.quantity == "RR"].set_index("variant")
        best = rm.loc[[v for v in ("M3", "M3b", "M4") if v in rm.index], "estimate"].idxmin()
        for v in order:
            if v == "M2":
                rows.append({"feature_set": f, "model": m, "variant": label[v], "RMSE": "n/a",
                             "G": "n/a", "RR": "n/a"})
                continue
            if v not in rm.index:
                continue
            r = rm.loc[v]
            s = fmt_ci(r.estimate, r.ci_low, r.ci_high, 4) + ("*" if v == best else "")
            rows.append({"feature_set": f, "model": m, "variant": label[v], "RMSE": s,
                         "G": fmt_ci(*gg.loc[v, ["estimate", "ci_low", "ci_high"]], 4)
                         if v in gg.index else "",
                         "RR": fmt_ci(*rr.loc[v, ["estimate", "ci_low", "ci_high"]], 2)
                         if v in rr.index else ""})
    t = pd.DataFrame(rows)
    t["note"] = "* lowest RMSE among the mitigation variants; M4 without RFI-aware training is M3"
    return save(t, "T09_nested_robustness")


def t10_performance() -> pd.DataFrame:
    """Performance by sample and variant (reference scenario)."""
    h = agg("stage4_headline")
    h = h[(h.analysis == "reference") & h.quantity.isin(["rmse", "ubrmsd", "r2"])
          & h.variant.isin(["M0", "M1", "M3", "M3b", "M4"])]
    h = h.assign(v=[fmt_ci(a, b, c, 3) for a, b, c in zip(h.estimate, h.ci_low, h.ci_high)])
    t = h.pivot_table(index=["evaluation", "feature_set", "model", "variant"], columns="quantity",
                      values="v", aggfunc="first").reset_index()
    return save(t, "T10_performance")


def t11_notation() -> pd.DataFrame:
    """Notation table for all symbols."""
    rows = [
        ("σ, σ̃", "clean and contaminated backscatter coefficient (linear)", "-"),
        ("VV, VH", "backscatter in co- and cross-polarisation", "dB"),
        ("SIR", "signal-to-interference ratio", "dB"),
        ("ρ", "RFI extent (affected fraction)", "-"),
        ("p_c", "probability that an acquisition is contaminated", "-"),
        ("ε", "injected backscatter error, 10 log10(σ̃ / σ)", "dB"),
        ("w, τ", "detection window (acquisitions) and z-score threshold", "-"),
        ("k", "number of previous clean acquisitions used for filling", "-"),
        ("λ", "weight of injected copies in RFI-aware training", "-"),
        ("rVV, rVH", "backscatter minus the station's robust median", "dB"),
        ("SM", "soil moisture, 0 to 5 cm", "m3/m3"),
        ("RMSE, MAE, ubRMSD", "root mean square, mean absolute and unbiased RMS difference", "m3/m3"),
        ("R, R2", "Pearson correlation and coefficient of determination", "-"),
        ("D, G, RR", "degradation, recovery, recovery ratio", "m3/m3, m3/m3, -"),
        ("M0 to M4", "variants: clean, contaminated, published (n/a), fill, discard, RFI-aware", "-"),
        ("F1, F2, F3", "feature configurations", "-"),
        ("a, b, c, c'", "mediation paths (treatment to mediator, mediator to outcome, total, direct)", "-"),
        ("E, T2, T3", "injected error and NDVI tercile indicators in the moderation model", "dB, -"),
        ("r", "Wilcoxon effect size Z / sqrt(N)", "-"),
        ("α", "significance level (0.05)", "-"),
    ]
    return save(pd.DataFrame(rows, columns=["Symbol", "Meaning", "Unit"]), "T11_notation")


def make_all() -> dict[str, pd.DataFrame]:
    """Write all tables T1 to T11."""
    return {f.__name__: f() for f in (t01_gap_table, t02_constructs, t03_base_vs_extended,
                                      t04_sample_profile, t05_configuration, t06_reliability,
                                      t07_structural, t08_hypotheses, t09_nested,
                                      t10_performance, t11_notation)}
