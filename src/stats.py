"""Statistical inference: cluster bootstrap over stations.

Observations from the same station are not independent (they share soil,
vegetation and sensor). All intervals are therefore computed with a cluster
bootstrap: whole stations are drawn with replacement, and the metric is
recomputed on all rows of the drawn stations.

The pooled metrics are computed from per-station sufficient statistics
(sums of errors, squares and cross-products), so that 5,000 resamples need
only one matrix product instead of 5,000 passes over the data.

Paired tests, mediation, moderation, power analysis and the learning curve
are added in Stage 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, rng
from .metrics import per_station_metrics

# Order of the per-station sufficient statistics.
_SUMS = ["n", "e", "e2", "ae", "y", "y2", "p", "p2", "yp"]


def station_sums(df: pd.DataFrame, y: str = "sm", p: str = "pred",
                 group: str = "station_id") -> tuple[np.ndarray, np.ndarray]:
    """Return the station labels and a (stations x 9) array of sums.

    Columns: count, sum(e), sum(e^2), sum(|e|), sum(y), sum(y^2), sum(p),
    sum(p^2), sum(y p), with e = p - y.
    """
    d = pd.DataFrame({"g": df[group].to_numpy(), "y": df[y].to_numpy(float),
                      "p": df[p].to_numpy(float)})
    d["e"] = d["p"] - d["y"]
    agg = pd.DataFrame({
        "n": d.groupby("g").size(),
        "e": d.groupby("g")["e"].sum(),
        "e2": (d["e"] ** 2).groupby(d["g"]).sum(),
        "ae": d["e"].abs().groupby(d["g"]).sum(),
        "y": d.groupby("g")["y"].sum(),
        "y2": (d["y"] ** 2).groupby(d["g"]).sum(),
        "p": d.groupby("g")["p"].sum(),
        "p2": (d["p"] ** 2).groupby(d["g"]).sum(),
        "yp": (d["y"] * d["p"]).groupby(d["g"]).sum(),
    })[_SUMS]
    return agg.index.to_numpy(), agg.to_numpy()


def metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    """Compute pooled metrics from summed sufficient statistics.

    ``s`` has shape (..., 9). Works for a single vector or for a matrix of
    bootstrap totals (one row per resample).
    """
    n, e, e2, ae, y, y2, p, p2, yp = (s[..., i] for i in range(9))
    with np.errstate(invalid="ignore", divide="ignore"):
        mse = e2 / n
        b = e / n
        ss_tot = y2 - y ** 2 / n
        cov = yp / n - (y / n) * (p / n)
        var_y = y2 / n - (y / n) ** 2
        var_p = p2 / n - (p / n) ** 2
        out = {
            "rmse": np.sqrt(mse),
            "mae": ae / n,
            "bias": b,
            "ubrmsd": np.sqrt(np.maximum(mse - b ** 2, 0.0)),
            "r2": 1.0 - e2 / ss_tot,
            "r": cov / np.sqrt(var_y * var_p),
        }
    return out


def draw_station_indices(n_stations: int, n_boot: int, *seed_keys) -> np.ndarray:
    """Draw ``n_boot`` resamples of station indices with replacement."""
    g = rng("cluster_bootstrap", *seed_keys)
    return g.integers(0, n_stations, size=(n_boot, n_stations))


def cluster_bootstrap(df: pd.DataFrame, *seed_keys, y: str = "sm", p: str = "pred",
                      group: str = "station_id", n_boot: int | None = None,
                      ci: float | None = None) -> pd.DataFrame:
    """Cluster bootstrap over stations for pooled and per-station metrics.

    Returns one row per metric with the point estimate (on the full sample),
    the bootstrap standard error and the percentile interval. Two families
    of metrics are reported:
      * pooled metrics (rmse, mae, bias, ubrmsd, r2, r) over all rows;
      * the median over stations of the per-station R and ubRMSD
        (``r_station_median``, ``ubrmsd_station_median``), which measure
        temporal agreement within stations.

    ``seed_keys`` identify the evaluation (for example the model name and
    the sample) so that each evaluation gets its own reproducible draws.
    """
    cfg = load_config()["stats"]
    n_boot = n_boot or cfg["n_bootstrap"]
    ci = ci or cfg["ci_level"]
    alpha = (1 - ci) / 2

    labels, sums = station_sums(df, y, p, group)
    n_st = len(labels)
    idx = draw_station_indices(n_st, n_boot, *seed_keys)

    # Count how often each station appears in each resample.
    counts = np.zeros((n_boot, n_st))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_st), idx.ravel()), 1.0)
    boot = metrics_from_sums(counts @ sums)
    point = metrics_from_sums(sums.sum(axis=0))

    # Per-station temporal metrics, summarised by the median over stations.
    ps = per_station_metrics(df, y, p, group).set_index(group).loc[labels]
    for m in ("r", "ubrmsd"):
        v = ps[m].to_numpy()
        point[f"{m}_station_median"] = np.nanmedian(v)
        boot[f"{m}_station_median"] = np.nanmedian(v[idx], axis=1)

    rows = []
    for m, est in point.items():
        b = boot[m][np.isfinite(boot[m])]
        rows.append({"metric": m, "estimate": float(est), "se": float(np.std(b, ddof=1)),
                     "ci_low": float(np.quantile(b, alpha)),
                     "ci_high": float(np.quantile(b, 1 - alpha)),
                     "n_stations": n_st, "n": int(sums[:, 0].sum()), "n_boot": n_boot})
    return pd.DataFrame(rows)


# ==========================================================================
# Stage 4: inference on the RFI experiment
# ==========================================================================
#
# Inputs are the locally stored Stage 4 recomputation outputs:
#   data/processed/stage4_station_sums.csv.gz  (per-station error sums)
#   data/processed/stage4_rows.csv.gz          (row-level errors)
# All intervals are cluster-bootstrap intervals over stations with
# ``stats.n_bootstrap`` resamples; seeds are pooled within a station (the
# ten injection repetitions are treated as repeated measurements of the
# same station). Outputs go to results/aggregate (aggregated only).

from itertools import product as _product

from scipy import stats as _sps

SUMS = ["n", "e", "e2", "ae", "y", "y2", "p", "p2", "yp"]
GROUP = ["evaluation", "feature_set", "model"]


def _alpha() -> float:
    return float(load_config()["stats"]["alpha"])


def _pct(b: np.ndarray, ci: float | None = None) -> tuple[float, float]:
    """Percentile interval of finite bootstrap values."""
    ci = ci or load_config()["stats"]["ci_level"]
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan, np.nan
    a = (1 - ci) / 2
    return float(np.quantile(b, a)), float(np.quantile(b, 1 - a))


def _boot_p(b: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'parameter = 0' (percentile method)."""
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return float(max(p, 1 / (len(b) + 1)))


def _counts(n_st: int, *keys) -> np.ndarray:
    """Bootstrap count matrix (resamples x stations) for a cluster bootstrap."""
    n_boot = load_config()["stats"]["n_bootstrap"]
    idx = draw_station_indices(n_st, n_boot, *keys)
    c = np.zeros((n_boot, n_st))
    np.add.at(c, (np.repeat(np.arange(n_boot), n_st), idx.ravel()), 1.0)
    return c


def station_table(sums: pd.DataFrame, scenario: int,
                  stations: set | None = None) -> pd.DataFrame:
    """Per-station error sums for one scenario, pooled over seeds and folds."""
    s = sums[sums["scenario"] == scenario]
    if stations is not None:
        s = s[s["station_id"].isin(stations)]
    return s.groupby(GROUP + ["variant", "station_id"], as_index=False)[SUMS].sum()


def station_rmse(tab: pd.DataFrame) -> pd.DataFrame:
    """Wide table of per-station RMSE (rows: group and station, columns: variant)."""
    t = tab.copy()
    t["rmse"] = np.sqrt(t["e2"] / t["n"])
    return t.pivot_table(index=GROUP + ["station_id"], columns="variant",
                         values="rmse").reset_index()


# --------------------------------------------------------------------------
# Headline metrics, degradation and recovery with intervals
# --------------------------------------------------------------------------

def variant_bootstrap(tab: pd.DataFrame, *keys) -> pd.DataFrame:
    """Bootstrap pooled metrics per variant and D, G, RR for one group.

    ``tab`` holds the per-station sums of one evaluation, feature set and
    model (all variants). Stations are resampled jointly for all variants,
    so D, G and RR are computed on the same resample (paired).
    """
    min_d = load_config()["stats"]["min_degradation"]
    variants = sorted(tab["variant"].unique())
    stations = np.sort(tab["station_id"].unique())
    arr = {v: (tab[tab.variant == v].set_index("station_id").reindex(stations)[SUMS]
               .fillna(0.0).to_numpy()) for v in variants}
    c = _counts(len(stations), "stage4_variants", *keys)
    point = {v: metrics_from_sums(a.sum(axis=0)) for v, a in arr.items()}
    boot = {v: metrics_from_sums(c @ a) for v, a in arr.items()}
    rows = []
    for v in variants:
        for m in ("rmse", "ubrmsd", "bias", "r2", "r"):
            lo, hi = _pct(boot[v][m])
            rows.append({"variant": v, "quantity": m, "estimate": float(point[v][m]),
                         "ci_low": lo, "ci_high": hi})
    if {"M0", "M1"} <= set(variants):
        d_pt = point["M1"]["rmse"] - point["M0"]["rmse"]
        d_b = boot["M1"]["rmse"] - boot["M0"]["rmse"]
        lo, hi = _pct(d_b)
        rows.append({"variant": "M1", "quantity": "D", "estimate": d_pt, "ci_low": lo,
                     "ci_high": hi, "p_boot": _boot_p(d_b)})
        for v in [x for x in ("M3", "M3b", "M4", "M4_nofill") if x in variants]:
            g_pt = point["M1"]["rmse"] - point[v]["rmse"]
            g_b = boot["M1"]["rmse"] - boot[v]["rmse"]
            rr_pt = g_pt / d_pt if d_pt >= min_d else np.nan
            rr_b = np.where(d_b >= min_d, g_b / np.where(d_b == 0, np.nan, d_b), np.nan)
            for q, pt, b in (("G", g_pt, g_b), ("RR", rr_pt, rr_b)):
                lo, hi = _pct(b)
                rows.append({"variant": v, "quantity": q, "estimate": pt, "ci_low": lo,
                             "ci_high": hi, "p_boot": _boot_p(b) if q == "G" else np.nan})
        if {"M3b", "M0_kept"} <= set(variants):
            gap = point["M3b"]["rmse"] - point["M0_kept"]["rmse"]
            lo, hi = _pct(boot["M3b"]["rmse"] - boot["M0_kept"]["rmse"])
            rows.append({"variant": "M3b", "quantity": "gap_to_M0_kept", "estimate": gap,
                         "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out["n_stations"], out["n_boot"] = len(stations), c.shape[0]
    return out


def headline_bootstrap(sums: pd.DataFrame, scenario: int, label: str,
                       stations: set | None = None) -> pd.DataFrame:
    """Run :func:`variant_bootstrap` for every evaluation, feature set and model."""
    tab = station_table(sums, scenario, stations)
    parts = []
    for key, g in tab.groupby(GROUP):
        b = variant_bootstrap(g, label, *key)
        for col, val in zip(GROUP, key):
            b.insert(0, col, val)
        parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "analysis", label)
    return out


# --------------------------------------------------------------------------
# Paired tests
# --------------------------------------------------------------------------

def wilcoxon_one_sided(x: np.ndarray, y: np.ndarray) -> dict:
    """One-sided Wilcoxon signed-rank test that x is larger than y.

    Uses the normal approximation and reports the effect size
    r = Z / sqrt(N), with N the number of non-zero paired differences.
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[np.isfinite(d)]
    nz = int((d != 0).sum())
    if nz < 5:
        return {"n": len(d), "median_diff": np.nan, "W": np.nan, "z": np.nan,
                "p": np.nan, "r": np.nan}
    res = _sps.wilcoxon(d, alternative="greater", method="approx")
    z = float(_sps.norm.isf(res.pvalue))
    return {"n": len(d), "median_diff": float(np.median(d)), "W": float(res.statistic),
            "z": z, "p": float(res.pvalue), "r": z / np.sqrt(nz)}


def station_slopes(sums: pd.DataFrame, scen: pd.DataFrame, role: str, xcol: str,
                   xfun) -> pd.DataFrame:
    """Per-station least-squares slope of RMSE_M1 against a severity measure.

    ``role`` selects the sweep scenarios, ``xcol`` the varied setting and
    ``xfun`` maps it to the severity axis (for the severity level: its mean
    excess power in dB, so that a larger value means stronger interference).
    """
    sc = scen[scen["roles"].str.contains(role)]
    parts = []
    for r in sc.itertuples():
        t = station_rmse(station_table(sums, r.scenario))[GROUP + ["station_id", "M1", "M0"]]
        t["x"] = xfun(getattr(r, xcol))
        parts.append(t)
    t = pd.concat(parts)

    def slope(g):
        x, y = g["x"].to_numpy(), g["M1"].to_numpy()
        return np.polyfit(x, y, 1)[0] if np.ptp(x) > 0 else np.nan

    return t.groupby(GROUP + ["station_id"]).apply(slope).rename("slope").reset_index()


# --------------------------------------------------------------------------
# Regression with a cluster bootstrap (mediation and moderation)
# --------------------------------------------------------------------------

def cluster_ols(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *keys,
                names: list[str]) -> pd.DataFrame:
    """OLS coefficients with a cluster bootstrap over stations.

    The per-station cross-products X'X and X'y are computed once, so each
    resample only sums them with the bootstrap counts and solves a small
    linear system. Returns the estimate, percentile interval and
    two-sided bootstrap p-value of every coefficient.
    """
    codes, labels = pd.factorize(groups)
    k = X.shape[1]
    xtx = np.zeros((len(labels), k, k))
    xty = np.zeros((len(labels), k))
    np.add.at(xtx, codes, X[:, :, None] * X[:, None, :])
    np.add.at(xty, codes, X * y[:, None])
    beta = np.linalg.solve(xtx.sum(0), xty.sum(0))
    c = _counts(len(labels), "cluster_ols", *keys)
    A = np.einsum("bs,sij->bij", c, xtx)
    v = c @ xty
    ok = np.linalg.matrix_rank(A) == k
    bb = np.full((c.shape[0], k), np.nan)
    bb[ok] = np.linalg.solve(A[ok], v[ok][..., None])[..., 0]
    rows = []
    for j, nm in enumerate(names):
        lo, hi = _pct(bb[:, j])
        rows.append({"term": nm, "estimate": float(beta[j]), "ci_low": lo, "ci_high": hi,
                     "p_boot": _boot_p(bb[:, j])})
    return pd.DataFrame(rows), bb


def _abs_err(rows: pd.DataFrame, fset: str, model: str, variant: str) -> np.ndarray:
    return np.abs(rows[f"{fset}|{model}|{variant}"].to_numpy(float) - rows["sm"].to_numpy())


def mediation_primary(rows: pd.DataFrame, fset: str, model: str, *keys) -> pd.DataFrame:
    """Primary mediation: mitigation -> residual backscatter error -> SM error.

    Each row of the reference scenario enters twice: as M1 (X = 0) and as M3
    (X = 1). M is the residual backscatter error, the mean over VV and VH of
    |observed - clean| (dB) of the input the model receives; Y is the
    absolute soil moisture error. Path a: M = a0 + a X. Path b and direct
    effect: Y = b0 + c' X + b M. Indirect effect a b, total effect c from
    Y = c0 + c X. All with a cluster bootstrap over stations (the same
    resamples for every path).
    """
    m1 = rows[["bs_M1_VV", "bs_M1_VH"]].mean(axis=1).to_numpy()
    m3 = rows[["bs_M3_VV", "bs_M3_VH"]].mean(axis=1).to_numpy()
    y1, y3 = _abs_err(rows, fset, model, "M1"), _abs_err(rows, fset, model, "M3")
    g = np.r_[rows["station_id"].to_numpy(), rows["station_id"].to_numpy()]
    X = np.r_[np.zeros(len(rows)), np.ones(len(rows))]
    M, Y = np.r_[m1, m3], np.r_[y1, y3]
    one = np.ones_like(X)
    a_tab, a_b = cluster_ols(np.c_[one, X], M, g, "med1", *keys, names=["a0", "a"])
    b_tab, b_b = cluster_ols(np.c_[one, X, M], Y, g, "med1", *keys, names=["b0", "c_prime", "b"])
    c_tab, c_b = cluster_ols(np.c_[one, X], Y, g, "med1", *keys, names=["c0", "c"])
    ind_b = a_b[:, 1] * b_b[:, 2]
    ind = a_tab.estimate[1] * b_tab.estimate[2]
    lo, hi = _pct(ind_b)
    tab = pd.concat([a_tab.iloc[[1]], b_tab.iloc[[1, 2]], c_tab.iloc[[1]]], ignore_index=True)
    tab = pd.concat([tab, pd.DataFrame([{
        "term": "indirect_ab", "estimate": ind, "ci_low": lo, "ci_high": hi,
        "p_boot": _boot_p(ind_b)}, {
        "term": "proportion_mediated", "estimate": ind / c_tab.estimate[1]
        if c_tab.estimate[1] != 0 else np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "p_boot": np.nan}])], ignore_index=True)
    tab["mediation"] = "primary"
    tab["note"] = ""
    return tab


SECONDARY_WARNING = ("Secondary mediation: the mediator eps is defined analytically from the "
                     "injection (eps = 10 log10(1 + rho 10^(-SIR/10))), so the path from "
                     "severity to eps is nearly deterministic. The meaningful quantity is "
                     "path b (the effect of eps on the soil moisture error), not the discovery "
                     "of a mediator.")


def mediation_secondary(rows: pd.DataFrame, scen: pd.DataFrame, fset: str, model: str,
                        *keys) -> pd.DataFrame:
    """Secondary mediation: severity -> injected error eps -> SM error (M1).

    Uses the row-level data of the severity sweep. X = severity of the level
    (mean OpenSAR excess power over VV and VH, dB; larger is stronger),
    M = mean of eps over VV and VH (0 if not contaminated), Y = absolute
    soil moisture error of M1. See ``SECONDARY_WARNING``.
    """
    from .rfi import injection_grid, severity_level_db

    grid = injection_grid(verbose=False)
    sw = scen[scen["roles"].str.contains("severity_sweep")]
    r = rows[rows["scenario"].isin(sw["scenario"])].merge(sw[["scenario", "severity_q"]],
                                                          on="scenario")
    X = r["severity_q"].map(lambda q: severity_level_db(grid, q)).to_numpy(float)
    M = r[["eps_VV", "eps_VH"]].mean(axis=1).to_numpy()
    Y = _abs_err(r, fset, model, "M1")
    g = r["station_id"].to_numpy()
    one = np.ones_like(X)
    a_tab, a_b = cluster_ols(np.c_[one, X], M, g, "med2", *keys, names=["a0", "a"])
    b_tab, b_b = cluster_ols(np.c_[one, X, M], Y, g, "med2", *keys, names=["b0", "c_prime", "b"])
    c_tab, _ = cluster_ols(np.c_[one, X], Y, g, "med2", *keys, names=["c0", "c"])
    ind_b = a_b[:, 1] * b_b[:, 2]
    lo, hi = _pct(ind_b)
    tab = pd.concat([a_tab.iloc[[1]], b_tab.iloc[[1, 2]], c_tab.iloc[[1]],
                     pd.DataFrame([{"term": "indirect_ab",
                                    "estimate": a_tab.estimate[1] * b_tab.estimate[2],
                                    "ci_low": lo, "ci_high": hi, "p_boot": _boot_p(ind_b)}])],
                    ignore_index=True)
    tab["mediation"] = "secondary"
    tab["note"] = SECONDARY_WARNING
    return tab


def moderation(rows: pd.DataFrame, scen: pd.DataFrame, fset: str, model: str,
               *keys) -> pd.DataFrame:
    """Moderation of the degradation by NDVI terciles and by polarisation.

    Outcome: row-level degradation dAE = |e_M1| - |e_M0| (m3/m3).
    NDVI (reference scenario): dAE ~ E + T2 + T3 + E:T2 + E:T3, with E the
    mean injected error over VV and VH (dB) and T2, T3 indicators of the
    middle and upper NDVI tercile (terciles of the evaluation's rows).
    Polarisation (VV-only and VH-only scenarios stacked): dAE ~ E + VH +
    E:VH, with E the injected error of the hit polarisation.
    """
    ref = scen[scen["roles"].str.contains("reference")]["scenario"].iloc[0]
    r = rows[rows["scenario"] == ref]
    dae = _abs_err(r, fset, model, "M1") - _abs_err(r, fset, model, "M0")
    E = r[["eps_VV", "eps_VH"]].mean(axis=1).to_numpy()
    q = np.quantile(r["ndvi"], [1 / 3, 2 / 3])
    t2 = ((r["ndvi"] > q[0]) & (r["ndvi"] <= q[1])).to_numpy(float)
    t3 = (r["ndvi"] > q[1]).to_numpy(float)
    one = np.ones_like(E)
    nd, _ = cluster_ols(np.c_[one, E, t2, t3, E * t2, E * t3], dae, r["station_id"].to_numpy(),
                        "mod_ndvi", *keys,
                        names=["intercept", "E", "T2", "T3", "E:T2", "E:T3"])
    nd["moderator"] = "ndvi_tercile"
    nd["note"] = f"NDVI tercile limits {q[0]:.3f}, {q[1]:.3f}"

    pv = scen[scen["roles"].isin(["pol_VV", "pol_VH"])]
    rp = rows[rows["scenario"].isin(pv["scenario"])].merge(pv[["scenario", "polarisation"]],
                                                          on="scenario")
    vh = (rp["polarisation"] == "VH").to_numpy(float)
    Ep = np.where(vh == 1, rp["eps_VH"], rp["eps_VV"])
    dp = _abs_err(rp, fset, model, "M1") - _abs_err(rp, fset, model, "M0")
    one = np.ones_like(Ep)
    po, _ = cluster_ols(np.c_[one, Ep, vh, Ep * vh], dp, rp["station_id"].to_numpy(),
                        "mod_pol", *keys, names=["intercept", "E", "VH", "E:VH"])
    po["moderator"] = "polarisation"
    po["note"] = "VV-only and VH-only scenarios stacked; VH = 1 for VH-only"
    return pd.concat([nd, po], ignore_index=True)


# --------------------------------------------------------------------------
# Learning curve and power
# --------------------------------------------------------------------------

def learning_curve(tab: pd.DataFrame, *keys) -> pd.DataFrame:
    """Station-count learning curve for one group (reference scenario).

    For each size n, ``learning_curve_repeats`` random subsets of n
    stations are drawn without replacement. For each subset: D, G (M3) and
    RR, and the width of the 95% cluster-bootstrap interval of G within the
    subset. Sizes larger than the number of stations are replaced by all
    stations (and marked).
    """
    cfg = load_config()["stage4"]
    stations = np.sort(tab["station_id"].unique())
    arr = {v: (tab[tab.variant == v].set_index("station_id").reindex(stations)[SUMS]
               .fillna(0.0).to_numpy()) for v in ("M0", "M1", "M3")}
    rows = []
    for size in cfg["learning_curve_sizes"]:
        n = len(stations) if size == "all" or size > len(stations) else int(size)
        g = rng("learning_curve", size, *keys)
        reps = 1 if n == len(stations) else cfg["learning_curve_repeats"]
        for rep in range(reps):
            pick = np.sort(g.choice(len(stations), n, replace=False))
            sub = {v: a[pick] for v, a in arr.items()}
            r = {v: np.sqrt(a[:, 2].sum() / a[:, 0].sum()) for v, a in sub.items()}
            d, gg = r["M1"] - r["M0"], r["M1"] - r["M3"]
            c = _counts(n, "learning_curve_boot", size, rep, *keys)
            rb = {v: np.sqrt((c @ a[:, 2]) / (c @ a[:, 0])) for v, a in sub.items()}
            lo, hi = _pct(rb["M1"] - rb["M3"])
            rows.append({"size": size, "n_stations": n, "repeat": rep, "D": d, "G": gg,
                         "RR": gg / d if d >= load_config()["stats"]["min_degradation"]
                         else np.nan, "G_ci_width": hi - lo,
                         "capped": size != "all" and int(size) > len(stations)})
    return pd.DataFrame(rows)


def power_analysis(diff: np.ndarray, n_station_days: int) -> dict:
    """Power of a one-sided paired t-test with the observed effect size.

    The paired t-test is used as an approximation to the Wilcoxon test
    (asymptotic relative efficiency about 0.95). ``diff`` holds the
    station-level differences in the hypothesised direction.
    """
    from statsmodels.stats.power import TTestPower

    d = diff[np.isfinite(diff)]
    dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    tp = TTestPower()
    alpha, target = _alpha(), load_config()["stage4"]["power_target"]
    if not np.isfinite(dz) or dz <= 0:
        return {"dz": dz, "n_stations": len(d), "station_days": n_station_days,
                "power": np.nan, "n_required_80": np.nan}
    power = float(tp.power(effect_size=dz, nobs=len(d), alpha=alpha, alternative="larger"))
    try:
        need = float(tp.solve_power(effect_size=dz, power=target, alpha=alpha,
                                    alternative="larger"))
    except Exception:
        need = np.nan
    return {"dz": dz, "n_stations": len(d), "station_days": n_station_days, "power": power,
            "n_required_80": need}


# --------------------------------------------------------------------------
# Stage 4 runner
# --------------------------------------------------------------------------

def _subset_stations(name: str, ev: str) -> set | None:
    """Stations of an evaluation subset (None means all stations)."""
    from .data import load_matchups

    m = load_matchups()
    if name == "grassland":
        return set(m.loc[m["lc"] == 30, "station_id"])
    if name == "min_n_75":
        c = m.groupby("station_id").size()
        return set(c[c >= load_config()["robustness"]["min_n_check"]].index)
    return None


def _rows_to_station_sums(r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-station error sums from row-level predictions (for row subsets)."""
    parts = []
    for c in cols:
        fset, model, variant = c.split("|")
        ok = r[c].notna()
        e = (r.loc[ok, c] - r.loc[ok, "sm"]).astype(float)
        y, p = r.loc[ok, "sm"].astype(float), r.loc[ok, c].astype(float)
        d = pd.DataFrame({"evaluation": r.loc[ok, "evaluation"], "station_id": r.loc[ok,
                          "station_id"], "n": 1.0, "e": e, "e2": e ** 2, "ae": e.abs(),
                          "y": y, "y2": y ** 2, "p": p, "p2": p ** 2, "yp": y * p})
        d = d.groupby(["evaluation", "station_id"], as_index=False)[SUMS].sum()
        d["feature_set"], d["model"], d["variant"] = fset, model, variant
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["scenario"] = -1
    return out


def run_stage4() -> dict[str, pd.DataFrame]:
    """Run all Stage 4 analyses and save the aggregated tables.

    Requires the Stage 3 outputs and the Stage 4 recomputation
    (:func:`src.experiment.run_stage4_recompute`).
    """
    from statsmodels.stats.multitest import multipletests

    from .config import path, write_version_log

    cfg = load_config()
    s4 = cfg["stage4"]
    alpha = _alpha()
    sums = pd.read_csv(path("processed") / "stage4_station_sums.csv.gz")
    rows = pd.read_csv(path("processed") / "stage4_rows.csv.gz")
    scen = pd.read_csv(path("results_aggregate") / "stage4_scenarios.csv")
    ref = int(scen[scen["roles"].str.contains("reference")]["scenario"].iloc[0])
    out: dict[str, pd.DataFrame] = {}

    # 1. Headline metrics and recovery (reference scenario), with intervals.
    out["stage4_headline"] = headline_bootstrap(sums, ref, "reference")

    # 2. Alternative checks.
    alt = []
    for r in scen[scen["roles"].str.contains("tau_check")].itertuples():
        b = headline_bootstrap(sums, r.scenario, f"tau={r.tau}")
        alt.append(b)
    for p in s4["polarisation_check"]:
        sc = int(scen[scen["roles"] == f"pol_{p}"]["scenario"].iloc[0])
        alt.append(headline_bootstrap(sums, sc, f"polarisation={p}"))
    for name in ("grassland", "min_n_75"):
        alt.append(headline_bootstrap(sums, ref, f"subset={name}", _subset_stations(name, "")))
    rr = rows[rows["scenario"] == ref]
    pcols = [c for c in rr.columns if c.count("|") == 2]
    if s4["subsets"] and "exclude_vh_low" in s4["subsets"]:
        ss = _rows_to_station_sums(rr[~rr["vh_low"].astype(bool)], pcols)
        alt.append(headline_bootstrap(ss, -1, "subset=exclude_vh_low"))
    if "ndvi_terciles" in s4["subsets"]:
        for ev, g in rr.groupby("evaluation"):
            q = np.quantile(g["ndvi"], [1 / 3, 2 / 3])
            lab = np.where(g["ndvi"] <= q[0], "low", np.where(g["ndvi"] <= q[1], "mid", "high"))
            for t in ("low", "mid", "high"):
                ss = _rows_to_station_sums(g[lab == t], pcols)
                alt.append(headline_bootstrap(ss, -1, f"subset=ndvi_{t}"))
    out["stage4_alternative_checks"] = pd.concat(alt, ignore_index=True)

    # 3. Hypothesis tests per evaluation, feature set and model.
    sr = station_rmse(station_table(sums, ref))
    from .rfi import injection_grid, severity_level_db

    grid = injection_grid(verbose=False)
    sl_sir = station_slopes(sums, scen, "severity_sweep", "severity_q",
                            lambda q: severity_level_db(grid, q))
    sl_ext = station_slopes(sums, scen, "extent_sweep", "extent", float)
    tests, meds, mods, powers = [], [], [], []
    n_days = rows[(rows["scenario"] == ref) & (rows["seed"] == 0)].groupby("evaluation").size()
    for key, g in sr.groupby(GROUP):
        ev, fset, model = key
        fam = []
        sel = (sl_sir[GROUP] == key).all(axis=1)
        fam.append(("H1", "RMSE_M1 slope vs severity (excess power, dB) > 0",
                    wilcoxon_one_sided(sl_sir.loc[sel, "slope"].to_numpy(), np.zeros(int(sel.sum())))))
        sel = (sl_ext[GROUP] == key).all(axis=1)
        fam.append(("H2", "RMSE_M1 slope vs extent > 0",
                    wilcoxon_one_sided(sl_ext.loc[sel, "slope"].to_numpy(), np.zeros(int(sel.sum())))))
        rk = rows[rows["evaluation"] == ev]
        med = mediation_primary(rk[rk["scenario"] == ref], fset, model, *key)
        med2 = mediation_secondary(rk, scen, fset, model, *key)
        for t in (med, med2):
            for col, val in zip(GROUP, key):
                t.insert(0, col, val)
        meds += [med, med2]
        ind = med.set_index("term").loc["indirect_ab"]
        fam.append(("H3", "indirect effect of mitigation via backscatter error != 0",
                    {"n": int(g.shape[0]), "median_diff": np.nan, "W": np.nan, "z": np.nan,
                     "p": ind.p_boot, "r": np.nan, "estimate": ind.estimate}))
        fam.append(("H4", "RMSE_M1 > RMSE_M3 (mitigation recovers accuracy)",
                    wilcoxon_one_sided(g["M1"].to_numpy(), g["M3"].to_numpy())))
        if "M4" in g and g["M4"].notna().any():
            fam.append(("H5", "RMSE_M3 > RMSE_M4 (RFI-aware training adds recovery)",
                        wilcoxon_one_sided(g["M3"].to_numpy(), g["M4"].to_numpy())))
        mo = moderation(rk, scen, fset, model, *key)
        for col, val in zip(GROUP, key):
            mo.insert(0, col, val)
        mods.append(mo)
        for term, lab in (("E:T3", "H6_ndvi"), ("E:VH", "H6_pol")):
            row = mo.set_index("term").loc[term]
            fam.append((lab, f"interaction {term} != 0",
                        {"n": int(g.shape[0]), "median_diff": np.nan, "W": np.nan,
                         "z": np.nan, "p": row.p_boot, "r": np.nan,
                         "estimate": row.estimate}))
        p = np.array([f[2]["p"] for f in fam], dtype=float)
        ok = np.isfinite(p)
        p_holm = np.full(len(p), np.nan)
        if ok.any():
            p_holm[ok] = multipletests(p[ok], alpha=alpha, method="holm")[1]
        for (h, desc, res), ph in zip(fam, p_holm):
            tests.append({"evaluation": ev, "feature_set": fset, "model": model,
                          "test": h, "description": desc, **res, "p_holm": ph})
        for lab, diff in (("M1_vs_M0", g["M1"] - g["M0"]), ("M1_vs_M3", g["M1"] - g["M3"]),
                          ("M3_vs_M4", g["M3"] - g["M4"] if "M4" in g else g["M3"] * np.nan)):
            powers.append({"evaluation": ev, "feature_set": fset, "model": model,
                           "comparison": lab, **power_analysis(diff.to_numpy(float),
                                                               int(n_days.get(ev, 0)))})
    tests = pd.DataFrame(tests)
    out["stage4_tests"] = tests
    out["stage4_mediation"] = pd.concat(meds, ignore_index=True)
    out["stage4_moderation"] = pd.concat(mods, ignore_index=True)
    out["stage4_power"] = pd.DataFrame(powers)
    out["stage4_hypotheses"] = hypothesis_verdicts(tests, out["stage4_mediation"], alpha)

    # 4. Learning curve (reference scenario, main evaluation).
    lc = []
    tab = station_table(sums, ref)
    for key, g in tab[tab["evaluation"] == s4["primary_evaluation"]].groupby(GROUP):
        if key[2] not in ("rf", "xgb", "cd"):
            continue
        t = learning_curve(g, *key)
        for col, val in zip(GROUP, key):
            t.insert(0, col, val)
        lc.append(t)
    lc = pd.concat(lc, ignore_index=True)
    out["stage4_learning_curve"] = lc
    out["stage4_learning_curve_summary"] = (
        lc.groupby(GROUP + ["size", "n_stations"], sort=False)
        .agg(G_ci_width_mean=("G_ci_width", "mean"), RR_mean=("RR", "mean"),
             RR_sd=("RR", "std"), D_mean=("D", "mean"), G_mean=("G", "mean"),
             repeats=("repeat", "size")).reset_index())

    # 5. Reliability and validity analogues.
    out["stage4_reliability"] = reliability_table(ref, scen)

    for name, t in out.items():
        t.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "stage4_tests.csv")
    return out


def hypothesis_verdicts(tests: pd.DataFrame, med: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Supported or not supported for H1 to H6 per evaluation, feature set, model.

    Rules (Holm-corrected p below alpha in every case):
      H1, H2, H4, H5: the one-sided test is significant (direction is built
      into the one-sided alternative);
      H3: the indirect effect is significant, path a is negative (mitigation
      reduces the backscatter error) and path b is positive (a larger
      backscatter error raises the soil moisture error);
      H6: at least one of the two interaction terms is significant.
    """
    rows = []
    for key, g in tests.groupby(GROUP):
        t = g.set_index("test")
        m = med[(med[GROUP] == key).all(axis=1) & (med["mediation"] == "primary")]
        m = m.set_index("term")
        verdict = {}
        for h in ("H1", "H2", "H4", "H5"):
            if h in t.index and np.isfinite(t.loc[h, "p_holm"]):
                verdict[h] = "supported" if t.loc[h, "p_holm"] < alpha else "not supported"
            else:
                verdict[h] = "not applicable"
        sig = t.loc["H3", "p_holm"] < alpha
        verdict["H3"] = ("supported" if sig and m.loc["a", "estimate"] < 0
                         and m.loc["b", "estimate"] > 0 else "not supported")
        h6 = [t.loc[x, "p_holm"] for x in ("H6_ndvi", "H6_pol") if x in t.index]
        verdict["H6"] = "supported" if any(p < alpha for p in h6) else "not supported"
        rows.append({**dict(zip(GROUP, key)), **verdict})
    return pd.DataFrame(rows)


def reliability_table(ref: int, scen: pd.DataFrame) -> pd.DataFrame:
    """Reliability and validity analogues.

    * stability: standard deviation over the injection seeds of the pooled
      RMSE per variant (reference scenario, from the Stage 3 pooled runs);
    * agreement with ISMN: clean-model R, ubRMSD and per-station R from
      Stage 2 (convergent validity analogue: satellite retrieval against
      independent in situ measurements);
    * station-wise cross-validation is the validation design (test stations
      are never used for training).
    Cronbach's alpha, AVE and HTMT do not apply to sensor data.
    """
    from .config import path

    r = scen[scen["scenario"] == ref].iloc[0]
    p = pd.read_csv(path("results") / "experiment_pooled.csv.gz")
    p = p[(p.severity_q == r.severity_q) & (p.extent == r.extent) & (p.p_c == r.p_c)
          & (p.polarisation == r.polarisation)]
    stab = (p.groupby(["evaluation", "feature_set", "model", "variant"])["rmse"]
            .agg(rmse_mean="mean", rmse_sd_over_seeds="std", seeds="size").reset_index())
    m2 = pd.read_csv(path("results_aggregate") / "stage2_metrics.csv")
    m2 = m2[(m2["scope"] == "all") & m2["metric"].isin(["r", "ubrmsd", "r_station_median"])]
    agree = m2.pivot_table(index=["setting", "feature_set", "model"], columns="metric",
                           values="estimate").reset_index().rename(
        columns={"setting": "evaluation", "r": "clean_R", "ubrmsd": "clean_ubRMSD",
                 "r_station_median": "clean_station_R"})
    return stab.merge(agree, on=["evaluation", "feature_set", "model"], how="left")
