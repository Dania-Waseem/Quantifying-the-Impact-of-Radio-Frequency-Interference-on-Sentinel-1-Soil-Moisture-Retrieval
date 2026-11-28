"""Stage 7 metrics on per-row predictions: conditional view and drought events.

Works on any per-row prediction file with this layout (the layout of
``data/processed/stage4_rows.csv.gz`` and of the per-row files written by
later stages):

  * identifier columns ``evaluation, station_id, scenario, seed``;
  * the observed soil moisture ``sm`` and the true contamination flags
    ``inj_VV, inj_VH``;
  * one prediction column per feature set, model and variant, named
    ``<feature set>|<model>|<variant>`` (for example ``F1|rf|M1``). A variant
    that drops rows (M3b) has missing values on the dropped rows.

and a scenario table with a ``scenario`` column (for example
``results/aggregate/stage4_scenarios.csv``).

Outputs (aggregated, one row per number with a 95% cluster-bootstrap
interval over stations):

  * conditional view: RMSE on contaminated overpasses only, the degradation
    D and the recovery G of each variant on those overpasses, and, for
    comparison, D over all overpasses and the hit probability. For models
    whose unaffected rows do not change, the mean squared error over all
    overpasses equals p times the mean squared error on hit overpasses, so
    D on hit overpasses is roughly D over all overpasses divided by p;
  * drought events: an alert is raised when soil moisture is below the
    station's own 30th percentile of observed soil moisture (the same
    threshold for observed and predicted values). F1 score, miss rate,
    false alarm rate and the rate at which a variant's alert differs from
    the M0 alert (flip rate), on all overpasses and on contaminated ones.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import load_config, path
from .stats import _counts, _pct

ID_COLS = ["evaluation", "station_id", "scenario", "seed", "sm", "inj_VV", "inj_VH"]
PRED_RE = re.compile(r"^([^|]+)\|([^|]+)\|([^|]+)$")


def prediction_columns(columns) -> pd.DataFrame:
    """Parse ``feature set|model|variant`` column names into a table."""
    rows = [dict(zip(("feature_set", "model", "variant"), m.groups()), column=c)
            for c in columns if (m := PRED_RE.match(c))]
    return pd.DataFrame(rows)


def load_rows(file) -> pd.DataFrame:
    """Read a per-row prediction file (predictions as float32 to save memory)."""
    cols = pd.read_csv(file, nrows=0).columns
    pc = prediction_columns(cols)["column"].tolist()
    keep = [c for c in ID_COLS if c in cols] + pc
    return pd.read_csv(file, usecols=keep, dtype={c: "float32" for c in pc})


def _station_sums(codes: np.ndarray, n_st: int, values: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-station sums of several quantities into an (n_st, k) array."""
    return np.column_stack([np.bincount(codes, weights=v, minlength=n_st)
                            for v in values.values()])


def _boot(sums: np.ndarray, fn, keys) -> tuple[dict, dict]:
    """Point estimates and bootstrap draws of ``fn`` applied to summed columns.

    ``sums`` is (stations, k); ``fn`` maps an array (..., k) of totals to a
    dict of metrics. The same resamples (keys) are used for every quantity
    of a group, so differences between variants are paired.
    """
    point = fn(sums.sum(axis=0))
    c = _counts(sums.shape[0], "stage7", *keys)
    boot = fn(c @ sums)
    return point, boot


def _records(point: dict, boot: dict, base: dict, n_st: int, notes: dict | None = None):
    out = []
    for q, est in point.items():
        b = np.asarray(boot[q], float)
        lo, hi = _pct(b[np.isfinite(b)]) if np.isfinite(b).any() else (np.nan, np.nan)
        out.append({**base, "quantity": q, "estimate": float(est), "ci_low": lo, "ci_high": hi,
                    "n_stations": n_st, "note": (notes or {}).get(q, "")})
    return out


def _rmse(s2, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(s2 / n)


# --------------------------------------------------------------------------
# Conditional view
# --------------------------------------------------------------------------

def conditional_metrics(rows: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """RMSE, D and G on contaminated overpasses, with bootstrap intervals."""
    pcols = prediction_columns(rows.columns)
    out = []
    for (ev, sc), g in rows.groupby(["evaluation", "scenario"], sort=True):
        codes, labels = pd.factorize(g["station_id"])
        n_st = len(labels)
        y = g["sm"].to_numpy(float)
        hit = (g["inj_VV"] | g["inj_VH"]).to_numpy(bool)
        for (fset, model), pc in pcols.groupby(["feature_set", "model"], sort=True):
            col = dict(zip(pc["variant"], pc["column"]))
            if not {"M0", "M1"} <= set(col):
                continue
            pred = {v: g[c].to_numpy(float) for v, c in col.items()}
            err2 = {v: (p - y) ** 2 for v, p in pred.items()}
            full_vars = [v for v in col if v not in ("M3b", "M0_kept")]
            vals = {"n_all": np.ones_like(y), "n_hit": hit.astype(float)}
            for v in full_vars:
                vals[f"all_{v}"] = err2[v]
                vals[f"hit_{v}"] = np.where(hit, err2[v], 0.0)
            if "M3b" in col:
                kept = np.isfinite(pred["M3b"])
                hk = hit & kept
                vals["n_hit_kept"] = hk.astype(float)
                vals["hit_M3b"] = np.where(hk, err2["M3b"], 0.0)
                vals["hit_M0_on_kept"] = np.where(hk, err2["M0"], 0.0)
            names = list(vals)
            S = _station_sums(codes, n_st, vals)
            ix = {k: i for i, k in enumerate(names)}
            min_d = load_config()["stats"]["min_degradation"]

            def fn(T, ix=ix, full_vars=full_vars, has_b="M3b" in col):
                r = {}
                for v in full_vars:
                    r[f"rmse_hit_{v}"] = _rmse(T[..., ix[f"hit_{v}"]], T[..., ix["n_hit"]])
                p = T[..., ix["n_hit"]] / T[..., ix["n_all"]]
                r["p_hit"] = p
                d_hit = r["rmse_hit_M1"] - r["rmse_hit_M0"]
                d_all = (_rmse(T[..., ix["all_M1"]], T[..., ix["n_all"]])
                         - _rmse(T[..., ix["all_M0"]], T[..., ix["n_all"]]))
                r["D_hit"], r["D_all"] = d_hit, d_all
                with np.errstate(invalid="ignore", divide="ignore"):
                    r["D_all_over_p"] = d_all / p
                for v in full_vars:
                    if v in ("M0", "M1"):
                        continue
                    g_ = r["rmse_hit_M1"] - r[f"rmse_hit_{v}"]
                    r[f"G_hit_{v}"] = g_
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r[f"RR_hit_{v}"] = np.where(d_hit >= min_d, g_ / d_hit, np.nan)
                if has_b:
                    nk = T[..., ix["n_hit_kept"]]
                    r["rmse_hit_M3b"] = _rmse(T[..., ix["hit_M3b"]], nk)
                    r["rmse_hit_M0_on_M3b_rows"] = _rmse(T[..., ix["hit_M0_on_kept"]], nk)
                    r["gap_hit_M3b"] = r["rmse_hit_M3b"] - r["rmse_hit_M0_on_M3b_rows"]
                    r["coverage_hit_M3b"] = nk / T[..., ix["n_hit"]]
                return r

            point, boot = _boot(S, fn, (ev, sc, fset, model, "conditional"))
            notes = {"D_hit": "D on contaminated overpasses only; about D_all / p_hit "
                              "(exact in mean squared error when unaffected rows do not change)",
                     "D_all_over_p": "D over all overpasses divided by the hit probability",
                     "gap_hit_M3b": "M3b against M0 on the same retained contaminated rows",
                     "coverage_hit_M3b": "share of contaminated overpasses kept by M3b"}
            base = {"evaluation": ev, "scenario": sc, "feature_set": fset, "model": model,
                    "n_rows_hit_all_seeds": int(hit.sum())}
            out += _records(point, boot, base, n_st, notes)
    t = pd.DataFrame(out)
    return t.merge(scen, on="scenario", how="left")


# --------------------------------------------------------------------------
# Drought events
# --------------------------------------------------------------------------

def event_metrics(rows: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """Drought-alert skill and flip rates against M0, with bootstrap intervals."""
    cfg = load_config()["v2_study"]["metrics"]
    q = cfg["drought_quantile"]
    flip_vars = cfg["flip_variants"]
    pcols = prediction_columns(rows.columns)
    # Threshold per station from the observed values of one scenario and seed
    # (the observations are the same in every scenario and seed).
    ref = rows[(rows["scenario"] == rows["scenario"].min()) & (rows["seed"] == rows["seed"].min())]
    thr = ref.groupby(["evaluation", "station_id"])["sm"].quantile(q)
    out = []
    for (ev, sc), g in rows.groupby(["evaluation", "scenario"], sort=True):
        codes, labels = pd.factorize(g["station_id"])
        n_st = len(labels)
        t = thr.loc[ev].reindex(labels).to_numpy()[codes]
        obs = g["sm"].to_numpy(float) < t
        hit = (g["inj_VV"] | g["inj_VH"]).to_numpy(bool)
        for (fset, model), pc in pcols.groupby(["feature_set", "model"], sort=True):
            col = dict(zip(pc["variant"], pc["column"]))
            if "M0" not in col:
                continue
            a0 = g[col["M0"]].to_numpy(float) < t
            vals, variants = {}, [v for v in ["M0"] + flip_vars if v in col]
            for v in variants:
                p = g[col[v]].to_numpy(float)
                ok = np.isfinite(p)
                a = p < t
                vals[f"n_{v}"] = ok.astype(float)
                vals[f"tp_{v}"] = (ok & a & obs).astype(float)
                vals[f"fp_{v}"] = (ok & a & ~obs).astype(float)
                vals[f"fn_{v}"] = (ok & ~a & obs).astype(float)
                vals[f"tn_{v}"] = (ok & ~a & ~obs).astype(float)
                if v != "M0":
                    flip = ok & (a != a0)
                    vals[f"flip_{v}"] = flip.astype(float)
                    vals[f"hitn_{v}"] = (ok & hit).astype(float)
                    vals[f"fliphit_{v}"] = (flip & hit).astype(float)
                    # A flip that makes the alert wrong (it was right under M0).
                    vals[f"badflip_{v}"] = (flip & hit & (a0 == obs)).astype(float)
            names = list(vals)
            S = _station_sums(codes, n_st, vals)
            ix = {k: i for i, k in enumerate(names)}

            def fn(T, ix=ix, variants=variants):
                r = {}
                with np.errstate(invalid="ignore", divide="ignore"):
                    for v in variants:
                        tp, fp = T[..., ix[f"tp_{v}"]], T[..., ix[f"fp_{v}"]]
                        fn_, tn = T[..., ix[f"fn_{v}"]], T[..., ix[f"tn_{v}"]]
                        r[f"f1_{v}"] = 2 * tp / (2 * tp + fp + fn_)
                        r[f"miss_rate_{v}"] = fn_ / (tp + fn_)
                        r[f"false_alarm_rate_{v}"] = fp / (fp + tn)
                        if v != "M0":
                            r[f"flip_rate_all_{v}"] = T[..., ix[f"flip_{v}"]] / T[..., ix[f"n_{v}"]]
                            r[f"flip_rate_hit_{v}"] = (T[..., ix[f"fliphit_{v}"]]
                                                       / T[..., ix[f"hitn_{v}"]])
                            r[f"bad_flip_rate_hit_{v}"] = (T[..., ix[f"badflip_{v}"]]
                                                           / T[..., ix[f"hitn_{v}"]])
                return r

            point, boot = _boot(S, fn, (ev, sc, fset, model, "events"))
            notes = {f"flip_rate_hit_{v}": "share of contaminated overpasses whose alert "
                                           "differs from the M0 alert" for v in variants}
            notes.update({f"bad_flip_rate_hit_{v}": "share of contaminated overpasses where "
                                                    "a correct M0 alert becomes wrong"
                          for v in variants})
            base = {"evaluation": ev, "scenario": sc, "feature_set": fset, "model": model,
                    "drought_quantile": q}
            out += _records(point, boot, base, n_st, notes)
    t = pd.DataFrame(out)
    return t.merge(scen, on="scenario", how="left")


def run_stage7_metrics(rows_file=None, scen_file=None, tag: str = "") -> dict[str, pd.DataFrame]:
    """Compute and save the conditional and event metrics for a per-row file.

    Defaults to the cached Stage 4 rows. ``tag`` is appended to the output
    names for later per-row files (for example ``_detectors``), so that no
    earlier output is overwritten.
    """
    rows_file = rows_file or path("processed") / "stage4_rows.csv.gz"
    scen_file = scen_file or path("results_aggregate") / "stage4_scenarios.csv"
    rows = load_rows(rows_file)
    scen = pd.read_csv(scen_file)
    cond = conditional_metrics(rows, scen)
    ev = event_metrics(rows, scen)
    cond.to_csv(path("results_aggregate") / f"stage7_conditional{tag}.csv", index=False)
    ev.to_csv(path("results_aggregate") / f"stage7_events{tag}.csv", index=False)
    return {"conditional": cond, "events": ev}
