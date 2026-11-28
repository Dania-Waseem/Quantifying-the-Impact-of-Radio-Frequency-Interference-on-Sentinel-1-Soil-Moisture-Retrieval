"""Stage 7 model comparison on the radar-reliant feature sets (F2, F3, F4).

Declared in ``config.yaml`` under ``v2_study`` (feature set F4, model
families and grids, nested selection, H1 test) before any run.

For every outer split of the Stage 3 evaluations (USCRN cross-validation,
OZNET cross-validation, OZNET transfer) and every feature set:

1. Nested selection on the outer training stations only: inner
   station-grouped folds, clean inner RMSE per hyperparameter setting,
   best setting per family, then the best family ("selected").
2. The chosen setting of every family is fitted on the outer training
   stations (random forest and XGBoost on F2 and F3 are reused from the
   Stage 2 cache; they have the same settings and seeds).
3. Clean (M0) and contaminated (M1) predictions on the outer test stations
   over the full Stage 3 injection grid with the Stage 3 seed streams.
   Saved: per-fold error sums per setting and seed (as in Stage 3), and
   per-station error sums pooled over seeds (for the cluster bootstrap).
   For the 7 Stage 4 row-level scenarios, per-row predictions are also kept.

Raw outputs are written per feature set, so the run can be split by
feature set without changing the data; ``aggregate_models`` builds the
tables from whatever feature sets have been run.
"""

from __future__ import annotations

import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import rfi
from .config import derive_seed, load_config, path, rng, write_version_log
from .data import add_derived_features, load_matchups
from .experiment import GZIP, SUM_COLS, _station_sums, evaluation_splits
from .models import features as v1_features
from .models import rf_predict
from .stats import _counts, _pct, metrics_from_sums, wilcoxon_one_sided

FAMILIES = ["rf", "xgb", "hgb", "ridge", "rf_grid"]
ID_COLS = ["network", "station_id", "t", "sm", "ndvi", "lc", "vh_low"]


def v2cfg() -> dict:
    return load_config()["v2_study"]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def add_f4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the F4 temporal features from past overpasses of each station.

    Requires ``rVV`` and ``rVH``. For each station in time order:
    previous relative backscatter, change since the previous overpass and
    the median of the previous ``past_window`` overpasses. Missing past
    values are set to the configured fill value (0, the reference level).
    """
    f4 = v2cfg()["feature_sets"]["F4"]
    w, fill = int(f4["past_window"]), float(f4["fill_missing_past"])
    out = df.copy()
    order = np.lexsort((out["t"].to_numpy(), out["station_id"].to_numpy()))
    s = out.iloc[order]
    g = s.groupby("station_id", sort=False)
    for p in ("VV", "VH"):
        r = s[f"r{p}"]
        prev = g[f"r{p}"].shift(1)
        med = g[f"r{p}"].transform(lambda x: x.shift(1).rolling(w, min_periods=1).median())
        prev_f = prev.fillna(fill)
        cols = {f"r{p}_prev": prev_f, f"d{p}": r - prev_f, f"r{p}_med5": med.fillna(fill)}
        for c, v in cols.items():
            arr = np.empty(len(out))
            arr[order] = v.to_numpy()
            out[c] = arr
    return out


def derive(df: pd.DataFrame, fset: str) -> pd.DataFrame:
    """All derived backscatter features needed by a feature set."""
    out = add_derived_features(df)
    return add_f4_features(out) if fset == "F4" else out


def features(fset: str) -> list[str]:
    if fset == "F4":
        return list(v2cfg()["feature_sets"]["F4"]["columns"])
    return v1_features(fset)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_v2_model(family: str, params: dict, seed: int):
    """Unfitted model of a family with given settings (single thread)."""
    if family in ("rf", "rf_grid"):
        return RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    if family == "xgb":
        return XGBRegressor(**params, tree_method="hist", random_state=seed, n_jobs=1)
    if family == "hgb":
        return HistGradientBoostingRegressor(**params, early_stopping=False, random_state=seed)
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    raise ValueError(family)


def predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, RandomForestRegressor):
        return rf_predict(model, X)
    return model.predict(X)


def model_path(ev: str, fset: str, family: str, k: int):
    """Outer model file (Stage 2 cache for rf and xgb on F2 and F3)."""
    if family in ("rf", "xgb") and fset in ("F2", "F3"):
        return path("models") / "stage2" / f"{ev}_{fset}_{family}_fold{k}.joblib"
    return path("models") / "stage7" / f"{ev}_{fset}_{family}_fold{k}.joblib"


def outer_seed(ev: str, family: str, k: int, fset: str, cfg_idx: int) -> int:
    """rf and xgb use the Stage 2 seeds (paired with F1 to F3); others their own."""
    if family in ("rf", "xgb"):
        return derive_seed("stage2", ev, family, k)
    return derive_seed("stage7", ev, fset, family, cfg_idx, k)


# --------------------------------------------------------------------------
# Nested selection and outer fitting
# --------------------------------------------------------------------------

def _select_task(ev: str, k: int, train: pd.DataFrame, fset: str) -> list[dict]:
    """Inner station-grouped CV on the outer training stations; fit the
    chosen setting of every family on all outer training stations."""
    from threadpoolctl import threadpool_limits

    fams = v2cfg()["models"]["families"]
    n_inner = int(v2cfg()["models"]["inner_folds"])
    with threadpool_limits(1):
        d = derive(train, fset).reset_index(drop=True)
        X, y = d[features(fset)].to_numpy(float), d["sm"].to_numpy(float)
        groups = d["station_id"].to_numpy()
        gkf = GroupKFold(n_splits=n_inner, shuffle=True,
                         random_state=derive_seed("stage7_inner", ev, k) % (2**31))
        inner = list(gkf.split(X, groups=groups))
        rows = []
        for fam in FAMILIES:
            for i, params in enumerate(fams[fam]):
                sse = 0.0
                for j, (tr, va) in enumerate(inner):
                    m = make_v2_model(fam, params,
                                      derive_seed("stage7_inner_fit", ev, k, fset, fam, i, j))
                    m.fit(X[tr], y[tr])
                    sse += float(((predict(m, X[va]) - y[va]) ** 2).sum())
                rows.append({"evaluation": ev, "fold": k, "feature_set": fset, "family": fam,
                             "config": i, "params": str(params),
                             "inner_rmse": float(np.sqrt(sse / len(y))),
                             "n_inner_folds": n_inner, "n_train_stations": len(set(groups))})
        sel = pd.DataFrame(rows)
        best_cfg = sel.loc[sel.groupby("family")["inner_rmse"].idxmin()]
        best_family = best_cfg.loc[best_cfg["inner_rmse"].idxmin(), "family"]
        sel["best_config_of_family"] = sel.index.isin(best_cfg.index)
        sel["selected_family"] = best_family
        (path("models") / "stage7").mkdir(parents=True, exist_ok=True)
        for r in best_cfg.itertuples():
            f = model_path(ev, fset, r.family, k)
            if f.parent.name == "stage2":
                assert f.exists(), f        # reuse the cached Stage 2 model
                continue
            m = make_v2_model(r.family, fams[r.family][r.config],
                              outer_seed(ev, r.family, k, fset, r.config))
            m.fit(X, y)
            joblib.dump(m, f)
    return sel.to_dict("records")


# --------------------------------------------------------------------------
# Outer predictions over the injection grid
# --------------------------------------------------------------------------

def _row_level_settings() -> set:
    """(severity_q, extent, p_c, polarisation) of the Stage 4 row-level scenarios."""
    s = pd.read_csv(path("results_aggregate") / "stage4_scenarios.csv")
    s = s[s["row_level"]]
    return {(int(r.severity_q), float(r.extent), float(r.p_c), r.polarisation): int(r.scenario)
            for r in s.itertuples()}


def _predict_task(ev: str, k: int, test: pd.DataFrame, fset: str, selected: str,
                  include_cd: bool) -> dict:
    """M0 and M1 over the full grid for one split and feature set."""
    from threadpoolctl import threadpool_limits

    cfg = load_config()
    grid = rfi.injection_grid(verbose=False)
    n_seeds = cfg["n_seeds"]
    rowlevel = _row_level_settings()
    cols = features(fset)
    with threadpool_limits(1):
        models = {fam: joblib.load(model_path(ev, fset, fam, k)) for fam in FAMILIES}
        if include_cd:
            cd = joblib.load(path("models") / "stage2" / f"{ev}_VV_cd_fold{k}.joblib")
        clean = derive(test, fset)
        X0 = clean[cols].to_numpy(float)
        y = clean["sm"].to_numpy(float)
        st = clean["station_id"].to_numpy()
        labels = pd.unique(st)
        p0 = {fam: predict(m, X0) for fam, m in models.items()}
        p0["selected"] = p0[selected]
        if include_cd:
            p0["cd"] = cd.predict(clean)

        def sums(p):
            return _station_sums(st, y, p).set_index("station_id").reindex(labels)[SUM_COLS]

        m0 = {name: sums(p) for name, p in p0.items()}
        runs, acc, rows = [], {}, []
        for q, rho, pc, pol in product(grid["severity_q"], grid["extent"], grid["p_c"],
                                       grid["polarisation"]):
            key = (q, rho, pc, pol)
            for s in range(n_seeds):
                g = rng("test_injection", ev, k, q, rho, pc, pol, s)
                obs = rfi.inject(test, grid["sir"][q], rho, pc, pol, g)
                x = derive(obs, fset)
                X1 = x[cols].to_numpy(float)
                same = (X1 == X0).all(axis=1)
                p1 = {}
                for fam, m in models.items():
                    p = p0[fam].copy()
                    if (~same).any():
                        p[~same] = predict(m, X1[~same])
                    p1[fam] = p
                p1["selected"] = p1[selected]
                if include_cd:
                    p1["cd"] = cd.predict(x)
                for name, p in p1.items():
                    ss = sums(p)
                    tot = ss.sum().to_dict()
                    runs.append({"evaluation": ev, "fold": k, "severity_q": q, "extent": rho,
                                 "p_c": pc, "polarisation": pol, "seed": s,
                                 "feature_set": "VV" if name == "cd" else fset, "model": name,
                                 "variant": "M1", **tot})
                    a = acc.setdefault((key, name), np.zeros((len(labels), len(SUM_COLS))))
                    a += ss.to_numpy()
                if (q, float(rho), float(pc), pol) in rowlevel:
                    r = clean[ID_COLS].copy()
                    r["evaluation"], r["fold"], r["seed"] = ev, k, s
                    r["scenario"] = rowlevel[(q, float(rho), float(pc), pol)]
                    for pp in ("VV", "VH"):
                        r[f"inj_{pp}"] = obs[f"inj_{pp}"].to_numpy()
                        r[f"eps_{pp}"] = obs[f"eps_{pp}"].to_numpy()
                    for name in p1:
                        fs = "VV" if name == "cd" else fset
                        r[f"{fs}|{name}|M0"] = p0[name].astype("float32")
                        r[f"{fs}|{name}|M1"] = p1[name].astype("float32")
                    rows.append(r)
        stsum = []
        for (key, name), a in acc.items():
            t = pd.DataFrame(a, columns=SUM_COLS)
            t.insert(0, "station_id", labels)
            q, rho, pc, pol = key
            t = t.assign(evaluation=ev, fold=k, severity_q=q, extent=rho, p_c=pc,
                         polarisation=pol, feature_set="VV" if name == "cd" else fset,
                         model=name, variant="M1", n_seeds=n_seeds)
            stsum.append(t)
        for name, t in m0.items():
            t = t.reset_index().assign(evaluation=ev, fold=k, severity_q=-1, extent=-1.0,
                                       p_c=-1.0, polarisation="clean",
                                       feature_set="VV" if name == "cd" else fset,
                                       model=name, variant="M0", n_seeds=1)
            stsum.append(t)
    return {"runs": runs, "station_sums": pd.concat(stsum, ignore_index=True),
            "rows": pd.concat(rows, ignore_index=True) if rows else None}


# --------------------------------------------------------------------------
# Runner (per feature set) and aggregation
# --------------------------------------------------------------------------

def run_models(fsets: list[str], n_jobs: int | None = None) -> None:
    """Selection, outer fitting and grid predictions for the given feature sets.

    Raw outputs per feature set (local, not tracked):
      data/processed/stage7_models_selection_<fset>.csv
      results/stage7_models_runs_<fset>.csv.gz        (per fold, setting, seed)
      data/processed/stage7_models_station_sums_<fset>.csv.gz
      data/processed/stage7_rows_models_<fset>.csv.gz (7 row-level scenarios)
    """
    cfg = load_config()
    n_jobs = cfg["experiment"]["n_jobs"] if n_jobs is None else n_jobs
    rfi.injection_grid()
    splits = evaluation_splits(load_matchups())
    for fset in fsets:
        t0 = time.time()
        sel = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_select_task)(ev, k, tr, fset) for ev, k, tr, _ in splits)
        sel = pd.DataFrame([r for s in sel for r in s])
        sel.to_csv(path("processed") / f"stage7_models_selection_{fset}.csv", index=False)
        print(f"{fset}: selection and fitting done in {time.time() - t0:.0f} s")
        chosen = sel.groupby(["evaluation", "fold"])["selected_family"].first().to_dict()
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=5)(
            delayed(_predict_task)(ev, k, te, fset, chosen[(ev, k)], fset == "F2")
            for ev, k, _, te in splits)
        print(f"{fset}: grid predictions done in {time.time() - t0:.0f} s")
        pd.DataFrame([r for x in res for r in x["runs"]]).to_csv(
            path("results") / f"stage7_models_runs_{fset}.csv.gz", index=False,
            compression=GZIP)
        pd.concat([x["station_sums"] for x in res], ignore_index=True).to_csv(
            path("processed") / f"stage7_models_station_sums_{fset}.csv.gz", index=False,
            compression=GZIP)
        rows = [x["rows"] for x in res if x["rows"] is not None]
        pd.concat(rows, ignore_index=True).to_csv(
            path("processed") / f"stage7_rows_models_{fset}.csv.gz", index=False,
            compression=GZIP)


def _boot_D(sums0: np.ndarray, sums1: np.ndarray, keys) -> dict:
    """RMSE_M0, RMSE_M1 and D with a cluster bootstrap over stations (paired)."""
    c = _counts(sums0.shape[0], "stage7_models", *keys)
    out = {}
    for name, S in (("rmse_M0", sums0), ("rmse_M1", sums1)):
        out[name] = (metrics_from_sums(S.sum(0))["rmse"], metrics_from_sums(c @ S)["rmse"])
    out["D"] = (out["rmse_M1"][0] - out["rmse_M0"][0], out["rmse_M1"][1] - out["rmse_M0"][1])
    return out


def _records(base: dict, res: dict) -> list[dict]:
    rows = []
    for q, (pt, b) in res.items():
        lo, hi = _pct(np.asarray(b, float))
        rows.append({**base, "quantity": q, "estimate": float(pt), "ci_low": lo, "ci_high": hi})
    return rows


def aggregate_models() -> dict[str, pd.DataFrame]:
    """Build the stage7_models_* tables from the raw outputs available."""
    fsets = [f for f in v2cfg()["models"]["feature_sets"]
             if (path("processed") / f"stage7_models_station_sums_{f}.csv.gz").exists()]
    st = pd.concat([pd.read_csv(path("processed") / f"stage7_models_station_sums_{f}.csv.gz")
                    for f in fsets], ignore_index=True)
    sel = pd.concat([pd.read_csv(path("processed") / f"stage7_models_selection_{f}.csv")
                     for f in fsets], ignore_index=True)
    st = st.groupby(["evaluation", "station_id", "severity_q", "extent", "p_c", "polarisation",
                     "feature_set", "model", "variant"], as_index=False)[SUM_COLS].sum()
    m0 = st[st.variant == "M0"]
    m1 = st[st.variant == "M1"]
    grid = rfi.injection_grid(verbose=False)
    sev_rank = {q: i + 1 for i, q in enumerate(grid["severity_q"])}
    ext_rank = {round(e, 6): i + 1 for i, e in enumerate(grid["extent"])}
    m1 = m1.assign(sev_rank=m1["severity_q"].map(sev_rank),
                   ext_rank=m1["extent"].round(6).map(ext_rank))
    outer, by_set, by_sev, by_ext = [], [], [], []
    for (ev, fs, mdl), g1 in m1.groupby(["evaluation", "feature_set", "model"]):
        g0 = m0[(m0.evaluation == ev) & (m0.feature_set == fs) & (m0.model == mdl)]
        labels = np.sort(g0["station_id"].unique())
        S0 = g0.set_index("station_id").reindex(labels)[SUM_COLS].to_numpy()
        n_set = g1[["severity_q", "extent", "p_c", "polarisation"]].drop_duplicates().shape[0]
        n_seeds = load_config()["n_seeds"]
        base = {"evaluation": ev, "feature_set": fs, "model": mdl, "n_stations": len(labels)}

        def m1_sums(sub, reps):
            S1 = sub.groupby("station_id")[SUM_COLS].sum().reindex(labels).fillna(0).to_numpy()
            return S1, S0 * reps         # M0 repeated for the same number of evaluations

        S1, S0r = m1_sums(g1, n_set * n_seeds)
        outer += _records({**base, "scope": "all settings"},
                          _boot_D(S0r, S1, (ev, fs, mdl, "all")))
        for (q, e), g in g1.groupby(["severity_q", "extent"]):
            reps = g[["p_c", "polarisation"]].drop_duplicates().shape[0] * n_seeds
            S1, S0r = m1_sums(g, reps)
            by_set += _records({**base, "severity_q": q, "extent": e},
                               _boot_D(S0r, S1, (ev, fs, mdl, "set", q, e)))
        for col, out, rank in (("severity_q", by_sev, "sev_rank"), ("extent", by_ext, "ext_rank")):
            for lv, g in g1.groupby(col):
                reps = g[["severity_q", "extent", "p_c", "polarisation"]].drop_duplicates().shape[0] * n_seeds
                S1, S0r = m1_sums(g, reps)
                out += _records({**base, col: lv, "rank": int(g[rank].iloc[0])},
                                _boot_D(S0r, S1, (ev, fs, mdl, col, lv)))
    outer = pd.DataFrame(outer)
    # Inner selection: share of outer folds in which a family was selected.
    pick = sel.groupby(["evaluation", "feature_set", "fold"])["selected_family"].first()
    share = (pick.reset_index().groupby(["evaluation", "feature_set"])["selected_family"]
             .value_counts(normalize=True).rename("selected_share").reset_index()
             .rename(columns={"selected_family": "model"}))
    outer = outer.merge(share, on=["evaluation", "feature_set", "model"], how="left")
    outer["selected_share"] = outer["selected_share"].fillna(0.0)
    out = {"stage7_models_outer": outer, "stage7_models_selection": sel,
           "stage7_models_by_setting": pd.DataFrame(by_set),
           "stage7_models_by_severity": pd.DataFrame(by_sev),
           "stage7_models_by_extent": pd.DataFrame(by_ext),
           "stage7_models_h1": h1_tests(m0, m1)}
    for name, t in out.items():
        t.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "stage7_models_outer.csv")
    return out


def h1_tests(m0: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """H1: station slopes of D over the ordered severity and extent levels."""
    from statsmodels.stats.multitest import multipletests

    n_seeds = load_config()["n_seeds"]
    prim = v2cfg()["h1_test"]["primary_evaluation"]
    rows = []
    for (ev, fs, mdl), g1 in m1.groupby(["evaluation", "feature_set", "model"]):
        g0 = m0[(m0.evaluation == ev) & (m0.feature_set == fs) & (m0.model == mdl)]
        r0 = g0.set_index("station_id")
        rmse0 = np.sqrt(r0["e2"] / r0["n"])
        for factor, rank in (("severity", "sev_rank"), ("extent", "ext_rank")):
            agg = g1.groupby(["station_id", rank])[["e2", "n"]].sum().reset_index()
            agg["D"] = np.sqrt(agg["e2"] / agg["n"]) - agg["station_id"].map(rmse0)
            slopes = agg.groupby("station_id").apply(
                lambda s: np.polyfit(s[rank], s["D"], 1)[0]).dropna()
            w = wilcoxon_one_sided(slopes.to_numpy(), np.zeros(len(slopes)))
            c = _counts(len(slopes), "stage7_h1", ev, fs, mdl, factor)
            b = (c @ slopes.to_numpy()) / len(slopes)
            lo, hi = _pct(b)
            rows.append({"evaluation": ev, "feature_set": fs, "model": mdl, "factor": factor,
                         "n_stations": len(slopes), "mean_slope": float(slopes.mean()),
                         "mean_slope_ci_low": lo, "mean_slope_ci_high": hi,
                         "median_slope": float(slopes.median()), "z": w["z"], "r": w["r"],
                         "p": w["p"], "primary": ev == prim})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    pm = t["primary"] & t["p"].notna()
    if pm.any():
        t.loc[pm, "p_holm"] = multipletests(t.loc[pm, "p"], method="holm")[1]
    t["note"] = np.where(t["primary"], "v2 Holm family (H1 tests so far; recomputed when "
                         "H2 to H6 tests are added)", "secondary evaluation, uncorrected")
    return t


def merge_rows(tag: str = "models") -> str:
    """Merge the per-feature-set row files into data/processed/stage7_rows_<tag>.csv.gz.

    The injection does not depend on the feature set (same seed streams), so
    the rows of the three runs line up; this is checked on the true flags.
    """
    keys = ["evaluation", "fold", "station_id", "t", "scenario", "seed"]
    merged = None
    for f in v2cfg()["models"]["feature_sets"]:
        p = path("processed") / f"stage7_rows_models_{f}.csv.gz"
        if not p.exists():
            continue
        r = pd.read_csv(p)
        if merged is None:
            merged = r
            continue
        new = [c for c in r.columns if c.count("|") == 2 and c not in merged.columns]
        chk = merged[keys + ["inj_VV", "inj_VH"]].merge(r[keys + ["inj_VV", "inj_VH"]],
                                                         on=keys, suffixes=("", "_b"))
        assert len(chk) == len(merged) == len(r), "row sets differ between feature sets"
        assert (chk["inj_VV"] == chk["inj_VV_b"]).all() and (chk["inj_VH"] == chk["inj_VH_b"]).all()
        merged = merged.merge(r[keys + new], on=keys)
    out = path("processed") / f"stage7_rows_{tag}.csv.gz"
    merged.to_csv(out, index=False, compression=GZIP)
    return str(out)
