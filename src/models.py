"""Retrieval models and station-wise cross-validation for the clean model.

Models:
  * random forest (scikit-learn);
  * XGBoost (gradient boosted trees);
  * a change-detection baseline in the style of the TU Wien method, with
    two scalings: per-station (``cd``, primary) and pooled (``cd_pooled``,
    the original version).

The tree models are fitted on each feature configuration declared in
``config.yaml`` (F1 primary, F2 radar-reliant). The change-detection
baseline uses VV only, so it is fitted once and labelled ``feature_set =
"VV"``. For a given setting, model and fold, F1 and F2 use the same model
seed and the same bootstrap draws, so that their comparison is paired.

Evaluation settings (Stage 2):
  * ``USCRN_cv``: station-wise 5-fold cross-validation within USCRN (main);
  * ``OZNET_transfer``: trained on all USCRN stations, tested on OZNET;
  * ``OZNET_cv``: station-wise 5-fold cross-validation within OZNET (small:
    12 stations, flagged as such in the outputs);
  * ``ALL_cv``: station-wise 5-fold cross-validation on both networks
    together, reported overall and per network.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from .config import derive_seed, load_config, path, write_version_log
from .metrics import all_metrics, per_station_metrics

TREE_MODELS = ["rf", "xgb"]
CD_MODELS = {"cd": "per_station", "cd_pooled": "pooled"}
CD_FEATURE_SET = "VV"   # label of the change-detection rows


def feature_sets() -> list[str]:
    """Return the names of the feature configurations, primary first."""
    cfg = load_config()
    names = list(cfg["feature_sets"])
    primary = cfg["primary_feature_set"]
    return [primary] + [n for n in names if n != primary]


def features(fset: str | None = None) -> list[str]:
    """Return the feature columns of a configuration (default: the primary one)."""
    cfg = load_config()
    return list(cfg["feature_sets"][fset or cfg["primary_feature_set"]]["columns"])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class ChangeDetection:
    """Change-detection baseline in the style of the TU Wien method.

    1. For each station, a dry and a wet reference are taken as the 5th and
       95th percentiles of its VV backscatter (dB).
    2. The relative wetness is ``m = (VV - dry) / (wet - dry)``, clipped to
       the range 0 to 1.
    3. ``m`` is mapped to soil moisture with one of two scalings:

       * ``per_station`` (model ``cd``, primary): ``m`` is standardised over
         each station's own series, ``z = (m - mean_s(m)) / sd_s(m)``, and
         ``sm = mu_sm + s_med * z``, where ``mu_sm`` is the mean soil moisture
         of all training rows and ``s_med`` is the median over training
         stations of the within-station standard deviation of soil
         moisture. This gives each station a realistic temporal spread.
       * ``pooled`` (model ``cd_pooled``, the original version): one mean and
         standard deviation matching over all training rows,
         ``sm = mu_sm + sd_sm * (m - mu_m) / sd_m``. Because each station's
         ``m`` spans 0 to 1, this gives every station the full spread of
         soil moisture across all stations and overstates the dynamics.

    Under station-wise cross-validation a test station has no rows in the
    training fold, so its references are computed from its own VV series.
    This uses radar input only and no in situ soil moisture, which is how
    change detection is applied in practice. All soil moisture statistics
    come from the training stations only. Incidence angle normalisation is
    not applied because each station uses a single relative orbit.

    ``fit`` and ``predict`` take the full data frame (they need the station
    labels), not a feature matrix.
    """

    def __init__(self, dry_pct: float = 5, wet_pct: float = 95, col: str = "VV",
                 scaling: str = "per_station"):
        if scaling not in ("per_station", "pooled"):
            raise ValueError(f"unknown scaling: {scaling}")
        self.dry_pct, self.wet_pct, self.col, self.scaling = dry_pct, wet_pct, col, scaling

    def references(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the dry and wet reference of each station in ``df``."""
        g = df.groupby("station_id")[self.col]
        return pd.DataFrame({"dry": g.quantile(self.dry_pct / 100),
                             "wet": g.quantile(self.wet_pct / 100)})

    def relative_wetness(self, df: pd.DataFrame, refs: pd.DataFrame | None = None) -> np.ndarray:
        """Return the clipped relative wetness of every row of ``df``."""
        refs = self.references(df) if refs is None else refs
        r = refs.loc[df["station_id"]].to_numpy()
        span = np.where(r[:, 1] - r[:, 0] > 0, r[:, 1] - r[:, 0], np.nan)
        return np.clip((df[self.col].to_numpy() - r[:, 0]) / span, 0.0, 1.0)

    @staticmethod
    def station_z(m: np.ndarray, stations: np.ndarray) -> np.ndarray:
        """Standardise ``m`` within each station (NaN for a constant series)."""
        s = pd.Series(m)
        g = s.groupby(stations)
        sd = g.transform(lambda x: x.std(ddof=0))
        return ((s - g.transform("mean")) / sd.where(sd > 0)).to_numpy()

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "ChangeDetection":
        """Estimate the scaling from the training rows only."""
        m = self.relative_wetness(df)
        self.mu_y_ = float(np.mean(y))
        if self.scaling == "pooled":
            self.mu_m_, self.sd_m_ = np.nanmean(m), np.nanstd(m)
            self.sd_y_ = float(np.std(y))
        else:
            # Median over training stations of the within-station spread.
            self.sd_y_ = float(pd.Series(y).groupby(df["station_id"].to_numpy())
                               .std(ddof=0).median())
        return self

    def predict(self, df: pd.DataFrame, refs: pd.DataFrame | None = None) -> np.ndarray:
        """Predict soil moisture; references default to each station's own VV."""
        m = self.relative_wetness(df, refs)
        if self.scaling == "pooled":
            return self.mu_y_ + self.sd_y_ * (m - self.mu_m_) / self.sd_m_
        return self.mu_y_ + self.sd_y_ * self.station_z(m, df["station_id"].to_numpy())


def make_model(name: str, seed: int):
    """Create an unfitted model with the settings from ``config.yaml``.

    Tree models use a single thread because parallelism is applied across
    folds instead; this keeps every fit deterministic.
    """
    mcfg = load_config()["models"]
    if name == "rf":
        c = mcfg["random_forest"]
        return RandomForestRegressor(n_estimators=c["n_estimators"],
                                     min_samples_leaf=c["min_samples_leaf"],
                                     random_state=seed, n_jobs=1)
    if name == "xgb":
        c = mcfg["xgboost"]
        return XGBRegressor(n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
                            max_depth=c["max_depth"], subsample=c["subsample"],
                            colsample_bytree=c["colsample_bytree"], tree_method="hist",
                            random_state=seed, n_jobs=1)
    if name in CD_MODELS:
        c = mcfg["change_detection"]
        return ChangeDetection(c["dry_percentile"], c["wet_percentile"],
                               scaling=CD_MODELS[name])
    raise ValueError(f"unknown model: {name}")


def fit_model(name: str, train: pd.DataFrame, seed: int, fset: str | None = None):
    """Fit a model on the training rows and return it."""
    model = make_model(name, seed)
    y = train["sm"].to_numpy()
    if name in CD_MODELS:
        return model.fit(train, y)
    return model.fit(train[features(fset)].to_numpy(), y)


def predict_model(name: str, model, test: pd.DataFrame, fset: str | None = None) -> np.ndarray:
    """Predict soil moisture for the test rows."""
    if name in CD_MODELS:
        return model.predict(test)
    return model.predict(test[features(fset)].to_numpy())


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

def station_folds(df: pd.DataFrame, n_folds: int, *seed_keys) -> np.ndarray:
    """Assign each row to a station-wise fold (all rows of a station share a fold).

    Uses ``GroupKFold`` with shuffling, seeded from the master seed, and
    asserts that no station appears in both the training and test part of
    any fold.
    """
    gkf = GroupKFold(n_splits=n_folds, shuffle=True,
                     random_state=derive_seed("folds", *seed_keys) % (2**31))
    fold = np.full(len(df), -1)
    groups = df["station_id"].to_numpy()
    for k, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        assert not set(groups[tr]) & set(groups[te]), "test station found in training fold"
        fold[te] = k
    assert (fold >= 0).all()
    return fold


# --------------------------------------------------------------------------
# Stage 2 runner
# --------------------------------------------------------------------------

def _fit_predict_task(setting: str, fset: str, name: str, k: int, train: pd.DataFrame,
                      test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit one model on one fold, save it, and return its test predictions.

    The seed does not depend on the feature set, so F1 and F2 are paired.
    """
    seed = derive_seed("stage2", setting, name, k)
    # No timing is stored, so that the run table is identical between runs.
    model = fit_model(name, train, seed, fset)
    pred = predict_model(name, model, test, fset)
    joblib.dump(model, path("models") / "stage2" / f"{setting}_{fset}_{name}_fold{k}.joblib")
    out = test[["network", "station_id", "t", "sm", "vh_low", "ndvi", "lc"]].copy()
    out["pred"] = pred
    out["setting"], out["feature_set"], out["model"], out["fold"] = setting, fset, name, k
    imp = {}
    if name in TREE_MODELS:
        imp = dict(zip(features(fset), model.feature_importances_))
    info = {"setting": setting, "feature_set": fset, "model": name, "fold": k, "seed": seed,
            "n_train": len(train), "n_test": len(test),
            "train_stations": train["station_id"].nunique(),
            "test_stations": test["station_id"].nunique(),
            **{f"imp_{f}": v for f, v in imp.items()}}
    return out, info


def build_tasks(df: pd.DataFrame) -> list[tuple]:
    """List every (setting, feature set, model, fold, train, test) combination."""
    cfg = load_config()
    n_folds = cfg["cv"]["n_folds"]
    main, second = cfg["samples"]["main"], cfg["samples"]["second"]
    tasks = []

    def cv(setting: str, sub: pd.DataFrame) -> None:
        sub = sub.reset_index(drop=True)
        fold = station_folds(sub, n_folds, setting)
        for k in range(n_folds):
            tr, te = sub[fold != k], sub[fold == k]
            for fset, name in model_grid():
                tasks.append((setting, fset, name, k, tr, te))

    cv(f"{main}_cv", df[df["network"] == main])
    cv(f"{second}_cv", df[df["network"] == second])
    cv("ALL_cv", df)
    # Transfer: one model per type, trained on all main-sample stations.
    for fset, name in model_grid():
        tasks.append((f"{second}_transfer", fset, name, 0, df[df["network"] == main],
                      df[df["network"] == second]))
    return tasks


def model_grid() -> list[tuple[str, str]]:
    """Return all (feature set, model) pairs.

    Tree models are fitted on every feature set; the change-detection models
    are fitted once, because they use VV only.
    """
    grid = [(fset, name) for fset in feature_sets() for name in TREE_MODELS]
    return grid + [(CD_FEATURE_SET, name) for name in CD_MODELS]


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bootstrap metrics for every setting, model and network.

    Returns the aggregated metric table (one row per setting, model, scope,
    metric) and the per-station metric table.
    """
    from .stats import cluster_bootstrap

    second = load_config()["samples"]["second"]
    rows, per_station = [], []
    for (setting, fset, name), g in pred.groupby(["setting", "feature_set", "model"],
                                                 sort=False):
        scopes = [("all", g)]
        if setting == "ALL_cv":
            scopes += [(net, gg) for net, gg in g.groupby("network")]
        for scope, gg in scopes:
            # Bootstrap keys exclude the feature set: F1 and F2 share draws.
            b = cluster_bootstrap(gg, "stage2", setting, name, scope)
            b.insert(0, "scope", scope)
            b.insert(0, "model", name)
            b.insert(0, "feature_set", fset)
            b.insert(0, "setting", setting)
            b["small_sample"] = setting == f"{second}_cv" or (
                setting == "ALL_cv" and scope == second)
            rows.append(b)
        ps = per_station_metrics(g)
        ps.insert(0, "model", name)
        ps.insert(0, "feature_set", fset)
        ps.insert(0, "setting", setting)
        per_station.append(ps)
    return pd.concat(rows, ignore_index=True), pd.concat(per_station, ignore_index=True)


def fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per fold (aggregated over the test stations of the fold)."""
    rows = []
    for (setting, fset, name, k), g in pred.groupby(["setting", "feature_set", "model",
                                                     "fold"]):
        rows.append({"setting": setting, "feature_set": fset, "model": name, "fold": k,
                     "test_stations": g["station_id"].nunique(),
                     **all_metrics(g["sm"].to_numpy(), g["pred"].to_numpy())})
    return pd.DataFrame(rows)


def run_stage2(n_jobs: int | None = None) -> dict[str, pd.DataFrame]:
    """Run the clean-model stage and save all outputs.

    Outputs:
      * fold models in ``models/stage2/``;
      * out-of-fold predictions in ``data/processed/stage2_predictions.csv``;
      * aggregated metrics with bootstrap intervals, per-fold metrics,
        feature importances and run information in ``results/aggregate/``;
      * per-station metrics in ``results/private/``.
    """
    from .data import add_derived_features, load_matchups

    cfg = load_config()
    n_jobs = cfg["n_jobs"] if n_jobs is None else n_jobs
    (path("models") / "stage2").mkdir(parents=True, exist_ok=True)
    # Derived backscatter features (VHVV, rVV, rVH) are computed from the clean
    # series here; in Stage 3 they are recomputed from the observed series.
    df = add_derived_features(load_matchups())
    tasks = build_tasks(df)
    print(f"Stage 2: {len(tasks)} fits on {n_jobs} workers")
    t0 = time.time()
    # max_nbytes=None disables memory-mapping of the data frames, which avoids
    # temporary-folder clean-up errors on Windows; the data are small.
    results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_fit_predict_task)(*t) for t in tasks)
    print(f"fits done in {time.time() - t0:.0f} s")

    pred = pd.concat([r[0] for r in results], ignore_index=True)
    runs = pd.DataFrame([r[1] for r in results])
    # Each cross-validation setting must predict every row of its sample once.
    for setting, g in pred.groupby("setting"):
        counts = g.groupby(["feature_set", "model"]).size()
        assert counts.nunique() == 1, f"unequal prediction counts in {setting}"
        assert not g.duplicated(["feature_set", "model", "station_id", "t"]).any(), setting

    metrics, per_station = evaluate(pred)
    folds = fold_metrics(pred)
    imp_cols = [c for c in runs if c.startswith("imp_")]
    importance = (runs[runs["model"].isin(TREE_MODELS)]
                  .groupby(["setting", "feature_set", "model"])[imp_cols]
                  .mean().reset_index())

    pred.to_csv(path("processed") / "stage2_predictions.csv", index=False)
    out_files = {"stage2_metrics": metrics, "stage2_fold_metrics": folds,
                 "stage2_feature_importance": importance,
                 "stage2_runs": runs.drop(columns=imp_cols)}
    for name, table in out_files.items():
        table.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    per_station.to_csv(path("results_private") / "stage2_station_metrics.csv", index=False)
    write_version_log(path("results_aggregate") / "stage2_metrics.csv")
    return {**out_files, "per_station": per_station, "pred": pred}


def summary_table(metrics: pd.DataFrame,
                  keep: tuple = ("rmse", "ubrmsd", "bias", "r2", "r",
                                 "r_station_median", "ubrmsd_station_median")) -> pd.DataFrame:
    """Format the metric table as 'estimate [low, high]' strings for display."""
    m = metrics[metrics["metric"].isin(keep)].copy()
    m["value"] = m.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
    idx = ["setting", "scope", "feature_set", "model"]
    t = m.pivot_table(index=idx, columns="metric", values="value", aggfunc="first", sort=False)
    n = metrics.groupby(idx, sort=False)[["n_stations", "n"]].first()
    return t[[k for k in keep if k in t.columns]].join(n)
