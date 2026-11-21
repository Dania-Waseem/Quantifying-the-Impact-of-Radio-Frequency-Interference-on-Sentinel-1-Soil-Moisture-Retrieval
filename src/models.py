"""Retrieval models and station-wise cross-validation for the clean model.

Models:
  * random forest (scikit-learn);
  * XGBoost (gradient boosted trees);
  * a change-detection baseline in the style of the TU Wien method.

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

MODEL_NAMES = ["rf", "xgb", "cd"]


def features() -> list[str]:
    """Return the list of model features defined in the configuration."""
    return list(load_config()["features"])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class ChangeDetection:
    """Change-detection baseline in the style of the TU Wien method.

    1. For each station, a dry and a wet reference are taken as the 5th and
       95th percentiles of its VV backscatter (dB).
    2. The relative wetness is ``m = (VV - dry) / (wet - dry)``, clipped to
       the range 0 to 1.
    3. ``m`` is mapped to soil moisture by matching its mean and standard
       deviation to those of the in situ soil moisture of the training
       stations: ``sm = mu_sm + sd_sm * (m - mu_m) / sd_m``, where ``mu_m``
       and ``sd_m`` are computed on the training stations.

    Under station-wise cross-validation a test station has no rows in the
    training fold, so its references are computed from its own VV series.
    This uses radar input only and no in situ soil moisture, which is how
    change detection is applied in practice. All soil moisture statistics
    come from the training stations only. Incidence angle normalisation is
    not applied because each station uses a single relative orbit.

    ``fit`` and ``predict`` take the full data frame (they need the station
    labels), not a feature matrix.
    """

    def __init__(self, dry_pct: float = 5, wet_pct: float = 95, col: str = "VV"):
        self.dry_pct, self.wet_pct, self.col = dry_pct, wet_pct, col

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

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "ChangeDetection":
        """Estimate the mean and standard deviation matching on training rows."""
        m = self.relative_wetness(df)
        self.mu_m_, self.sd_m_ = np.nanmean(m), np.nanstd(m)
        self.mu_y_, self.sd_y_ = float(np.mean(y)), float(np.std(y))
        return self

    def predict(self, df: pd.DataFrame, refs: pd.DataFrame | None = None) -> np.ndarray:
        """Predict soil moisture; references default to each station's own VV."""
        m = self.relative_wetness(df, refs)
        return self.mu_y_ + self.sd_y_ * (m - self.mu_m_) / self.sd_m_


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
    if name == "cd":
        c = mcfg["change_detection"]
        return ChangeDetection(c["dry_percentile"], c["wet_percentile"])
    raise ValueError(f"unknown model: {name}")


def fit_model(name: str, train: pd.DataFrame, seed: int):
    """Fit a model on the training rows and return it."""
    model = make_model(name, seed)
    y = train["sm"].to_numpy()
    if name == "cd":
        return model.fit(train, y)
    return model.fit(train[features()].to_numpy(), y)


def predict_model(name: str, model, test: pd.DataFrame) -> np.ndarray:
    """Predict soil moisture for the test rows."""
    if name == "cd":
        return model.predict(test)
    return model.predict(test[features()].to_numpy())


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

def _fit_predict_task(setting: str, name: str, k: int, train: pd.DataFrame,
                      test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit one model on one fold, save it, and return its test predictions."""
    seed = derive_seed("stage2", setting, name, k)
    t0 = time.time()
    model = fit_model(name, train, seed)
    pred = predict_model(name, model, test)
    joblib.dump(model, path("models") / "stage2" / f"{setting}_{name}_fold{k}.joblib")
    out = test[["network", "station_id", "t", "sm", "vh_low", "ndvi", "lc"]].copy()
    out["pred"] = pred
    out["setting"], out["model"], out["fold"] = setting, name, k
    imp = {}
    if name in ("rf", "xgb"):
        imp = dict(zip(features(), model.feature_importances_))
    info = {"setting": setting, "model": name, "fold": k, "seed": seed,
            "n_train": len(train), "n_test": len(test),
            "train_stations": train["station_id"].nunique(),
            "test_stations": test["station_id"].nunique(),
            "seconds": time.time() - t0, **{f"imp_{f}": v for f, v in imp.items()}}
    return out, info


def build_tasks(df: pd.DataFrame) -> list[tuple]:
    """List every (setting, model, fold, train, test) combination of Stage 2."""
    cfg = load_config()
    n_folds = cfg["cv"]["n_folds"]
    main, second = cfg["samples"]["main"], cfg["samples"]["second"]
    tasks = []

    def cv(setting: str, sub: pd.DataFrame) -> None:
        sub = sub.reset_index(drop=True)
        fold = station_folds(sub, n_folds, setting)
        for k in range(n_folds):
            tr, te = sub[fold != k], sub[fold == k]
            for name in MODEL_NAMES:
                tasks.append((setting, name, k, tr, te))

    cv(f"{main}_cv", df[df["network"] == main])
    cv(f"{second}_cv", df[df["network"] == second])
    cv("ALL_cv", df)
    # Transfer: one model per type, trained on all main-sample stations.
    for name in MODEL_NAMES:
        tasks.append((f"{second}_transfer", name, 0, df[df["network"] == main],
                      df[df["network"] == second]))
    return tasks


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute bootstrap metrics for every setting, model and network.

    Returns the aggregated metric table (one row per setting, model, scope,
    metric) and the per-station metric table.
    """
    from .stats import cluster_bootstrap

    second = load_config()["samples"]["second"]
    rows, per_station = [], []
    for (setting, name), g in pred.groupby(["setting", "model"], sort=False):
        scopes = [("all", g)]
        if setting == "ALL_cv":
            scopes += [(net, gg) for net, gg in g.groupby("network")]
        for scope, gg in scopes:
            b = cluster_bootstrap(gg, "stage2", setting, name, scope)
            b.insert(0, "scope", scope)
            b.insert(0, "model", name)
            b.insert(0, "setting", setting)
            b["small_sample"] = setting == f"{second}_cv" or (
                setting == "ALL_cv" and scope == second)
            rows.append(b)
        ps = per_station_metrics(g)
        ps.insert(0, "model", name)
        ps.insert(0, "setting", setting)
        per_station.append(ps)
    return pd.concat(rows, ignore_index=True), pd.concat(per_station, ignore_index=True)


def fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per fold (aggregated over the test stations of the fold)."""
    rows = []
    for (setting, name, k), g in pred.groupby(["setting", "model", "fold"]):
        rows.append({"setting": setting, "model": name, "fold": k,
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
    from .data import load_matchups

    cfg = load_config()
    n_jobs = cfg["n_jobs"] if n_jobs is None else n_jobs
    (path("models") / "stage2").mkdir(parents=True, exist_ok=True)
    df = load_matchups()
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
        counts = g.groupby("model").size()
        assert counts.nunique() == 1, f"unequal prediction counts in {setting}"
        assert not g.duplicated(["model", "station_id", "t"]).any(), setting

    metrics, per_station = evaluate(pred)
    folds = fold_metrics(pred)
    imp_cols = [c for c in runs if c.startswith("imp_")]
    importance = (runs.dropna(subset=imp_cols).groupby(["setting", "model"])[imp_cols]
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
    t = m.pivot_table(index=["setting", "scope", "model"], columns="metric", values="value",
                      aggfunc="first", sort=False)
    n = metrics.groupby(["setting", "scope", "model"], sort=False)[["n_stations", "n"]].first()
    return t[[k for k in keep if k in t.columns]].join(n)
