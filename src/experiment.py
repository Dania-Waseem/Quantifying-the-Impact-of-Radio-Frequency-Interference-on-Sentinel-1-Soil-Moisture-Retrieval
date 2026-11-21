"""Stage 3: RFI injection experiment over variants M0 to M4.

For every evaluation (USCRN cross-validation, OZNET cross-validation, OZNET
transfer), fold, feature set and model, the clean Stage 2 model is applied
to test inputs that have been contaminated with RFI and, depending on the
variant, mitigated:

  * M0: clean input (upper bound);
  * M1: contaminated input, no mitigation;
  * M2: published baseline suppression (placeholder, not available);
  * M3: detect-and-fill;
  * M3b: detect-and-discard (``M0_kept`` is M0 on the same retained rows, so
    that M3b can be compared on equal rows);
  * M4: detect-and-fill input with an RFI-aware model, trained on the clean
    training fold plus an injected, filled copy of it.

Injection settings are all combinations of severity level (each level has
its own SIR for VV and for VH), extent, p_c and polarisation from
:func:`src.rfi.injection_grid`, each repeated with ``n_seeds`` seeds. Injection is applied to test inputs only (and, with
different seeds, to the M4 training copies).

Per run, the sufficient statistics of the errors are stored (count and sums
of errors, squares and cross-products), so that metrics pooled over folds
can be computed exactly without keeping every prediction.
"""

from __future__ import annotations

import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import rfi
from .config import derive_seed, load_config, path, rng, write_version_log
from .data import add_derived_features, load_matchups
from .metrics import recovery_ratio
from .models import (CD_FEATURE_SET, CD_MODELS, TREE_MODELS, feature_sets, features,
                     make_model, predict_model, rf_predict, station_folds)
from .stats import metrics_from_sums

SUM_COLS = ["n", "e", "e2", "ae", "y", "y2", "p", "p2", "yp"]
# Gzip with a fixed header time, so that reruns write identical files.
GZIP = {"method": "gzip", "mtime": 0}
KEY_SETTING = ["evaluation", "severity_q", "extent", "p_c", "polarisation", "seed"]
VARIANTS_RUN = ["M1", "M3", "M3b", "M0_kept", "M4"]


# --------------------------------------------------------------------------
# Splits (identical to Stage 2)
# --------------------------------------------------------------------------

def evaluation_splits(df: pd.DataFrame) -> list[tuple[str, int, pd.DataFrame, pd.DataFrame]]:
    """Recreate the Stage 2 train and test splits for the Stage 3 evaluations.

    The subsets, their row order and the fold seeds are the same as in
    :func:`src.models.build_tasks`, so the saved Stage 2 models belong to
    exactly these splits.
    """
    cfg = load_config()
    n_folds = cfg["cv"]["n_folds"]
    main, second = cfg["samples"]["main"], cfg["samples"]["second"]
    out = []
    for ev in cfg["experiment"]["evaluations"]:
        if ev.endswith("_transfer"):
            out.append((ev, 0, df[df["network"] == main], df[df["network"] == second]))
            continue
        net = ev.replace("_cv", "")
        sub = df[df["network"] == net].reset_index(drop=True)
        fold = station_folds(sub, n_folds, ev)
        for k in range(n_folds):
            out.append((ev, k, sub[fold != k], sub[fold == k]))
    return out


def model_file(kind: str, ev: str, fset: str, name: str, k: int):
    """Path of a saved model (kind 'stage2' or 'stage3_m4')."""
    return path("models") / kind / f"{ev}_{fset}_{name}_fold{k}.joblib"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def error_sums(y: np.ndarray, p: np.ndarray) -> dict:
    """Sufficient statistics of the errors e = p - y (see ``SUM_COLS``)."""
    e = p - y
    return {"n": len(y), "e": e.sum(), "e2": (e ** 2).sum(), "ae": np.abs(e).sum(),
            "y": y.sum(), "y2": (y ** 2).sum(), "p": p.sum(), "p2": (p ** 2).sum(),
            "yp": (y * p).sum()}


def backscatter_error(x: pd.DataFrame) -> dict:
    """Sum of absolute backscatter errors against the clean values (dB)."""
    return {f"bs_abs_{p}": float(np.abs(x[p] - x[f"{p}_clean"]).sum()) for p in rfi.POLS}


def mitigate(obs: pd.DataFrame, tau: float | None = None) -> dict[str, pd.DataFrame]:
    """Return the model inputs of M1, M3 and M3b for an observed series."""
    flags = rfi.detect(obs, tau=tau)
    return {"flags": flags,
            "M1": add_derived_features(obs),
            "M3": add_derived_features(rfi.fill(obs, flags)),
            "M3b": add_derived_features(rfi.discard(obs, flags))}


# --------------------------------------------------------------------------
# RFI-aware training (M4)
# --------------------------------------------------------------------------

def augmented_training(train: pd.DataFrame, ev: str, k: int) -> pd.DataFrame:
    """Return an injected and filled copy of a training fold.

    Each station gets one setting (severity level, extent, p_c,
    polarisation) drawn uniformly from the injection grid; each polarisation
    is injected with the SIR of the drawn severity level. The seed stream differs from all
    test-injection seeds (it includes the configured seed offset and a
    separate label).
    """
    grid = rfi.injection_grid(verbose=False)
    offset = load_config()["rfi_aware_training"]["seed_offset"]
    g = rng("train_injection", offset, ev, k)
    stations = np.sort(train["station_id"].unique())
    draw = pd.DataFrame({
        "station_id": stations,
        "sev": g.choice(grid["severity_q"], len(stations)),
        "rho": g.choice(grid["extent"], len(stations)),
        "pc": g.choice(grid["p_c"], len(stations)),
        "pol": g.choice(np.array(grid["polarisation"], dtype=object), len(stations))})
    d = train[["station_id"]].merge(draw, on="station_id", how="left")
    sir = {p: d["sev"].map(lambda q: grid["sir"][q][p]).to_numpy() for p in rfi.POLS}
    obs = rfi.inject(train, sir, d["rho"].to_numpy(), d["pc"].to_numpy(),
                     d["pol"].to_numpy(), g)
    filled = rfi.fill(obs, rfi.detect(obs))
    return add_derived_features(filled)


def _train_m4_task(ev: str, k: int, train: pd.DataFrame) -> list[str]:
    """Fit and save the M4 models of one split for all feature sets."""
    lam = load_config()["rfi_aware_training"]["mix_lambda"]
    clean = add_derived_features(train)
    aug = augmented_training(train, ev, k)
    y = np.r_[clean["sm"].to_numpy(), aug["sm"].to_numpy()]
    w = np.r_[np.full(len(clean), 1 - lam), np.full(len(aug), lam)]
    done = []
    for fset, name in product(feature_sets(), TREE_MODELS):
        cols = features(fset)
        X = np.vstack([clean[cols].to_numpy(), aug[cols].to_numpy()])
        model = make_model(name, derive_seed("stage3_m4", ev, name, k))
        model.fit(X, y, sample_weight=w)
        f = model_file("stage3_m4", ev, fset, name, k)
        joblib.dump(model, f)
        done.append(f.name)
    return done


def train_m4(splits: list, n_jobs: int) -> None:
    """Fit the RFI-aware (M4) models for every split, in parallel."""
    (path("models") / "stage3_m4").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_train_m4_task)(ev, k, tr) for ev, k, tr, _ in splits)
    print(f"M4 models fitted in {time.time() - t0:.0f} s")


# --------------------------------------------------------------------------
# Experiment task
# --------------------------------------------------------------------------

def _load_models(ev: str, k: int) -> dict:
    """Load the Stage 2 and M4 models of one split."""
    models = {}
    for fset, name in product(feature_sets(), TREE_MODELS):
        models[(fset, name, "clean")] = joblib.load(model_file("stage2", ev, fset, name, k))
        models[(fset, name, "m4")] = joblib.load(model_file("stage3_m4", ev, fset, name, k))
    for name in CD_MODELS:
        models[(CD_FEATURE_SET, name, "clean")] = joblib.load(
            model_file("stage2", ev, CD_FEATURE_SET, name, k))
    return models


def _predict(models: dict, fset: str, name: str, kind: str, x: pd.DataFrame) -> np.ndarray:
    """Predict with a loaded model on derived inputs."""
    return predict_model(name, models[(fset, name, kind)], x,
                         None if name in CD_MODELS else fset)


class CachedPredictor:
    """Exact prediction with reuse of the predictions on clean input.

    A tree model's prediction for a row depends only on that row's features.
    Rows whose feature values are identical to the clean input (rows that
    were neither contaminated nor changed by mitigation) therefore get the
    cached prediction on clean input, and the model is only applied to the
    rows that changed. The result is identical to predicting every row; it
    only saves time. Change-detection models depend on the whole series of
    a station and are always applied in full.
    """

    def __init__(self, models: dict, clean: pd.DataFrame):
        self.models, self.clean = models, clean
        self.X0 = {f: clean[features(f)].to_numpy(float) for f in feature_sets()}
        self.p0: dict = {}

    def clean_prediction(self, fset: str, name: str, kind: str) -> np.ndarray:
        """Prediction of a model on the clean input (computed once)."""
        key = (fset, name, kind)
        if key not in self.p0:
            self.p0[key] = _predict(self.models, fset, name, kind, self.clean)
        return self.p0[key]

    def __call__(self, fset: str, name: str, kind: str, x: pd.DataFrame) -> np.ndarray:
        if name in CD_MODELS:
            return _predict(self.models, fset, name, kind, x)
        pos = self.clean.index.get_indexer(x.index)
        X = x[features(fset)].to_numpy(float)
        same = (X == self.X0[fset][pos]).all(axis=1)
        out = self.clean_prediction(fset, name, kind)[pos].copy()
        if (~same).any():
            m = self.models[(fset, name, kind)]
            out[~same] = rf_predict(m, X[~same]) if name == "rf" else m.predict(X[~same])
        return out


def _experiment_task(ev: str, k: int, test: pd.DataFrame, q,
                     grid: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Run every setting with one severity level on one split.

    Returns the run rows (error sums per feature set, model and variant),
    the M0 rows (only for the first severity level, as M0 does not depend on the
    injection) and the detection rows.
    """
    n_seeds = load_config()["n_seeds"]
    models = _load_models(ev, k)
    pairs = [(f, n) for f in feature_sets() for n in TREE_MODELS] + \
            [(CD_FEATURE_SET, n) for n in CD_MODELS]
    clean = add_derived_features(test)
    y_all = clean["sm"].to_numpy()
    cp = CachedPredictor(models, clean)
    p0 = {pr: pd.Series(cp(*pr, "clean", clean), index=clean.index) for pr in pairs}

    m0_rows = []
    if q == grid["severity_q"][0]:
        for fset, name in pairs:
            m0_rows.append({"evaluation": ev, "fold": k, "feature_set": fset, "model": name,
                            "variant": "M0", **error_sums(y_all, p0[(fset, name)].to_numpy())})

    runs, dets = [], []
    for rho, pc, pol, s in product(grid["extent"], grid["p_c"], grid["polarisation"],
                                   range(n_seeds)):
        g = rng("test_injection", ev, k, q, rho, pc, pol, s)
        obs = rfi.inject(test, grid["sir"][q], rho, pc, pol, g)
        inp = mitigate(obs)
        key = {"evaluation": ev, "fold": k, "severity_q": q, "extent": rho, "p_c": pc,
               "polarisation": pol, "seed": s}
        n_obs, n_kept = len(obs), len(inp["M3b"])
        hit = obs["inj_VV"] | obs["inj_VH"]
        dets.append({**key, **rfi.detection_counts(obs, inp["flags"]), "n_obs": n_obs,
                     "n_kept": n_kept, "n_contaminated": int(hit.sum()),
                     "eps_sum_VV": float(obs["eps_VV"].sum()),
                     "eps_sum_VH": float(obs["eps_VH"].sum())})
        bs = {v: backscatter_error(inp[v]) for v in ("M1", "M3", "M3b")}
        y_kept = inp["M3b"]["sm"].to_numpy()
        for fset, name in pairs:
            preds = {v: cp(fset, name, "clean", inp[v]) for v in ("M1", "M3", "M3b")}
            preds["M0_kept"] = p0[(fset, name)].loc[inp["M3b"].index].to_numpy()
            if name in TREE_MODELS:
                preds["M4"] = cp(fset, name, "m4", inp["M3"])
            for v, p in preds.items():
                y = y_kept if v in ("M3b", "M0_kept") else y_all
                row = {**key, "feature_set": fset, "model": name, "variant": v,
                       **error_sums(y, p)}
                row.update(bs.get("M3" if v == "M4" else v, {}))
                runs.append(row)
    return runs, m0_rows, dets


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add RMSE, MAE, bias, ubRMSD, R2 and R computed from the error sums."""
    m = metrics_from_sums(df[SUM_COLS].to_numpy(float))
    out = df.copy()
    for k, v in m.items():
        out[k] = v
    return out


def pool_folds(runs: pd.DataFrame, m0: pd.DataFrame) -> pd.DataFrame:
    """Sum the error statistics over folds and compute pooled metrics.

    Returns one row per evaluation, setting, seed, feature set, model and
    variant, including M0 (repeated for every setting so that the
    degradation can be computed row by row).
    """
    keys = KEY_SETTING + ["feature_set", "model", "variant"]
    extra = [c for c in runs.columns if c.startswith("bs_abs_")]
    pooled = runs.groupby(keys, as_index=False)[SUM_COLS + extra].sum()
    m0p = m0.groupby(["evaluation", "feature_set", "model"], as_index=False)[SUM_COLS].sum()
    settings = pooled[KEY_SETTING + ["feature_set", "model"]].drop_duplicates()
    m0_rep = settings.merge(m0p, on=["evaluation", "feature_set", "model"])
    m0_rep["variant"] = "M0"
    return add_metrics(pd.concat([pooled, m0_rep], ignore_index=True))


def derived_quantities(pooled: pd.DataFrame) -> pd.DataFrame:
    """Degradation D, recovery G and recovery ratio RR per pooled run.

    D = RMSE_M1 - RMSE_M0 and, for each mitigation variant,
    G = RMSE_M1 - RMSE_variant and RR = G / D (NaN when D is below the
    configured minimum). For M3b, which is evaluated on fewer rows, the
    comparison on equal rows is also given: D_kept uses M0 on the retained
    rows (``M0_kept``) instead of M0.
    """
    min_d = load_config()["stats"]["min_degradation"]
    keys = KEY_SETTING + ["feature_set", "model"]
    w = pooled.pivot_table(index=keys, columns="variant", values="rmse").reset_index()
    rows = []
    for v in ("M3", "M3b", "M4"):
        if v not in w:
            continue
        d = w["M1"] - w["M0"]
        g = w["M1"] - w[v]
        rr = [recovery_ratio(a, b, c, min_d) for a, b, c in zip(w["M0"], w["M1"], w[v])]
        part = w[keys].copy()
        part["variant"] = v
        part["rmse_M0"], part["rmse_M1"], part["rmse_variant"] = w["M0"], w["M1"], w[v]
        part["D"], part["G"], part["RR"] = d, g, rr
        if v == "M3b":
            part["rmse_M0_kept"] = w["M0_kept"]
            part["gap_to_M0_kept"] = w[v] - w["M0_kept"]
        rows.append(part.dropna(subset=["rmse_variant"]))
    out = pd.concat(rows, ignore_index=True)
    # M2 is reserved: add explicit rows with missing values and a note.
    m2 = out[out["variant"] == "M3"].copy()
    m2[["rmse_variant", "G", "RR"]] = np.nan
    m2["variant"] = "M2"
    out = pd.concat([out, m2], ignore_index=True)
    out["note"] = np.where(out["variant"] == "M2", "not available", "")
    return out


def summarise(derived: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and spread over seeds (and any factors not in ``by``).

    Reports the mean, the standard deviation, a 95% interval of the mean
    across the repetitions (t-based), the number of repetitions, and the
    recovery ratio as the ratio of mean recovery to mean degradation (more
    stable than the mean of per-run ratios when D is small).
    """
    from scipy import stats as sps

    def ci(x):
        x = x.dropna()
        if len(x) < 2:
            return pd.Series({"ci_low": np.nan, "ci_high": np.nan})
        h = sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        return pd.Series({"ci_low": x.mean() - h, "ci_high": x.mean() + h})

    g = derived.groupby(by, dropna=False)
    out = g.agg(D_mean=("D", "mean"), D_sd=("D", "std"), G_mean=("G", "mean"),
                G_sd=("G", "std"), RR_median=("RR", "median"),
                rmse_M0=("rmse_M0", "mean"), rmse_M1=("rmse_M1", "mean"),
                rmse_variant=("rmse_variant", "mean"), runs=("D", "size")).reset_index()
    out["RR_of_means"] = np.where(out["D_mean"] >= load_config()["stats"]["min_degradation"],
                                  out["G_mean"] / out["D_mean"], np.nan)
    dci = g["D"].apply(ci).unstack().add_prefix("D_").reset_index()
    gci = g["G"].apply(ci).unstack().add_prefix("G_").reset_index()
    return out.merge(dci, on=by).merge(gci, on=by)


def detection_summary(det: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Precision, recall and F1 per polarisation, pooled over folds.

    Counts are summed over folds per seed; the metrics are then averaged
    over seeds (and over factors not in ``by``), with their standard
    deviation. Also reports the share of acquisitions lost by discarding.
    """
    keys = KEY_SETTING
    cnt = [c for c in det.columns if c[:3] in ("tp_", "fp_", "fn_", "tn_")]
    pooled = det.groupby(keys, as_index=False)[cnt + ["n_obs", "n_kept",
                                                      "n_contaminated"]].sum()
    for p in rfi.POLS:
        pr, rc, f1 = rfi.precision_recall_f1(pooled[f"tp_{p}"], pooled[f"fp_{p}"],
                                             pooled[f"fn_{p}"])
        pooled[f"precision_{p}"], pooled[f"recall_{p}"], pooled[f"f1_{p}"] = pr, rc, f1
        pooled[f"false_alarm_rate_{p}"] = pooled[f"fp_{p}"] / (pooled[f"fp_{p}"]
                                                              + pooled[f"tn_{p}"])
    pooled["share_lost"] = 1 - pooled["n_kept"] / pooled["n_obs"]
    vals = [f"{m}_{p}" for m in ("precision", "recall", "f1", "false_alarm_rate")
            for p in rfi.POLS] + ["share_lost"]
    g = pooled.groupby(by)[vals]
    return g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()


# --------------------------------------------------------------------------
# Stage 3 runner
# --------------------------------------------------------------------------

def run_stage3(n_jobs: int | None = None) -> dict[str, pd.DataFrame]:
    """Run the Stage 3 experiment and save all outputs.

    Outputs (``results/`` files other than ``aggregate`` are not tracked):
      * ``results/experiment_runs.csv.gz``: one row per evaluation, fold,
        setting, seed, feature set, model and variant, with metrics;
      * ``results/experiment_pooled.csv.gz``: the same pooled over folds;
      * ``results/experiment_detection.csv.gz``: detection counts per run;
      * ``results/aggregate/stage3_*.csv``: summaries over seeds.
    """
    cfg = load_config()
    n_jobs = cfg["experiment"]["n_jobs"] if n_jobs is None else n_jobs
    grid = rfi.injection_grid()
    df = load_matchups()
    splits = evaluation_splits(df)
    print(f"Stage 3: {len(splits)} splits, feature sets {feature_sets()}, grid source "
          f"{grid['source']}, {n_jobs} workers")

    train_m4(splits, n_jobs)

    t0 = time.time()
    tasks = [(ev, k, te, q) for ev, k, _, te in splits for q in grid["severity_q"]]
    res = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=5)(
        delayed(_experiment_task)(ev, k, te, q, grid) for ev, k, te, q in tasks)
    print(f"experiment tasks done in {time.time() - t0:.0f} s")

    runs = pd.DataFrame([r for t in res for r in t[0]])
    m0 = pd.DataFrame([r for t in res for r in t[1]])
    det = pd.DataFrame([r for t in res for r in t[2]])
    checks(runs, m0, det, splits, grid)

    runs_m = add_metrics(pd.concat([runs, m0], ignore_index=True))
    pooled = pool_folds(runs, m0)
    derived = derived_quantities(pooled)
    det_sum = {
        "stage3_detection_by_setting": detection_summary(
            det, ["evaluation", "severity_q", "extent", "p_c", "polarisation"]),
        "stage3_detection_by_severity": detection_summary(det, ["evaluation", "severity_q"]),
        "stage3_detection_by_extent": detection_summary(det, ["evaluation", "extent"]),
    }
    base = ["evaluation", "feature_set", "model", "variant"]
    summaries = {
        "stage3_summary_by_setting": summarise(
            derived, base + ["severity_q", "extent", "p_c", "polarisation"]),
        "stage3_summary_by_severity": summarise(derived, base + ["severity_q"]),
        "stage3_summary_by_extent": summarise(derived, base + ["extent"]),
        "stage3_summary_by_pc": summarise(derived, base + ["p_c"]),
        "stage3_summary_by_polarisation": summarise(derived, base + ["polarisation"]),
        "stage3_summary_overall": summarise(derived, base),
    }
    share_lost = (det.groupby(KEY_SETTING)[["n_obs", "n_kept"]].sum()
                  .assign(share_lost=lambda d: 1 - d.n_kept / d.n_obs).reset_index())

    runs_m.to_csv(path("results") / "experiment_runs.csv.gz", index=False, compression=GZIP)
    pooled.to_csv(path("results") / "experiment_pooled.csv.gz", index=False, compression=GZIP)
    derived.to_csv(path("results") / "experiment_derived.csv.gz", index=False, compression=GZIP)
    det.to_csv(path("results") / "experiment_detection.csv.gz", index=False, compression=GZIP)
    share_lost.to_csv(path("results") / "experiment_share_lost.csv.gz", index=False, compression=GZIP)
    write_version_log(path("results") / "experiment_runs.csv")
    for name, table in {**summaries, **det_sum}.items():
        table.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "stage3_summary_overall.csv")
    gt = rfi.grid_table(grid)
    gt["notes"] = grid["notes"]
    gt.to_csv(path("results_aggregate") / "stage3_grid.csv", index=False)
    return {**summaries, **det_sum, "derived": derived, "pooled": pooled}


def checks(runs: pd.DataFrame, m0: pd.DataFrame, det: pd.DataFrame, splits: list,
           grid: dict) -> None:
    """Consistency checks on the raw experiment output.

    * every expected setting and seed is present exactly once per split;
    * M0 on each split equals the Stage 2 out-of-fold predictions (the saved
      models are the ones of these splits);
    * no test station of a split is in its training part.
    """
    n_set = (len(grid["severity_q"]) * len(grid["extent"]) * len(grid["p_c"])
             * len(grid["polarisation"]) * load_config()["n_seeds"])
    per_split = det.groupby(["evaluation", "fold"]).size()
    assert (per_split == n_set).all(), "missing or duplicated settings"
    assert not det.duplicated(["evaluation", "fold", "severity_q", "extent", "p_c",
                               "polarisation", "seed"]).any()
    for ev, k, tr, te in splits:
        assert not set(tr["station_id"]) & set(te["station_id"]), "test station in training"
    s2 = pd.read_csv(path("processed") / "stage2_predictions.csv")
    s2 = s2[s2["setting"].isin(m0["evaluation"].unique())]
    s2["e"] = s2["pred"] - s2["sm"]
    ref = (s2.groupby(["setting", "fold", "feature_set", "model"])["e"].apply(
        lambda e: (e ** 2).sum()).rename("e2_ref").reset_index()
        .rename(columns={"setting": "evaluation"}))
    chk = m0.merge(ref, on=["evaluation", "fold", "feature_set", "model"])
    assert len(chk) == len(m0), "M0 rows without Stage 2 reference"
    assert np.allclose(chk["e2"], chk["e2_ref"], rtol=1e-9), "M0 differs from Stage 2"
    print(f"checks passed: {len(det)} settings x splits, M0 equals Stage 2 on "
          f"{len(chk)} split-model pairs")


# --------------------------------------------------------------------------
# Stage 4 recomputation: per-station and row-level errors
# --------------------------------------------------------------------------

S4_VARIANTS = ["M0", "M1", "M3", "M3b", "M0_kept", "M4", "M4_nofill"]


def stage4_scenarios() -> pd.DataFrame:
    """List the Stage 4 scenarios declared in ``config.yaml``.

    Each scenario is a setting (severity level, extent, p_c, polarisation)
    plus a detection threshold ``tau``. A scenario can serve several roles (for
    example the reference is also one point of each sweep); it is computed
    once. ``row_level`` marks the scenarios whose row-level errors are kept
    (reference, severity sweep and polarisation check), used for mediation
    and moderation.
    """
    cfg = load_config()
    s4, grid = cfg["stage4"], rfi.injection_grid(verbose=False)
    tau0 = cfg["detection"]["tau"]
    ref = s4["reference"]
    base = (ref["severity_q"], rfi.extent_value(grid, ref["extent_q"]), ref["p_c"],
            ref["polarisation"], tau0)
    items: dict[tuple, set] = {base: {"reference"}}

    def add(key, role):
        items.setdefault(key, set()).add(role)

    if s4["severity_sweep"]:
        for v in grid["severity_q"]:
            add((v, base[1], base[2], base[3], tau0), "severity_sweep")
    if s4["extent_sweep"]:
        for v in grid["extent"]:
            add((base[0], v, base[2], base[3], tau0), "extent_sweep")
    for p in s4["polarisation_check"]:
        add((base[0], base[1], base[2], p, tau0), f"pol_{p}")
    if s4["tau_check"]:
        for t in cfg["detection"]["tau_grid"]:
            add((base[0], base[1], base[2], base[3], t), "tau_check")
    rows = []
    for i, (key, roles) in enumerate(items.items()):
        rows.append({"scenario": i, "severity_q": key[0], "extent": key[1], "p_c": key[2],
                     "polarisation": key[3], "tau": key[4], "roles": ";".join(sorted(roles)),
                     "row_level": bool({"reference", "severity_sweep", "pol_VV",
                                        "pol_VH"} & roles)})
    return pd.DataFrame(rows)


def _station_sums(station: np.ndarray, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Per-station error sums (rows with missing predictions are skipped)."""
    ok = np.isfinite(p)
    codes, labels = pd.factorize(station[ok])
    e, yy, pp = p[ok] - y[ok], y[ok], p[ok]
    cols = {"n": np.ones_like(e), "e": e, "e2": e ** 2, "ae": np.abs(e), "y": yy,
            "y2": yy ** 2, "p": pp, "p2": pp ** 2, "yp": yy * pp}
    out = pd.DataFrame({k: np.bincount(codes, weights=v, minlength=len(labels))
                        for k, v in cols.items()})
    out.insert(0, "station_id", labels)
    return out


def _stage4_task(ev: str, k: int, test: pd.DataFrame,
                 scen: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Recompute the Stage 4 scenarios on one split.

    Returns per-station error sums for every scenario, seed, feature set,
    model and variant, and row-level errors for the row-level scenarios.
    """
    n_seeds = load_config()["n_seeds"]
    grid = rfi.injection_grid(verbose=False)
    models = _load_models(ev, k)
    pairs = [(f, n) for f in feature_sets() for n in TREE_MODELS] + \
            [(CD_FEATURE_SET, n) for n in CD_MODELS]
    clean = add_derived_features(test)
    y = clean["sm"].to_numpy()
    st = clean["station_id"].to_numpy()
    cp = CachedPredictor(models, clean)
    p0 = {pr: cp(*pr, "clean", clean) for pr in pairs}
    sums, rows = [], []
    for sc in scen.itertuples():
        for s in range(n_seeds):
            g = rng("test_injection", ev, k, sc.severity_q, sc.extent, sc.p_c,
                    sc.polarisation, s)
            obs = rfi.inject(test, grid["sir"][sc.severity_q], sc.extent, sc.p_c,
                             sc.polarisation, g)
            inp = mitigate(obs, tau=sc.tau)
            kept = clean.index.isin(inp["M3b"].index)
            preds = {}
            for fset, name in pairs:
                pr = {"M0": p0[(fset, name)],
                      "M1": cp(fset, name, "clean", inp["M1"]),
                      "M3": cp(fset, name, "clean", inp["M3"])}
                full = np.full(len(clean), np.nan)
                full[kept] = cp(fset, name, "clean", inp["M3b"])
                pr["M3b"] = full
                pr["M0_kept"] = np.where(kept, pr["M0"], np.nan)
                if name in TREE_MODELS:
                    pr["M4"] = cp(fset, name, "m4", inp["M3"])
                    pr["M4_nofill"] = cp(fset, name, "m4", inp["M1"])
                for v, p in pr.items():
                    ss = _station_sums(st, y, p)
                    ss.insert(1, "variant", v)
                    ss.insert(1, "model", name)
                    ss.insert(1, "feature_set", fset)
                    preds[f"{fset}|{name}|{v}"] = p
                    sums.append(ss.assign(evaluation=ev, fold=k, scenario=sc.scenario, seed=s))
            if sc.row_level:
                r = clean[["network", "station_id", "t", "sm", "ndvi", "lc", "vh_low"]].copy()
                r = r.assign(evaluation=ev, fold=k, scenario=sc.scenario, seed=s, kept=kept)
                for p in rfi.POLS:
                    r[f"inj_{p}"] = obs[f"inj_{p}"].to_numpy()
                    r[f"eps_{p}"] = obs[f"eps_{p}"].to_numpy()
                    for v in ("M1", "M3"):
                        r[f"bs_{v}_{p}"] = np.abs(inp[v][p] - inp[v][f"{p}_clean"]).to_numpy()
                for c, p in preds.items():
                    r[c] = p.astype("float32")
                rows.append(r)
    return sums, rows


def run_stage4_recompute(n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the Stage 4 scenarios on all splits and save them locally.

    Writes ``data/processed/stage4_station_sums.csv.gz`` and
    ``data/processed/stage4_rows.csv.gz`` (both private) and the scenario
    list to ``results/aggregate/stage4_scenarios.csv``. Checks that the
    reference scenario reproduces the Stage 3 pooled RMSE.
    """
    cfg = load_config()
    n_jobs = cfg["experiment"]["n_jobs"] if n_jobs is None else n_jobs
    rfi.injection_grid()      # prints the provisional-grid warning
    scen = stage4_scenarios()
    splits = evaluation_splits(load_matchups())
    print(f"Stage 4 recompute: {len(scen)} scenarios x {cfg['n_seeds']} seeds x "
          f"{len(splits)} splits")
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_stage4_task)(ev, k, te, scen) for ev, k, _, te in splits)
    print(f"done in {time.time() - t0:.0f} s")
    sums = pd.concat([s for r in res for s in r[0]], ignore_index=True)
    rows = pd.concat([x for r in res for x in r[1]], ignore_index=True)
    sums.to_csv(path("processed") / "stage4_station_sums.csv.gz", index=False, compression=GZIP)
    rows.to_csv(path("processed") / "stage4_rows.csv.gz", index=False, compression=GZIP)
    scen.to_csv(path("results_aggregate") / "stage4_scenarios.csv", index=False)

    # Consistency with Stage 3: pooled RMSE of the reference scenario.
    ref = scen[scen["roles"].str.contains("reference")].iloc[0]
    st3 = pd.read_csv(path("results") / "experiment_pooled.csv.gz")
    st3 = st3[(st3.severity_q == ref.severity_q) & (st3.extent == ref.extent)
              & (st3.p_c == ref.p_c)
              & (st3.polarisation == ref.polarisation) & st3.variant.isin(["M1", "M3", "M4"])]
    s4 = sums[sums.scenario == ref.scenario]
    s4 = s4.groupby(["evaluation", "seed", "feature_set", "model", "variant"])[SUM_COLS].sum()
    s4 = add_metrics(s4.reset_index())
    chk = st3.merge(s4, on=["evaluation", "seed", "feature_set", "model", "variant"],
                    suffixes=("_s3", "_s4"))
    assert len(chk) > 0 and np.allclose(chk.rmse_s3, chk.rmse_s4, rtol=1e-9), \
        "Stage 4 reference differs from Stage 3"
    print(f"reference scenario reproduces Stage 3 on {len(chk)} pooled runs")
    return sums, rows
