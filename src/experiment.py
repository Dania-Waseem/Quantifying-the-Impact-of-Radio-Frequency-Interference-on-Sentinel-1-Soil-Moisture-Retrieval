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

Injection settings are all combinations of SIR, extent, p_c and
polarisation from :func:`src.rfi.injection_grid`, each repeated with
``n_seeds`` seeds. Injection is applied to test inputs only (and, with
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
                     make_model, predict_model, station_folds)
from .stats import metrics_from_sums

SUM_COLS = ["n", "e", "e2", "ae", "y", "y2", "p", "p2", "yp"]
KEY_SETTING = ["evaluation", "sir_db", "extent", "p_c", "polarisation", "seed"]
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

    Each station gets one setting (SIR, extent, p_c, polarisation) drawn
    uniformly from the injection grid. The seed stream differs from all
    test-injection seeds (it includes the configured seed offset and a
    separate label).
    """
    grid = rfi.injection_grid(verbose=False)
    offset = load_config()["rfi_aware_training"]["seed_offset"]
    g = rng("train_injection", offset, ev, k)
    stations = np.sort(train["station_id"].unique())
    draw = pd.DataFrame({
        "station_id": stations,
        "sir": g.choice(grid["sir_db"], len(stations)),
        "rho": g.choice(grid["extent"], len(stations)),
        "pc": g.choice(grid["p_c"], len(stations)),
        "pol": g.choice(np.array(grid["polarisation"], dtype=object), len(stations))})
    d = train[["station_id"]].merge(draw, on="station_id", how="left")
    obs = rfi.inject(train, d["sir"].to_numpy(), d["rho"].to_numpy(), d["pc"].to_numpy(),
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


def _experiment_task(ev: str, k: int, test: pd.DataFrame, sir: float,
                     grid: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Run every setting with one SIR value on one split.

    Returns the run rows (error sums per feature set, model and variant),
    the M0 rows (only for the first SIR value, as M0 does not depend on the
    injection) and the detection rows.
    """
    n_seeds = load_config()["n_seeds"]
    models = _load_models(ev, k)
    pairs = [(f, n) for f in feature_sets() for n in TREE_MODELS] + \
            [(CD_FEATURE_SET, n) for n in CD_MODELS]
    clean = add_derived_features(test)
    y_all = clean["sm"].to_numpy()
    p0 = {pr: pd.Series(_predict(models, *pr, "clean", clean), index=clean.index)
          for pr in pairs}

    m0_rows = []
    if sir == grid["sir_db"][0]:
        for fset, name in pairs:
            m0_rows.append({"evaluation": ev, "fold": k, "feature_set": fset, "model": name,
                            "variant": "M0", **error_sums(y_all, p0[(fset, name)].to_numpy())})

    runs, dets = [], []
    for rho, pc, pol, s in product(grid["extent"], grid["p_c"], grid["polarisation"],
                                   range(n_seeds)):
        g = rng("test_injection", ev, k, sir, rho, pc, pol, s)
        obs = rfi.inject(test, sir, rho, pc, pol, g)
        inp = mitigate(obs)
        key = {"evaluation": ev, "fold": k, "sir_db": sir, "extent": rho, "p_c": pc,
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
            preds = {v: _predict(models, fset, name, "clean", inp[v]) for v in ("M1", "M3", "M3b")}
            preds["M0_kept"] = p0[(fset, name)].loc[inp["M3b"].index].to_numpy()
            if name in TREE_MODELS:
                preds["M4"] = _predict(models, fset, name, "m4", inp["M3"])
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
    tasks = [(ev, k, te, sir) for ev, k, _, te in splits for sir in grid["sir_db"]]
    res = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=5)(
        delayed(_experiment_task)(ev, k, te, sir, grid) for ev, k, te, sir in tasks)
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
            det, ["evaluation", "sir_db", "extent", "p_c", "polarisation"]),
        "stage3_detection_by_sir": detection_summary(det, ["evaluation", "sir_db"]),
        "stage3_detection_by_extent": detection_summary(det, ["evaluation", "extent"]),
    }
    base = ["evaluation", "feature_set", "model", "variant"]
    summaries = {
        "stage3_summary_by_setting": summarise(
            derived, base + ["sir_db", "extent", "p_c", "polarisation"]),
        "stage3_summary_by_sir": summarise(derived, base + ["sir_db"]),
        "stage3_summary_by_extent": summarise(derived, base + ["extent"]),
        "stage3_summary_by_pc": summarise(derived, base + ["p_c"]),
        "stage3_summary_by_polarisation": summarise(derived, base + ["polarisation"]),
        "stage3_summary_overall": summarise(derived, base),
    }
    share_lost = (det.groupby(KEY_SETTING)[["n_obs", "n_kept"]].sum()
                  .assign(share_lost=lambda d: 1 - d.n_kept / d.n_obs).reset_index())

    runs_m.to_csv(path("results") / "experiment_runs.csv.gz", index=False)
    pooled.to_csv(path("results") / "experiment_pooled.csv.gz", index=False)
    derived.to_csv(path("results") / "experiment_derived.csv.gz", index=False)
    det.to_csv(path("results") / "experiment_detection.csv.gz", index=False)
    share_lost.to_csv(path("results") / "experiment_share_lost.csv.gz", index=False)
    write_version_log(path("results") / "experiment_runs.csv")
    for name, table in {**summaries, **det_sum}.items():
        table.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "stage3_summary_overall.csv")
    (path("results_aggregate") / "stage3_grid.txt").write_text(
        f"source: {grid['source']}\n" + "\n".join(f"{k}: {v}" for k, v in grid.items()
                                                 if k != "source") + "\n",
        encoding="utf-8")
    return {**summaries, **det_sum, "derived": derived, "pooled": pooled}


def checks(runs: pd.DataFrame, m0: pd.DataFrame, det: pd.DataFrame, splits: list,
           grid: dict) -> None:
    """Consistency checks on the raw experiment output.

    * every expected setting and seed is present exactly once per split;
    * M0 on each split equals the Stage 2 out-of-fold predictions (the saved
      models are the ones of these splits);
    * no test station of a split is in its training part.
    """
    n_set = (len(grid["sir_db"]) * len(grid["extent"]) * len(grid["p_c"])
             * len(grid["polarisation"]) * load_config()["n_seeds"])
    per_split = det.groupby(["evaluation", "fold"]).size()
    assert (per_split == n_set).all(), "missing or duplicated settings"
    assert not det.duplicated(["evaluation", "fold", "sir_db", "extent", "p_c",
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
