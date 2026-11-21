"""Exploratory follow-up experiments (declared in ``config.yaml``, section
``followup``), run after the main analysis.

1. Oracle mitigation: fill, discard and the RFI-aware model with the TRUE
   injection flags instead of the detector (upper bound on recovery).
2. Persistent RFI: contamination of one contiguous block of each station's
   series, or of every acquisition of a fraction of the stations.
3. Stress test: severity and extent beyond the OpenSAR P90 (up to the
   largest observed values), with a degradation breakpoint.

Nothing here changes an earlier result: the outputs are written to
``results/aggregate/followup_*.csv`` (aggregated), to
``data/processed/followup_station_sums.csv.gz`` (local) and to figures
15 to 17. The tests form their own Holm family and are exploratory.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import rfi
from .config import load_config, path, rng, write_version_log
from .data import add_derived_features, load_matchups
from .experiment import (GZIP, SUM_COLS, CachedPredictor, _load_models, _station_sums,
                         evaluation_splits, mitigate)
from .models import CD_FEATURE_SET, CD_MODELS, TREE_MODELS, feature_sets
from .stats import (GROUP, _boot_p, _counts, _pct, metrics_from_sums, station_rmse,
                    station_table, wilcoxon_one_sided)

ORACLE_VARIANTS = ["M3_oracle", "M3b_oracle", "M0_kept_oracle", "M4_oracle"]


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

def stress_levels() -> tuple[dict, dict]:
    """SIR per polarisation and extent for the stress percentiles.

    Same rules as the main grid: severity percentiles per polarisation over
    the patches with contrast above the minimum; extent percentiles over all
    patches; 100 means the largest observed value.
    """
    cfg = load_config()
    st = cfg["followup"]["stress"]
    stats = rfi.load_opensar_stats()
    if stats is None:
        raise RuntimeError("the stress test needs data/processed/opensar_rfi_stats.csv")
    cmin = cfg["injection"]["opensar"]["min_contrast_db"]
    sev = {}
    for q in st["severity_pct"]:
        sev[q] = {}
        for p in rfi.POLS:
            c = stats[f"severity_db_{p.lower()}"]
            c = c[c > cmin]
            cq = float(np.percentile(c, q))
            sev[q][p] = {"contrast_db": cq, "sir": float(rfi.sir_from_contrast(cq))}
    ext = {q: float(np.percentile(stats["extent"], q)) for q in st["extent_pct"]}
    return sev, ext


def scenarios() -> pd.DataFrame:
    """All follow-up scenarios with their injection parameters."""
    cfg = load_config()["followup"]
    grid = rfi.injection_grid(verbose=False)
    rows = []

    def add(exp, label, sev_label, sir, rho, mode, frac, pol, seed_keys):
        rows.append({"experiment": exp, "label": label, "severity": sev_label,
                     "sir_VV": sir["VV"], "sir_VH": sir["VH"], "extent": rho, "mode": mode,
                     "fraction": frac, "polarisation": pol, "seed_keys": seed_keys})

    o = cfg["oracle"]
    for q, e in product(o["severity_q"], o["extent_q"]):
        rho = rfi.extent_value(grid, e)
        # Same seed keys as Stage 3, so M1 and M3 reproduce Stage 3 exactly.
        add("oracle", f"P{q} severity, P{e} extent", f"P{q}", grid["sir"][q], rho, "random",
            o["p_c"], o["polarisation"], ("test_injection", q, rho, o["p_c"], o["polarisation"]))
    p = cfg["persistent"]
    for mode, frac, q, e in product(p["modes"], p["fraction"], p["severity_q"], p["extent_q"]):
        rho = rfi.extent_value(grid, e)
        add("persistent", f"{mode} {frac}, P{q} severity, P{e} extent", f"P{q}",
            grid["sir"][q], rho, mode, frac, p["polarisation"],
            ("followup_persistent", mode, frac, q, e))
    s = cfg["stress"]
    sev, ext = stress_levels()
    for q, e, pc in product(s["severity_pct"], s["extent_pct"], s["p_c"]):
        add("stress", f"P{q} severity, P{e} extent, p_c {pc}", f"P{q}",
            {k: sev[q][k]["sir"] for k in rfi.POLS}, ext[e], "random", pc, s["polarisation"],
            ("followup_stress", q, e, pc))
    t = pd.DataFrame(rows)
    t.insert(0, "scenario", range(len(t)))
    t["eps_VV"] = rfi.eps_db(t["sir_VV"], t["extent"])
    t["eps_VH"] = rfi.eps_db(t["sir_VH"], t["extent"])
    t["eps_mean"] = (t["eps_VV"] + t["eps_VH"]) / 2
    return t


# --------------------------------------------------------------------------
# Injection with structured contamination
# --------------------------------------------------------------------------

def contamination_mask(df: pd.DataFrame, mode: str, frac: float,
                       g: np.random.Generator) -> np.ndarray:
    """Boolean mask of contaminated acquisitions (aligned with ``df``).

    ``block``: one contiguous run covering ``frac`` of each station's
    time-ordered series, with a random start; ``station``: all acquisitions
    of each station with probability ``frac``.
    """
    mask = np.zeros(len(df), bool)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for sid in np.sort(df["station_id"].unique()):
        idx = df.index[df["station_id"] == sid]
        order = pos[idx].to_numpy()[np.argsort(df.loc[idx, "t"].to_numpy())]
        n = len(order)
        if mode == "block":
            length = int(round(frac * n))
            start = int(g.integers(0, n - length + 1)) if length < n else 0
            mask[order[start:start + length]] = True
        elif mode == "station":
            if g.random() < frac:
                mask[order] = True
        else:
            raise ValueError(mode)
    return mask


def inject_mask(df: pd.DataFrame, sir: dict, extent: float, mask: np.ndarray,
                polarisation: str) -> pd.DataFrame:
    """Inject RFI into the acquisitions in ``mask`` (same model as rfi.inject)."""
    out = df.copy()
    for p in rfi.POLS:
        hit = mask & (polarisation in (p, "both"))
        sigma = 10 ** (out[p].to_numpy() / 10)
        factor = 1 + extent * 10 ** (-sir[p] / 10)
        out[f"{p}_clean"] = out[p]
        out[p] = np.where(hit, 10 * np.log10(sigma * factor), out[p].to_numpy())
        out[f"inj_{p}"] = hit
        out[f"eps_{p}"] = out[p] - out[f"{p}_clean"]
    return out


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------

def _task(ev: str, k: int, test: pd.DataFrame, scen: pd.DataFrame) -> tuple[list, list]:
    """Run the given scenarios on one split; return station sums and detection rows."""
    n_seeds = load_config()["followup"]["n_seeds"]
    models = _load_models(ev, k)
    clean = add_derived_features(test)
    cp = CachedPredictor(models, clean)
    y, st = clean["sm"].to_numpy(), clean["station_id"].to_numpy()
    pairs = [(f, n) for f in feature_sets() for n in TREE_MODELS] + \
            [(CD_FEATURE_SET, n) for n in CD_MODELS]
    p0 = {pr: cp(*pr, "clean", clean) for pr in pairs}
    sums, dets = [], []
    for sc in scen.itertuples():
        sir = {"VV": sc.sir_VV, "VH": sc.sir_VH}
        for s in range(n_seeds):
            keys = tuple(sc.seed_keys)
            if sc.mode == "random":
                # Stage 3 key order: label, split, then setting, then seed.
                g = rng(keys[0], ev, k, *keys[1:], s)
                obs = rfi.inject(test, sir, sc.extent, sc.fraction, sc.polarisation, g)
            else:
                g = rng(keys[0], ev, k, *keys[1:], s)
                obs = inject_mask(test, sir, sc.extent,
                                  contamination_mask(test, sc.mode, sc.fraction, g),
                                  sc.polarisation)
            inp = mitigate(obs)
            kept = clean.index.isin(inp["M3b"].index)
            oracle = sc.experiment == "oracle"
            if oracle:
                truth = pd.DataFrame({"flag_VV": obs["inj_VV"].to_numpy(),
                                      "flag_VH": obs["inj_VH"].to_numpy()}, index=obs.index)
                o_fill = add_derived_features(rfi.fill(obs, truth))
                o_disc = add_derived_features(rfi.discard(obs, truth))
                o_kept = clean.index.isin(o_disc.index)
            dets.append({"evaluation": ev, "fold": k, "scenario": sc.scenario, "seed": s,
                         **rfi.detection_counts(obs, inp["flags"]),
                         "n_obs": len(obs), "n_kept": int(kept.sum()),
                         "n_contaminated": int((obs["inj_VV"] | obs["inj_VH"]).sum())})
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
                if oracle:
                    pr["M3_oracle"] = cp(fset, name, "clean", o_fill)
                    full = np.full(len(clean), np.nan)
                    full[o_kept] = cp(fset, name, "clean", o_disc)
                    pr["M3b_oracle"] = full
                    pr["M0_kept_oracle"] = np.where(o_kept, pr["M0"], np.nan)
                    if name in TREE_MODELS:
                        pr["M4_oracle"] = cp(fset, name, "m4", o_fill)
                for v, p in pr.items():
                    ss = _station_sums(st, y, p)
                    ss.insert(1, "variant", v)
                    ss.insert(1, "model", name)
                    ss.insert(1, "feature_set", fset)
                    sums.append(ss.assign(evaluation=ev, fold=k, scenario=sc.scenario, seed=s))
    return sums, dets


# --------------------------------------------------------------------------
# Aggregation and statistics
# --------------------------------------------------------------------------

def per_seed_summary(sums: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE per seed, then D, G and RR, summarised over seeds."""
    min_d = load_config()["stats"]["min_degradation"]
    keys = ["evaluation", "scenario", "seed", "feature_set", "model", "variant"]
    p = sums.groupby(keys, as_index=False)[SUM_COLS].sum()
    p["rmse"] = metrics_from_sums(p[SUM_COLS].to_numpy(float))["rmse"]
    w = p.pivot_table(index=keys[:-1], columns="variant", values="rmse").reset_index()
    rows = []
    for v in [c for c in w.columns if c.startswith(("M3", "M4"))]:
        d = w["M1"] - w["M0"]
        g = w["M1"] - w[v]
        part = w[keys[:-1]].assign(variant=v, rmse_M0=w["M0"], rmse_M1=w["M1"],
                                   rmse_variant=w[v], D=d, G=g)
        part["RR"] = np.where(d >= min_d, g / d.where(d != 0), np.nan)
        rows.append(part.dropna(subset=["rmse_variant"]))
    der = pd.concat(rows, ignore_index=True)
    g = der.groupby(["evaluation", "scenario", "feature_set", "model", "variant"])
    out = g.agg(rmse_M0=("rmse_M0", "mean"), rmse_M1=("rmse_M1", "mean"),
                rmse_variant=("rmse_variant", "mean"), D_mean=("D", "mean"),
                D_sd=("D", "std"), G_mean=("G", "mean"), G_sd=("G", "std"),
                RR_median=("RR", "median"), seeds=("D", "size")).reset_index()
    out["D_rel"] = out["D_mean"] / out["rmse_M0"]
    out["RR_of_means"] = np.where(out["D_mean"] >= min_d, out["G_mean"] / out["D_mean"], np.nan)
    return out.merge(scen.drop(columns="seed_keys"), on="scenario")


def bootstrap_scenario(sums: pd.DataFrame, scenario: int) -> pd.DataFrame:
    """Cluster bootstrap over stations of RMSE, D, G and RR for one scenario."""
    min_d = load_config()["stats"]["min_degradation"]
    tab = station_table(sums, scenario)
    rows = []
    for key, g in tab.groupby(GROUP):
        stations = np.sort(g["station_id"].unique())
        arr = {v: gg.set_index("station_id").reindex(stations)[SUM_COLS].fillna(0).to_numpy()
               for v, gg in g.groupby("variant")}
        c = _counts(len(stations), "followup", scenario, *key)
        pt = {v: metrics_from_sums(a.sum(0))["rmse"] for v, a in arr.items()}
        bt = {v: metrics_from_sums(c @ a)["rmse"] for v, a in arr.items()}
        d_pt, d_b = pt["M1"] - pt["M0"], bt["M1"] - bt["M0"]

        def rec(variant, quantity, est, b, p=np.nan):
            lo, hi = _pct(b)
            rows.append({**dict(zip(GROUP, key)), "scenario": scenario, "variant": variant,
                         "quantity": quantity, "estimate": float(est), "ci_low": lo,
                         "ci_high": hi, "p_boot": p, "n_stations": len(stations)})
        for v in arr:
            rec(v, "rmse", pt[v], bt[v])
        rec("M1", "D", d_pt, d_b, _boot_p(d_b))
        for v in [x for x in arr if x.startswith(("M3", "M4"))]:
            g_pt, g_b = pt["M1"] - pt[v], bt["M1"] - bt[v]
            rec(v, "G", g_pt, g_b, _boot_p(g_b))
            rr_b = np.where(d_b >= min_d, g_b / np.where(d_b == 0, np.nan, d_b), np.nan)
            rec(v, "RR", g_pt / d_pt if d_pt >= min_d else np.nan, rr_b)
    return pd.DataFrame(rows)


def tests(sums: pd.DataFrame, scen: pd.DataFrame) -> pd.DataFrame:
    """Exploratory one-sided Wilcoxon tests with a Holm correction per family."""
    from statsmodels.stats.multitest import multipletests

    alpha = load_config()["stats"]["alpha"]
    cfg = load_config()["followup"]
    plan = []
    o = scen[scen.experiment == "oracle"]
    for q, e in ((50, 50), (90, 90)):
        sid = o[o.label == f"P{q} severity, P{e} extent"]["scenario"].iloc[0]
        plan += [(sid, "M1", "M3_oracle", f"oracle fill recovers (P{q}/P{e})"),
                 (sid, "M1", "M3b_oracle", f"oracle discard recovers (P{q}/P{e})")]
    pe = scen[(scen.experiment == "persistent") & (scen.severity == "P90")
              & (scen.extent == scen[scen.experiment == "persistent"]["extent"].max())]
    for r in pe.itertuples():
        plan += [(r.scenario, "M1", "M0", f"degradation, {r.mode} {r.fraction}"),
                 (r.scenario, "M1", "M3", f"detector fill recovers, {r.mode} {r.fraction}")]
    sx = scen[scen.experiment == "stress"].sort_values("eps_mean").iloc[-1]
    plan.append((sx.scenario, "M1", "M0", f"degradation at strongest stress ({sx.label})"))

    rows = []
    srm = {sid: station_rmse(station_table(sums, sid)) for sid in {p[0] for p in plan}}
    for key in srm[plan[0][0]][GROUP].drop_duplicates().itertuples(index=False):
        fam = []
        for sid, a, b, desc in plan:
            t = srm[sid]
            t = t[(t[GROUP] == tuple(key)).all(axis=1)]
            if a not in t or b not in t or t[b].isna().all():
                continue
            fam.append((sid, desc, wilcoxon_one_sided(t[a].to_numpy(), t[b].to_numpy())))
        p = np.array([f[2]["p"] for f in fam], float)
        ok = np.isfinite(p)
        ph = np.full(len(p), np.nan)
        if ok.any():
            ph[ok] = multipletests(p[ok], alpha=alpha, method="holm")[1]
        for (sid, desc, res), h in zip(fam, ph):
            rows.append({**dict(zip(GROUP, key)), "scenario": sid, "test": desc, **res,
                         "p_holm": h, "significant_holm": bool(h < alpha) if np.isfinite(h)
                         else False, "note": "exploratory follow-up"})
    return pd.DataFrame(rows)


def breakpoints(summary: pd.DataFrame) -> pd.DataFrame:
    """Weakest setting (by mean injected error) at which D reaches each threshold."""
    thr = load_config()["followup"]["stress"]["breakpoint_relative_d"]
    s = summary[summary.variant == "M3"]
    rows = []
    for (ev, f, m, exp, pc), g in s.groupby(["evaluation", "feature_set", "model", "experiment",
                                             "fraction"]):
        if exp not in ("oracle", "stress"):
            continue
        g = g.sort_values("eps_mean")
        for t in thr:
            hit = g[g["D_rel"] >= t]
            r = hit.iloc[0] if len(hit) else None
            rows.append({"evaluation": ev, "feature_set": f, "model": m, "grid": exp,
                         "p_c": pc, "threshold_D_rel": t, "reached": r is not None,
                         "setting": r["label"] if r is not None else "not reached",
                         "eps_mean_db": r["eps_mean"] if r is not None else np.nan,
                         "D": r["D_mean"] if r is not None else np.nan,
                         "max_D_rel_in_grid": g["D_rel"].max()})
    return pd.DataFrame(rows)


def run_followup(n_jobs: int | None = None) -> dict[str, pd.DataFrame]:
    """Run the three follow-up experiments and save their outputs separately."""
    cfg = load_config()
    n_jobs = cfg["experiment"]["n_jobs"] if n_jobs is None else n_jobs
    scen = scenarios()
    splits = evaluation_splits(load_matchups())
    print(f"Follow-up: {len(scen)} scenarios x {cfg['followup']['n_seeds']} seeds x "
          f"{len(splits)} splits (exploratory)")
    t0 = time.time()
    tasks = [(ev, k, te, scen[scen.experiment == e]) for ev, k, _, te in splits
             for e in ("oracle", "persistent", "stress")]
    res = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=5)(
        delayed(_task)(*t) for t in tasks)
    print(f"done in {time.time() - t0:.0f} s")
    sums = pd.concat([s for r in res for s in r[0]], ignore_index=True)
    det = pd.DataFrame([d for r in res for d in r[1]])
    sums.to_csv(path("processed") / "followup_station_sums.csv.gz", index=False,
                compression=GZIP)

    # Consistency: the oracle grid shares Stage 3 seeds, so M1 and M3 must match.
    st3 = pd.read_csv(path("results") / "experiment_pooled.csv.gz")
    o = scen[scen.experiment == "oracle"]
    chk = []
    for r in o.itertuples():
        a = sums[(sums.scenario == r.scenario) & sums.variant.isin(["M1", "M3"])]
        a = a.groupby(["evaluation", "seed", "feature_set", "model", "variant"])[SUM_COLS].sum()
        a = a.reset_index()
        a["rmse"] = metrics_from_sums(a[SUM_COLS].to_numpy(float))["rmse"]
        q = int(r.severity[1:])
        b = st3[(st3.severity_q == q) & np.isclose(st3.extent, r.extent)
                & (st3.p_c == r.fraction) & (st3.polarisation == r.polarisation)
                & st3.variant.isin(["M1", "M3"])]
        chk.append(a.merge(b, on=["evaluation", "seed", "feature_set", "model", "variant"],
                           suffixes=("", "_s3")))
    chk = pd.concat(chk)
    assert len(chk) > 0 and np.allclose(chk["rmse"], chk["rmse_s3"], rtol=1e-9), \
        "oracle grid does not reproduce Stage 3"
    print(f"oracle grid reproduces Stage 3 M1 and M3 on {len(chk)} pooled runs")

    summary = per_seed_summary(sums, scen)
    boot = pd.concat([bootstrap_scenario(sums, s) for s in scen["scenario"]],
                     ignore_index=True).merge(
        scen[["scenario", "experiment", "label", "eps_mean"]], on="scenario")
    tst = tests(sums, scen)
    bp = breakpoints(summary)
    dsum = det.groupby("scenario")[[c for c in det.columns if c[:3] in ("tp_", "fp_", "fn_")]
                                   + ["n_obs", "n_kept"]].sum().reset_index()
    for p in rfi.POLS:
        pr, rc, f1 = rfi.precision_recall_f1(dsum[f"tp_{p}"], dsum[f"fp_{p}"], dsum[f"fn_{p}"])
        dsum[f"precision_{p}"], dsum[f"recall_{p}"], dsum[f"f1_{p}"] = pr, rc, f1
    dsum["share_lost"] = 1 - dsum["n_kept"] / dsum["n_obs"]
    dsum = dsum.merge(scen[["scenario", "experiment", "label"]], on="scenario")
    out = {"followup_scenarios": scen.drop(columns="seed_keys"), "followup_summary": summary,
           "followup_bootstrap": boot, "followup_tests": tst, "followup_breakpoints": bp,
           "followup_detection": dsum}
    for name, t in out.items():
        t.to_csv(path("results_aggregate") / f"{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "followup_summary.csv")
    return out
