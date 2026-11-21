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
