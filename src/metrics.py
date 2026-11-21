"""Accuracy metrics for soil moisture retrieval and RFI degradation measures.

Conventions:
  * errors are ``e = predicted - observed`` (a positive bias means
    overestimation);
  * ubRMSD is the unbiased root mean square difference,
    ``sqrt(RMSE^2 - bias^2)``;
  * R2 is the coefficient of determination ``1 - SS_res / SS_tot`` (it can
    be negative), and R is the Pearson correlation coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["rmse", "mae", "bias", "ubrmsd", "r2", "r"]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    """Mean error (predicted minus observed)."""
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def ubrmsd(y: np.ndarray, p: np.ndarray) -> float:
    """Unbiased RMSD: the RMSE after removing the mean bias."""
    e = np.asarray(p) - np.asarray(y)
    return float(np.sqrt(np.mean((e - e.mean()) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination 1 - SS_res / SS_tot."""
    y, p = np.asarray(y), np.asarray(p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def pearson_r(y: np.ndarray, p: np.ndarray) -> float:
    """Pearson correlation coefficient (NaN if either series is constant)."""
    y, p = np.asarray(y), np.asarray(p)
    if y.std() == 0 or p.std() == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Return all accuracy metrics and the sample size as a dictionary."""
    return {"rmse": rmse(y, p), "mae": mae(y, p), "bias": bias(y, p),
            "ubrmsd": ubrmsd(y, p), "r2": r2(y, p), "r": pearson_r(y, p), "n": len(y)}


def per_station_metrics(df: pd.DataFrame, y: str = "sm", p: str = "pred",
                        group: str = "station_id") -> pd.DataFrame:
    """Compute all metrics separately for each station.

    Per-station (temporal) R and ubRMSD are the usual measures in soil
    moisture validation, because pooled R also reflects differences in mean
    wetness between stations.
    """
    rows = []
    for sid, g in df.groupby(group):
        rows.append({group: sid, **all_metrics(g[y].to_numpy(), g[p].to_numpy())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Degradation and recovery (used from Stage 3 onwards)
# --------------------------------------------------------------------------

def degradation(rmse_m1: float, rmse_m0: float) -> float:
    """Loss of accuracy caused by RFI: D = RMSE_M1 - RMSE_M0."""
    return rmse_m1 - rmse_m0


def recovery(rmse_m1: float, rmse_variant: float) -> float:
    """Accuracy recovered by a mitigation variant: G = RMSE_M1 - RMSE_variant."""
    return rmse_m1 - rmse_variant


def recovery_ratio(rmse_m0: float, rmse_m1: float, rmse_variant: float,
                   min_d: float = 1e-4) -> float:
    """Recovery ratio RR = G / D.

    Returns NaN when the degradation D is smaller than ``min_d``, because the
    ratio is then dominated by noise (division by a near-zero number).
    RR = 1 means full recovery, RR = 0 no recovery, RR < 0 a further loss.
    """
    d = degradation(rmse_m1, rmse_m0)
    if not np.isfinite(d) or d < min_d:
        return np.nan
    return recovery(rmse_m1, rmse_variant) / d
