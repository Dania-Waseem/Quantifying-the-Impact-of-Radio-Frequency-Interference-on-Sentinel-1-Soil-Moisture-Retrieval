"""RFI injection, detection, filling and discarding.

Injection model (linear power domain, per polarisation):

    sigma_tilde = sigma * (1 + rho * 10^(-SIR / 10))

where ``rho`` is the affected fraction of the footprint (extent) and SIR is
the signal-to-interference ratio in dB. The resulting backscatter error in
dB is ``eps = 10 log10(sigma_tilde / sigma) = 10 log10(1 + rho 10^(-SIR/10))``,
which is always positive: interference adds power. Each acquisition is
contaminated with probability ``p_c``; the polarisation hit is VV, VH or both.

Detection: a rolling robust z-score per station on the time-ordered dB
series, ``z = (x - median) / max(1.4826 MAD, floor)``, with the median and
MAD taken over the previous ``w`` observed acquisitions of the same orbit
(each station uses a single relative orbit). Only increases are flagged by
default, because RFI raises the backscatter.

Mitigation: fill replaces a flagged value by the median of the previous
``k`` unflagged values of the same station and polarisation; discard drops
acquisitions with any flagged polarisation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import load_config, path

POLS = ("VV", "VH")
OPENSAR_COLUMNS = ["patch_id", "split", "extent", "severity_db_vv", "severity_db_vh"]


class NotAvailableError(RuntimeError):
    """Raised for components that are reserved but not implemented."""


# --------------------------------------------------------------------------
# M2 placeholder
# --------------------------------------------------------------------------

def m2_published_suppression(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for M2, the RFI suppression of the published baseline.

    The baseline reproduction is not part of the current work. Every table
    and figure treats M2 as missing.
    """
    raise NotAvailableError("M2 (published baseline RFI suppression) is not available: "
                            "the baseline reproduction has not been implemented yet.")


# --------------------------------------------------------------------------
# Injection grid (provisional or from OpenSAR statistics)
# --------------------------------------------------------------------------

def load_opensar_stats() -> pd.DataFrame | None:
    """Read ``data/processed/opensar_rfi_stats.csv`` if it exists.

    The file must have one row per patch and the columns ``patch_id, split,
    extent, severity_db_vv, severity_db_vh``. Returns None if the file does
    not exist. Raises ValueError if columns are missing or values are
    invalid (extent outside 0 to 1, or negative severity).
    """
    p = path("opensar_stats")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [c for c in OPENSAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing columns: {missing}")
    if not df["extent"].between(0, 1).all():
        raise ValueError(f"{p}: extent must lie between 0 and 1")
    if (df[["severity_db_vv", "severity_db_vh"]] < 0).any().any():
        raise ValueError(f"{p}: severity (dB increase) must not be negative")
    return df


def sir_from_severity(severity_db: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Convert a backscatter increase in dB at a given extent into SIR (dB).

    Inverts ``eps = 10 log10(1 + rho 10^(-SIR/10))``:
    ``SIR = -10 log10((10^(eps/10) - 1) / rho)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return -10 * np.log10((10 ** (np.asarray(severity_db) / 10) - 1) / np.asarray(extent))


def injection_grid(verbose: bool = True) -> dict:
    """Return the injection grid (SIR, extent, p_c, polarisation).

    If the OpenSAR statistics exist, the SIR and extent values are the
    configured quantiles of the patch statistics (SIR derived from the
    severity of both polarisations at the patch extent). Otherwise the
    PROVISIONAL grid from ``config.yaml`` is used and a warning is printed
    every time. ``p_c`` and the polarisation always come from the
    configuration. The returned dictionary has a ``source`` entry.
    """
    cfg = load_config()["injection"]
    grid = {"p_c": list(cfg["p_c"]), "polarisation": list(cfg["polarisation"])}
    stats = load_opensar_stats()
    if stats is None:
        msg = ("PROVISIONAL injection grid in use: data/processed/opensar_rfi_stats.csv "
               "does not exist. SIR and extent values are placeholders and must be "
               "replaced by OpenSAR statistics.")
        if verbose:
            print("WARNING:", msg)
        warnings.warn(msg, stacklevel=2)
        grid.update(sir_db=list(cfg["sir_db"]), extent=list(cfg["extent"]),
                    source="provisional")
        return grid
    q = cfg["opensar_quantiles"]
    sir = np.concatenate([sir_from_severity(stats[f"severity_db_{p.lower()}"], stats["extent"])
                          for p in POLS])
    sir = sir[np.isfinite(sir)]
    grid.update(sir_db=[float(v) for v in np.quantile(sir, q)],
                extent=[float(v) for v in np.quantile(stats["extent"], q)],
                source="opensar")
    if verbose:
        print("Injection grid derived from OpenSAR statistics:", grid)
    return grid


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------

def eps_db(sir_db: float | np.ndarray, extent: float | np.ndarray) -> np.ndarray:
    """Backscatter increase in dB caused by interference at a given SIR and extent."""
    return 10 * np.log10(1 + np.asarray(extent) * 10 ** (-np.asarray(sir_db) / 10))


def pols_hit(polarisation: str) -> tuple[str, ...]:
    """Return the polarisations affected by a setting ('VV', 'VH' or 'both')."""
    return POLS if polarisation == "both" else (polarisation,)


def inject(df: pd.DataFrame, sir_db: np.ndarray | float, extent: np.ndarray | float,
           p_c: np.ndarray | float, polarisation, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``df`` with RFI injected into the backscatter.

    Parameters may be scalars (one setting for all rows) or per-row arrays
    (used for the RFI-aware training copies, where each station has its own
    setting). ``polarisation`` is a string or a per-row array of strings.

    Adds the columns ``VV_clean``, ``VH_clean`` (the original values),
    ``inj_VV``, ``inj_VH`` (true contamination flags) and ``eps_VV``,
    ``eps_VH`` (injected error in dB, 0 where not contaminated).
    """
    out = df.copy()
    n = len(out)
    contaminated = rng.random(n) < np.broadcast_to(p_c, (n,))
    pol = np.broadcast_to(np.asarray(polarisation, dtype=object), (n,))
    for p in POLS:
        hit = contaminated & ((pol == p) | (pol == "both"))
        sigma = 10 ** (out[p].to_numpy() / 10)                     # dB to linear power
        factor = 1 + np.broadcast_to(extent, (n,)) * 10 ** (-np.broadcast_to(sir_db, (n,)) / 10)
        out[f"{p}_clean"] = out[p]
        # Back to dB for contaminated rows only; other rows keep their exact
        # original value (no round trip through the linear domain).
        out[p] = np.where(hit, 10 * np.log10(sigma * factor), out[p].to_numpy())
        out[f"inj_{p}"] = hit
        out[f"eps_{p}"] = out[p] - out[f"{p}_clean"]
    return out


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def _past_windows(x: np.ndarray, station_start: np.ndarray, w: int) -> np.ndarray:
    """Return an (n, w) array of the previous ``w`` values of each position.

    Values from before the start of the position's station are NaN.
    ``x`` must be sorted by station and time.
    """
    n = len(x)
    idx = np.arange(n)[:, None] - w + np.arange(w)[None, :]        # i-w .. i-1
    valid = idx >= station_start[:, None]
    vals = x[np.clip(idx, 0, n - 1)]
    return np.where(valid, vals, np.nan)


def _sorted_view(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the sort order (station, time) and each row's station start."""
    order = np.lexsort((df["t"].to_numpy(), df["station_id"].to_numpy()))
    st = df["station_id"].to_numpy()[order]
    first = np.r_[True, st[1:] != st[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(st)), 0))
    return order, start


def detect(df: pd.DataFrame, tau: float | None = None, w: int | None = None) -> pd.DataFrame:
    """Flag suspected RFI with a rolling robust z-score per station.

    Returns a data frame aligned with ``df`` with boolean columns
    ``flag_VV`` and ``flag_VH`` and the z-scores ``z_VV`` and ``z_VH``.
    Positions with fewer than ``min_history`` previous acquisitions are never
    flagged. The median and MAD use the observed (possibly contaminated)
    past values; both statistics are robust to a minority of outliers.
    """
    cfg = load_config()["detection"]
    tau = cfg["tau"] if tau is None else tau
    w = cfg["window"] if w is None else w
    order, start = _sorted_view(df)
    out = pd.DataFrame(index=df.index)
    for p in POLS:
        x = df[p].to_numpy(float)[order]
        win = _past_windows(x, start, w)
        count = np.isfinite(win).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
            med = np.nanmedian(win, axis=1)
            mad = np.nanmedian(np.abs(win - med[:, None]), axis=1)
        scale = np.maximum(1.4826 * mad, cfg["mad_floor_db"])
        z = (x - med) / scale
        enough = count >= cfg["min_history"]
        flag = enough & ((z > tau) if cfg["one_sided"] else (np.abs(z) > tau))
        z_out, f_out = np.full(len(x), np.nan), np.zeros(len(x), bool)
        z_out[order], f_out[order] = np.where(enough, z, np.nan), flag
        out[f"z_{p}"], out[f"flag_{p}"] = z_out, f_out
    return out


def detection_counts(df: pd.DataFrame, flags: pd.DataFrame) -> dict:
    """Return true/false positives and negatives per polarisation.

    The truth is the injected flag ``inj_<pol>`` of each acquisition.
    """
    out = {}
    for p in POLS:
        truth, pred = df[f"inj_{p}"].to_numpy(), flags[f"flag_{p}"].to_numpy()
        out[f"tp_{p}"] = int((truth & pred).sum())
        out[f"fp_{p}"] = int((~truth & pred).sum())
        out[f"fn_{p}"] = int((truth & ~pred).sum())
        out[f"tn_{p}"] = int((~truth & ~pred).sum())
    return out


def precision_recall_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple:
    """Precision, recall and F1 from counts (NaN where undefined)."""
    tp, fp, fn = (np.asarray(v, float) for v in (tp, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return prec, rec, f1


# --------------------------------------------------------------------------
# Mitigation
# --------------------------------------------------------------------------

def fill(df: pd.DataFrame, flags: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Replace flagged values by the median of the previous ``k`` unflagged values.

    Works per station and polarisation on the time-ordered series. A
    flagged value with no earlier unflagged value (at the start of a
    series) is replaced by the median of all unflagged values of the
    station. Returns a copy of ``df``; the column ``filled_<pol>`` marks
    replaced values.
    """
    k = load_config()["fill"]["k"] if k is None else k
    out = df.copy()
    order, _ = _sorted_view(df)
    st = df["station_id"].to_numpy()[order]
    for p in POLS:
        x = df[p].to_numpy(float)[order]
        f = flags[f"flag_{p}"].to_numpy()[order]
        clean = pd.Series(np.where(f, np.nan, x))
        # Rolling median over the last k unflagged values of the station,
        # placed at the unflagged positions, then carried forward.
        rm = (clean.dropna().groupby(st[~f]).transform(
            lambda s: s.rolling(k, min_periods=1).median()))
        carried = pd.Series(np.nan, index=clean.index)
        carried[rm.index] = rm.to_numpy()
        carried = carried.groupby(st).ffill()
        fallback = clean.groupby(st).transform("median")
        rep = carried.fillna(fallback).to_numpy()
        new = np.where(f, rep, x)
        col = np.empty(len(x))
        col[order] = new
        filled = np.zeros(len(x), bool)
        filled[order] = f
        out[p] = col
        out[f"filled_{p}"] = filled
    return out


def discard(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Drop acquisitions where any polarisation is flagged."""
    keep = ~(flags["flag_VV"].to_numpy() | flags["flag_VH"].to_numpy())
    return df.loc[keep].copy()
