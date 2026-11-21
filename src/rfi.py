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
# Injection grid (from OpenSAR statistics, provisional fallback)
# --------------------------------------------------------------------------

def load_opensar_stats() -> pd.DataFrame | None:
    """Read ``data/processed/opensar_rfi_stats.csv`` if it exists.

    The file has one row per patch with at least the columns ``patch_id,
    split, extent, severity_db_vv, severity_db_vh``. ``severity_db_*`` is
    the measured excess power (dB) of the masked RFI pixels against the
    unmasked pixels; it can be zero or negative when a patch shows no power
    excess in that polarisation. Other columns (for example ``sir_db_*``)
    are ignored. Returns None if the file does not exist; raises ValueError
    if columns are missing, values are missing, or extent is outside 0 to 1.
    """
    p = path("opensar_stats")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [c for c in OPENSAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing columns: {missing}")
    if df[OPENSAR_COLUMNS].isna().any().any():
        raise ValueError(f"{p} has missing values in {OPENSAR_COLUMNS}")
    if not df["extent"].between(0, 1).all():
        raise ValueError(f"{p}: extent must lie between 0 and 1")
    return df[OPENSAR_COLUMNS].copy()


def sir_from_contrast(contrast_db: np.ndarray) -> np.ndarray:
    """SIR (dB) of fully affected pixels with a measured excess power C (dB).

    Inverts ``C = 10 log10(1 + 10^(-SIR/10))``:
    ``SIR = -10 log10(10^(C/10) - 1)``. Defined for C > 0 only.
    """
    c = np.asarray(contrast_db, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, -10 * np.log10(10 ** (c / 10) - 1), np.nan)


def injection_grid(verbose: bool = True) -> dict:
    """Return the injection grid.

    Keys:
      ``severity_q``: severity level labels (percentiles);
      ``sir``: {level: {"VV": SIR, "VH": SIR}} in dB, one SIR per
        polarisation;
      ``contrast_db``: {level: {"VV": C, "VH": C}}, the excess power of the
        level (dB);
      ``extent_q``, ``extent``: extent level labels and values;
      ``p_c``, ``polarisation``: from the configuration;
      ``source``: "opensar" or "provisional"; ``notes``: text.

    With the OpenSAR file: severity levels are percentiles of
    ``severity_db_<pol>`` over the patches with C above the configured
    minimum contrast in that polarisation, converted to SIR per
    polarisation; extent levels are percentiles of extent over all patches.
    Without the file: the provisional fallback lists, with a warning printed
    every time.
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
        fb = cfg["provisional_fallback"]
        levels = list(range(1, len(fb["sir_db"]) + 1))
        grid.update(severity_q=levels,
                    sir={q: {"VV": float(s), "VH": float(s)} for q, s in zip(levels, fb["sir_db"])},
                    contrast_db={q: {p: float(eps_db(s, 1.0)) for p in POLS}
                                 for q, s in zip(levels, fb["sir_db"])},
                    extent_q=list(range(1, len(fb["extent"]) + 1)), extent=list(fb["extent"]),
                    source="provisional", notes=msg)
        return grid
    oc = cfg["opensar"]
    pct = list(oc["percentiles"])
    cmin = float(oc["min_contrast_db"])
    contrast, excluded = {}, {}
    for p in POLS:
        c = stats[f"severity_db_{p.lower()}"]
        keep = c > cmin
        excluded[p] = int((~keep).sum())
        contrast[p] = np.percentile(c[keep], pct)
    ext = np.percentile(stats["extent"], pct)
    grid.update(
        severity_q=pct,
        contrast_db={q: {p: float(contrast[p][i]) for p in POLS} for i, q in enumerate(pct)},
        sir={q: {p: float(sir_from_contrast(contrast[p][i])) for p in POLS}
             for i, q in enumerate(pct)},
        extent_q=pct, extent=[float(v) for v in ext], source="opensar",
        n_patches=len(stats), excluded=excluded,
        notes=(f"OpenSAR: {len(stats)} patches; severity levels use patches with contrast "
               f"above {cmin} dB (excluded: VV {excluded['VV']}, VH {excluded['VH']}); extent "
               f"over all patches (range {stats['extent'].min():.3f} to "
               f"{stats['extent'].max():.3f}, {int((stats['extent'] > 0.85).sum())} patches "
               f"above 0.85)."))
    if verbose:
        print("Injection grid from OpenSAR statistics.", grid["notes"])
    return grid


def grid_table(grid: dict | None = None) -> pd.DataFrame:
    """The injection grid as a table (one row per severity or extent level)."""
    grid = grid or injection_grid(verbose=False)
    rows = [{"factor": "severity", "level": q, "contrast_db_VV": grid["contrast_db"][q]["VV"],
             "contrast_db_VH": grid["contrast_db"][q]["VH"], "sir_db_VV": grid["sir"][q]["VV"],
             "sir_db_VH": grid["sir"][q]["VH"]} for q in grid["severity_q"]]
    rows += [{"factor": "extent", "level": q, "extent": v}
             for q, v in zip(grid["extent_q"], grid["extent"])]
    rows += [{"factor": "p_c", "level": i + 1, "p_c": v} for i, v in enumerate(grid["p_c"])]
    t = pd.DataFrame(rows)
    t["source"] = grid["source"]
    return t


def extent_value(grid: dict, q) -> float:
    """Extent value of an extent level label."""
    return grid["extent"][grid["extent_q"].index(q)]


def severity_level_db(grid: dict, q) -> float:
    """Mean excess power of a severity level over VV and VH (dB), used as the
    severity axis of the dose-response analyses."""
    return float(np.mean([grid["contrast_db"][q][p] for p in POLS]))


# --------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------

def eps_db(sir_db: float | np.ndarray, extent: float | np.ndarray) -> np.ndarray:
    """Backscatter increase in dB caused by interference at a given SIR and extent."""
    return 10 * np.log10(1 + np.asarray(extent) * 10 ** (-np.asarray(sir_db) / 10))


def pols_hit(polarisation: str) -> tuple[str, ...]:
    """Return the polarisations affected by a setting ('VV', 'VH' or 'both')."""
    return POLS if polarisation == "both" else (polarisation,)


def inject(df: pd.DataFrame, sir_db, extent: np.ndarray | float,
           p_c: np.ndarray | float, polarisation, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``df`` with RFI injected into the backscatter.

    ``sir_db`` is either one value for both polarisations or a dictionary
    ``{"VV": ..., "VH": ...}`` so that each polarisation is injected with its
    own SIR. Parameters may be scalars (one setting for all rows) or per-row
    arrays (used for the RFI-aware training copies, where each station has
    its own setting). ``polarisation`` is a string or a per-row array.

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
        sir_p = sir_db[p] if isinstance(sir_db, dict) else sir_db
        factor = 1 + np.broadcast_to(extent, (n,)) * 10 ** (-np.broadcast_to(sir_p, (n,)) / 10)
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
