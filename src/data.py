"""Data loading, matchup building and data screening.

The main entry point is :func:`build_matchups`, which produces the single
analysis table ``data/processed/matchups.csv``. Each row is one Sentinel-1
acquisition at one station, matched to in situ soil moisture, NDVI,
antecedent precipitation and static site attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import load_config, path, rng

# Columns needed by the models; rows missing any of these are dropped.
FEATURES = ["VV", "VH", "VHVV", "angle", "ndvi", "p7", "p30", "elev", "slope", "doy"]


# --------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------

def report(df: pd.DataFrame, step: str, log: list | None = None) -> None:
    """Print the row count and the null counts of a table after a step.

    If ``log`` is given, a record of the step is appended to it so that the
    screening summary can list every transformation.
    """
    nulls = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    stations = df["station_id"].nunique() if "station_id" in df else np.nan
    print(f"[{step}] rows={len(df):,} stations={stations} nulls={nulls or 'none'}")
    if log is not None:
        log.append({"step": step, "rows": len(df), "stations": stations,
                    "nulls": "; ".join(f"{k}={v}" for k, v in nulls.items()) or "none"})


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_stations() -> pd.DataFrame:
    """Read the ISMN station list (network, station_id, lat, lon, days)."""
    return pd.read_csv(path("stations"))


def load_s1() -> pd.DataFrame:
    """Read the Sentinel-1 GRD table and parse the acquisition time as UTC."""
    df = pd.read_csv(path("s1"))
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def load_ndvi() -> pd.DataFrame:
    """Read the Sentinel-2 NDVI table. Dates are interpreted as 00:00 UTC."""
    df = pd.read_csv(path("ndvi"))
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def load_precip() -> pd.DataFrame:
    """Read the ERA5-Land daily precipitation table and convert it to mm.

    For stations where ERA5-Land has no value (for example coastal pixels),
    the export contains only two fields per line, so the station name ends
    up in the precipitation column and the station column is empty. These
    lines are repaired here: the station name is moved back to its column
    and the precipitation is set to missing. Missing values are not filled.
    """
    df = pd.read_csv(path("precip"), dtype={"total_precipitation_sum": str})
    value = pd.to_numeric(df["total_precipitation_sum"], errors="coerce")
    shifted = value.isna() & df["station_id"].isna()
    df.loc[shifted, "station_id"] = df.loc[shifted, "total_precipitation_sum"]
    df["precip_mm"] = value * 1000.0          # metres of water to millimetres
    df["date"] = pd.to_datetime(df["t"]).dt.date
    n_shift = int(shifted.sum())
    if n_shift:
        bad = sorted(df.loc[shifted, "station_id"].unique())
        print(f"[load_precip] repaired {n_shift} lines without a value; "
              f"stations with no ERA5-Land precipitation: {bad}")
    return df[["station_id", "date", "precip_mm"]]


def load_static() -> pd.DataFrame:
    """Read the static site table (land cover class, elevation, slope)."""
    return pd.read_csv(path("static"))


def load_ismn() -> pd.DataFrame:
    """Read the hourly ISMN soil moisture table (m3/m3) and parse time as UTC.

    Both ``ismn_sm_0_5cm.csv`` and ``ismn_sm_0_5cm.csv.gz`` are accepted.
    """
    p = path("ismn_sm")
    if not p.exists():
        alt = Path(str(p) + ".gz") if p.suffix != ".gz" else p.with_suffix("")
        if alt.exists():
            p = alt
    df = pd.read_csv(p, usecols=["station_id", "time", "sm"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# --------------------------------------------------------------------------
# Matchup steps
# --------------------------------------------------------------------------

def select_main_orbit(s1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep, for each station, only the relative orbit with most observations.

    Ties are broken by the lowest orbit number so that the choice is
    deterministic. Returns the filtered table and a table of the chosen
    orbit per station.
    """
    counts = (s1.groupby(["station_id", "orbit"]).size().rename("n").reset_index()
              .sort_values(["station_id", "n", "orbit"], ascending=[True, False, True]))
    chosen = counts.drop_duplicates("station_id")[["station_id", "orbit", "n"]]
    out = s1.merge(chosen[["station_id", "orbit"]], on=["station_id", "orbit"])
    return out, chosen.rename(columns={"n": "n_orbit_obs"})


def match_ismn(s1: pd.DataFrame, sm: pd.DataFrame, tol_min: int) -> pd.DataFrame:
    """Attach the nearest hourly ISMN value within ``tol_min`` minutes.

    ``merge_asof`` requires both tables to be sorted by the time key.
    Acquisitions without an ISMN value in the window get ``sm = NaN``.
    """
    left = s1.sort_values("t")
    right = sm.rename(columns={"time": "t_ismn"}).sort_values("t_ismn")
    out = pd.merge_asof(left, right, left_on="t", right_on="t_ismn", by="station_id",
                        direction="nearest", tolerance=pd.Timedelta(minutes=tol_min))
    return out


def match_ndvi(df: pd.DataFrame, ndvi: pd.DataFrame, tol_days: int) -> pd.DataFrame:
    """Attach NDVI from the nearest Sentinel-2 date within ``tol_days`` days.

    Missing NDVI values in the source are removed first so that the nearest
    valid observation is used.
    """
    right = (ndvi.dropna(subset=["ndvi"]).rename(columns={"t": "t_ndvi"})
             .sort_values("t_ndvi"))
    out = pd.merge_asof(df.sort_values("t"), right, left_on="t", right_on="t_ndvi",
                        by="station_id", direction="nearest",
                        tolerance=pd.Timedelta(days=tol_days))
    out["ndvi_lag_days"] = (out["t"] - out["t_ndvi"]).dt.total_seconds() / 86400.0
    return out


def antecedent_precip(precip: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute antecedent precipitation sums over the days before each date.

    For a window of ``w`` days, the value on date ``d`` is the sum over the
    days ``d - w`` to ``d - 1``. The day ``d`` itself is excluded because a
    daily total also contains rain that fell after the satellite overpass.
    A sum is only reported when all ``w`` days are present; otherwise it is
    missing (no partial sums at the start of the record and no sums for
    stations without data).
    """
    parts = []
    for sid, g in precip.groupby("station_id"):
        g = g.sort_values("date").set_index(pd.to_datetime(g["date"]))
        # Reindex to a complete daily calendar so that gaps count as missing.
        full = g["precip_mm"].asfreq("D")
        out = pd.DataFrame({"date": full.index.date, "station_id": sid})
        for w in windows:
            # The rolling sum ends on day d (inclusive); shifting by one day
            # makes it end on day d - 1.
            out[f"p{w}"] = full.rolling(w, min_periods=w).sum().shift(1).to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def apply_quality_rules(df: pd.DataFrame, qcfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows that fail the row-level quality rules in ``config.yaml``.

    Rules (each can be switched off in the ``quality`` section):
      * VV exactly equal to VH: an export error, since the two polarisations
        are measured separately and never coincide to full precision;
      * NDVI below 0: invalid for these vegetated or bare land sites (cloud,
        snow or water contamination);
      * soil moisture exactly 0: a likely sensor fault.

    A row that fails several rules is listed once, under the first rule.
    Returns the kept rows and a table of the removed rows with the reason.
    """
    rules = []
    if qcfg.get("drop_vv_equals_vh"):
        rules.append(("VV equals VH exactly (export error)", df["VV"] == df["VH"]))
    if qcfg.get("drop_negative_ndvi"):
        rules.append(("NDVI below 0 (invalid NDVI)", df["ndvi"] < 0))
    if qcfg.get("drop_zero_sm"):
        rules.append(("soil moisture exactly 0 (likely sensor fault)", df["sm"] == 0))

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, mask in rules:
        reason = reason.mask(mask & reason.isna(), label)
    drop = reason.notna()
    removed = df.loc[drop, ["network", "station_id", "t", "VV", "VH", "ndvi", "sm"]].copy()
    removed["reason"] = reason[drop]
    return df.loc[~drop].copy(), removed.reset_index(drop=True)


def build_matchups(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the analysis table and return it with a screening summary.

    Steps (as listed in the project plan):
      1. read Sentinel-1, drop rows with missing VV, VH or angle, and merge
         duplicate rows of the same acquisition;
      2. keep one relative orbit per station (the most frequent);
      3. match the nearest ISMN hourly value within the time tolerance;
      4. add NDVI from the nearest Sentinel-2 date within the tolerance;
      5. add 7-day and 30-day antecedent precipitation from ERA5-Land
         (days before the acquisition date only);
      6. join static attributes, add VHVV = VH - VV and day of year;
      7. drop rows with any missing feature, apply the row-level quality
         rules, flag VH below the noise floor, and drop stations with fewer
         than MIN_N matchups.
    """
    cfg = load_config()
    mcfg = cfg["matchup"]
    log: list = []
    info: dict = {}

    stations = load_stations()

    # Step 1: Sentinel-1 and removal of incomplete rows.
    s1 = load_s1()
    report(s1, "1a read Sentinel-1", log)
    n0 = len(s1)
    s1 = s1.dropna(subset=["VV", "VH", "angle"])
    info["s1_missing_removed"] = n0 - len(s1)
    report(s1, "1b drop missing VV/VH/angle", log)

    # Step 1c: duplicate acquisitions. The export contains a few rows with
    # the same station, time and orbit (overlapping or reprocessed scenes of
    # one acquisition, with equal or nearly equal values). They are merged
    # into one row by averaging, so that each acquisition counts once.
    key = ["station_id", "t", "orbit", "pass"]
    dup = s1.duplicated(key, keep=False)
    info["s1_duplicates_merged"] = int(dup.sum() - s1[dup].drop_duplicates(key).shape[0])
    s1 = s1.groupby(key, as_index=False)[["VV", "VH", "angle"]].mean()
    print(f"[1c duplicates] merged {info['s1_duplicates_merged']} duplicate acquisition rows")
    report(s1, "1c merge duplicate acquisitions", log)

    # Step 2: one relative orbit per station.
    s1, chosen = select_main_orbit(s1)
    info["chosen_orbit"] = chosen
    report(s1, "2 keep most frequent orbit", log)

    # Step 3: in situ soil moisture.
    sm = load_ismn()
    df = match_ismn(s1, sm, mcfg["ismn_tolerance_minutes"])
    del sm
    report(df, "3 match ISMN within tolerance", log)

    # Step 4: NDVI.
    df = match_ndvi(df, load_ndvi(), mcfg["ndvi_tolerance_days"])
    report(df, "4 match NDVI within tolerance", log)

    # Step 5: antecedent precipitation before the acquisition date (UTC date).
    ap = antecedent_precip(load_precip(), mcfg["precip_windows_days"])
    df["date"] = df["t"].dt.date
    df = df.merge(ap, on=["station_id", "date"], how="left")
    report(df, "5 add antecedent precipitation", log)

    # Step 6: static attributes, network and derived features.
    df = df.merge(load_static(), on="station_id", how="left")
    df = df.merge(stations[["station_id", "network", "lat", "lon"]], on="station_id",
                  how="left")
    df["VHVV"] = df["VH"] - df["VV"]
    df["doy"] = df["t"].dt.dayofyear
    report(df, "6 join static, add VHVV and doy", log)

    # Step 7a: rows with any missing feature or target.
    needed = FEATURES + ["sm"]
    miss_by_col = {c: int(df[c].isna().sum()) for c in needed if df[c].isna().any()}
    info["missing_by_column_before_drop"] = miss_by_col
    # Stations that lose every row because one feature is always missing.
    all_missing = df.groupby("station_id")[needed].apply(lambda g: g.isna().all())
    info["stations_feature_always_missing"] = {
        sid: [c for c in needed if row[c]] for sid, row in all_missing.iterrows() if row.any()}
    n0 = len(df)
    df = df.dropna(subset=needed)
    info["rows_missing_feature_removed"] = n0 - len(df)
    report(df, "7a drop rows with any missing feature", log)

    # Step 7b: row-level quality rules. Each removed row is kept in a table
    # with the reason, so that every exclusion can be traced.
    df, removed = apply_quality_rules(df, cfg["quality"])
    info["quality_removed"] = removed
    for reason, n in removed["reason"].value_counts().sort_index().items():
        print(f"[7b quality] removed {n} rows: {reason}")
    report(df, "7b apply row-level quality rules", log)

    # Step 7c: flag (do not remove) VH below the Sentinel-1 noise floor.
    floor = cfg["quality"]["vh_noise_floor_db"]
    df["vh_low"] = df["VH"] < floor
    print(f"[7c flag] vh_low (VH < {floor} dB): {int(df['vh_low'].sum())} rows flagged")

    # Step 7d: stations with too few matchups.
    counts = df.groupby("station_id").size()
    small = counts[counts < mcfg["min_n"]]
    all_ids = set(stations["station_id"])
    no_rows = sorted(all_ids - set(counts.index))
    info["dropped_small_stations"] = small.sort_values().to_dict()
    info["dropped_no_rows"] = no_rows
    df = df[~df["station_id"].isin(small.index)]
    report(df, f"7d drop stations with < {mcfg['min_n']} matchups", log)

    # Final column order and sort.
    cols = ["network", "station_id", "t", "orbit", "pass", "VV", "VH", "VHVV", "angle",
            "ndvi", "ndvi_lag_days", "p7", "p30", "elev", "slope", "lc", "lat", "lon",
            "doy", "sm", "t_ismn", "vh_low"]
    df = df[cols].sort_values(["network", "station_id", "t"]).reset_index(drop=True)
    assert not df.duplicated(["station_id", "t"]).any(), "duplicate station and time rows"
    info["log"] = pd.DataFrame(log)

    if save:
        out = path("matchups")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {out} ({len(df):,} rows)")
    return df, info


def load_matchups() -> pd.DataFrame:
    """Read the saved analysis table and parse the time columns."""
    df = pd.read_csv(path("matchups"))
    for c in ("t", "t_ismn"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


# --------------------------------------------------------------------------
# Data screening
# --------------------------------------------------------------------------

def iqr_outliers(x: pd.Series, k: float) -> pd.Series:
    """Return a boolean mask of values outside [Q1 - k IQR, Q3 + k IQR]."""
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def screening_summary(df: pd.DataFrame, info: dict) -> dict[str, pd.DataFrame]:
    """Compute the data-screening tables for the final analysis table.

    Returns a dictionary of tables: counts per network, counts per station,
    outlier flags (values are flagged, never removed), and normality checks.
    The Shapiro-Wilk test is computed on a reproducible random subsample of
    at most 5,000 values because the test is not defined for larger samples.
    """
    k = load_config()["matchup"]["outlier_iqr_factor"]
    out: dict[str, pd.DataFrame] = {}

    out["per_network"] = (df.groupby("network")
                          .agg(stations=("station_id", "nunique"), matchups=("sm", "size"),
                               vh_low=("vh_low", "sum"), sm_mean=("sm", "mean"),
                               sm_sd=("sm", "std"))
                          .reset_index())
    out["per_station"] = (df.groupby(["network", "station_id"])
                          .agg(matchups=("sm", "size"), orbit=("orbit", "first"),
                               lc=("lc", "first"), sm_mean=("sm", "mean"),
                               sm_sd=("sm", "std"), VV_mean=("VV", "mean"),
                               ndvi_mean=("ndvi", "mean"), vh_low=("vh_low", "sum"))
                          .reset_index())

    rows = []
    for c in ["sm", "VV", "VH", "VHVV", "angle", "ndvi", "p7", "p30"]:
        mask = iqr_outliers(df[c], k)
        q1, q3 = df[c].quantile([0.25, 0.75])
        # Quartiles and fences only: the minimum and maximum are single-row
        # values and are not written to the shareable tables.
        rows.append({"variable": c, "n": len(df), "n_outliers": int(mask.sum()),
                     "share": mask.mean(), "q1": q1, "q3": q3,
                     "lower_fence": q1 - k * (q3 - q1), "upper_fence": q3 + k * (q3 - q1)})
    out["outliers"] = pd.DataFrame(rows)

    g = rng("shapiro")
    rows = []
    for c in ["sm", "VV", "VH"]:
        x = df[c].to_numpy()
        sub = g.choice(x, size=min(5000, len(x)), replace=False)
        w, p = sps.shapiro(sub)
        rows.append({"variable": c, "skewness": sps.skew(x), "excess_kurtosis": sps.kurtosis(x),
                     "shapiro_W": w, "shapiro_p": p, "shapiro_n": len(sub)})
    out["normality"] = pd.DataFrame(rows)
    out["steps"] = info["log"]
    return out


def write_codebook(dest: str | Path | None = None) -> Path:
    """Write ``docs/codebook.csv`` describing every column of the matchup table."""
    rows = [
        ("network", "ISMN network of the station", "-", "ISMN"),
        ("station_id", "ISMN station name", "-", "ISMN"),
        ("t", "Sentinel-1 acquisition time", "UTC", "Sentinel-1 GRD (Google Earth Engine)"),
        ("orbit", "Sentinel-1 relative orbit (one per station)", "-", "Sentinel-1 GRD"),
        ("pass", "Orbit direction (ASCENDING or DESCENDING)", "-", "Sentinel-1 GRD"),
        ("VV", "Backscatter coefficient sigma0, VV polarisation, 3x3 pixel mean", "dB", "Sentinel-1 GRD"),
        ("VH", "Backscatter coefficient sigma0, VH polarisation, 3x3 pixel mean", "dB", "Sentinel-1 GRD"),
        ("VHVV", "Cross-polarisation ratio VH - VV", "dB", "Derived"),
        ("angle", "Local incidence angle at the station", "degree", "Sentinel-1 GRD"),
        ("ndvi", "Normalised difference vegetation index, nearest date within 30 days", "-", "Sentinel-2"),
        ("ndvi_lag_days", "Acquisition time minus NDVI date", "day", "Derived"),
        ("p7", "Precipitation sum over the 7 days before the acquisition date (acquisition day excluded)", "mm", "ERA5-Land"),
        ("p30", "Precipitation sum over the 30 days before the acquisition date (acquisition day excluded)", "mm", "ERA5-Land"),
        ("elev", "Elevation", "m", "static.csv (DEM source to be confirmed)"),
        ("slope", "Terrain slope", "degree", "static.csv (DEM source to be confirmed)"),
        ("lc", "ESA WorldCover land cover class", "-", "ESA WorldCover"),
        ("lat", "Station latitude", "degree", "ISMN"),
        ("lon", "Station longitude", "degree", "ISMN"),
        ("doy", "Day of year of the acquisition", "day", "Derived"),
        ("sm", "In situ soil moisture, 0 to 5 cm, good-quality flag", "m3/m3", "ISMN"),
        ("t_ismn", "Time of the matched ISMN hourly value", "UTC", "ISMN"),
        ("vh_low", "Flag: VH below the noise floor in config.yaml (-30 dB); kept, sensitivity run in Stage 4", "boolean", "Derived"),
    ]
    dest = Path(dest) if dest else path("docs") / "codebook.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["variable", "meaning", "unit", "source"]).to_csv(dest, index=False)
    return dest


def run_stage1() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the full data-cleaning stage and save all outputs.

    Writes the matchup table, the codebook, and the screening tables
    ``screening_*.csv`` (aggregated tables in ``results/aggregate``,
    per-station and per-row tables in ``results/private``), with a package
    version log. Returns the matchup table and the screening tables.
    """
    from .config import ensure_dirs, write_version_log

    ensure_dirs()
    df, info = build_matchups(save=True)
    write_version_log(path("matchups"))
    scr = screening_summary(df, info)

    # Table of excluded stations and the reason for each exclusion.
    excl = [{"station_id": s, "reason": "feature always missing: " + ", ".join(c)}
            for s, c in info["stations_feature_always_missing"].items()]
    excl += [{"station_id": s, "reason": f"fewer than MIN_N matchups ({n})"}
             for s, n in info["dropped_small_stations"].items()]
    scr["excluded_stations"] = pd.DataFrame(excl)
    scr["removed"] = pd.DataFrame(
        [{"item": "Sentinel-1 rows with missing VV, VH or angle",
          "rows": info["s1_missing_removed"]},
         {"item": "Sentinel-1 duplicate acquisition rows merged",
          "rows": info["s1_duplicates_merged"]},
         {"item": "matchup rows with any missing feature or soil moisture",
          "rows": info["rows_missing_feature_removed"]}]
        + [{"item": f"quality rule: {r}", "rows": int(n)} for r, n in
           info["quality_removed"]["reason"].value_counts().sort_index().items()]
        + [{"item": f"missing before drop: {c}", "rows": n}
           for c, n in info["missing_by_column_before_drop"].items()])
    info["chosen_orbit"].to_csv(path("results_private") / "screening_chosen_orbit.csv",
                                index=False)
    scr["quality_removed"] = info["quality_removed"]

    # Data policy: tables with per-station or per-row values stay in the
    # local results/private folder; only aggregated tables go to
    # results/aggregate, which may be committed.
    private = {"per_station", "excluded_stations", "quality_removed"}
    for name, table in scr.items():
        folder = path("results_private") if name in private else path("results_aggregate")
        table.to_csv(folder / f"screening_{name}.csv", index=False)
    write_version_log(path("results_aggregate") / "screening_steps.csv")
    write_codebook()
    return df, scr
