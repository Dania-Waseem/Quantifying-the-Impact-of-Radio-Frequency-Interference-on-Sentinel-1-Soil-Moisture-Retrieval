# Quantifying the Impact of Radio Frequency Interference on Sentinel-1 Soil Moisture Retrieval

**A controlled study of degradation and recovery after mitigation**

Dania Waseem (23I-2622) and Taiba Tariq (23I-2618)
FAST-NUCES, Methods in Business Research (MG 3010)

## Motivation

Radio frequency interference (RFI) from ground-based emitters contaminates
C-band synthetic aperture radar (SAR) images, including those of Sentinel-1.
Research on RFI usually ends at the image: methods are judged by detection
accuracy or image quality. Soil moisture retrieval, in turn, usually assumes
that its radar input is clean. As a result, it is not known how much RFI
degrades a downstream geophysical product, how this depends on the severity
and extent of the interference, and how much of the loss mitigation recovers.

This repository injects controlled, physically based RFI into Sentinel-1
backscatter at in situ soil moisture stations. It then measures the change in
retrieval accuracy and evaluates several mitigation strategies.

## Research gaps

- **G1.** RFI research stops at the image and is not evaluated on a product
  such as soil moisture.
- **G2.** Soil moisture retrieval assumes usable input; RFI has not been added
  deliberately to measure its effect on retrieval accuracy.
- **G3.** For Sentinel-1, the effect of RFI severity and extent on retrieval
  error, and the share recovered by mitigation, has not been measured.

## Research questions

- **RQ1.** How does RFI contamination affect the accuracy of Sentinel-1 based
  soil moisture retrieval?
- **RQ2.** How does the loss in accuracy change with the severity and the
  spatial extent of RFI?
- **RQ3.** To what extent can RFI mitigation recover the performance lost to
  interference?
- **RQ4 (optional).** Does the impact differ across land cover or vegetation
  conditions?

## Data

All inputs are public secondary data. They are not part of this repository
and must be placed in `data/` as described below.

| File | Content | Source |
|---|---|---|
| `data/external/s1_grd.csv` | Sentinel-1 GRD sigma0 (VV, VH, dB), 3x3 pixel mean at each station, incidence angle, relative orbit, pass | Copernicus Sentinel-1, accessed through Google Earth Engine |
| `data/external/s2_ndvi.csv` | Sentinel-2 NDVI at each station | Copernicus Sentinel-2, accessed through Google Earth Engine |
| `data/external/era5_precip.csv` | Daily total precipitation (m) | ERA5-Land daily aggregates (Munoz-Sabater et al., 2021) |
| `data/external/static.csv` | ESA WorldCover class, elevation, slope | ESA WorldCover (Zanaga et al., 2022) and a digital elevation model |
| `data/external/ismn_sm_0_5cm.csv` | Hourly soil moisture (m3/m3), 0 to 5 cm, good-quality flag only | International Soil Moisture Network (Dorigo et al., 2021) |
| `data/raw/ismn/` | ISMN download (networks USCRN and OZNET, 2019 to 2023) and `stations.csv` | ISMN, https://ismn.earth |

Networks: USCRN, the U.S. Climate Reference Network (Bell et al., 2013), is
the main sample. OZNET, in the Murrumbidgee catchment, Australia (Smith et al.,
2012), is the second sample. The study period is 2019 to 2023.

**How to obtain the data.**
1. ISMN: register at https://ismn.earth, select the networks USCRN and OZNET,
   the period 2019-01-01 to 2023-12-31, the variable soil moisture, depths
   0 to 0.05 m, and the good-quality flag only. Extract the archive to
   `data/raw/ismn/`. The hourly 0 to 5 cm values are collected into
   `data/external/ismn_sm_0_5cm.csv` with the columns
   `network, station_id, time, sm`.
2. Sentinel-1, Sentinel-2, ERA5-Land and ESA WorldCover: sample each product at
   the station coordinates in `data/raw/ismn/stations.csv` (Google Earth Engine)
   and export the tables listed above.

The variables of the analysis table are described in
[`docs/codebook.csv`](docs/codebook.csv).

### References

- Bell, J. E., et al. (2013). U.S. Climate Reference Network soil moisture and
  temperature observations. *Journal of Hydrometeorology*, 14(3), 977-988.
- Dorigo, W., et al. (2021). The International Soil Moisture Network: serving
  Earth system science for over a decade. *Hydrology and Earth System
  Sciences*, 25, 5749-5804.
- Munoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global
  reanalysis dataset for land applications. *Earth System Science Data*, 13,
  4349-4383.
- Smith, A. B., et al. (2012). The Murrumbidgee soil moisture monitoring
  network data set. *Water Resources Research*, 48, W07701.
- Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo.

## Method overview

1. **Matchups.** Each Sentinel-1 acquisition of the most frequent relative
   orbit at a station is matched to the nearest hourly ISMN value within
   60 minutes, the nearest Sentinel-2 NDVI within 30 days, and the 7-day and
   30-day ERA5-Land precipitation sums over the days before the acquisition
   date (the acquisition day is excluded, so rain after the overpass is not
   counted). Rows with VV exactly equal to VH (export error), negative NDVI or
   soil moisture of exactly 0 are removed and logged. Rows with VH below
   -30 dB (the noise floor) are flagged and kept. Stations with fewer than 50
   matchups are excluded; the stricter threshold of 100 would leave no OZNET
   station.
2. **Clean model.** Random forest, XGBoost and a change-detection baseline are
   evaluated with station-wise 5-fold cross-validation.
   Three feature configurations are fixed in `config.yaml` before any RFI experiment and
   are reported side by side in every later stage: F1 (primary) uses `VV, VH,
   VHVV, angle, ndvi, p7, p30, elev, slope, doy`; F2 (radar-reliant) removes
   the precipitation features `p7` and `p30`; F3 (secondary) uses relative
   backscatter, `rVV` and `rVH` (VV and VH minus a robust, MAD-clipped median
   of the station's own series, computed without soil moisture data), with
   `VHVV, angle, ndvi, doy` and without site or precipitation features. Under
   RFI injection the reference median is computed from the series the model
   actually receives (contaminated, filled or after discarding), never from
   the clean values. The change-detection baseline
   uses VV only. Its relative wetness is standardised within each station and
   scaled to the mean soil moisture of the training stations and to their
   median within-station standard deviation (`cd`); the original version,
   with one pooled mean and standard deviation matching (`cd_pooled`), is kept
   as a second row in the results. USCRN is the main
   sample. OZNET is used as a transfer test (trained on USCRN, tested on
   OZNET); within-OZNET cross-validation is also reported but is based on few
   stations and matchups.
3. **RFI injection.** Interference is added in the linear power domain,
   `sigma_tilde = sigma * (1 + rho * 10^(-SIR/10))`, where `rho` is the affected
   fraction and SIR the signal-to-interference ratio in dB.
4. **Mitigation variants.** M0 clean input, M1 contaminated input, M2 published
   RFI suppression (reserved), M3 detect-and-fill, M3b detect-and-discard,
   M4 detect-and-fill with RFI-aware training.
5. **Statistics.** Cluster bootstrap over stations (5,000 resamples), Wilcoxon
   signed-rank tests, mediation and moderation analysis, nested robustness
   checks and a station-count learning curve. Significance level 0.05.

## Repository structure

```
config.yaml        all settings (paths, seed, tolerances, grids, bootstrap size)
requirements.txt   pinned Python packages
run_all.py         full reproduction of all stages in order
src/               analysis modules
  config.py        configuration, paths and seeds
  data.py          loading, matchup building and data screening
  metrics.py       accuracy, degradation and recovery metrics
  models.py        random forest, XGBoost and change-detection baseline
  rfi.py           RFI injection, detection, filling and discarding
  experiment.py    variants M0 to M4 over folds, seeds, severity and extent
  stats.py         bootstrap, paired tests, mediation, moderation, power
  tables.py        CSV and LaTeX table writers
  plotstyle.py     shared figure style
  plots.py         result figures
notebooks/         one thin notebook per stage
tools/             helper scripts (notebook generation)
docs/              codebook and notation table
data/processed/    generated tables (not tracked)
models/            saved models (not tracked)
results/           CSV and LaTeX results and logs
figures/           PNG (400 dpi) and PDF figures; diagrams/ holds editable sources
```

## Environment

Python 3.12 on Windows, CPU only.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name rfi_project
```

## How to run

Run the notebooks in order (each one runs top to bottom after a kernel
restart):

1. `notebooks/01_data_cleaning.ipynb`: builds `data/processed/matchups.csv` and
   the data-screening summary.
2. `notebooks/02_base_model.ipynb`: clean models and cross-validation.
3. `notebooks/03_rfi_statistics.ipynb`: RFI statistics from OpenSAR (input
   pending).
4. `notebooks/04_extended_model.ipynb`: RFI injection and mitigation variants.
5. `notebooks/05_robustness_statistics.ipynb`: statistics and robustness.
6. `notebooks/06_plots_tables.ipynb`: figures and tables.

Alternatively, `python run_all.py` runs all stages in order. All random seeds
are derived from the single seed in `config.yaml`.

## Outputs

- `data/processed/matchups.csv`: analysis table (one row per acquisition).
- `results/aggregate/screening_*.csv`: data-screening summary (counts,
  quality rules, outlier flags, normality checks).
- `results/aggregate/stage2_*.csv`: clean-model metrics with 95%
  cluster-bootstrap intervals over stations, per-fold metrics and feature
  importances.
- Further result tables and figures are added as the stages are completed.

Only aggregated results are part of the repository. Per-station and per-row
outputs (`results/private/`, `data/processed/`) and fitted models
(`models/`) are regenerated locally, because the source data may not be
redistributed.
