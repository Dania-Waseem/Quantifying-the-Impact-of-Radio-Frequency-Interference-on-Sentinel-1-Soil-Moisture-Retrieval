# Focused Dataset Audit Report: Experiments 5A–5C

**Project:** SAR RFI Baseline Reproduction  
**Reference Paper:** Fang et al., *Standardized Dataset and Image-Subspace-Based Method for Strip-Mode Synthetic Aperture Radar Radio Frequency Interference Suppression*, Remote Sensing, 2025, 17, 3688.  
**Audit Scope:** On-disk datasets (`SAR-RFI-DATASET`, `RFI-Patch`, `Sentinel Polluted-SAR`), directory structures, sample formats, dimensions, annotations, and splitting logic.  
**Audit Date:** 2026-11-21  

---

## 1. Executive Summary

This dataset audit provides an empirical evaluation of the physical data files available in the workspace against the requirements of the reference paper. 

### Key Findings
1. **Core Laboratory Benchmark (`SAR-RFI-DATASET`) is Completely Missing Locally (`BLOCKED`):** The 640 full-sized laboratory slices ($512\times 1500$) described in Section 3.1 were hosted on Baidu Netdisk (`pan.baidu.com/s/14BjmzpkJREBbmaAOPyjbyg?pwd=nh3f`) and were not downloaded. Zero full laboratory slices exist on local disk.
2. **Local RFI-Patch Dataset is a Single-Category Subset (`PARTIALLY VERIFIED`):** Local disk contains 16 MATLAB v5/v7 files in `Data/RFI-Patch/RFI-1`. Each file is a 5-slice stack of shape `(5, 512, 1500)` representing 1 clean label and 4 degraded levels ($0, -10, -20, -30$ dB), yielding 64 labeled pairs. However, all 16 files cover only one RFI pattern (`RFI-1`) out of the 8 patterns described in the paper.
3. **Sentinel-1 Test Dataset is Incomplete (`PARTIALLY VERIFIED / BLOCKED`):** Local disk contains 12 slices in `Data/Scenario 1..3` ($3\text{ scenarios}\times 4\text{ files}$). The paper specifies 30 slices across 4 scenarios. 18 slices and Scenario 4 are completely missing.
4. **Data Integrity is 100% Valid:** Across all 28 on-disk `.mat` files, zero NaNs and zero Infs were detected. All arrays are stored in native `complex64` format.

---

## 2. Dataset Inventory & Sample Verification

Every `.mat` file on disk was inspected and cataloged in [`outputs/phase2/dataset_manifest.csv`](file:///C:/Users/hp/Pictures/BR%20Baseline%20implementation/sar_rfi_baseline/outputs/phase2/dataset_manifest.csv).

```
+--------------------------------------------------------------------------------------------------------------------------------------------+
|                                                         DATASET AVAILABILITY AUDIT                                                         |
+--------------------------+-----------------------+---------------------+-------------------+---------------------+-------------------------+
| Dataset Name             | Paper Requirement     | Local Disk Count    | Array Dimensions  | Dtype & Key         | Status                  |
+--------------------------+-----------------------+---------------------+-------------------+---------------------+-------------------------+
| SAR-RFI-DATASET (Lab)    | 640 slices (4 SIRs,   | 0 slices            | None              | None                | BLOCKED (Data Missing)  |
|                          | 4 scenarios, 8 types) |                     |                   |                     |                         |
| RFI-Patch (RFI-1)        | Unmentioned in paper; | 16 stacks (64 pairs)| (5, 512, 1500)    | complex64, 'ss'     | PARTIALLY VERIFIED      |
|                          | official patch sample |                     |                   |                     | (Single RFI type)       |
| Sentinel Polluted-SAR    | 30 slices (4 scenarios| 12 slices           | Scen 1: 512x1536  | complex64, 's4'     | PARTIALLY VERIFIED      |
|                          | maritime/port/urb/mtn)| (3 scenarios)       | Scen 2/3: 512x1500| complex64, 's3'     | (18 slices missing)     |
+--------------------------+-----------------------+---------------------+-------------------+---------------------+-------------------------+
```

### Detailed Inspection of Representative Samples

#### 1. RFI-Patch (`Data/RFI-Patch/RFI-1/s1.mat`, `first.mat`)
* **Format:** MATLAB 5.0 / v7 compressed file.
* **Variable Key:** `ss`.
* **Array Dimensions:** `(5, 512, 1500)`.
* **Data Type:** `complex64` (real and imaginary components present).
* **Integrity:** `has_nan = False`, `has_inf = False`.
* **Magnitude Range:** Min $|ss| = 0.0$, Max $|ss| = 2.329\times 10^4$.
* **Stack Structure (per `explaination.docx` & `ReadPatch.m`):**
  * `ss[0]`: Clean SAR image (Ground Truth Label).
  * `ss[1]`: Polluted image at documented $\text{SNR} = 0\text{ dB}$.
  * `ss[2]`: Polluted image at documented $\text{SNR} = -10\text{ dB}$.
  * `ss[3]`: Polluted image at documented $\text{SNR} = -20\text{ dB}$.
  * `ss[4]`: Polluted image at documented $\text{SNR} = -30\text{ dB}$.
* **Radar Geometry Orientation:** 
  * Empirical energy distribution confirms the **1500-sample axis is Range** (95% of RFI energy bandwidth is aligned along this axis) and the **512-sample axis is Azimuth**.
  * This refutes the conflicting comment in the author's `explaination.docx` (`% range=512, azimuth=1500`).

#### 2. Sentinel Polluted-SAR (`Data/Scenario 1/snr0.mat`, `snr15.mat`)
* **Format:** MATLAB 5.0 / v7 file.
* **Variable Key:** `s4`.
* **Dimensions:** `(512, 1536)` (Note: range dimension is 1536 samples, not 1500).
* **Data Type:** `complex64`.
* **Integrity:** `has_nan = False`, `has_inf = False`.
* **Magnitude Range:** Min $|s4| = 1.089\times 10^{-2}$, Max $|s4| = 4.958\times 10^3$.
* **Label Status:** Unlabeled (real Sentinel-1 field data; no clean reference image exists).

#### 3. Sentinel Polluted-SAR (`Data/Scenario 2/snr0.mat` & `Scenario 3/snr0.mat`)
* **Format:** MATLAB 5.0 / v7 file.
* **Variable Key:** `s3`.
* **Dimensions:** `(512, 1500)` (matches canonical paper dimension).
* **Data Type:** `complex64`.
* **Integrity:** `has_nan = False`, `has_inf = False`.
* **Magnitude Range:** Min $|s3| = 0.0$, Max $|s3| = 2.418\times 10^3$.
* **Label Status:** Unlabeled.

---

## 3. Dataset Splitting & Leakage Verification

### Paper Specification
* **Paper Text:** Section 5.1 trains on constructed data and tests on 30 Sentinel-1 slices. Section 5.2 trains at $-20$ dB and tests at $0, -10, -20, -30$ dB. **The paper does not state the split ratio, sample count, or random seed.**
* **Official Code:** The authors pre-split data into separate files (`train.mat`, `test.mat`) on local storage before training. The code contains no online partitioning.

### Repository Implementation (`sar_rfi_baseline/src/data/dataset.py`)
* **Function:** `assign_scene_splits(scene_ids, val_ratio=0.1, test_ratio=0.1, seed=42)`.
* **Split Policy:** 
  * Explicit **scene-level holdout** (default 80% train, 10% val, 10% test).
  * Grouped strictly by `scene_id` (derived from the file stem).
  * Enforces that **all 4 SIR variants of a given scene remain in the identical split**.
* **Empirical Execution on Local RFI-Patch (16 Scenes):**
  * Total Scenes: 16 (`fifth`, `first`, `fourth`, `s1`, `s10`, `s2`, `s3`, `s4`, `s5`, `s6`, `s7`, `s8`, `s9`, `second`, `sixth`, `third`).
  * `train` Scenes (12 scenes, 48 slice variants): `s6`, `s1`, `fifth`, `s4`, `s9`, `s2`, `fourth`, `s10`, `sixth`, `first`, `second`, `s5`.
  * `val` Scenes (2 scenes, 8 slice variants): `s8`, `s7`.
  * `test` Scenes (2 scenes, 8 slice variants): `s3`, `third`.
* **Leakage Verification:**
  * Cross-split intersection: $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$.
  * Zero scene-level or target-structure leakage. Enforced and validated by unit tests in `tests/test_dataset.py` and `tests/test_rfi_patch.py`.

---

## 4. Dataset Readiness Verdict for Experiments 5A–5C

| Experiment | Required Dataset | Available on Local Disk | Readiness Status | Actionable Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 5A (Sentinel-1 Generalization)** | 640 lab train slices; 30 Sentinel-1 test slices | 0/640 lab slices; 12/30 Sentinel-1 slices | **BLOCKED (Data Missing)** | Plumbing is verified via `run_exp1.py --smoke`. Full evaluation requires downloading the Baidu Netdisk dataset and acquiring the remaining 18 S1 slices. |
| **Exp 5B (Same Distribution)** | 640 lab slices across 4 SIRs (trained at -20 dB) | 16 RFI-Patch stacks (RFI-1 only) | **PARTIALLY VIABLE (Proxy) / BLOCKED (Faithful)** | Full 640-slice benchmark is blocked. Can run a proxy benchmark on 16 RFI-Patch scenes using `--source rfi_patch`. |
| **Exp 5C (Domain Shift)** | Clean SAR scenes + shift parameter distributions | 16 clean scenes from RFI-Patch; synthetic simulator | **APPROXIMATION (Executable)** | End-to-end executable via `run_exp3.py` using synthetic disjoint parameter sets (`exp3_train` vs. `exp3_test`). |
