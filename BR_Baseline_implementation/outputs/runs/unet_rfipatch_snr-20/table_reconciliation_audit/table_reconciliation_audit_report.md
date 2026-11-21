# Scientific Audit and Reconciliation Report: UNet Baseline Evaluation Tables

**Target Investigation:** Discrepancy between Table 1 and Table 2 in `SAR_RFI_UNet_Experiments_and_Results`  
**Run Directory:** `outputs/runs/unet_rfipatch_snr-20`  
**Audit Date:** 2026-11-21  
**Status:** Audit Complete & Tables Reconciled  

---

## 1. Executive Summary

This audit investigates and resolves the numerical discrepancies between Table 1 and Table 2 in `SAR_RFI_UNet_Experiments_and_Results` (`research_results.md` and `.docx`), specifically for the documented -20 dB condition:
- **Input MSE:** Table 1 reported **693.54**, whereas Table 2 reported **645.74** ($\Delta = -47.80$).
- **Output MSE:** Table 1 reported **721.16**, whereas Table 2 reported **671.39** ($\Delta = -49.77$).

### Key Findings:
1. **Origin of Discrepancy:**
   - Table 1 was generated dynamically in `generate_paper_evaluation.py` from the per-scene test evaluation arrays of held-out scenes `s3` (Input MSE: 1028.59, Output MSE: 1069.33) and `third` (Input MSE: 358.50, Output MSE: 372.99), yielding the true arithmetic mean of **693.54** (input) and **721.16** (output).
   - Table 2 in `generate_paper_evaluation.py` (lines 405–410) and `generate_docx_report.py` was **hardcoded as a static string** containing outdated placeholder values (Input MSE: 645.74, Output MSE: 671.39). The script failed to dynamically populate Table 2 from the computed results dataframe `df_all`.
2. **Authoritative Values:**
   - The authoritative 8-bit quantized values for the -20 dB condition across the test split (`s3`, `third`) are **Input MSE = 693.54**, **Output MSE = 721.16** ($\Delta = +27.62$).
   - The authoritative Float64 continuous normalized values are **Input MSE = 0.01066**, **Output MSE = 0.01109** ($\Delta = +0.00042$).
3. **Protocol Mixing:**
   - Table 2 labeled the columns as `Input MSE (u8)` and `UNet MSE (u8)`, but the figures were neither valid 8-bit quantized MSEs nor valid Float64 continuous MSEs. Furthermore, Table 2 omitted Float64 continuous metrics entirely.
4. **Reproducibility:**
   - Re-running inference directly from `best.pt` reproduces `test_metrics.csv` bit-for-bit with 0.000000 error.

---

## 2. Inventory of Evaluation Scripts and Artifacts

| Script / Artifact | Path | Role in Evaluation | Audit Status |
| :--- | :--- | :--- | :--- |
| `generate_paper_evaluation.py` | `sar_rfi_baseline/scripts/generate_paper_evaluation.py` | Primary generator for test evaluation, metric CSVs, summary JSON, and research results markdown. | **Flawed Template:** Table 1 dynamically computed; Table 2 hardcoded static string. |
| `generate_docx_report.py` | `sar_rfi_baseline/scripts/generate_docx_report.py` | Generates formatted `.docx` report. | **Flawed Template:** Copied hardcoded Table 2 from `generate_paper_evaluation.py`. |
| `run_test_evaluation.py` | `sar_rfi_baseline/scripts/run_test_evaluation.py` | Earlier evaluation script. | Generated PSNR/SSIM CSVs; omitted MSE computation. |
| `eval_proxy_unet.py` | `sar_rfi_baseline/scripts/eval_proxy_unet.py` | Diagnostic visualizer and evaluator. | Computed PSNR/SSIM across levels; saved `eval_results.csv`. |
| `test_metrics.csv` | `outputs/runs/unet_rfipatch_snr-20/test_evaluation/test_metrics.csv` | Raw per-slice metric log across all 4 levels and both test scenes (8 rows). | **Authoritative Ground Truth.** Contains exact bit-level metrics. |
| `per_scene_metrics.csv` | `outputs/runs/unet_rfipatch_snr-20/test_evaluation/per_scene_metrics.csv` | Raw metric log for -20 dB condition (2 rows: `s3`, `third`). | **Authoritative Ground Truth** for target condition. |
| `evaluation_summary.json` | `outputs/runs/unet_rfipatch_snr-20/test_evaluation/evaluation_summary.json` | Structured JSON summary of model, config, and target -20 dB metrics. | Matches Table 1 and `per_scene_metrics.csv`. |

---

## 3. Dataset, Split, and Sample Size Verification

- **Dataset Source:** RFI-Patch benchmark (`Data/RFI-Patch/RFI-1/`) consisting of 16 MAT stacks of shape `(5, 512, 1500)` complex64.
- **Split Configuration (`splits.json`):**
  - **Training Split (12 scenes):** `fifth`, `first`, `fourth`, `s1`, `s10`, `s2`, `s4`, `s5`, `s6`, `s9`, `second`, `sixth`.
  - **Validation Split (2 scenes):** `s7`, `s8`.
  - **Held-Out Test Split (2 scenes):** `s3`, `third`.
- **Sample Counts:**
  - **Table 1 (Target Level -20 dB):** Exactly 2 samples (`s3` at index 3, `third` at index 3).
  - **Table 2 (Cross-Level Benchmark):** Exactly 8 samples (2 scenes $\times$ 4 levels: 0 dB, -10 dB, -20 dB, -30 dB).
- **Leakage Check:** Confirmed 0 overlap between test, validation, and training scenes.

---

## 4. Verification of Preprocessing, Normalization, Quantization, and Formulas

Both tables are based on inference executions sharing identical preprocessing:
1. **Time-Frequency Processing:**
   - 1D Range-line STFT using Hamming window ($N=1500$, win_len $= 187$, hop $= 46$, FFT length $= 187$).
   - Periodic Hamming window satisfying COLA (invertibility error $< 1.16 \times 10^{-7}$).
2. **Spectrogram Normalization:**
   - Peak magnitude normalization per slice: $\alpha = \max |\text{STFT}(Y)|$.
   - Network evaluated in batches of 128 rows on $(2, F, T)$ real and imaginary channels.
   - Denormalization by $\alpha$ prior to ISTFT synthesis.
3. **Metric Formulas:**
   - **8-bit Quantized Protocol (Official Author Benchmark Protocol):**
     - Scales magnitude array $M = |S|$ by $\text{peak} = \max(|S_{\text{clean}}|)$.
     - Maps to $[0, 255]$: $M_{u8} = \text{round}\left(\text{clip}\left(\frac{M}{\text{peak}} \times 255.0, 0, 255\right)\right)$.
     - $\text{MSE}_{u8} = \frac{1}{N} \sum (M_{\text{clean}, u8} - M_{\text{est}, u8})^2$.
     - $\text{PSNR}_{u8} = 20 \log_{10}\left(\frac{255.0}{\sqrt{\text{MSE}_{u8}}}\right)$.
     - $\text{SSIM}_{u8}$: standard SSIM with $7 \times 7$ Gaussian window ($K_1=0.01, K_2=0.03, L=255$).
   - **Float64 Continuous Protocol (Unquantized Floating-Point Precision):**
     - Normalizes magnitude array: $M_{\text{norm}} = M / \text{peak} \in [0, 1]$.
     - $\text{MSE}_{\text{norm}} = \frac{1}{N} \sum (M_{\text{clean, norm}} - M_{\text{est, norm}})^2$.
     - $\text{PSNR}_{f64} = 20 \log_{10}\left(\frac{1.0}{\sqrt{\text{MSE}_{\text{norm}}}}\right)$.
     - $\text{SSIM}_{f64}$: standard SSIM on normalized float64 arrays ($L=1.0$).

---

## 5. Detailed Breakdown of Inconsistencies

### A. The -20 dB Discrepancy (693.54 vs. 645.74)
- In `test_metrics.csv`:
  - `s3` at -20 dB: $\text{MSE}_{u8} = 1028.589379$ (input), $1069.334632$ (output).
  - `third` at -20 dB: $\text{MSE}_{u8} = 358.498255$ (input), $372.985185$ (output).
  - Arithmetic Mean:
    $$\text{Mean Input MSE}_{u8} = \frac{1028.589379 + 358.498255}{2} = 693.543817 \to \mathbf{693.54}$$
    $$\text{Mean Output MSE}_{u8} = \frac{1069.334632 + 372.985185}{2} = 721.159908 \to \mathbf{721.16}$$
- **Table 1** reported **693.54** and **721.16**, exactly matching the mathematical mean.
- **Table 2** reported **645.74** and **671.39**.
- **Root Cause:** In `sar_rfi_baseline/scripts/generate_paper_evaluation.py`, lines 405–410, Table 2 was hardcoded inside the multiline Markdown string template instead of being dynamically formatted from `df_all.groupby("snr_db").mean()`.

### B. Other Cross-Level Discrepancies in Table 2
The same hardcoding error affected every other row in Table 2:
- **0.0 dB Level:** Table 2 reported Input MSE **3.90** and Output MSE **4.37**. True means are **4.39** and **4.86**.
- **-10.0 dB Level:** Table 2 reported Input MSE **52.88** and Output MSE **55.51**. True means are **59.21** and **62.09**.
- **-30.0 dB Level:** Table 2 reported Input MSE **6695.42** and Output MSE **6940.75**. True means are **6693.78** and **6930.31**.

---

## 6. Authoritative Reconciled Quantitative Tables

### Table 1 (Reconciled): Target Condition Performance (Documented SIR = -20 dB)
*Evaluated on held-out test scenes `s3` and `third` under strict scene-level isolation.*

| Test Scene | Evaluation Protocol | Noisy Input PSNR (dB) | UNet Output PSNR (dB) | $\Delta$ PSNR (dB) | Noisy Input SSIM | UNet Output SSIM | $\Delta$ SSIM | Noisy Input MSE | UNet Output MSE | $\Delta$ MSE | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`s3`** | 8-bit Quantized | 18.01 | 17.84 | **-0.17** | 0.5121 | 0.4879 | **-0.0243** | 1028.59 | 1069.33 | **+40.75** | Degraded |
| **`s3`** | Float64 Continuous | 18.01 | 17.84 | **-0.17** | 0.5127 | 0.4892 | **-0.0235** | 0.01582 | 0.01644 | **+0.00063** | Degraded |
| **`third`** | 8-bit Quantized | 22.59 | 22.41 | **-0.17** | 0.5352 | 0.5163 | **-0.0190** | 358.50 | 372.99 | **+14.49** | Degraded |
| **`third`** | Float64 Continuous | 22.59 | 22.42 | **-0.17** | 0.5359 | 0.5180 | **-0.0179** | 0.00551 | 0.00573 | **+0.00022** | Degraded |
| **AVERAGE** | **8-bit Quantized** | **20.30** | **20.13** | **-0.17** | **0.5237** | **0.5021** | **-0.0216** | **693.54** | **721.16** | **+27.62** | **Degraded** |
| **AVERAGE** | **Float64 Continuous** | **20.30** | **20.13** | **-0.17** | **0.5243** | **0.5036** | **-0.0207** | **0.01066** | **0.01109** | **+0.00042** | **Degraded** |

---

### Table 2 (Reconciled): Cross-Level Generalization Performance Across All SIR Levels
*Averaged across the 2 held-out test scenes (`s3`, `third`) for both 8-bit Quantized and Float64 Continuous protocols.*

| Documented SIR | Protocol | Input PSNR (dB) | UNet PSNR (dB) | $\Delta$ PSNR (dB) | Input SSIM | UNet SSIM | $\Delta$ SSIM | Input MSE | UNet MSE | $\Delta$ MSE | Improved |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 dB** | 8-bit Quantized | 42.26 | 41.76 | **-0.50** | 0.9046 | 0.8964 | **-0.0082** | 4.39 | 4.86 | **+0.48** | 0 / 2 |
| | Float64 Cont. | 42.40 | 41.94 | **-0.45** | 0.9061 | 0.8988 | **-0.0074** | 0.00007 | 0.00007 | **+0.00001** | 0 / 2 |
| **-10.0 dB** | 8-bit Quantized | 30.98 | 30.76 | **-0.21** | 0.6791 | 0.6696 | **-0.0095** | 59.21 | 62.09 | **+2.89** | 0 / 2 |
| | Float64 Cont. | 30.99 | 30.78 | **-0.21** | 0.6801 | 0.6715 | **-0.0087** | 0.00091 | 0.00095 | **+0.00004** | 0 / 2 |
| **-20.0 dB** | **8-bit Quantized** | **20.30** | **20.13** | **-0.17** | **0.5237** | **0.5021** | **-0.0216** | **693.54** | **721.16** | **+27.62** | **0 / 2** |
| | **Float64 Cont.** | **20.30** | **20.13** | **-0.17** | **0.5243** | **0.5036** | **-0.0207** | **0.01066** | **0.01109** | **+0.00042** | **0 / 2** |
| **-30.0 dB** | 8-bit Quantized | 10.45 | 10.29 | **-0.16** | 0.4399 | 0.3658 | **-0.0741** | 6693.78 | 6930.31 | **+236.53** | 0 / 2 |
| | Float64 Cont. | 10.44 | 10.27 | **-0.16** | 0.4403 | 0.3669 | **-0.0735** | 0.10329 | 0.10725 | **+0.00396** | 0 / 2 |

---

## 7. Artifact Directory and Verification Paths

All audit deliverables and reconciled datasets are saved in the isolated audit directory:
`outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/`

- **Per-Scene Reconciled CSV:** [`reconciled_per_scene_metrics.csv`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/reconciled_per_scene_metrics.csv)
- **Cross-Level Summary CSV:** [`reconciled_cross_level_summary.csv`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/reconciled_cross_level_summary.csv)
- **Reconciled JSON Summary:** [`reconciled_evaluation_summary.json`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/reconciled_evaluation_summary.json)
- **Audit Report Document:** [`table_reconciliation_audit_report.md`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/table_reconciliation_audit_report.md)
- **Reconciled Research Results:** [`reconciled_research_results.md`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/outputs/runs/unet_rfipatch_snr-20/table_reconciliation_audit/reconciled_research_results.md)
