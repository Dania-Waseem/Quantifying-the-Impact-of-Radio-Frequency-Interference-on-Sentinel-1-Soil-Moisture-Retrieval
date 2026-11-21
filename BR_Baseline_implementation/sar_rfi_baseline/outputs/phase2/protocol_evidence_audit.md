# Focused Evidence Audit: Experimental Protocol & Reproduction Analysis

**Project:** SAR RFI Baseline Reproduction  
**Reference Paper:** Fang et al., *Standardized Dataset and Image-Subspace-Based Method for Strip-Mode Synthetic Aperture Radar Radio Frequency Interference Suppression*, Remote Sensing, 2025, 17, 3688.  
**Official Codebase:** `SAR-RFI-DATASET-main`  
**Repository Working Copy:** `sar_rfi_baseline`  
**Audit Date:** 2026-11-21  

---

## 1. Executive Summary

This focused evidence audit verifies the empirical basis of every claim in `outputs/phase2/paper_protocol.md` against:
1. The exact text, equations, tables, and figures of the published paper (*Fang et al., Remote Sensing 2025, 17, 3688*).
2. The authors' official source code files and line numbers (`SAR-RFI-DATASET-main` / `third_party/FDNet`).
3. The repository implementation in `sar_rfi_baseline`.

No model architectures, training configurations, datasets, or experiment scripts were modified during this audit. Every discrepancy, omission, and approximation has been identified and classified into standardized reproduction tiers.

---

## 2. Resolution of Key Audit Issues

### Issue 1: Dataset Splitting (Repository 80/10/10 vs. Paper's Unspecified Split)
* **Paper Verification:** Section 3.1 describes 640 laboratory slices ($512\times 1500$); Section 5.1 describes training on constructed data and testing on 30 Sentinel-1 slices; Section 5.2 describes training at $-20$ dB and testing at $0, -10, -20, -30$ dB. **The paper text nowhere specifies the training/validation/test split ratio or slice partitioning.**
* **Official Code Verification:** In `SAR-RFI-DATASET-main/.../extracted/Code/FDNet/dataset/mydataset.py:7-40`, the code loads pre-partitioned MATLAB arrays from hardcoded author paths (`train.mat`, `snr15.mat`). The code contains no split logic because partitioning was performed offline before data creation.
* **Repository Implementation:** `sar_rfi_baseline/src/data/dataset.py:assign_scene_splits()` implements an explicit scene-level partition with default ratios `val_ratio=0.1, test_ratio=0.1` (yielding an 80% train, 10% val, 10% test split). Scenes are grouped by `scene_id` to strictly prevent data leakage across splits.
* **Audit Verdict:** **REASONABLE IMPLEMENTATION CHOICE**. The paper's split is **NOT SPECIFIED**. The repository's 80/10/10 scene-level split is a sound engineering decision that improves upon the unstated official procedure by guaranteeing zero cross-scene evaluation contamination.

---

### Issue 2: Optimizer and Learning Rate per Model Architecture
* **Paper Verification:** Section 4.3 (Page 14, lines 17–20) states globally:
  > *"For standardized comparison, each model underwent 100-epoch training cycles with a uniformly applied batch size of 4, ensuring consistent computational throughput and memory utilization patterns across all experimental iterations. The initial learning rate is set to 0.0002, and the AdamW optimizer is adopted."*
  The paper text describes this configuration uniformly across all four architectures (DNCNN, PISNet, UNet, FDNet).
* **Official Code Verification:**
  * **FDNet / UNet / DCNet** (`extracted/Code/FDNet/options.py:16-17`):
    `--optimizer`: default `'adamw'`  
    `--lr_initial`: default `0.0002`  
    `--nepoch`: default `101`  
  * **PISNet / IMN** (`extracted/Code/PISNet+IMN/train.py:72`):
    `optimizer = torch.optim.Adam(Net.parameters(), lr=LR)` where `LR = 0.0002`.  
    *Discrepancy:* Official PISNet code uses `torch.optim.Adam`, **not `AdamW`**.
  * **FDNet 2024 Precedent** (`remotesensing-16-02540-v2.pdf`, Table 2 & `train_FDNet.py:96`):
    In the authors' earlier 2024 FDNet paper, FDNet was trained with learning rate `2e-3` (0.002) and a `StepLR(50, 0.5)` schedule.
* **Repository Implementation (`src/train/train.py:495-497`):**
  * DNCNN, PISNet, UNet, DCNet, IMN: AdamW, initial lr = `2e-4`, weight decay = `0.01`, no step schedule. (Matches paper Section 4.3).
  * FDNet default: AdamW, initial lr = `2e-3`, weight decay = `0.02`, StepLR(50, 0.5). (Preserves the 2024 official code behavior unless `--lr 2e-4` is explicitly passed).
* **Audit Verdict:**
  * DNCNN / PISNet / UNet: **PAPER-MATCHED** (AdamW, lr 2e-4).
  * Official PISNet code divergence: **REPRESENTATIVE DIVERGENCE** (Official code used Adam; paper reported AdamW).
  * FDNet lr 2e-3 in repo: **REASONABLE IMPLEMENTATION CHOICE** (Aligns with official code behavior, but requires `--lr 2e-4` for strict Section 4.3 adherence).

---

### Issue 3: Universality of Batch Size 4 and 100 Epochs
* **Paper Verification:** Section 4.3 (Page 14) explicitly states:
  > *"each model underwent 100-epoch training cycles with a uniformly applied batch size of 4 ... across all experimental iterations."*
  Sections 5.1 (Exp 5A), 5.2 (Exp 5B), 5.3 (Exp 5C), and 5.4 (Exp 5D) introduce no deviations or alternative epoch/batch settings.
* **Official Code Verification:**
  * `extracted/Code/FDNet/options.py:10, 72`: `--batch_size 4`, `--nepoch 101` (1-based indexing for 100 complete epochs).
  * `extracted/Code/PISNet+IMN/train.py:26, 50`: `EPOCH = 101`, `batch_size = opt.batch_size` (default 4).
* **Repository Implementation:** `configs/default.yaml` sets `epochs: 100`, `batch_size: 4`. `src/train/train.py` defaults to 100 epochs and batch size 4.
* **Audit Verdict:** **PAPER-MATCHED**. Batch size 4 and 100 epochs apply universally to all models and experiments.

---

### Issue 4: Recoverability of Experiment 5C Domain Shift Distributions
* **Paper Verification:** Section 5.3 ("Experiment Under the Domain Shift Condition", Pages 18–19) describes testing under inconsistent interference parameters and modulation modes. **The paper text withholds all numerical values**—no bandwidths, pulse widths, chirp rates, or modulation categories are specified.
* **Official Code Verification:** An exhaustive search of `SAR-RFI-DATASET-main/extracted/Code` revealed **zero scripts, zero test loaders, and zero parameter definitions for Experiment 5C**. The author codebase contains no implementation of this experiment.
* **Supplementary Material Audit:** The paper references online supplementary information, but no domain shift parameter definitions exist locally on disk.
* **Repository Implementation:** `src/data/param_sets.py` constructs synthetic disjoint parameter intervals:
  * `exp3_train`: chirp wideband only ($B \in [20, 50]$ MHz, $\tau \in [40, 100]\ \mu$s, $|k_{rfi}| \in [2\times 10^{11}, 1.25\times 10^{12}]$ Hz/s).
  * `exp3_test`: chirp wideband ($B \in [60, 100]$ MHz, $\tau \in [5, 20]\ \mu$s, $|k_{rfi}| \in [3\times 10^{12}, 2\times 10^{13}]$ Hz/s) plus narrowband and sinusoidal modulation.
* **Audit Verdict:** **APPROXIMATION**. The exact distributions cannot be recovered from paper, figures, or code release. The repository configuration is an engineering proxy that reproduces the qualitative behavior (performance collapse at $\le -20$ dB SIR) but cannot guarantee exact numerical replication of Figure 12.

---

### Issue 5: Experiment 5D (Echo Domain vs. Image Subspace) Component Separation
* **Paper Verification:** Section 5.4 (Pages 19–20) compares the image subspace against the echo domain across 4 SIR levels, noting a $+0.96$ dB PSNR and $+0.06$ SSIM improvement at $-30$ dB.
  * *Unspecified in Paper:*
    1. Neural network architecture evaluated on echo domain (presumably FDNet, but unstated).
    2. Exact radar echo simulator or raw echo data source.
    3. SAR imaging processor utilized (Range-Doppler, Chirp Scaling, or Back-Projection).
    4. Echo-domain input dimensions and preprocessing.
    5. Training protocol (optimizer, loss, epochs) for echo-domain suppression.
* **Official Code Release Verification:**
  * *Absent from Official Code Release:*
    1. No raw radar echo dataset is included in `SAR-RFI-DATASET-main`.
    2. No radar chirp simulator or echo generator code is provided.
    3. No echo-domain training or testing scripts exist.
    4. No SAR imaging algorithms (range compression, RCMC, azimuth compression) are provided.
* **Repository Implementation:** Unimplemented.
* **Audit Verdict:** **BLOCKED (Missing Core Components)**. The experiment relies on components that are simultaneously **NOT SPECIFIED** in the paper and **ABSENT** from the official release.

---

### Issue 6: 8-Bit Metric Quantization Peak Normalization
* **Official Code Verification (`extracted/Code/FDNet/test/A_test_fdnet.py:184-203`):**
  ```python
  max1 = np.max(img_H)                                      # Ground-truth peak
  img_net = np.uint8((img_net / max1 * 255.0).round())      # Prediction scaled by max1
  img_H   = np.uint8((img_H   / max1 * 255.0).round())      # Ground truth scaled by max1
  psnr = util.calculate_psnr(img_net, img_H, border=border)
  ssim = util.calculate_ssim(img_net, img_H, border=border)
  ```
  The official code unequivocally scales **both** the prediction (`img_net`) and the ground truth (`img_H`) by the **same peak value** (`max1 = np.max(img_H)`).
* **Repository Implementation (`src/eval/metrics.py:116-121, 152-157`):**
  ```python
  peak = float(ref.max()) if data_range is None else float(data_range)
  ref_u8 = quantize_to_uint8(ref, peak=peak).astype(np.float64)
  est_u8 = quantize_to_uint8(est, peak=peak).astype(np.float64)
  ```
  The repository explicitly computes `peak = float(ref.max())` and passes that identical peak to both `quantize_to_uint8(ref, peak=peak)` and `quantize_to_uint8(est, peak=peak)`.
* **Audit Verdict:** **PAPER-MATCHED & CODE-ALIGNED**. The repository faithfully reproduces the official evaluation logic by applying the ground-truth peak ceiling to both images.

---

### Issue 7: STFT / ISTFT Reconstruction, Window, Hop, Padding & Bins
* **Paper Verification:** Section 4.1 (Algorithm 1, Steps II and IV; Equations 20 and 22, Page 12):
  * Window type: **Hamming window**.
  * Window length: **$L/8$** (with $L=1500$ range samples, $L/8 = 187.5 \approx 187$ samples).
  * Hop size / Overlap: **NOT SPECIFIED in paper text.**
  * FFT length ($N_{fft}$): **NOT SPECIFIED in paper text.**
  * Padding / Centering: **NOT SPECIFIED in paper text.**
  * Frequency-bin representation: **NOT SPECIFIED in paper text.**
* **Official Code Verification:**
  * The authors' released Python codebase contains **no STFT or ISTFT code whatsoever**. The official Python training scripts loaded static real-valued 2D float32 crops (`[N, 1, 512, 512]`) generated offline.
* **Repository Implementation (`src/utils/tf.py:38-104`):**
  * `win_len = length // 8` ($187$ for $L=1500$). Matches paper $L/8$.
  * `hop = win_len // 4` ($46$ samples, 75% overlap). Fulfills the Constant Overlap-Add (COLA) requirement for perfect reconstruction in `torch.istft`.
  * `n_fft = win_len` ($187$ frequency bins).
  * `center = True` with `pad_mode="reflect"`.
  * `onesided = False` (Preserves both positive and negative frequencies for complex SAR signals, yielding complex spectrograms of shape `(..., 187, 33)`).
  * Invertibility: Unit tests (`tests/test_tf.py`) confirm exact machine-precision roundtrip reconstruction ($\ell_\infty < 10^{-6}$).
* **Audit Verdict:** **REASONABLE IMPLEMENTATION CHOICE**. The forward/inverse pipeline strictly honors the paper's Hamming $L/8$ window specification. Because the hop, padding, and FFT parameters were omitted in the paper and absent in the code release, the repository's 75% COLA-compliant implementation is a valid, invertible engineering choice, but cannot be claimed as an exact duplicate of the authors' private MATLAB generator.

---

## 3. Comprehensive List of Reproduction Deviations

Every deviation between the published paper, official code, and repository implementation is classified into one of five standard tiers:

```
+--------------------------------------------------------------------------------------------------------------------------------------------+
|                                                  REPRODUCTION DEVIATION CLASSIFICATION MATRIX                                              |
+----+--------------------------------+-----------------------------------+-----------------------------------+------------------------------+
| #  | Component                      | Classification                    | Paper / Official Baseline         | Repository Implementation    |
+----+--------------------------------+-----------------------------------+-----------------------------------+------------------------------+
| 1  | Model Architectures            | PAPER-MATCHED                     | Custom UNet (dim 32), DCNet,      | Exact state-dict equivalence |
|    |                                |                                   | FDUformer, PISNet (no sigmoid)    | and forward output match     |
| 2  | Training Epochs (100)          | PAPER-MATCHED                     | Section 4.3: 100 epochs           | 100 epochs default           |
| 3  | Batch Size (4)                 | PAPER-MATCHED                     | Section 4.3: batch size 4         | batch_size = 4 default       |
| 4  | Baseline Optimizer (AdamW)     | PAPER-MATCHED                     | Section 4.3: AdamW, lr 2e-4       | AdamW, lr 2e-4               |
| 5  | Metric Quantization (8-bit)    | PAPER-MATCHED                     | Official code: uint8 [0, 255]     | quantize_8bit=True matches   |
|    |                                |                                   | scaled by max|X|                  | identical ref peak scaling   |
| 6  | Peak Normalization (Eq. 26)    | PAPER-MATCHED                     | Eq. 26: X / max(X), Y / max(X)    | normalize() divides by peak  |
| 7  | STFT Window & Length           | PAPER-MATCHED                     | Hamming window, length L/8        | Hamming, win_len = L // 8    |
| 8  | STFT Hop / Overlap / Padding   | REASONABLE IMPLEMENTATION CHOICE  | Not specified in paper text;      | 75% COLA overlap (L/32),     |
|    |                                |                                   | absent in official Python code    | reflect padding, two-sided   |
| 9  | Data Representation            | REASONABLE IMPLEMENTATION CHOICE  | Official code: real 512x512 crops;| Dynamic on-the-fly STFT/ISTFT|
|    |                                |                                   | Paper: Algorithm 1 STFT           | 2-channel complex (2, F, T)  |
| 10 | Scene-Level Splitting (80/10/10| REASONABLE IMPLEMENTATION CHOICE  | Not specified; official code used | Strict zero-leakage scene    |
|    |                                |                                   | pre-split local MAT files         | holdout (0.8 / 0.1 / 0.1)    |
| 11 | Validation Policy              | REASONABLE IMPLEMENTATION CHOICE  | Official validation code was dead | Active validation per epoch; |
|    |                                |                                   | (never evaluated or saved best)   | best checkpoint tracking     |
| 12 | FDNet Default Learning Rate    | REASONABLE IMPLEMENTATION CHOICE  | Paper: 2e-4; 2024 paper & code:   | Defaults to 2e-3 (StepLR 50);|
|    |                                |                                   | 2e-3 with StepLR(50, 0.5)         | accepts --lr 2e-4 for paper  |
| 13 | Official PISNet Optimizer      | REASONABLE IMPLEMENTATION CHOICE  | Official code used Adam (not AdamW)| Uses AdamW per paper text   |
| 14 | Domain Shift Parameters (5C)   | APPROXIMATION                     | Withheld in paper text;           | Disjoint parameter intervals |
|    |                                |                                   | absent in official release        | in param_sets.py             |
| 15 | Lab Dataset (640 Slices)       | BLOCKED (Data Missing)            | 640 full slices on Baidu Netdisk  | 0 / 640 slices locally       |
| 16 | Sentinel-1 Test Set            | BLOCKED (Data Missing)            | 30 slices in Section 3.2          | 12 / 30 slices locally       |
| 17 | Pretrained Checkpoints         | BLOCKED (Data Missing)            | Not released by authors           | 0 checkpoints locally        |
| 18 | Experiment 5D (Echo Domain)    | BLOCKED (Missing Core Components) | Unspecified in paper; absent in   | Completely unimplemented     |
|    |                                |                                   | code release                      | (needs radar simulator & BPA)|
+----+--------------------------------+-----------------------------------+-----------------------------------+------------------------------+
```

---

## 4. Summary of Remaining Blockers

1. **Upstream Data Availability:** Complete replication of Experiments 5A and 5B requires downloading the 640 laboratory slices from the official Baidu Netdisk link and acquiring the missing 18 Sentinel-1 test slices.
2. **Compute Hardware Requirements:** Full 100-epoch training across 4 architectures requires an NVIDIA CUDA GPU with $\ge 16$ GB VRAM (approx. 4–6 hours on RTX 3090 vs. multiple weeks on CPU).
3. **Experiment 5C Parameter Invariance:** Numerical curves for Figure 12 can only be approximated due to withheld parameter intervals.
4. **Experiment 5D Implementation Deficit:** Experiment 5D is blocked by the absence of a raw radar echo synthesizer and a Range-Doppler / Back-Projection imaging engine.
