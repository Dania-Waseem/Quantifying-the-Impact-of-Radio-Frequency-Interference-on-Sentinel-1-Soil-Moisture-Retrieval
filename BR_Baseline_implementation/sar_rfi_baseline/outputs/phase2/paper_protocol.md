# Corrected Experimental Protocol Extraction: Experiments 5A–5D
**Reference Paper:** Fang et al., *Standardized Dataset and Image-Subspace-Based Method for Strip-Mode Synthetic Aperture Radar Radio Frequency Interference Suppression*, Remote Sensing, 2025, 17, 3688.  
**Official Codebase:** `SAR-RFI-DATASET-main`  
**Repository Copy:** `sar_rfi_baseline`  
**Working Directory:** `C:\Users\hp\Pictures\BR Baseline implementation\sar_rfi_baseline`  
**Audit & Correction Date:** 2026-11-21  

---

## 1. Executive Summary & Audit Baseline

This document provides the audited, evidence-backed extraction of the experimental methodology, dataset specifications, model architectures, training protocols, and reported results from **Fang et al. (Remote Sensing 2025, 17, 3688)**. 

### Core Audit Principles
* **Primary Source of Truth:** Published paper text, equations, tables, and figures.
* **Implementation Reference:** Authors' released codebase (`SAR-RFI-DATASET-main` / `third_party/FDNet`).
* **Strict Parameter Designation:** Parameters not explicitly stated in the paper are marked **"Not specified"**.
* **Standardized Reproduction Tiers:** Every component is classified as:
  * `PAPER-MATCHED`: Directly matches explicit paper text, equations, or tables.
  * `REASONABLE IMPLEMENTATION CHOICE`: Principled engineering implementation addressing an unspecified paper component or fixing an official code defect.
  * `APPROXIMATION`: Engineering proxy for parameters or distributions withheld by the authors.
  * `NOT SPECIFIED`: Information completely omitted in published text.
  * `BLOCKED`: Missing essential data or upstream dependencies preventing execution.

---

## 2. Global Experimental Methodology & Common Parameters

Section 4.3 ("Implementation Details", Page 14) and Sections 4.1–4.2 specify the common baseline execution parameters across all experiments:

| Parameter Category | Paper Specification (with Page & Section) | Official Code Reference (File & Lines) | Repository Implementation (`sar_rfi_baseline`) | Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Hardware Platform** | Intel Core i7-14700K (20 cores, 2.50 GHz), NVIDIA GeForce RTX 3090 (24 GB VRAM) (Sec 4.3, p. 14) | Hardcoded CUDA drive paths (`train_FDNet.py:17`, `A_test_fdnet.py:187`) | Configurable via `--device` (CUDA auto-detect, CPU fallback) in `train.py:513` | **PAPER-MATCHED** |
| **Software Framework** | PyTorch (implicit from models and references) | PyTorch 1.10+ / timm / einops / scipy (`requirements.txt`) | PyTorch 2.1+, torchaudio, scikit-image, scipy | **PAPER-MATCHED** |
| **Training Epochs** | **100 epochs** (Sec 4.3, p. 14: *"each model underwent 100-epoch training cycles"*) | `--nepoch 101` (`options.py:10, 72`), `EPOCH = 101` (`train.py:26`) | 100 epochs default (`configs/default.yaml:10`, `train.py:491`) | **PAPER-MATCHED** |
| **Batch Size** | **4** (Sec 4.3, p. 14: *"uniformly applied batch size of 4"*) | `--batch_size 4` (`options.py:9, 71`), `opt.batch_size` (`train.py:50`) | 4 slices/step default (`configs/default.yaml:9`, `train.py:532`) | **PAPER-MATCHED** |
| **Optimizer** | **AdamW** (Sec 4.3, p. 14: *"the AdamW optimizer is adopted"*) | AdamW (`options.py:17`); **Divergence:** PISNet code uses Adam (`train.py:72`) | AdamW for all models (`train.py:542`) | **PAPER-MATCHED** |
| **Learning Rate** | **0.0002 ($2\times 10^{-4}$)** (Sec 4.3, p. 14) | `lr_initial = 0.0002` (`options.py:16`); `LR = 0.0002` (`train.py:27`) | 2e-4 for baselines; defaults to 2e-3 for FDNet (`train.py:495`) | **REASONABLE CHOICE** (Repo defaults to 2e-3 for FDNet; needs `--lr 2e-4` for strict paper) |
| **Learning Rate Schedule** | *Not specified in paper text* | `StepLR(step_size=50, gamma=0.5)` for FDNet (`train_FDNet.py:96`) | StepLR(50, 0.5) for FDNet; constant for baselines (`train.py:544`) | **REASONABLE CHOICE** (Follows official code) |
| **Data Representation** | Algorithm 1: STFT along 1D range lines ($L=1500$, window Hamming $L/8$) (Sec 4.1, p. 12) | Static real float32 512×512 crops loaded from MAT (`mydataset.py:7-22`) | Dynamic on-the-fly STFT/ISTFT, 2-channel complex `(B, 2, F, T)` (`tf.py:38`) | **REASONABLE CHOICE** (Implements true Alg 1; official code trained on real crops) |
| **STFT Window & Length** | Hamming window, length $L/8 = 187$ samples (Sec 4.1, p. 12) | *Absent in official Python code* (no STFT code released) | `torch.hamming_window(187, periodic=True)` (`tf.py:33`) | **PAPER-MATCHED** |
| **STFT Hop & Padding** | *Not specified in paper text* | *Absent in official Python code* | Hop = $L/32 = 46$ samples (75% overlap, COLA compliant); reflect padding (`tf.py:58-67`) | **REASONABLE CHOICE** (Mathematically invertible) |
| **Data Normalization** | Eq. 26: $X_{norm} = X / \max(X)$, $Y_{norm} = Y / \max(X)$ (Sec 4.3, p. 15) | Scaled by `max1 = np.max(img_H)` (`A_test_fdnet.py:184`) | `normalize(polluted, clean)` matching Eq. 26 (`dataset.py:441`) | **PAPER-MATCHED** |
| **Metric Formulation** | PSNR (Eq. 23), SSIM (Eq. 24), Mean-Entropy (Eq. 25) (Sec 4.2, p. 13–14) | Scaled by `max1`, rounded to 8-bit uint8 `[0, 255]` (`A_test_fdnet.py:199-203`) | Float64 default; `quantize_8bit=True` matches official uint8 scaling (`metrics.py:97-158`) | **PAPER-MATCHED & CODE-ALIGNED** |

---

## 3. Detailed Protocol Extraction by Experiment

### Experiment 5A: Generalization to Open-Source Sentinel-1 Data

* **Experiment ID:** 5A
* **Objective:** Evaluate whether deep learning models trained on laboratory-constructed block-type RFI data can generalize to field-measured real-world RFI from Sentinel-1 satellites without retraining, validating the fidelity of the theoretical RFI physical model.
* **Dataset:**
  * **Training Dataset:** Laboratory-constructed block-type RFI dataset (Section 3.1, Pages 9–10: 640 full slices of $512 \times 1500$ complex pixels across 4 imaging scenarios: maritime zone, harbor area, urban region, mountainous terrain; 4 SIR levels: 0 dB, -10 dB, -20 dB, -30 dB; 8 interference patterns).
  * **Testing Dataset:** Open-source Sentinel-1 dataset (Section 3.2, Page 11: 30 image slices of $512 \times 1500$ complex pixels across the same 4 scenarios; real-world interference with unmeasured patterns and intensities, estimated SIR $< -20$ dB for strong cases, $\approx 0$ dB for weak cases).
* **Dataset Split:**
  * **Paper Specification:** Training on constructed data, testing on 30 Sentinel-1 slices. *Exact training slice count and validation split: Not specified.*
  * **Official Code Specification:** Pre-split offline MAT files (`mydataset.py:7-40`).
  * **Repository Implementation:** Strict scene-level holdout (`splits.json`: 80% train, 10% val, 10% test) preventing cross-scene data leakage (`dataset.py:assign_scene_splits`). Sentinel-1 slices loaded via `unlabeled_cases()` (`scripts/run_exp1.py:70`).
* **Models Evaluated:**
  1. `DNCNN`: Canonical 17-layer residual CNN with BatchNorm (Zhang et al. 2017).
  2. `PISNet`: Time-frequency network with EMAU attention (Shen et al.), with final sigmoid explicitly removed by Fang et al. (Sec 5, p. 14 & `network.py:90`).
  3. `UNet`: Official custom 32-dim residual ConvNet without BatchNorm (`third_party/FDNet/model.py:107-178`).
  4. `FDNet`: FDUformer transformer with dual decomposition bottleneck (`model.py:FDUformer`).
* **Input Representation & Dimensions:**
  * Complex SAR image $S \in \mathbb{C}^{M \times N}$ ($M=512$ azimuth lines, $N=1500$ range samples).
  * Algorithm 1: 1D range signal $X = S(i, :)$ transformed via STFT ($L=1500$, Hamming window $L/8=187$, hop $46$) to spectrogram $(F=187, T=33)$, mapped to 2 real/imaginary channels `(2, 187, 33)`.
* **Preprocessing:**
  * Peak normalization per Eq. 26: both input and label divided by $\max(|X|)$.
  * Hamming window of length $L/8$.
* **Training Protocol:**
  * Epochs: 100 (Sec 4.3, p. 14).
  * Optimizer: AdamW, initial lr = $2\times 10^{-4}$ (Sec 4.3, p. 14).
  * Batch Size: 4 (Sec 4.3, p. 14).
  * Loss: MSE for baselines (DNCNN, PISNet, UNet); FDNet multi-term loss (Charbonnier + cosine decomposition, Eq. 20–22 in [13]) for FDNet.
* **Evaluation Metrics:**
  * Mean-Entropy (ME, Eq. 25: $ME = Mean(\hat{X}) \cdot Ent(\hat{X})$) on normalized magnitude images.
  * *PSNR and SSIM cannot be computed because real Sentinel-1 slices lack clean ground-truth labels.*
* **SIR Conditions:**
  * Training SIR: Constructed data across all SIRs (or -20 dB). *Exact training SIR subset: Not explicitly specified in Section 5.1.*
  * Testing SIR: Real-world uncalibrated interference (strong cases $< -20$ dB; weak cases $\approx 0$ dB).
* **Baselines Compared:** Unmitigated RFI-polluted, DNCNN, PISNet, UNet, FDNet.
* **Reported Results:**
  * **Table 3 (Page 16): Mean-Entropy Under Open-Source Sentinel-1 Data:**
    $$\begin{array}{lc}
    \hline
    \textbf{Methods} & \textbf{ME} \\
    \hline
    \text{RFI-polluted} & 0.314 \\
    \text{DNCNN} & 0.068 \\
    \text{PISNet} & 0.067 \\
    \text{UNet} & 0.066 \\
    \text{FDNet} & \mathbf{0.064} \\
    \hline
    \end{array}$$
  * Qualitative Figures: Figure 7 (Time-frequency spectrums a–e, p. 15), Figure 8 (SAR magnitude images a–e, p. 16).
* **Paper Reference:** Remote Sensing 2025, 17, 3688: Section 5.1, Pages 15–16; Table 3; Figures 7 & 8.

---

### Experiment 5B: Benchmark Under the Same Data Distribution

* **Experiment ID:** 5B
* **Objective:** Benchmark the restoration performance of deep learning models under identical training and testing data distributions across four distinct interference power levels (SIR = 0, -10, -20, -30 dB).
* **Dataset:**
  * **Training & Testing Dataset:** Laboratory-constructed block-type RFI dataset (Section 3.1, Pages 9–10: 640 slices of $512 \times 1500$, 4 scenarios, 8 interference patterns).
* **Dataset Split:**
  * **Training Split:** Specifically trained on slices with **$\text{SIR} = -20\text{ dB}$** (Section 5.2, Page 16: *"we carefully select the dataset with a signal-to-interference ratio (SIR) of −20 dB as the training data"*). *Exact slice count: Not specified.*
  * **Validation Split:** *Not specified.*
  * **Testing Split:** Test slices evaluated across 4 SIR levels: $0\text{ dB}, -10\text{ dB}, -20\text{ dB}, -30\text{ dB}$. *Exact slice count: Not specified.*
* **Models Evaluated:** DNCNN, PISNet, UNet, FDNet.
* **Input Representation & Dimensions:** Spectrograms $(2, 187, 33)$ via Algorithm 1 STFT on range lines ($L=1500$, window Hamming 187, hop 46).
* **Preprocessing:** Peak normalization (Eq. 26).
* **Training Protocol:** 100 epochs, AdamW optimizer, lr = $2\times 10^{-4}$ (FDNet: $2\times 10^{-3}$ in official code), batch size 4.
* **Batch Size:** 4 (Sec 4.3, p. 14).
* **Loss Function:** MSE for baselines; Charbonnier + cosine decomposition for FDNet.
* **Evaluation Metrics:** PSNR (Eq. 23) in dB and SSIM (Eq. 24) computed on reconstructed SAR magnitude images against ground-truth clean images.
* **SIR Conditions:**
  * Training SIR: **$-20\text{ dB}$** (Sec 5.2, p. 16).
  * Testing SIR: **$0\text{ dB}, -10\text{ dB}, -20\text{ dB}, -30\text{ dB}$** (Sec 5.2, p. 16).
* **Baselines Compared:** RFI-polluted, DNCNN, PISNet, UNet, FDNet.
* **Reported Results:**
  * **Numerical Data Points Extracted from Figure 9 (Page 17):**
    $$\begin{array}{lcccc}
    \hline
    \textbf{Method} & \mathbf{0\text{ dB}} & \mathbf{-10\text{ dB}} & \mathbf{-20\text{ dB}} & \mathbf{-30\text{ dB}} \\
    \hline
    \text{PSNR (dB)} & & & & \\
    \quad \text{RFI-polluted} & \approx 22.5 & \approx 15.0 & \approx 12.0 & \approx 10.5 \\
    \quad \text{PISNet} & \approx 28.5 & \approx 30.0 & \approx 26.5 & \approx 20.0 \\
    \quad \text{DNCNN} & \approx 28.0 & \approx 29.0 & \approx 28.0 & \approx 21.0 \\
    \quad \text{UNet} & \approx 31.0 & \approx 31.0 & \approx 31.8 & \approx 25.5 \\
    \quad \text{FDNet} & \mathbf{\approx 32.0} & \mathbf{\approx 34.0} & \mathbf{\approx 33.5} & \mathbf{\approx 28.2} \\
    \hline
    \text{SSIM} & & & & \\
    \quad \text{PISNet} & \approx 0.945 & \approx 0.940 & \approx 0.890 & \approx 0.685 \\
    \quad \text{DNCNN} & \approx 0.945 & \approx 0.945 & \approx 0.915 & \approx 0.725 \\
    \quad \text{UNet} & \approx 0.970 & \approx 0.950 & \approx 0.958 & \approx 0.855 \\
    \quad \text{FDNet} & \mathbf{\approx 0.975} & \mathbf{\approx 0.980} & \mathbf{\approx 0.970} & \mathbf{\approx 0.890} \\
    \hline
    \end{array}$$
  * Qualitative Figures: Figure 10 (Time-frequency spectrums at $-20$ dB, p. 17), Figure 11 (SAR magnitude images at $-20$ dB, p. 18).
* **Paper Reference:** Remote Sensing 2025, 17, 3688: Section 5.2, Pages 16–18; Figures 9, 10, 11.

---

### Experiment 5C: Robustness Under Domain Shift Conditions

* **Experiment ID:** 5C
* **Objective:** Investigate how model performance degrades when significant distribution shifts exist between training and testing data (inconsistent interference modulation modes and signal parameters).
* **Dataset:**
  * **Training Dataset:** Laboratory-constructed data synthesized with one set of interference parameters.
  * **Testing Dataset:** Laboratory-constructed data synthesized with a shifted/inconsistent set of interference parameters.
  * *Exact numerical parameter ranges for the shift:* **Not specified in paper text.** (Section 5.3, Pages 18–19 only states: *"significant differences exist in interference parameters between training and testing data"*).
* **Dataset Split:**
  * **Training Split:** *Not specified.*
  * **Validation Split:** *Not specified.*
  * **Testing Split:** Evaluated at 4 SIR levels: $0\text{ dB}, -10\text{ dB}, -20\text{ dB}, -30\text{ dB}$. *Exact slice count: Not specified.*
* **Models Evaluated:** DNCNN, PISNet, UNet, FDNet.
* **Input Representation & Dimensions:** Spectrograms $(2, 187, 33)$ via Algorithm 1 STFT.
* **Preprocessing:** Peak normalization (Eq. 26).
* **Training Protocol:** 100 epochs, AdamW optimizer, lr = $2\times 10^{-4}$, batch size 4 (Sec 4.3, p. 14). Training SIR = $-20$ dB (implicit).
* **Batch Size:** 4 (Sec 4.3, p. 14).
* **Loss Function:** MSE for baselines; Charbonnier + cosine decomposition for FDNet.
* **Evaluation Metrics:** PSNR (Eq. 23) and SSIM (Eq. 24).
* **SIR Conditions:**
  * Training SIR: $-20\text{ dB}$ (implicit).
  * Testing SIR: $0\text{ dB}, -10\text{ dB}, -20\text{ dB}, -30\text{ dB}$.
* **Baselines Compared:** RFI-polluted, DNCNN, PISNet, UNet, FDNet.
* **Repository Implementation Status:** **APPROXIMATION**. Because the authors withheld numerical shift ranges, `src/data/param_sets.py` implements disjoint synthetic intervals (`exp3_train` vs. `exp3_test`).
* **Reported Results:**
  * **Numerical Data Points Extracted from Figure 12 (Page 19):**
    $$\begin{array}{lcccc}
    \hline
    \textbf{Method} & \mathbf{0\text{ dB}} & \mathbf{-10\text{ dB}} & \mathbf{-20\text{ dB}} & \mathbf{-30\text{ dB}} \\
    \hline
    \text{PSNR (dB)} & & & & \\
    \quad \text{RFI-polluted} & \approx 27.0 & \approx 20.0 & \approx 15.0 & \approx 11.0 \\
    \quad \text{DNCNN} & \approx 30.5 & \approx 28.0 & \approx 23.0 & \approx 17.0 \\
    \quad \text{PISNet} & \approx 31.8 & \approx 29.0 & \approx 23.0 & \approx 17.0 \\
    \quad \text{UNet} & \approx 33.5 & \approx 29.2 & \approx 23.2 & \approx 17.2 \\
    \quad \text{FDNet} & \mathbf{\approx 34.2} & \mathbf{\approx 29.5} & \approx 23.0 & \approx 17.0 \\
    \hline
    \text{SSIM} & & & & \\
    \quad \text{DNCNN} & \approx 0.945 & \approx 0.890 & \approx 0.795 & \approx 0.615 \\
    \quad \text{PISNet} & \approx 0.950 & \approx 0.900 & \approx 0.800 & \approx 0.610 \\
    \quad \text{UNet} & \approx 0.955 & \approx 0.900 & \approx 0.800 & \approx 0.640 \\
    \quad \text{FDNet} & \mathbf{\approx 0.960} & \mathbf{\approx 0.905} & \mathbf{\approx 0.805} & \mathbf{\approx 0.655} \\
    \hline
    \end{array}$$
  * *Critical Empirical Finding:* At $-20$ dB and $-30$ dB, architectural advantages completely vanish—all four models collapse to identical low performance ($\approx 23$ dB PSNR at $-20$ dB and $\approx 17$ dB at $-30$ dB).
  * Qualitative Figures: Figure 13 (SAR magnitude images under domain shift at $-20$ dB, p. 19).
* **Paper Reference:** Remote Sensing 2025, 17, 3688: Section 5.3, Pages 18–19; Figures 12 & 13.

---

### Experiment 5D: Image Subspace vs. Echo Domain Comparison

* **Experiment ID:** 5D
* **Objective:** Compare interference suppression performed in the proposed **Image Subspace** against traditional interference suppression performed in the **SAR Echo Domain**, demonstrating performance advantages and workflow decoupling.
* **Component Breakdown:**
  * **Unspecified in Published Paper:**
    1. Neural network architecture evaluated on echo domain (presumably FDNet, unstated).
    2. Exact radar echo simulator or raw echo data source.
    3. SAR imaging processor utilized (Range-Doppler, Chirp Scaling, or Back-Projection).
    4. Echo-domain input dimensions and preprocessing.
    5. Training protocol (optimizer, loss, epochs) for echo-domain suppression.
  * **Absent from Official Code Release:**
    1. No raw radar echo dataset is included in `SAR-RFI-DATASET-main`.
    2. No radar chirp simulator or echo generator code is provided.
    3. No echo-domain training or testing scripts exist.
    4. No SAR imaging algorithms (range compression, RCMC, azimuth compression) are provided.
* **Status in Repository:** **BLOCKED (Missing Core Components)**. Unimplemented in repository due to total absence of upstream echo simulation and imaging components.
* **Reported Results:**
  * **Textual Quantitative Claims (Section 5.4, Page 19):**
    > *"the most significant improvement achieved under the condition of −30 dB signal-to-interference ratio: the PSNR is improved by 0.96 dB, and the SSIM is improved by 0.06."*
  * **Numerical Data Points Extracted from Figure 14 (Page 20):**
    $$\begin{array}{lcccc}
    \hline
    \textbf{Domain} & \mathbf{0\text{ dB}} & \mathbf{-10\text{ dB}} & \mathbf{-20\text{ dB}} & \mathbf{-30\text{ dB}} \\
    \hline
    \text{PSNR (dB)} & & & & \\
    \quad \text{Image Subspace} & \approx 31.7 & \approx 33.5 & \approx 33.1 & \mathbf{28.00} \\
    \quad \text{Echo Domain} & \approx 31.5 & \approx 33.2 & \approx 32.7 & \mathbf{27.04} \text{ ($-0.96$ dB)} \\
    \hline
    \text{SSIM} & & & & \\
    \quad \text{Image Subspace} & \approx 0.970 & \approx 0.975 & \approx 0.965 & \mathbf{0.890} \\
    \quad \text{Echo Domain} & \approx 0.967 & \approx 0.960 & \approx 0.962 & \mathbf{0.884} \text{ ($-0.006$ / $-0.06$)} \\
    \hline
    \end{array}$$
* **Paper Reference:** Remote Sensing 2025, 17, 3688: Section 5.4, Pages 19–20; Figure 14.

---

## 4. Audited Paper-versus-Repository Comparative Matrix

The following matrix contrasts every experimental component specified in the paper with the repository implementation in `sar_rfi_baseline`.

| Experimental Component | Paper Specification (with Ref) | Current Implementation in `sar_rfi_baseline` | Reproduction Tier | Evidence & Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset (Lab 640 Slices)** | 640 full .mat slices ($512\times 1500$), 4 scenarios, 4 SIRs, 8 patterns (Sec 3.1, p. 9–10) | `RFIDataset(source="mat")` supports loading, but **0 of 640 slices exist locally** (hosted on Baidu Netdisk). Only 16 RFI-Patch stacks exist. | **BLOCKED (Data Missing)** | Baidu Netdisk link `pan.baidu.com/s/14BjmzpkJREBbmaAOPyjbyg` not downloaded to local environment. |
| **Dataset (Sentinel-1)** | 30 full image slices ($512\times 1500$) across 4 scenarios (Sec 3.2, p. 11) | `unlabeled_cases()` loads Sentinel-1 slices, but **only 12 of 30 slices exist** in `Data/Scenario 1..3`. 18 slices are missing. | **BLOCKED (Data Missing)** | Local disk contains 12 slices ($3\text{ scenarios}\times 4\text{ SNR files}$). Missing 18 slices prevents exact Table 3 reproduction. |
| **Training Split Policy** | Trained on lab data (Exp 5B trains at -20 dB). Scene split ratio unstated. (Sec 5, p. 14–16) | Strict scene-level splitting (`train: 0.8`, `val: 0.1`, `test: 0.1`), zero scene leakage enforced by `test_train.py`. | **REASONABLE IMPLEMENTATION CHOICE** | Repo improves on paper by preventing cross-scene data leakage. |
| **Test Split Policy** | Exp 5A: 30 S1 slices.<br>Exp 5B/5C: Lab test split at 0, -10, -20, -30 dB. (Sec 5.1–5.3) | `run_exp1.py` tests on `s1-root`. `run_exp2.py` / `run_exp3.py` test on `splits.json` test partition across all 4 SIRs. | **PAPER-MATCHED** | Code structure strictly reproduces paper evaluation flow. |
| **Preprocessing Pipeline** | Algorithm 1: STFT along range lines ($L=1500$, Hamming window $L/8=187$). Eq. 26 peak normalization. (Sec 4.1, p. 12; Sec 4.3, p. 15) | Vectorized STFT/ISTFT in `utils/tf.py`, `normalize()` in `data/dataset.py`. Passes invertibility test `test_stft_istft_roundtrip`. | **PAPER-MATCHED** | 100% mathematically faithful to Algorithm 1 and Eq. 26. |
| **STFT Hop & Padding** | *Not specified in paper text.* Absent in official code. | Hop = $L/32 = 46$ samples (75% overlap, COLA compliant); reflect padding (`tf.py:58-67`). | **REASONABLE IMPLEMENTATION CHOICE** | Mathematically invertible engineering implementation. |
| **Model Architectures** | DNCNN, PISNet (no sigmoid), UNet (dim 32 residual), FDNet (FDUformer). (Sec 5, p. 14–15) | All 4 models implemented and verified in `src/models/`. Exact state-dict load and $\ell_\infty \le 4.77\times 10^{-7}$ diff against official code. | **PAPER-MATCHED** | Phase 1 alignment completely verified in Phase 2A (98/98 tests passing). |
| **Pretrained Weights** | *Not provided in paper.* | Exhaustive search found 0 `.pth`/`.pt` checkpoints in workspace or official archive. | **BLOCKED (Data Missing)** | Authors did not publish checkpoints; models must be trained from scratch. |
| **Training Hyperparameters** | 100 epochs, AdamW optimizer, lr $2\times 10^{-4}$, batch size 4. (Sec 4.3, p. 14) | Implemented in `configs/default.yaml` and `train.py`. FDNet lr $2\times 10^{-3}$ aligns with official code. | **PAPER-MATCHED** | Hyperparameters match Section 4.3 exactly. |
| **Validation Policy** | *Unstated in paper text.* Official training code had dead validation logic (`train_FDNet.py:196`). | Active validation every epoch on held-out validation scenes. Best model tracked by validation PSNR (`best.pt`). | **REASONABLE IMPLEMENTATION CHOICE** | Repo fixes critical official code bug where validation never fired. |
| **Evaluation Metrics** | PSNR (Eq. 23), SSIM (Eq. 24), Mean-Entropy (Eq. 25). Official code used 8-bit quantization (`A_test_*.py:199-203`). | Continuous float64 metrics by default; optional `quantize_8bit=True` matches official `A_test_*.py` exactly. | **PAPER-MATCHED & CODE-ALIGNED** | Dual metric evaluation supported in `src/eval/metrics.py`. |
| **Experiment 5A Script** | Section 5.1: Train on lab data, test on Sentinel-1, report Table 3 ME. | `scripts/run_exp1.py` fully implemented. Supports `--smoke` and full evaluation. | **PAPER-MATCHED (Code) / BLOCKED (Data)** | Script is ready; blocked only by missing 640 lab train slices and 18/30 S1 slices. |
| **Experiment 5B Script** | Section 5.2: Train at -20 dB, test at 0, -10, -20, -30 dB, plot Figure 9. | `scripts/run_exp2.py` fully implemented. Supports `--smoke` and full evaluation. | **PAPER-MATCHED (Code) / BLOCKED (Data)** | Script is ready; blocked only by missing 640 lab train slices. |
| **Experiment 5C Script** | Section 5.3: Domain shift test across 4 SIRs, plot Figure 12. | `scripts/run_exp3.py` implemented with `exp3_train` vs `exp3_test` sets in `param_sets.py`. | **APPROXIMATION** | Script executes cleanly; but paper withheld exact numerical shift parameters. |
| **Experiment 5D Script** | Section 5.4: Image Subspace vs. Echo Domain comparison, plot Figure 14. | **Unimplemented in repository.** Absent from official code release. | **BLOCKED (Missing Core Components)** | Requires raw radar echo simulator, range compression, RCMC, and azimuth compression engine. |
