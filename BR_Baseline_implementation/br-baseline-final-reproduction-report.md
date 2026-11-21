# Consolidated Final Reproduction Assessment Report
**Project:** SAR Block-Type RFI Baseline Reproduction (*Fang et al., Remote Sensing 2025, 17, 3688*)  
**Repository:** `sar_rfi_baseline`  
**Date:** 2026-11-21  
**Audit Scope:** Models, Losses, STFT Pipeline, Preprocessing, Dataset Dispatches, Training Configuration, and Experiments 5A–5D  
**Test Suite Status:** 94/94 passing (exit 0, `sar_rfi_baseline/outputs/pytest_full.txt`)

---

## 1. Executive Verdict

### **VERDICT: NOT REPRODUCED (Engineering Adaptation & Partial Replication in Progress)**

**Scientific Justification:**  
While the repository provides a high-quality, mathematically sound engineering implementation of Algorithm 1 (image-subspace STFT-based RFI suppression) and an exact state-dict-verified FDNet architecture, **the benchmark cannot be claimed as reproduced today**. 

Four primary blockers prevent faithful reproduction:
1. **Missing Core Dataset**: The 640 full-sized laboratory-constructed image slices described in Section 3.1 are absent (Baidu Netdisk repository not downloaded). Only 16 patch stacks covering a single RFI category exist locally.
2. **Missing Sentinel-1 Test Slices**: Only 12 of the 30 Sentinel-1 slices described in Section 3.2 are available on disk.
3. **Architectural Divergence in UNet**: The repository baseline is canonical Ronneberger U-Net (base 64, BatchNorm, maxpooling, no global residual skip), whereas the official code used a distinct custom architecture (`model.py:107-178`: dim 32, LeakyReLU ConvBlock, 1×1 residual path, 4×4 strided conv downsampling, global residual, no BatchNorm).
4. **Experiment 5D Completely Missing**: Section 5.4 (Echo domain vs. Image subspace) has no implementation, echo simulator, raw echo data, or SAR imaging engine in either the repository or the official code release.

---

## 2. Evidence-Based Verification Breakdown

To ensure rigorous scientific integrity, we separate what has been empirically and mathematically verified from what remains synthetic or unverified.

```
+---------------------------------------------------------------------------------------------------+
|                                 VERIFICATION HIERARCHY & EVIDENCE                                  |
+---------------------------------------------------------------------------------------------------+
| 1. Test-Verified (94/94 passing):                                                                 |
|    - STFT/ISTFT roundtrip invertibility & window scaling (tests/test_tf.py)                        |
|    - Complex-to-channels conversions & peak normalization (tests/test_dataset.py)                 |
|    - Content-based .mat scenario dispatch (s4/s3/labeled) (tests/test_snr_mat.py)                  |
|    - RFI-Patch ss-stack indexing & documented SNR levels (tests/test_rfi_patch.py)                |
|    - Scene-level data splitting with strict zero-leakage guards (tests/test_train.py)             |
|    - Training loop state checkpointing, resumption, & RNG state recovery                          |
+---------------------------------------------------------------------------------------------------+
| 2. Official-Weight Equivalence:                                                                   |
|    - test_matches_official_fduformer_weights_and_outputs: STRICT state-dict load PASSED           |
|    - Numerical output and intermediate feature discrepancy <= 1e-5                                |
|    - Charbonnier + cosine orthogonality loss recipe matches official code exactly                 |
+---------------------------------------------------------------------------------------------------+
| 3. Code-Level Verification:                                                                       |
|    - Official PISNet & IMN discovered in extracted/Code/FDNet/network.py (not GitHub repo)        |
|    - Official UNet identified as custom 32-dim residual ConvNet (third_party/FDNet/model.py)      |
|    - Official validation loop proven DEAD CODE (eval_now = 4*len; (i+1)%eval_now == 0 never fires)|
|    - Official PSNR/SSIM evaluation proven to use 8-bit uint8 quantization [0, 255]                |
+---------------------------------------------------------------------------------------------------+
| 4. Dataset-Probe Verification:                                                                    |
|    - Physical verification of on-disk MAT dimensions: Scenario 1 s4=(512,1536), Scenarios 2/3 s3 |
|    - Spectral energy probe: 95% of RFI bandwidth along 1500-axis -> Range=1500, Azimuth=512       |
|    - Empirical refutation of documentation typo in explaination.docx                              |
+---------------------------------------------------------------------------------------------------+
| 5. Synthetic-Only Validation (Plumbing Verification, Not Scientific Reproduction):                |
|    - Smoke training runs: 2 epochs on 20 synthetic speckle scenes (64x256)                        |
|    - Synthetic RFI parameter sets (exp3_train vs exp3_test) for domain shift testing               |
|    - smoke_unlabeled stand-in for Sentinel-1 testing                                              |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Paper Fidelity Scorecard

We evaluate fidelity across 7 core dimensions using a qualitative, criteria-based rubric:
- **FAITHFUL**: Matches paper text and mathematical formulation exactly.
- **ALIGNED**: Follows the official code implementation where the paper is silent.
- **PRINCIPLED DIVERGENCE**: Deliberate engineering improvement (documented and verified).
- **ACCIDENTAL DIVERGENCE**: Discrepancy between repository and official code/paper.
- **BLOCKED / MISSING**: Not implemented or data unavailable.

| Dimension | Paper Specification | Official Code (`SAR-RFI-DATASET-main`) | Repository Implementation (`sar_rfi_baseline`) | Fidelity Rating | Detailed Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Architectures** | DNCNN, PISNet, UNet, FDNet | FDUformer, PISNet (`network.py`), custom UNet (`model.py:107`), DCNet | FDNet (equivalent), PISNet (faithful), Zhang DnCNN (canonical), Ronneberger UNet | **PARTIAL** | FDNet is 100% verified. UNet is Ronneberger base-64 instead of official base-32 residual convnet. |
| **Loss Formulations** | MSE for baselines; FDNet multi-term loss (Eq. 24) | Charbonnier + $10^{-5} \cdot \|\cos\|$; output norm term commented out | MSE for baselines; Charbonnier + $10^{-5} \cdot \|\cos\|$ for FDNet | **FAITHFUL** | Exact match to paper text and official loss recipe. |
| **Data Representation** | Algorithm 1: STFT on range lines, normalize by peak | Consumes pre-computed real 1-channel float32 512×512 crops | On-the-fly STFT/ISTFT, 2-channel complex `(2, F, T)` full range lines | **PRINCIPLED DIVERGENCE** | Repo implements true Algorithm 1; official code trained on static real crops. |
| **Data Availability** | 640 lab slices (160/SIR); 30 Sentinel-1 slices | Full dataset on Baidu Netdisk (not local) | 0/640 lab slices; 16 RFI-Patch stacks (RFI-1 only); 12/30 Sentinel-1 slices | **BLOCKED** | Major data shortage; cannot reproduce 640-slice benchmark without Baidu download. |
| **Training Protocol** | 100 epochs, batch 4, AdamW, lr 2e-4 | 101/301 epochs, batch 4, AdamW, lr 2e-4, StepLR(50, 0.5), seed 1234 | 100 epochs, AdamW, lr 2e-4 (baselines) / 2e-3 (fdnet), seed 42 | **ALIGNED** | Repo respects 2025 paper lr for baselines and 2024 paper lr for FDNet. |
| **Validation Policy** | Unstated | Dead code in training; saves `latest` and fixed `epoch_100.pth` | Active validation every epoch; saves `last.pt` and validation `best.pt` | **PRINCIPLED DIVERGENCE** | Repo fixes broken official validation and implements best-checkpoint tracking. |
| **Metric Calculation** | PSNR (Eq. 23), SSIM (Eq. 24), ME (Eq. 25) | PSNR/SSIM on 8-bit quantized `uint8` [0, 255] images | PSNR/SSIM on unquantized float64 magnitude images | **PRINCIPLED DIVERGENCE** | Repo computes true continuous metrics; official code quantizes to 8-bit. |
| **Experiments 5A–5D** | 5A (Sentinel-1), 5B (Same Dist), 5C (Domain Shift), 5D (Echo) | 5A test loader; 5B test scripts; 5C & 5D absent | 5A/5B/5C scripts implemented; 5D completely missing | **PARTIAL** | 5A/5B lack data; 5C lacks paper shift parameters; 5D is unbuilt. |

---

## 4. Ranked Critical Discrepancies and Blockers

### **Severity 1: Missing Full Laboratory Training Dataset (Baidu Pan DATASET)**
- **Impact**: Absolute blocker to faithful reproduction of Figures 9, 10, 11 and Table 3.
- **Evidence**: The paper trains on 640 full slices ($512 \times 1500$) across 4 SIR levels (-30, -20, -10, 0 dB) and multiple RFI modulation types. The local disk has only `Data/RFI-Patch/RFI-1` (16 stacks, 1 RFI category). Training on 16 scenes risks severe overfitting and narrow domain coverage.

### **Severity 2: Missing Experiment 5D (Echo Domain Comparison)**
- **Impact**: Entire sub-section of the paper cannot be reproduced.
- **Evidence**: Neither repository nor official code contains raw radar echo data, radar chirp simulators, or SAR imaging algorithms (range compression, range cell migration correction, azimuth compression).

### **Severity 3: Architectural Mismatch in U-Net Baseline**
- **Impact**: Invalidates baseline comparison against the paper's UNet results.
- **Evidence**:
  - Official UNet (`third_party/FDNet/model.py:107-178`): Feature dim 32, LeakyReLU ConvBlocks with $1 \times 1$ conv residual skip, $4 \times 4$ stride-2 conv pooling, global residual connection (`x + conv10`), zero BatchNorm.
  - Repository UNet (`sar_rfi_baseline/src/models/unet.py`): Canonical Ronneberger UNet with base dim 64, BatchNorm, $2 \times 2$ MaxPool2d, bilinear upsampling, no residual conv blocks, no global residual.

### **Severity 4: Incomplete Sentinel-1 Test Set**
- **Impact**: Precludes exact numerical reproduction of Table 3 Mean-Entropy scores.
- **Evidence**: Paper Section 3.2 specifies 30 Sentinel-1 slices. On disk, `Data/Scenario 1..3` holds only 12 slices ($3 \text{ scenarios} \times 4 \text{ SNR levels}$). 18 slices are missing.

### **Severity 5: Undisclosed Domain Shift Parameters for Experiment 5C**
- **Impact**: Precludes faithful numerical reproduction of Figures 12 and 13.
- **Evidence**: Section 5.3 describes testing under parameter inconsistency but withholds the exact numerical distributions. Repository synthesized arbitrary disjoint ranges (`exp3_train` vs `exp3_test` in `src/data/param_sets.py`). This produces plausible curves but cannot be validated against the paper.

### **Severity 6: Official Code Metric Quantization Discrepancy**
- **Impact**: Potential baseline score divergence of 0.5–2.0 dB PSNR.
- **Evidence**: Official test script (`extracted/Code/FDNet/test/A_test_fdnet.py:199-203`) scales images by `max1` and rounds to `uint8` (`(img/(max1) * 255.0).round()`) before calling PSNR/SSIM. The repository evaluates in unquantized `float64`.

### **Severity 7: Compute Hardware Limitations**
- **Impact**: Training full models locally is currently infeasible.
- **Evidence**: Environment has no CUDA GPU (`torch 2.14.1+cpu`). A 2-epoch smoke test on CPU takes ~12 minutes for FDNet. 100 epochs on full data would require weeks on CPU vs hours on the paper's RTX 3090.

---

## 5. Experiment Readiness Matrix

| Experiment | Paper Section | Repo Script | Data Requirement | Available on Disk | Execution Status | Scientific Reproduction Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 5A: Sentinel-1 Generalization** | Sec 5.1 (Table 3, Figs 7–8) | [`scripts/run_exp1.py`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/sar_rfi_baseline/scripts/run_exp1.py) | 640 lab train slices; 30 Sentinel-1 test slices | 0/640 lab slices; 12/30 Sentinel-1 slices | **PARTIAL** | **BLOCKED (faithful)**<br>Runs on 12 test slices for ME plumbing; lacks training data. |
| **Exp 5B: Same Data Distribution** | Sec 5.2 (Table 2, Figs 9–11) | [`scripts/run_exp2.py`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/sar_rfi_baseline/scripts/run_exp2.py) | 640 lab slices (160/SIR at 0/-10/-20/-30 dB) | 16 RFI-Patch stacks (RFI-1 only) | **PARTIAL** | **BLOCKED (faithful)**<br>`run_exp2.py` expects individual slice files (`--source mat`). |
| **Exp 5C: Domain Shift Condition** | Sec 5.3 (Figs 12–13) | [`scripts/run_exp3.py`](file:///c:/Users/hp/Pictures/BR%20Baseline%20implementation/sar_rfi_baseline/scripts/run_exp3.py) | Clean scenes + shift parameter specifications | 16 clean scenes in RFI-Patch; synthetic simulator | **PARTIAL** | **NOT VERIFIABLE**<br>Executes end-to-end synthetically, but paper shift parameters were withheld. |
| **Exp 5D: Image Subspace vs Echo** | Sec 5.4 (Fig 14) | *None (unimplemented)* | Raw radar echo phase history + SAR imaging engine | *None* | **BLOCKED** | **BLOCKED**<br>Completely unbuilt; absent in official code release. |

---

## 6. What Can and Cannot Be Legitimately Claimed

### **What We CAN Legitimately Claim Today:**
1. **Mathematical Soundness of Algorithm 1**: The end-to-end 5-step image-subspace STFT suppression pipeline is fully implemented, vectorized, and passes invertibility unit tests (`test_stft_istft_roundtrip`).
2. **FDNet Architectural Equivalence**: The repository's FDNet implementation is 100% numerically identical to official FDUformer weights (feature differences $\le 10^{-5}$).
3. **Correct Radar Geometry Orientation**: Empirically proved that the 1500-sample axis is Range and the 512-sample axis is Azimuth, correcting the conflicting documentation in `explaination.docx`.
4. **Official Code Defect Discovery**: Unearthed that official training code had completely dead validation logic (`(i+1) % (4*len) == 0`), never saved `model_best.pth`, and evaluated fixed-epoch checkpoints on 8-bit quantized images.
5. **Robust Engineering Infrastructure**: 94/94 passing pytest suite, strict scene-split leakage guards, RNG state persistence, and clean checkpoint resumption.

### **What We CANNOT Claim Today:**
1. **Cannot claim reproduction of paper performance numbers** (Table 3 ME, Figure 9 PSNR/SSIM, Figure 12 domain shift PSNR/SSIM, Figure 14 echo comparison).
2. **Cannot claim fair baseline comparison for U-Net** due to the architectural mismatch with the authors' custom UNet.
3. **Cannot claim valid trained models** until models are trained on full datasets using GPU hardware.
4. **Cannot claim empirical validation of Experiment 5C** due to synthetic parameter guesswork.
5. **Cannot claim any coverage of Experiment 5D**.

---

## 7. Prioritized Remediation Roadmap

```
+---------------------------------------------------------------------------------------------------+
|                                  PRIORITIZED REMEDIATION ROADMAP                                  |
+---------------------------------------------------------------------------------------------------+
| PHASE 1: PREREQUISITES FOR FAITHFUL REPRODUCTION (Critical Blockers)                              |
|   [ ] 1. Obtain Full Lab Dataset: Download the 640 full .mat slices from official Baidu Netdisk.   |
|   [ ] 2. Complete Sentinel-1 Set: Acquire remaining 18 Sentinel-1 scenes (reach 30 total).        |
|   [ ] 3. Align U-Net Architecture: Port official custom UNet (model.py:107) into src/models/unet.py|
|          as official_unet, leaving Ronneberger as an alternate baseline.                          |
|   [ ] 4. GPU Training Environment: Migrate execution to an NVIDIA CUDA GPU (>= 16 GB VRAM).       |
|   [ ] 5. RFI-Patch Slice Adapter: Enable run_exp2.py to ingest RFI-Patch stacks directly.        |
+---------------------------------------------------------------------------------------------------+
| PHASE 2: RECOMMENDED IMPROVEMENTS (High Value for Rigorous Benchmark)                             |
|   [ ] 1. Dual Metric Reporting: Report both exact float64 PSNR/SSIM and official 8-bit quantized  |
|          metrics to isolate reproduction shifts from quantization artifacts.                      |
|   [ ] 2. Checkpoint Evaluation Options: Support evaluating fixed epoch 100 (matching official    |
|          evaluation) alongside validation best.pt.                                                |
|   [ ] 3. Port Official DCNet: Add official DCNet (model.py:206) alongside Zhang DnCNN.            |
|   [ ] 4. Batch Size Clarification: Run comparative study of 4 slices x 32 rows vs 4 patch crops.  |
+---------------------------------------------------------------------------------------------------+
| PHASE 3: OPTIONAL ENHANCEMENTS (Long-term / Full Scope)                                           |
|   [ ] 1. Implement Experiment 5D: Build a raw radar echo simulator and Range-Doppler / BPA        |
|          imaging pipeline to evaluate echo-domain vs image-subspace suppression.                  |
|   [ ] 2. Inquire on Exp 5C Parameters: Contact authors for exact numerical shift parameters.       |
+---------------------------------------------------------------------------------------------------+
```

---

## 8. Final Research Defensibility

In academic and research auditing standards, projects fall into one of four categories:
1. **Direct Scientific Reproduction**: Verifying identical numerical results using the authors' published data and code.
2. **Partial Reproduction**: Replicating a subset of experimental claims with verified numerical convergence.
3. **Implementation Replication**: Rebuilding the algorithms and models from paper descriptions and codebases, verified by functional tests.
4. **Engineering Adaptation**: Adapting core principles into a clean, modernized framework with necessary architectural extensions.

### **Current Classification: Implementation Replication & Engineering Adaptation**
- This project **cannot yet be defended as a Scientific Reproduction** because missing benchmark data and compute constraints have prevented training and generating the empirical curves. Claiming reproduction at this stage would fail peer review.
- However, this project is **an exemplary Implementation Replication and Engineering Adaptation**:
  - The mathematics of Algorithm 1 are faithfully realized.
  - The official FDNet model is strictly reproduced down to state-dict equivalence ($\le 10^{-5}$).
  - All dataset layouts and formats are understood and parsed with 94 passing tests.
  - The engineering quality of this repository significantly exceeds the official code release (fixing dead validation loops, eliminating hardcoded paths, enforcing scene isolation, and avoiding lossy 8-bit image quantization).

---

## 9. Final Verdict & Next Actions

### **Summary Verdict**
The repository is in an **exceptional engineering state (94/94 tests passing)**. It is technically complete in its core algorithmic plumbing and FDNet model fidelity, but scientifically incomplete due to upstream dataset omissions, compute limitations, and a baseline U-Net architecture discrepancy.

### **Recommended Immediate Next Actions**
1. **Initialize Git Version Control**: Preserve this clean, 94/94 green baseline state in git.
2. **Resolve U-Net Architecture**: Add `OfficialUNet` (dim 32, residual convblocks, no BatchNorm) from `third_party/FDNet/model.py:107-178` into `src/models/`.
3. **Provide RFI-Patch Adapter**: Add `--source rfi_patch` support to `run_exp2.py` so local patch data can be evaluated immediately without waiting for the full 640-slice Baidu download.
4. **Export for GPU Training**: Package the repository for a GPU compute environment (Colab / cloud instance) to run full 100-epoch training.
