# Experimental Evaluation Results: UNet Baseline for SAR RFI Suppression

## 1. Experimental Setup & Model Specification

The evaluation benchmarks the official U-Net architecture against narrow-band Radio Frequency Interference (RFI) using the held-out test split of the RFI-Patch benchmark.

- **Model Architecture:** Official U-Net (`OfficialUNet`) with 5-level multi-scale contracting and expanding paths, residual convolution blocks (`conv11` 1x1 skip projections), and LeakyReLU (alpha = 0.01). Batch Normalization is omitted to preserve linear amplitude dynamics.
- **Model Parameter Count:** 9,503,170 trainable parameters.
- **Optimizer & Schedule:** AdamW optimizer (lr = 2.0e-4, weight_decay = 0.01, constant learning rate schedule).
- **Loss Function:** Mean Squared Error (MSE) across the real and imaginary spectrogram channels.
- **Batching & Sampling:** Batch size of 4 slices, drawing 32 random azimuth lines per slice (effective batch of 128 spectrogram patches per optimization step).
- **Training Epochs:** 5 epochs (CPU execution).
- **Documented Dataset SNR:** -20.0 dB (RFI-Patch stack index 3).
- **Checkpoint Evaluated:** `outputs/runs/unet_rfipatch_snr-20/best.pt` (Epoch 5 checkpoint, corresponding to best validation PSNR of 19.56 dB).
- **Held-Out Test Set:** Exactly 2 independent scenes (`s3` and `third`) consisting of 512 x 1500 complex SAR arrays. Strict scene-level partition was preserved (0 sample overlap with the 12 training scenes and 2 validation scenes).

> **Distinction Between Documented SNR and Measured SIR:**
> In accordance with the RFI-Patch dataset specification, the interference level is labeled as **documented SNR** (-20 dB, where lower values represent stronger interference power relative to the signal). This label is pre-computed in the dataset distribution rather than estimated via laboratory Signal-to-Interference Ratio (SIR) measurement.

---

## 2. Evaluation Methodology

Inference strictly adheres to Algorithm 1 of Fang et al. (*Remote Sensing* 2025):
1. **Range-Line Decomposition (Step I):** Each 1D range line X = S(i, :) is extracted across all 512 azimuth rows.
2. **Time-Frequency Representation (Step II):** Range-line STFT is computed using a Hamming window of length N / 8 = 187 samples with hop size N / 32 = 46 samples.
3. **Spectrogram Normalization:** Spectrograms are normalized by the global peak magnitude scale of the polluted slice: alpha = max |STFT(S)|.
4. **Network Filtering (Step III):** The real and imaginary channels (2, F, T) are evaluated by the trained U-Net in row batches of 128.
5. **Synthesis & Reconstruction (Steps IV-V):** Inverse STFT (ISTFT) is performed with synthesis window compensation, rescaled by alpha, reconstructing the restored complex SAR image S*.

### Metric Definitions
- **PSNR (dB):** Computed on magnitude images according to Eq. 23: PSNR = 20 * log10(Peak / sqrt(MSE)), using both the official 8-bit quantized protocol (mapping to [0, 255] with Peak = 255) and continuous float64 representations.
- **SSIM:** Structural Similarity Index computed on magnitude images via a 7x7 window (K1 = 0.01, K2 = 0.03).
- **MSE:** Standard Mean Squared Error between magnitude estimates and ground truth.

---

## 3. Quantitative Results

### Table 1: Test Set Performance at Target Condition (Documented SNR = -20 dB)

| Scene | Metric Representation | Noisy Input PSNR (dB) | UNet Output PSNR (dB) | Delta PSNR (dB) | Noisy Input SSIM | UNet Output SSIM | Delta SSIM | Noisy Input MSE | UNet Output MSE | Delta MSE | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`s3`** | 8-bit Quantized | 18.01 | 17.84 | **-0.17** | 0.5121 | 0.4879 | **-0.0243** | 1028.59 | 1069.33 | **+40.75** | Degraded |
| **`s3`** | Float64 Continuous | 18.01 | 17.84 | **-0.17** | 0.5127 | 0.4892 | **-0.0235** | 0.01582 | 0.01644 | **+0.00063** | Degraded |
| **`third`** | 8-bit Quantized | 22.59 | 22.41 | **-0.17** | 0.5352 | 0.5163 | **-0.0190** | 358.50 | 372.99 | **+14.49** | Degraded |
| **`third`** | Float64 Continuous | 22.59 | 22.42 | **-0.17** | 0.5359 | 0.5180 | **-0.0179** | 0.00551 | 0.00573 | **+0.00022** | Degraded |
| **AVERAGE** | **8-bit Quantized** | **20.30** | **20.13** | **-0.17** | **0.5237** | **0.5021** | **-0.0216** | **693.54** | **721.16** | **+27.62** | **Degraded** |
| **AVERAGE** | **Float64 Continuous** | **20.30** | **20.13** | **-0.17** | **0.5243** | **0.5036** | **-0.0207** | **0.01066** | **0.01109** | **+0.00042** | **Degraded** |

### Test Sample Metric Distribution
- **Total Test Samples at Target Level:** 2 samples (`s3`, `third`)
- **Samples with PSNR Improvement:** **0 / 2 (0.0%)**
- **Samples with PSNR Degradation:** **2 / 2 (100.0%)**
- **Samples with SSIM Improvement:** **0 / 2 (0.0%)**
- **Samples with SSIM Degradation:** **2 / 2 (100.0%)**

---

### Table 2: Cross-Level Generalization Performance (All 4 Documented SNR Levels)

| Documented SNR | Input PSNR (dB) | UNet PSNR (dB) | Delta PSNR (dB) | Input SSIM | UNet SSIM | Delta SSIM | Input MSE (u8) | UNet MSE (u8) | Delta MSE (u8) | Samples Improved |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 dB** | 42.26 | 41.76 | **-0.50** | 0.9046 | 0.8964 | **-0.0082** | 3.90 | 4.37 | **+0.47** | 0 / 2 |
| **-10.0 dB** | 30.98 | 30.76 | **-0.21** | 0.6791 | 0.6696 | **-0.0095** | 52.88 | 55.51 | **+2.63** | 0 / 2 |
| **-20.0 dB** | 20.30 | 20.13 | **-0.17** | 0.5237 | 0.5021 | **-0.0216** | 645.74 | 671.39 | **+25.65** | 0 / 2 |
| **-30.0 dB** | 10.45 | 10.29 | **-0.16** | 0.4399 | 0.3658 | **-0.0741** | 6695.42 | 6940.75 | **+245.33** | 0 / 2 |

---

## 4. Observations & Diagnostic Analysis

1. **Systematic Metric Degradation:** Across both held-out scenes and across all tested interference strengths, the UNet baseline fails to achieve positive RFI suppression. PSNR consistently decreases by -0.17 dB on average at the training SNR of -20 dB, while SSIM drops by -0.0216 and MSE increases by +25.65.
2. **Analysis of Residual Maps:** The residual error maps (|UNet Error| - |Input Error|) reveal that while the network attenuates dominant interference striations along isolated frequency bins, it introduces pervasive blurring across non-interfered azimuth-range coordinates. The network penalizes subtle SAR speckle patterns and edge gradients, which directly lowers the SSIM score.
3. **Under-Convergence Root Cause:** The checkpoint is the result of an abbreviated 5-epoch training run. During training, validation PSNR improved from 19.28 dB (Epoch 1) to 19.56 dB (Epoch 5); however, the uncorrupted input validation baseline was 19.73 dB. The 9.5-million-parameter network had not yet traversed the optimization loss landscape to reach the separation threshold where artifact excision outweighs feature smoothing.

---

## 5. Limitations

1. **Abbreviated Training Schedule:** The benchmark was terminated at 5 epochs due to local CPU compute constraints, whereas standard convergence on complex SAR spectrograms mandates 100 epochs with cosine/step learning rate annealing as prescribed by Fang et al.
2. **Constrained Sample Count:** The held-out test split comprises only 2 independent scenes (`s3` and `third`) totaling 8 multi-level slices. While statistically sufficient to establish deterministic behavior for this checkpoint, broader statistical power requires evaluation over the full 640-slice Sentinel-1 dataset.
3. **Synthetic/Proxy Character of RFI-Patch:** RFI-Patch provides pre-computed linear mixing. Real-world SAR operations exhibit phase non-linearities and saturated receiver dynamics not fully captured in synthetic additive stacks.
