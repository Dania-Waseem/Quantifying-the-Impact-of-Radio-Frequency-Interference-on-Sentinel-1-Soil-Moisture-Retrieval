# UNet Baseline: Preliminary Experimental Report (5-Epoch Evaluation)

## 1. Experimental Setup & Model Specification

The evaluation benchmarks the official U-Net architecture against narrow-band Radio Frequency Interference (RFI) using the held-out test split of the RFI-Patch benchmark.

- **Model Architecture:** Official U-Net (`OfficialUNet`) with 5-level multi-scale contracting and expanding paths, residual convolution blocks (`conv11` 1x1 skip projections), and LeakyReLU ($\alpha = 0.01$). Batch Normalization is omitted to preserve linear amplitude dynamics.
- **Model Parameter Count:** 9,503,170 trainable parameters.
- **Optimizer & Schedule:** AdamW optimizer ($\text{lr} = 2.0 \times 10^{-4}$, $\text{weight\_decay} = 0.01$, constant learning rate schedule).
- **Loss Function:** Mean Squared Error (MSE) across real and imaginary spectrogram channels.
- **Batching & Sampling:** Batch size of 4 slices, drawing 32 random azimuth lines per slice (effective batch of 128 spectrogram patches per optimization step).
- **Training Epochs:** 5 epochs (completed run on CPU).
- **Target Training Condition:** Documented $\text{SIR} = -20.0\text{ dB}$ (RFI-Patch stack index 3, `stack[3]`).
- **Checkpoint Evaluated:** `outputs/runs/unet_rfipatch_snr-20/best.pt` (Epoch 5 checkpoint, corresponding to best validation PSNR of 19.56 dB).
- **Held-Out Test Set:** Exactly 2 independent scenes (`s3` and `third`) consisting of $512 \times 1500$ complex SAR arrays. Strict scene-level partition was preserved (0 sample overlap with the 12 training scenes and 2 validation scenes).

> **Clarification of Interference Metric (Local SIR vs. Full-Scene Ratio):**
> In accordance with Section 3.1 (p. 10) of Fang et al. (*Remote Sensing* 2025), the interference level is defined as the **Signal-to-Interference Ratio (SIR)** evaluated over the localized interfered area ($\Omega_{\text{RFI}}$):
> $$\text{SIR} = 10 \log_{10}\left(\frac{P_{\text{target}}(\Omega_{\text{RFI}})}{P_{\text{interference}}(\Omega_{\text{RFI}})}\right)$$
> Under this standard definition, the benchmark target condition corresponds to $\text{SIR} = -20\text{ dB}$ (empirically measured as $-19.90\text{ dB}$ on held-out test scene `s3` and $-19.93 \pm 1.05\text{ dB}$ across all benchmark scenes). When evaluated globally over the entire unmasked $512 \times 1500$ scene, uncorrupted margins dilute the interference power, resulting in a full-scene power ratio of $-16.49\text{ dB}$.

---

## 2. Evaluation Methodology

Inference strictly adheres to Algorithm 1 of Fang et al. (*Remote Sensing* 2025):
1. **Range-Line Decomposition (Step I):** Each 1D range line $X = S(i, :)$ is extracted across all 512 azimuth rows.
2. **Time-Frequency Representation (Step II):** Range-line STFT is computed using a periodic Hamming window of length $N / 8 = 187$ samples with hop size $N / 32 = 46$ samples.
3. **Spectrogram Normalization:** Spectrograms are normalized by the global peak magnitude scale of the polluted slice: $\alpha = \max |\text{STFT}(Y)|$.
4. **Network Filtering (Step III):** The real and imaginary channels $(2, F, T)$ are evaluated by the trained U-Net in row batches of 128.
5. **Synthesis & Reconstruction (Steps IV–V):** Inverse STFT (ISTFT) is performed with synthesis window compensation, rescaled by $\alpha$, reconstructing the restored complex SAR image $\hat{S}$.

### Metric Protocols (Strictly Distinguished)
- **8-Bit Quantized Protocol (Official Author Benchmark Protocol):**
  - Following `A_test_fdnet.py:199-203`, magnitude arrays $M = |S|$ are scaled by $\text{peak} = \max(|S_{\text{clean}}|)$ and quantized to integer range $[0, 255]$:
    $$M_{u8} = \text{round}\left(\text{clip}\left(\frac{M}{\text{peak}} \times 255.0, 0, 255\right)\right)$$
  - $\text{MSE}_{u8} = \frac{1}{N} \sum (M_{\text{clean}, u8} - M_{\text{est}, u8})^2$ (range $[0, 65025]$).
  - $\text{PSNR}_{u8} = 20 \log_{10}(255.0 / \sqrt{\text{MSE}_{u8}})$.
  - $\text{SSIM}_{u8}$: standard SSIM with $7 \times 7$ window, $L = 255$.
- **Float64 Continuous Protocol (Unquantized Continuous Domain):**
  - Computed directly on normalized double-precision magnitude arrays $M_{\text{norm}} = M / \text{peak} \in [0, 1.0]$.
  - $\text{MSE}_{\text{norm}} = \frac{1}{N} \sum (M_{\text{clean, norm}} - M_{\text{est, norm}})^2$ (range $[0, 1.0]$).
  - $\text{PSNR}_{f64} = 20 \log_{10}(1.0 / \sqrt{\text{MSE}_{\text{norm}}})$.
  - $\text{SSIM}_{f64}$: standard SSIM on normalized float64 arrays, $L = 1.0$.

---

## 3. Reconciled Quantitative Results

### Table 1: Test Set Performance at Target Condition (Documented SIR = -20 dB)

| Scene | Evaluation Protocol | Noisy Input PSNR (dB) | UNet Output PSNR (dB) | $\Delta$ PSNR (dB) | Noisy Input SSIM | UNet Output SSIM | $\Delta$ SSIM | Noisy Input MSE | UNet Output MSE | $\Delta$ MSE | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`s3`** | 8-bit Quantized | 18.01 | 17.84 | **-0.17** | 0.5121 | 0.4879 | **-0.0243** | 1028.59 | 1069.33 | **+40.75** | Degraded |
| **`s3`** | Float64 Continuous | 18.01 | 17.84 | **-0.17** | 0.5127 | 0.4892 | **-0.0235** | 0.01582 | 0.01644 | **+0.00063** | Degraded |
| **`third`** | 8-bit Quantized | 22.59 | 22.41 | **-0.17** | 0.5352 | 0.5163 | **-0.0190** | 358.50 | 372.99 | **+14.49** | Degraded |
| **`third`** | Float64 Continuous | 22.59 | 22.42 | **-0.17** | 0.5359 | 0.5180 | **-0.0179** | 0.00551 | 0.00573 | **+0.00022** | Degraded |
| **AVERAGE** | **8-bit Quantized** | **20.30** | **20.13** | **-0.17** | **0.5237** | **0.5021** | **-0.0216** | **693.54** | **721.16** | **+27.62** | **Degraded** |
| **AVERAGE** | **Float64 Continuous** | **20.30** | **20.13** | **-0.17** | **0.5243** | **0.5036** | **-0.0207** | **0.01066** | **0.01109** | **+0.00042** | **Degraded** |

#### Target Condition Summary Statistics
- **Total Test Scenes at Target Condition:** 2 independent scenes (`s3`, `third`)
- **Samples Showing PSNR Improvement:** **0 / 2 (0.0%)**
- **Samples Showing PSNR Degradation:** **2 / 2 (100.0%)**
- **Samples Showing SSIM Improvement:** **0 / 2 (0.0%)**
- **Samples Showing SSIM Degradation:** **2 / 2 (100.0%)**

---

### Table 2: Cross-Level Generalization Performance Across All SIR Levels (Reconciled)

*Arithmetic averages across held-out test scenes `s3` and `third`. Both 8-bit Quantized and Float64 Continuous protocols are explicitly distinguished.*

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

## 4. Observations & Scientific Interpretation

1. **Systematic Metric Degradation:** Across both test scenes and all four interference levels, the 5-epoch UNet baseline slightly degrades reconstruction quality compared to the unprocessed input. At the target condition of $\text{SIR} = -20\text{ dB}$, PSNR decreases by $-0.17\text{ dB}$ on average, SSIM decreases by $-0.0216$, and 8-bit MSE increases by $+27.62$ ($+0.00042$ in normalized float64).
2. **Analysis of Spatial Error Residuals:** Residual difference maps ($|\hat{S} - S_{\text{clean}}| - |Y - S_{\text{clean}}|$) indicate that while the network attenuates high-amplitude RFI spikes along dominant frequency bins, it introduces slight high-frequency over-smoothing across background regions. This smoothing penalizes fine radar speckle and subtle edge transitions, lowering SSIM.
3. **Convergence Context:** Under-convergence is a plausible explanation for the observed degradation, as the 9.50M parameter network completed only 5 training epochs (approx. 5% of the standard 100-epoch budget). However, further training and evaluation across the full 100-epoch schedule are required to establish whether it converges to positive interference suppression.

---

## 5. Experimental Limitations

1. **Abbreviated Training Duration:** The run was restricted to 5 epochs due to local CPU execution constraints. The reference paper specifies a uniform 100-epoch schedule.
2. **Limited Test Sample Size:** The held-out test split comprises 2 scenes (`s3` and `third`). While sufficient for preliminary implementation audit and deterministic pipeline verification, strong generalization claims require evaluation over a larger benchmark set.
3. **Reference Benchmark Comparability:** Published figures in Fang et al. (~31.8 dB PSNR, ~0.958 SSIM for UNet) reflect training on a full 640-slice benchmark dataset across multiple RFI modulation categories. The local 16-scene RFI-Patch subset represents a restricted single-category proxy, so reference numbers should not be treated as guaranteed training targets for this subset.
