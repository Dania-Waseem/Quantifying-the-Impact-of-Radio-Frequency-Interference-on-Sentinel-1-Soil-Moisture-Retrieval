# Test-Only Evaluation Report: UNet RFI-Patch (SNR -20 dB)

## 1. Executive Summary

A rigorous **test-only evaluation** of the completed UNet baseline run on the RFI-Patch dataset was conducted strictly adhering to the held-out test protocol. No model training, fine-tuning, or hyperparameter alterations were performed. Existing scene-level dataset splits were strictly respected to prevent data leakage.

- **Run Directory:** `outputs/runs/unet_rfipatch_snr-20` (junction to `sar_rfi_baseline/outputs/rfi_patch_proxy_unet`)
- **Evaluated Checkpoint:** `outputs/runs/unet_rfipatch_snr-20/best.pt`
- **Model Architecture:** Official UNet (9,503,170 parameters)
- **Checkpoint Epoch:** Epoch 5 (Best Val PSNR: 19.5597 dB)
- **Held-Out Test Scenes:** 2 scenes (`s3` and `third`)
- **Total Test Samples Evaluated:**
  - **Target Training Level (SNR -20 dB):** 2 test patches / scenes (`s3`, `third`)
  - **Comprehensive Multi-Level Benchmark:** 8 test patches / scenes (4 SNR levels across 2 scenes)

---

## 2. Quantitative Evaluation Results

### 2.1 Target Condition (Documented SNR = -20 dB)

The primary test evaluation matches the training distribution of the model (trained at SNR = -20 dB):

| Scene | Metric Representation | Input PSNR (dB) | UNet Output PSNR (dB) | Δ PSNR (dB) | Input SSIM | UNet Output SSIM | Δ SSIM | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **s3** | 8-bit Quantized (Official) | 18.01 dB | 17.84 dB | **-0.17 dB** | 0.5121 | 0.4879 | **-0.0243** | Degradation |
| **s3** | Float64 Continuous | 18.01 dB | 17.84 dB | **-0.17 dB** | 0.5127 | 0.4892 | **-0.0235** | Degradation |
| **third** | 8-bit Quantized (Official) | 22.59 dB | 22.41 dB | **-0.17 dB** | 0.5352 | 0.5163 | **-0.0190** | Degradation |
| **third** | Float64 Continuous | 22.59 dB | 22.42 dB | **-0.17 dB** | 0.5359 | 0.5180 | **-0.0179** | Degradation |
| **AVERAGE** | **8-bit Quantized (Official)** | **20.30 dB** | **20.13 dB** | **-0.17 dB** | **0.5237** | **0.5021** | **-0.0216** | **Degradation** |
| **AVERAGE** | **Float64 Continuous** | **20.30 dB** | **20.13 dB** | **-0.17 dB** | **0.5243** | **0.5036** | **-0.0207** | **Degradation** |

---

### 2.2 Cross-Level Generalization (All 4 Documented SNR Levels)

Evaluated across the 2 held-out test scenes for all documented SNR levels in RFI-Patch:

| SNR Level | Input PSNR (dB) | UNet Output PSNR (dB) | Δ PSNR (dB) | Input SSIM | UNet Output SSIM | Δ SSIM | Avg Inference Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 dB** | 42.26 dB | 41.76 dB | -0.50 dB | 0.9046 | 0.8964 | -0.0082 | 37.9 s |
| **-10.0 dB** | 30.98 dB | 30.76 dB | -0.21 dB | 0.6791 | 0.6696 | -0.0095 | 34.7 s |
| **-20.0 dB** | 20.30 dB | 20.13 dB | -0.17 dB | 0.5237 | 0.5021 | -0.0216 | 34.4 s |
| **-30.0 dB** | 10.45 dB | 10.29 dB | -0.16 dB | 0.4399 | 0.3658 | -0.0741 | 34.1 s |

---

## 3. Visual Comparison Artifacts

All visual comparisons were generated using **identical visualization scales** (`vmin=0`, unified `vmax` across noisy input, UNet output, and clean reference ground truth) and saved with colorbars verifying scale consistency:

1. **Scene `s3` (SNR -20 dB):**
   - Full SAR Image Comparison: `figures/s3_snr-20dB_full_comparison.png`
   - Center 256x256 Patch Zoom: `figures/s3_snr-20dB_patch_zoom.png`
   - Peak Interfered Range Line STFT: `figures/s3_snr-20dB_spectrogram.png`
2. **Scene `third` (SNR -20 dB):**
   - Full SAR Image Comparison: `figures/third_snr-20dB_full_comparison.png`
   - Center 256x256 Patch Zoom: `figures/third_snr-20dB_patch_zoom.png`
   - Peak Interfered Range Line STFT: `figures/third_snr-20dB_spectrogram.png`
3. **Reference Figures (0 dB, -10 dB, -30 dB):**
   - Full SAR comparisons, patch zooms, and STFT spectrograms for all levels are saved in `figures/`.

---

## 4. Key Findings and Conclusion

1. **Does UNet improve RFI suppression on this checkpoint?**
   **No.** The checkpoint does **not** improve RFI suppression. On both held-out test scenes, the UNet output degrades PSNR by **-0.17 dB** and degrades SSIM by **-0.0216** relative to the uncorrected noisy input.
2. **Underlying Cause:**
   The training run was limited to only **5 epochs** on CPU. While validation PSNR steadily increased during training (19.28 dB at Epoch 1 to 19.56 dB at Epoch 5), it had not yet surpassed the raw uncorrupted input baseline (19.73 dB on validation, 20.30 dB on test). The model is behaving as an under-trained regularizer that slightly blurs high-frequency SAR texture without having learned the precise time-frequency manifold of the RFI component.
3. **Implication:**
   To achieve positive RFI suppression and reproduce published benchmarks, the model requires full convergence (100 epochs as specified in Section 4.3 of Fang et al.) executed on GPU hardware.
