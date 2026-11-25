# Protocol report: Fang et al., Remote Sensing 2025, 17, 3688

This report separates what the paper states, what the authors' code does, and what is on disk.
No score below was adjusted to look like Figure 9. The corrected Section 5.2 model was not trained,
because the laboratory set required for that experiment is not here.

## What is on disk

`Data/RFI-Patch` contains one folder, `RFI-1`, with 16 `.mat` files. Each file is a complex
`ss` array of shape `(5, 512, 1500)`: `ss[0]` is the RFI-free image and `ss[1..4]` are the
same scene at the documented SIRs 0, −10, −20 and −30 dB. That is 64 polluted slices.

Checked on every file:

- The difference from the label is exactly zero outside one block, azimuth 237–1260 and the full range axis.
- The same block is used at all four SIRs, and the interference power steps by about 10 dB per SIR.
- The energy-normalized complex residual at −20 dB has inner product 0.99998 between every pair of files.
  All 16 scenes carry one copied interference pattern. The audit reports `residual_clusters = 1`.
- A mid-block range line (scene `s3`, azimuth 749) has a single horizontal tone in the spectrogram.
  That is not the paper's set of eight patterns.

The paper's laboratory set is 640 slices, eight interference patterns, and four scene types
(maritime, harbor, urban, mountainous). Section 3.1 also says each `.mat` stores the polluted
image, the clean image, and the interference parameters. These files store only `ss`. They do
not store those parameters.

## Where the missing data is, and why it was not generated

The authors' repository is https://github.com/fangfuping/SAR-RFI-DATASET. Its README points to
the Baidu archive https://pan.baidu.com/s/14BjmzpkJREBbmaAOPyjbyg?pwd=nh3f (code `nh3f`).
Unpack every `RFI-*` folder under `Data/RFI-Patch`. The loader already reads every
`RFI-*/*.mat` it finds, and files that share a label image stay in the same split.

The missing patterns were not synthesized. Section 2 gives the chirp imaging model. Table 2
gives only bandwidth 0–100 MHz and the four SIRs. The eight chirp rates, amplitudes, block
positions, scene list, and seeds are not in the paper or the released code. Choosing them
would create a different dataset. The two local Sentinel scenes are the unlabeled open-source
data of Section 5.1. They are not substitutes for the 640 labeled slices.

## What was already the paper's procedure, and was left in place

| Item | Status | Source |
|---|---|---|
| Hamming window, length L/8 = 64 | kept | paper Section 4.1 |
| U-Net, 1 input, 1 output, base width 32, 9,502,561 parameters | kept | authors' `model.py` |
| 100 epochs, batch size 4, AdamW, learning rate 0.0002 | kept | paper Section 4.3 |
| Charbonnier loss, ε = 1e-3, weight decay 0.02, StepLR(50, 0.5) with the initial `scheduler.step()` | kept | authors' training code, not stated in the paper |
| Reported checkpoint is epoch 100 | kept | authors' `A_test_unet.py` |
| Input and label divided by the maximum of the polluted map | kept | paper Eq. 26; the paper says "image", the 2024 paper defines it on the spectrogram. The map scope is the one that matches a network sample |
| 8-bit PSNR and SSIM, mean over maps, peak 255 after scaling by the label maximum | kept | authors' `A_test_unet.py`. Paper Eq. 23 is the float formula and does not say uint8 or whether the score is on the spectrogram or the SAR image. Both are reported by the existing evaluator; the uint8 map score is the authors' script |

STFT length 512, hop 1, centred Hamming window, and fftshift are still inferences from the
512 × 512 maps in the authors' loader. They were not changed.

## What was wrong, and what the code does now

1. **Dataset size.** Training now stops unless the audit sees at least 640 polluted slices, or
   640 files. This disk has 64 and 16.
2. **SIR exposure.** The default config is `config_official_unet_sir20_amp.yaml`. Training uses
   −20 dB only. Validation and test use 0, −10, −20 and −30 dB. The old all-SIR config remains
   as `config_official_unet.yaml` and is no longer the default. The previous run used that
   all-SIR config.
3. **Interference diversity.** The audit clusters additive residuals. One cluster blocks a
   paper-protocol run. No generator was added.
4. **Split.** The paper does not publish its slice list, so that list was not invented.
   The local holdout (`configs/splits/rfipatch_rfi1_seed42_frozen.json`, seed 42) is unchanged:
   12 train, 2 validation (`s7`, `s8`), 2 test (`s3`, `third`). The audit found no file in
   both test and train. New files, once downloaded, cannot share a label image with another part.
   This holdout is not claimed to be the authors' split.
5. **Evaluation.** `scripts/evaluate.py` is unchanged. It still follows `A_test_unet.py`.

`python repro_magmap/scripts/train.py` writes `protocol_audit.json` and exits with status 3
while the set is short. `--allow-incomplete-data` exists for a smoke test and must not be
compared with Figure 9. A dry run on this disk returned status 3 before any epoch.

## Comparison

Figure 9 is a plot. The paper has no numeric table for it. The Figure 9 column is the plot
reading already stored with the previous evaluation, not a new digitization and not an official
number. The previous column is the epoch-100 U-Net on RFI-1 only, trained on all four SIRs,
scored with the authors' uint8 map PSNR and SSIM on 8192 test maps
(`repro_magmap_runs-20261125T155216Z-1-001`, `eval_epoch_100/summary.json`).

| SIR | Figure 9 U-Net, plot reading | Previous RFI-1 run, all SIRs in training | Corrected Section 5.2 run |
|---|---|---|---|
| 0 dB | 30.7 / 0.965 | 43.68 / 0.9936 | not trained |
| −10 dB | 30.8 / 0.948 | 41.92 / 0.9899 | not trained |
| −20 dB | 31.5 / 0.953 | 39.21 / 0.9808 | not trained |
| −30 dB | 25.4 / 0.852 | 34.23 / 0.9455 | not trained |

The previous scores are higher because the test is 16 copies of one interference block, and
the network also saw the other three SIRs of the training scenes. They are not evidence that
the implementation beat the paper.

## Images

`repro_magmap/protocol_audit/image_clean_polluted_residual.png` shows scene `s3`: the clean
magnitude, the −20 dB magnitude, and the additive interference. The interference is one
azimuth block. There is no restored panel.

`repro_magmap/protocol_audit/spectrogram_clean_vs_polluted.png` shows the polluted and clean
spectrograms of azimuth line 749. The polluted map is one tone. A restored spectrogram was
not produced.

The earlier run's restoration figures remain in
`repro_magmap_runs-20261125T155216Z-1-001`. They belong to the all-SIR, one-pattern run.

## What a faithful run still needs

Download the archive above and unpack it under `Data/RFI-Patch`. Then:

```
python repro_magmap/scripts/audit_protocol.py --out-dir repro_magmap/protocol_audit
python repro_magmap/scripts/train.py --out-dir <new run directory> --device cuda
python repro_magmap/scripts/evaluate.py --run <new run directory> --device cuda
```

Training starts only after the audit reports 640 slices and 8 residual clusters, with −20 dB
alone in the training split and no shared test file. The resulting numbers are a Section 5.2
protocol on a local scene holdout. They become a numerical reproduction of Figure 9 only if
the authors' unpublished split and the unpublished scoring domain turn out to match. That is
not known from the paper.
