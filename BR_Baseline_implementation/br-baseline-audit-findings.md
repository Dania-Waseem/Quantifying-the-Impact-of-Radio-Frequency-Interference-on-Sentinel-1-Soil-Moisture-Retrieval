---
name: br-baseline-audit-findings
description: Hard-won provenance and audit facts for the SAR RFI baseline
  (official code locations, equivalence evidence, architecture mismatches)
metadata:
  node_type: memory
  type: project
  originSessionId: sess_0254f19c-25b4-45e4-a4e0-81a93b31f7eb
---

Audit facts established 2026-11-21 (paper-vs-repo audit of [[br-baseline-project-state]]):
- `third_party/FDNet` is a clean git clone of github.com/fangfuping/FDNet (3 commits). Its `model.py`/`losses.py` differ from `SAR-RFI-DATASET-main/.../extracted/Code/FDNet/` by **line endings only**; `options.py` differs for real (older snapshot, epochs defaults).
- The **official PISNet + IMN architectures live in `extracted/Code/FDNet/network.py`** (PISNet line 53, EMAU 111, IMN 203) — not in the GitHub FDNet clone, not in PISNet+IMN folder (that folder has only KAIR-style test harnesses, no model file).
- Repo FDNet port is **verified equivalent** to official FDUformer: `test_matches_official_fduformer_weights_and_outputs` (strict state-dict load + outputs/features ≤1e-5) PASSED.
- Repo UNet (Ronnegger, base 64, BN, maxpool, no global residual) is **NOT the official UNet** the paper used (official = `third_party/FDNet/model.py:107-178`: dim 32, LeakyReLU ConvBlock + 1×1 residual path, 4×4 s2 conv pooling, global residual `x + conv10`, no BN). Biggest reproduction gap found.
- Repo DnCNN = canonical Zhang DnCNN; official code offers DCNet (`model.py:206-302`, no BN, LeakyReLU, no skip-cats, no output residual) — paper's "DNCNN" reference is ambiguous between the two.
- FDNet paper (rs-16-2540 p.12): AdamW, 100 epochs, **lr 0.002**, wd 0.02. Official code default lr 0.0002 contradicts its own paper; repo follows paper text (fdnet→2e-3, others→2e-4 per the 2025 paper).
- Official FDNet loss recipe: Charbonnier(eps 1e-3) + 1e-5·|cos(signal,jam)|, norm term commented out (`train/train_FDNet.py:187-203`) — repo defaults match exactly.
- FDNet gradient checkpointing: official `use_checkpoint=False` default; repo always-on in training (`src/models/fdnet.py:317-321`) — numerically equivalent, faithfulness deviation.
- Full suite: 94/94 PASSED, exit 0 (outputs/pytest_full.txt, 2026-11-21 12:36).

Dataset audit (2026-11-21, probe-verified):
- On-disk truth: Scenario 1 `s4` (512,**1536**) c64; Scenarios 2/3 `s3` (512,1500) c64; RFI-Patch/RFI-1 = 16 stacks `ss` (5,512,1500) c64, stems first..sixth + s1..s10; all MATLAB v5.
- **Axis orientation resolved empirically**: pure RFI (ss[4]−ss[0]) has all spectral content along the 1500-axis (95%-band 0.10) and is near-constant along 512-axis (0.000) → 1500 = range, 512 = azimuth → repo's STFT axis is RIGHT; `explaination.docx`'s "range=512, azimuth=1500" is contradicted by its own data.
- Data availability vs paper: labeled lab set 640 slices → 0 full slices downloaded (Baidu pan DATASET absent); only RFI-Patch RFI-1 (64 labeled pairs, one of ≥2 RFI types; RFI-2 missing). Sentinel 30 slices → 3 (Scenario 1/2/3 × snr{0,5,10,15}; that SNR grid matches neither Table 2 nor the docx labels). exp1/exp2 runnable only at small scale; exp4 (echo-domain) has no simulator at all.
- RFI-Patch semantics match docx exactly (Label=ss(1,:,:), Snr 0/−10/−20/−30 = ss(2..5)) — repo constants `RFI_PATCH_CLEAN_INDEX=0`, `RFI_PATCH_SNR_DB` correct.
- Official pipeline (both mydataset.py copies): consumes PRE-COMPUTED REAL 1-channel [N,1,512,512] float32 TF maps, 512×512 crops (0:512 or 211:723) — repo's complex 2-channel on-the-fly STFT over full rows is a documented, principled deviation, not what the paper's networks actually consumed.
- Splits: scene-level (stem-is-scene for RFI-Patch) with leakage guards (train.py:283-285) and passing tests; stem↔true-scene mapping NOT VERIFIABLE (leakage risk if stems are patches of shared scenes).
- train.py defaults: epochs 100 (2 smoke), batch-size 4 slices/step, rows-per-image 32 (8 fdnet) — batch semantics = 4 slices × rows spectrograms, vs official 4 patches; interpretation documented (train.py:28-35).


Training configuration and experiments audit (2026-11-21, Sections 4.3 and 5A–5D):
- **1. Optimizers, Learning Rates, Weight Decay, Schedulers, Epochs, Batch Size, Seeds**:
  - *Paper text (Remote Sens. 2025, 17, 3688, Sec 4.3, p. 14)*: Platform standardized on 20-core Intel Core i7-14700K CPU (2.50 GHz base) + NVIDIA GeForce RTX 3090 GPU (24 GB). Uniform 100-epoch training cycles, batch size 4. AdamW optimizer, initial learning rate 0.0002 (`2e-4`). Weight decay, learning rate scheduler, and random seeds are completely unstated in the 2025 paper text (earlier 2024 FDNet paper rs-16-02540 p. 12 reported AdamW, lr 0.002, wd 0.02, 100 epochs).
  - *Official code (`extracted/Code/FDNet/options.py:10-19`, `train/train_FDNet.py:60-98`)*: AdamW (`options.py:16`, `train_FDNet.py:77`), initial lr 0.0002 (`options.py:17`), weight decay 0.02 (`options.py:19`). Hardcoded seed 1234 (`train_FDNet.py:60-63`). Scheduler is `StepLR(optimizer, step_size=50, gamma=0.5)` (`train_FDNet.py:96`); note the official script executes a premature `scheduler.step()` at line 97 before training begins. Total epochs configured as 101 (`options.py:11`) or 301 (`HWOptions:76`).
  - *Repository (`configs/default.yaml:7-11`, `src/train/train.py:435-503, 542-545`)*: AdamW across all models. Default epochs 100 (`train.py:491`; 2 with `--smoke`). Seed defaults to 42 (`train.py:457`, `utils/seed.py:7-12`), seeding random, numpy, torch, cuda, and DataLoader generator (`train.py:536`). LR and weight decay: baselines use paper lr 2e-4, wd 0.01, constant lr (`lr_step=0`); FDNet follows FDNet paper recipe: lr 2e-3, wd 0.02, `StepLR(step_size=50, gamma=0.5)` stepped cleanly at epoch end (`train.py:604`). Batch size: `--batch-size 4` slices per step, with `--rows-per-image` (default 32, 8 for fdnet) random azimuth rows sampled per slice (`train.py:32-35, 502, 531-536`), giving 128 (or 32) spectrograms per step vs official 4 pre-cropped 512x512 patches.

- **2. Train / Validation / Test Split and Validation Usage**:
  - *Paper text*: No validation split ratio or validation methodology is specified in the paper. Section 5 specifies training on lab data and testing on Sentinel-1 (5.1) or testing across 4 SIRs (5.2, 5.3).
  - *Official implementation dead code (`train/train_FDNet.py:167, 193-224`)*: In `train_FDNet.py`, evaluation interval is set to `eval_now = len(train_loader) * 4` (line 167). The loop condition is `if (i+1)%eval_now==0 and i>0:` (line 193). Because batch iteration index `i` only runs `0 <= i < len(train_loader)`, `i+1` reaches at most `len(train_loader)` and can never reach `4 * len(train_loader)`. Consequently, the validation block **never executed during training in the official code**. Moreover, lines 198-199 inverted `target = data_val[0]` and `input_ = data_val[1]` relative to train lines 182-183 (`target = data[1]`, `input_ = data[0]`). Validation was completely non-functional.
  - *Repository implementation (`src/train/train.py:262, 283-285, 351-376, 527-540, 605`)*: Enforces an explicit scene-level train/val/test split (default 80% train / 10% val / 10% test) with strict scene-disjoint assertions (`overlap` check, line 283). Validation executes every epoch across all rows of validation slices, computing both TF MSE and image-domain PSNR against ground truth.

- **3. Checkpoint Selection (Best, Last, or Fixed Epoch)**:
  - *Paper text*: Mentions 100-epoch cycles without specifying checkpoint selection criteria.
  - *Official code (`train/train_FDNet.py:212-215, 231-240`, `test/A_test_*.py`)*: Because the validation block was dead code, `model_best.pth` (line 215) was never saved. Only `model_latest.pth` (line 234) and periodic `model_epoch_{epoch}.pth` (saved every `opt.checkpoint=20` epochs, lines 236-240) were written. All official test evaluation scripts (`A_test_fdnet.py:77, 109`, `A_test_unet.py:77`, etc.) load **fixed-epoch checkpoints** (specifically `model_epoch_100.pth`, `model_epoch_160.pth`, or `model_epoch_220.pth`).
  - *Repository implementation (`src/train/train.py:635-637`, `src/eval/experiments.py:99-111`)*: Explicitly tracks `best_val_psnr` and saves both `last.pt` and `best.pt`. All experiment runner scripts (`run_exp1.py`, `run_exp2.py`, `run_exp3.py`) load **`best.pt`** via `load_models` (`experiments.py:99, 111`).

- **4. Early Stopping and Reproducibility**:
  - *Early stopping*: Completely absent in paper, official code (`train_FDNet.py:173`), and repo (`train.py:596`). All run for full configured epochs.
  - *Reproducibility*: Official code sets seeds (1234) but enables `torch.backends.cudnn.benchmark = True` (`train_FDNet.py:23`) without setting deterministic mode. Repo exposes `--seed` (default 42), seeds PyTorch DataLoader generators (`train.py:536`), and serializes exact `torch_rng` tensor state into checkpoints (`train.py:633, 574`) ensuring resumption matches uninterrupted runs.

- **5. Experiments 5A–5D Audit (Implementation, I/O, Metrics, Required Data)**:
  - **Experiment 5A (Section 5.1: Real Sentinel-1 Generalization)**:
    - *Paper protocol*: Train on 640 lab slices; test on 30 Sentinel-1 slices (unlabeled, 512x1500). Metric: Mean-Entropy (ME, Eq. 25, Table 3: polluted 0.314, DNCNN 0.068, PISNet 0.067, UNet 0.066, FDNet 0.064). Visuals: Fig 7 (TF line spectrum), Fig 8 (SAR image).
    - *Official code*: `mydataset.py:42-56` (`loadtestall()` loads `testsentile.mat`, crops 211:723). No standalone exp1 script.
    - *Repo script*: `scripts/run_exp1.py`. Trains on lab data (or `--smoke`), runs inference via `suppress_rfi`, evaluates ME on magnitude images (`src/eval/metrics.py:127-148`), outputs `exp1_per_slice.csv`, `exp1_me.csv`, and figure panels (`_sar.png`, `_tf.png`).
    - *Data required*: 640 full lab slices for training; 30 Sentinel-1 slices for testing.
  - **Experiment 5B (Section 5.2: Same Data Distribution Benchmark)**:
    - *Paper protocol*: Train on lab data at SIR = -20 dB; test on lab data at 0 dB, -10 dB, -20 dB, -30 dB (160 slices/level). Metrics: PSNR (Eq. 23) and SSIM (Eq. 24). Visuals: Fig 9 (PSNR/SSIM vs SIR), Fig 10 (TF at -20 dB), Fig 11 (SAR at -20 dB).
    - *Official code*: `test/A_test_*.py` on test .mat files (`testsnr10.mat`, etc.). Metrics computed on 8-bit quantized `uint8` images [0, 255] (`A_test_fdnet.py:199-203`).
    - *Repo script*: `scripts/run_exp2.py`. Trains at SIR = -20 dB (`--train-sir -20.0`), tests across test scenes at 0, -10, -20, -30 dB (`eval/experiments.py:mat_cases`). Outputs `exp2_per_slice.csv`, `exp2_summary.csv`, and curves `exp2_psnr.png`, `exp2_ssim.png`. Metrics computed in unquantized float64 magnitude domain (`src/eval/metrics.py:81-124`).
    - *Data required*: 640 labeled lab slices (160 per SIR).
  - **Experiment 5C (Section 5.3: Domain Shift Condition)**:
    - *Paper protocol*: Train at SIR = -20 dB on one interference modulation distribution; test at 0/-10/-20/-30 dB on shifted interference distribution. Visuals: Fig 12 (PSNR/SSIM curves), Fig 13 (SAR image at -20 dB). *Crucial gap*: Paper text never disclosed the exact mathematical parameter distributions of the training vs shifted testing sets.
    - *Official code*: No domain shift scripts, configs, or data in repository.
    - *Repo script*: `scripts/run_exp3.py` & `src/data/param_sets.py`. Synthesizes disjoint parameter sets `exp3_train` (chirp rates [0.5, 1.5] x 1e11) vs `exp3_test` (chirp rates [2.5, 4.0] x 1e11 + sinusoidal/narrowband). Tested via `eval/experiments.py:synthetic_cases`. Outputs `exp3_per_slice.csv`, `exp3_summary.csv`, `exp3_shifted_by_modulation.csv`, `exp3_psnr.png`, `exp3_ssim.png`.
    - *Data required*: Clean SAR scene images + synthetic RFI simulator.
  - **Experiment 5D (Section 5.4: Different Domains - Image Subspace vs Echo Domain)**:
    - *Paper protocol*: Compares Algorithm 1 (image subspace) against echo-domain RFI suppression prior to SAR image formation across 0/-10/-20/-30 dB (Fig 14). Reports +0.96 dB PSNR and +0.06 SSIM improvement at -30 dB.
    - *Official code*: Completely absent. No raw radar echo simulator, raw echo data, or SAR imaging pipeline (range compression, RCMC, azimuth compression).
    - *Repo script*: Unimplemented (no `run_exp4.py`, no echo dataset loader, no radar imaging engine).
    - *Data required*: Raw SAR radar echo signal dataset (uncompressed raw phase history) + SAR imaging processor.

- **6. Execution Feasibility with Currently Available Data**:
  - *On-disk data truth*: `Data/RFI-Patch/RFI-1` has 16 stacks of shape `(5, 512, 1500)` complex64 (64 labeled pairs, 1 RFI class only; RFI-2 missing). `Data/Scenario 1..3` has 12 Sentinel-1 slices (4 slices x 3 scenarios; 18 slices missing). The 640 full lab slices from Baidu Netdisk are absent.
  - *Exp 5A*: **PARTIAL**. Sentinel-1 evaluation can run over the 12 available slices for ME scoring, but missing 18/30 test slices and all 640 lab training slices (must train on RFI-Patch or synthetic surrogate).
  - *Exp 5B*: **PARTIAL / BLOCKED as written**. `run_exp2.py` requires `--source mat` with individual `.mat` slices. The 16 stacks in `Data/RFI-Patch/RFI-1` can provide 16 training scenes at SNR -20 dB and test scenes at 0/-10/-20/-30 dB, but require slicing/adapter scripts. Faithfully reproducing the paper's 640-slice benchmark is strictly blocked.
  - *Exp 5C*: **PARTIAL / SYNTHETIC ONLY**. Can execute end-to-end using clean scenes extracted from RFI-Patch (`ss[0]`) or `--smoke`, but only under synthetic RFI injection.
  - *Exp 5D*: **BLOCKED**. Zero echo-domain code or data exist in repo or official release.

- **7. Synthetic Stand-Ins vs Faithful Reproduction**:
  - *`--smoke` runs (Exp 1, 2, 3)*: Runs 2 epochs on 20 tiny synthetic complex scenes `(64, 256)` with synthetic chirp RFI. Serves solely as pipeline verification; provides zero empirical reproduction value.
  - *Synthetic RFI in Exp 5C (`exp3_train` vs `exp3_test`)*: While mathematically rigorous and proven disjoint by unit tests, the parameter ranges were arbitrarily chosen by the repo authors. Because the paper withheld the true modulation parameters, no synthetic run can claim faithful numerical reproduction of Figures 12-13.
  - *RFI-Patch as Lab Training Surrogate (Exp 5A & 5B)*: Contains only 16 scenes for a single RFI category (RFI-1), representing <= 10% of the paper's sample size and omitting multi-class interference. Models trained on this subset suffer domain narrowing and cannot reproduce paper benchmark numbers.
  - *Synthetic Unlabeled Stand-in (`smoke_unlabeled`)*: `run_exp1.py:35-44` synthesizes unlabeled scenes by stripping clean labels from synthetic images; purely a code test.

- **8. Discrepancy Table**:

| Configuration / Dimension | Paper Setting (Fang et al. 2025) | Official Implementation (`SAR-RFI-DATASET-main`) | Repository Setting (`sar_rfi_baseline`) | Discrepancy / Reproduction Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Compute Hardware** | Intel i7-14700K CPU, RTX 3090 GPU (24GB) (p. 14) | Single CUDA GPU via `opt.gpu` (`train_FDNet.py:21`) | CPU-only detected (torch 2.14.1+cpu), `--device auto` | Real 100-epoch training infeasible without GPU; smoke runs only on current machine |
| **Training Epochs** | 100 epochs (Sec 4.3, p. 14) | `nepoch = 101` (`options.py:11`) / `301` (`HWOptions:76`) | Default 100 (`train.py:491`); 2 with `--smoke` | Repo matches paper text (100); official code had 101 or 301 |
| **Batch Size & Input Unit** | Batch size 4 (Sec 4.3, p. 14) | `batch_size = 4` patches (`options.py:10`), pre-cropped 512x512 | `batch_size = 4` slices, 32 rows/slice (8 for fdnet) | Repo steps over 128 (or 32) row spectrograms vs official 4 pre-computed 2D patches |
| **Optimizer** | AdamW (Sec 4.3, p. 14) | `optim.AdamW(..., betas=(0.9, 0.999), eps=1e-8)` (`train_FDNet.py:77`) | `torch.optim.AdamW(..., betas=(0.9, 0.999), eps=1e-8)` (`train.py:542`) | Faithful match across paper, official code, and repo |
| **Initial Learning Rate** | 0.0002 (`2e-4`) (Sec 4.3, p. 14) | 0.0002 (`options.py:17`, `train_FDNet.py:77`) | 2e-4 for baselines; 2e-3 for FDNet (`train.py:495`) | Repo differentiates FDNet (2e-3 per 2024 paper) from 2025 paper baselines (2e-4) |
| **Weight Decay** | Unstated in 2025 paper (0.02 in 2024 paper) | 0.02 (`options.py:19`, `train_FDNet.py:77`) | 0.01 for baselines; 0.02 for FDNet (`train.py:496`) | Repo uses PyTorch AdamW default (0.01) for baselines, 0.02 for FDNet |
| **LR Scheduler** | Unstated in 2025 paper | `StepLR(step_size=50, gamma=0.5)` with premature step at L97 | Constant (`lr_step=0`) for baselines; `StepLR(50, 0.5)` for FDNet | Official code stepped lr before training; repo steps at epoch end |
| **RNG Seeds** | Unstated in paper | Hardcoded 1234 (`train_FDNet.py:60-63`) | Configurable `--seed 42` (`train.py:457, 511`) | Repo adds full generator seeding & RNG state saving in checkpoints |
| **Validation During Training** | Unstated in paper | Dead code (`(i+1) % (4*len) == 0` never triggers; L193) | Full validation every epoch on val slices (`train.py:605`) | Official validation was completely broken; repo actively validates |
| **Checkpoint Selection** | 100 epochs mentioned | Fixed epoch (`model_epoch_100.pth`, `A_test_*.py:109`) | `best.pt` (highest validation PSNR) (`experiments.py:99`) | Repo selects model on validation PSNR; official used fixed epoch 100 |
| **Early Stopping** | None | None | None | Consistent (no early stopping) |
| **Input Representation** | STFT spectrogram (Alg 1, p. 13) | Pre-computed real float32 512x512 crops (`mydataset.py:14`) | On-the-fly STFT/ISTFT, 2-channel complex `(2, F, T)` | Principled architectural divergence; official networks saw real crops |
| **Evaluation Domain & Quantization** | PSNR/SSIM/ME on magnitude images (Sec 4.2) | 8-bit quantized `uint8` [0, 255] (`A_test_fdnet.py:199-203`) | Unquantized float64 magnitude images (`metrics.py:81-124`) | Official code quantizes images to 8-bit before PSNR/SSIM; repo computes exact float64 |

- **9. Concise Experiment Readiness Matrix**:

| Experiment | Paper Section | Official Script / Code | Repo Script | Required Data | Available Data on Disk | Execution Status | Reproduction Fidelity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 5A** | Sec 5.1 (Sentinel-1 Test) | `dataset/mydataset.py:loadtestall` | `scripts/run_exp1.py` | 640 lab train slices; 30 Sentinel-1 test slices | 0 of 640 lab slices; 12 of 30 Sentinel-1 slices | **PARTIAL** | **BLOCKED (faithful)** / Runnable as plumbing/small scale |
| **Exp 5B** | Sec 5.2 (Same Distribution) | `test/A_test_*.py` | `scripts/run_exp2.py` | 640 lab slices (160 per SIR at 0/-10/-20/-30 dB) | 16 RFI-Patch stacks (64 labeled pairs, RFI-1 only) | **PARTIAL** | **BLOCKED (faithful)** / Blocked as written (`run_exp2.py` expects `--source mat`) |
| **Exp 5C** | Sec 5.3 (Domain Shift) | None | `scripts/run_exp3.py` | Clean scenes + parameter shift definitions | 16 clean scenes in RFI-Patch; synthetic simulator | **PARTIAL** | **NOT VERIFIABLE** (Paper parameters withheld; runs with repo synthetic RFI) |
| **Exp 5D** | Sec 5.4 (Echo vs Image Subspace) | None | None (no script) | Raw radar echo phase history + SAR imaging processor | None (no raw echo data or imaging code) | **BLOCKED** | **BLOCKED** (Completely unimplemented in repo and official code) |
