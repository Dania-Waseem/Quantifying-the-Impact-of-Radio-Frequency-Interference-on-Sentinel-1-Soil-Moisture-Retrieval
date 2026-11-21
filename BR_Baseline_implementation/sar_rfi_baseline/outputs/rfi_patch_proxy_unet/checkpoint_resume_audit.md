# Checkpoint Resume Audit — UNet, RFI-Patch, SNR −20 dB

Date: 2026-11-21 · Scope: inspection only. No training was run, and no code or checkpoint was modified.
I could not do the controlled resume test because PyTorch was not available in the audit environment
(see §6 for the command to run it yourself).

## 1. Checkpoint contents (`sar_rfi_baseline/outputs/rfi_patch_proxy_unet/`)

| Field | last.pt | best.pt |
|---|---|---|
| epoch | 5 | 5 |
| model_name | unet (72 tensors, 9,503,170 params) | same |
| optimizer | AdamW state, 72 params (step, exp_avg, exp_avg_sq), lr 2e-4, wd 0.01 | same |
| scaler | None (run was CPU, amp=False) | None |
| scheduler | None (lr_step = 0, constant LR) | None |
| best_val_psnr / val_psnr | 19.5597 dB | 19.5597 dB |
| global_step | 15 | 15 |
| torch_rng | CPU RNG state (5056 bytes) | same |
| args | epochs 5, device cpu, amp False, snr_db [−20], seed 42, split_file None | same |

SHA-256: last.pt `bdffb884…7befc1b`, best.pt `6f440351…16660caaf3` (keep these as a check that the originals are unchanged).

Note: **global_step = 15**. With 12 training slices at batch size 4, each epoch is only 3 optimizer steps,
so the "5-epoch" model has received just 15 gradient updates. Validation PSNR (19.56 dB) is still
below the input PSNR (19.73 dB) at every epoch in metrics.csv.

## 2. Resume code (`src/train/train.py`)

| Item | Lines | Status |
|---|---|---|
| `--resume` arg (path or `auto` → `<out-dir>/last.pt`) | 460, 552–559 | OK |
| model state | 562–564 | Restored (checks model_name) |
| optimizer state | 565 | Restored |
| GradScaler | 566–567 | Restored only if saved and AMP on; here none saved → fresh scaler |
| scheduler | 568–569 | Not used (constant LR); nothing to restore |
| start epoch | 570 | `start_epoch = state["epoch"] + 1` → **6** |
| best_val_psnr, global_step | 571–572 | Restored (19.5597, 15) |
| torch CPU RNG | 573–574 | Restored |
| loop bound | 596 | `range(start_epoch, args.epochs + 1)` → epochs 6…100 |
| best.pt update | 607, 636–637 | Only when `val_psnr > best_psnr` — correct |

## 3. Requirement checklist

| # | Requirement | Result |
|---|---|---|
| 1 | Checkpoint records epoch 5 + model state | ✅ |
| 2 | Optimizer state present and restorable | ✅ |
| 3 | Scheduler restored if used | ✅ N/A (no scheduler) |
| 4 | RNG states restored | ⚠️ Partial: torch CPU only. Python `random`, NumPy, CUDA RNG and the DataLoader shuffle generator are **not** saved |
| 5 | Resume starts at epoch 6 | ✅ (by code: 5 + 1) |
| 6 | Target is 100 total, not 100 more | ✅ `--epochs` is the final epoch number; 95 epochs would run |
| 7 | Config/split/SIR/preprocessing unchanged | ⚠️ See §4 |
| 8 | best.pt only on improvement | ✅ |

## 4. Problems to fix before training

**P1 — Nothing currently resumes from epoch 5.** The configured 100-epoch run is a fresh run:
- `outputs/runs/unet_rfipatch_snr-20_100epochs/config.json` has `"resume": null`.
- The Colab notebook (cell 14) only adds `--resume auto` if `RUN_DIR/last.pt` exists, and
  `RUN_DIR = outputs/colab_runs/unet_<source>_level-20`, a new folder with no checkpoint.
  So it will start at epoch 1.

**P2 — Overwrite risk.** If `--resume auto` is pointed at the original folder
(`--out-dir .../rfi_patch_proxy_unet`), the run overwrites `last.pt`, `best.pt`, `metrics.csv`,
`config.json` and `splits.json` there. `build_datasets` always rewrites `splits.json`, and `main` always rewrites `config.json`.

**P3 — Protocol changes on resume.** Epochs 1–5 ran on CPU in fp32. The 100-epoch config/notebook use
CUDA with AMP. Resuming would mix the two in one run. Resume also checks only `model_name`, not the
data, split or SNR args.

**P4 — Data order is not continued.** The DataLoader generator is re-seeded with `seed=42` on every
launch, so epoch 6 sees the same shuffle as epoch 1. The row sampling in `RowSampler` uses worker RNGs
that are not saved either. Neither is harmful, but the result is not bit-identical to an uninterrupted run.

**P5 — Notebook passes no `--split-file`.** The split is regenerated from seed 42. It should reproduce the
same split (s3, third = test; s7, s8 = val), but pass `--split-file` explicitly to guarantee it.

**P6 — History/best.pt in a new folder.** Resuming into a new `--out-dir` starts an empty metrics.csv
(epochs 1–5 missing) and writes `best.pt` there only once validation PSNR beats 19.5597 dB.

Splits verified: `rfi_patch_proxy_unet/splits.json` and `unet_rfipatch_snr-20_100epochs/splits.json` are byte-identical.

## 5. Recommendation

The 5 epochs amount to 15 optimizer steps, a few seconds of GPU time. Resuming saves almost nothing and creates
the CPU/fp32 → GPU/AMP mix in P3. **A fresh 100-epoch run** (same split file, same seed, same hyperparameters)
gives one consistent protocol. Keep the 5-epoch checkpoint as the preliminary result for comparison.
For fresh training, the current config is fine, but add `--split-file`.

If you do want a true resume, use a new out-dir and seed it with the old history:

```
copy  rfi_patch_proxy_unet\metrics.csv  -> <NEW_DIR>\metrics.csv
python src/train/train.py --model unet --source rfi_patch --root Data/RFI-Patch/RFI-1 \
  --snr-db -20 --epochs 100 --batch-size 4 --rows-per-image 32 --lr 2e-4 --weight-decay 0.01 \
  --seed 42 --split-file outputs/rfi_patch_proxy_unet/splits.json \
  --resume outputs/rfi_patch_proxy_unet/last.pt --out-dir <NEW_DIR> --device cpu --no-amp
```
(Use `--device cuda --amp` instead only if you accept the precision change, and document it.)

## 6. Minimal resume test (run yourself, ~1 epoch)

Copy `last.pt` into a scratch folder and run the command above with `--epochs 6 --out-dir <SCRATCH>`.
Expected: the log prints `resumed from … at epoch 6 (best val PSNR 19.560 dB)`, then exactly one line
`epoch   6/6`. The scratch `last.pt` should then have epoch 6 and global_step 18. Re-check the original's SHA-256 afterwards.
