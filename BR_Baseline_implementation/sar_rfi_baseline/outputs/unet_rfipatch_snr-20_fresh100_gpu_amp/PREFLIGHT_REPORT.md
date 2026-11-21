# Preflight — Fresh 100-Epoch UNet (RFI-Patch RFI-1, documented −20 dB)

Prepared 2026-11-21. **Status: prepared, NOT launched. No training has run.**

## 1. GPU and VRAM — NOT VERIFIED
These could not be checked from here. The shell on this PC would not start because the disk has less than
the ~17 GB it needs, and no PyTorch was available elsewhere. The 5-epoch run used `device: cpu`.
Run `scripts/preflight_fresh100.py` on the machine that will train (this PC or Colab). It prints the GPU name
and free/total VRAM, and fails if CUDA is missing or less than 4 GiB VRAM is free.

## 2. Disk space — PARTLY VERIFIED
This PC has less than ~17 GB free (that is why the shell failed). The exact free space is unknown.
The run needs about 0.5 GB: best.pt and last.pt are 114 MB each, plus a .tmp copy during each save and small logs.
The preflight script requires at least 2 GiB free.

## 3. Dataset pairing — VERIFIED (all 16 files)
- `Data/RFI-Patch/explaination.docx` says: Label = `ss(1,:,:)`, snr = −20 dB = `ss(4,:,:)` (MATLAB 1-based).
  In Python that is **clean = `stack[0]`, −20 dB = `stack[3]`**.
- Code: `RFI_PATCH_CLEAN_INDEX = 0`, `RFI_PATCH_SNR_DB = {1: 0, 2: −10, 3: −20, 4: −30}`. The loader (`dataset.py`, lines 840–842) uses `stack[ss_index]` vs `stack[0]`.
  Training at −20 dB keeps only `ss_index = 3`.
- All 16 files are complex64, shape (5, 512, 1500).
- Measured 10·log10(‖clean‖² / ‖stack[k] − clean‖²) for k = 1…4 drops by about 10 dB per step in every file.
  For `stack[3]` it ranges from −14.8 dB (third) to −18.2 dB (s7). So the "−20 dB" is the dataset's **documented label**,
  about 2–5 dB above the measured ratio. Report it as "documented −20 dB" and don't call it a measured SIR.

## 4. Split — VERIFIED, frozen
- `configs/splits/rfipatch_rfi1_seed42_frozen.json` is a byte copy of
  `outputs/rfi_patch_proxy_unet/splits.json` (SHA-256 `15a7f511…56d95f`). It is passed with `--split-file`, so nothing is regenerated.
- Train (12): fifth, first, fourth, s1, s10, s2, s4, s5, s6, s9, second, sixth · Val: s7, s8 · Test: s3, third.
- The copy lives under `configs/` so the Colab zip includes it. `train.py` rewrites `splits.json` in its run folder,
  so the frozen file is stored under a different name.

## 5. Test isolation — VERIFIED in code
`build_datasets` returns only train and val records (train.py, lines 265–266). `validate()` runs on the val loader only,
and `best.pt` is chosen by val PSNR (lines 607 and 636–637). Test scenes are touched only by the separate evaluation run after training.

## 6. Training configuration
| Setting | 5-epoch run | Fresh run |
|---|---|---|
| model | UNet, 9,503,170 params | same |
| data | rfi_patch, RFI-1, documented −20 dB (ss[3]) | same |
| split | seed 42 (generated) | same split, from the frozen file |
| epochs | 5 | **100** |
| batch / rows per slice / val chunk | 4 / 32 / 128 | same |
| optimizer | AdamW, lr 2e-4, betas (0.9, 0.999), eps 1e-8, wd 0.01 | same |
| LR schedule / loss / grad clip | constant / MSE / off | same |
| seed / workers | 42 / 2 | same |
| device / precision | CPU / fp32 | **CUDA / AMP** (intended change) |
| resume | none | none (fresh start) |
| tensorboard | off | off (stdout is saved to train_log.txt) |

## 7. New output directory
`sar_rfi_baseline/outputs/unet_rfipatch_snr-20_fresh100_gpu_amp/`
train.py will write config.json, splits.json, metrics.csv (train loss, val loss, val PSNR and input PSNR per epoch), last.pt and best.pt here.
Tee the console to train_log.txt.
The original `outputs/rfi_patch_proxy_unet/` is not referenced by the command. Its hashes are in `original_hashes.txt`, and the preflight script re-checks them.
(The older `outputs/runs/unet_rfipatch_snr-20_100epochs/` is an unused stub holding only config.json and splits.json. It is not used.)

## 8. Launch command (not executed — awaiting approval)
Run from the workspace root (the folder containing `Data/` and `sar_rfi_baseline/`):
```
python sar_rfi_baseline/scripts/preflight_fresh100.py

python sar_rfi_baseline/src/train/train.py --model unet --source rfi_patch \
  --root Data/RFI-Patch/RFI-1 \
  --split-file sar_rfi_baseline/configs/splits/rfipatch_rfi1_seed42_frozen.json \
  --snr-db -20 --epochs 100 --batch-size 4 --rows-per-image 32 --val-row-chunk 128 \
  --lr 2e-4 --weight-decay 0.01 --loss mse --lr-step 0 --grad-clip 0 \
  --seed 42 --num-workers 2 --device cuda --amp --no-tensorboard \
  --out-dir sar_rfi_baseline/outputs/unet_rfipatch_snr-20_fresh100_gpu_amp \
  2>&1 | tee sar_rfi_baseline/outputs/unet_rfipatch_snr-20_fresh100_gpu_amp/train_log.txt
```
`--device cuda` (not `auto`) makes the run stop with an error if no GPU is present, instead of falling back to CPU.
