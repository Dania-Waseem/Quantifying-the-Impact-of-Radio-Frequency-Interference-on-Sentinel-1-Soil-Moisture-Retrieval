# repro_magmap — U-Net baseline in the authors' representation

A separate reproduction track. It reuses nothing from the earlier complex 64 × 33 proxy pipeline except the U-Net
class (`src/models/unet.py`, instantiated with 1 input and 1 output channel) and, for the
supplementary image-domain metrics, `src/eval/metrics.py`. Both are imported and left unchanged.

| piece | file | follows |
|---|---|---|
| 512 × 512 magnitude maps of one range line | `magmap/stft.py` | paper (Hamming L/8), authors' loader ([N, 512, 512], crop [0:512]), rest inferred or chosen |
| data, pairing checks, scene split, samples | `magmap/data.py`, `magmap/pipeline.py` | dataset documentation, authors' logs (2048 / 1024 maps) |
| U-Net 1 → 1 | `magmap/model.py` | authors' `model.py` (checked weight-for-weight) |
| training loop | `magmap/trainer.py`, `scripts/train.py` | authors' `train/train_FDNet.py` and `options.py` |
| 8-bit map PSNR / SSIM | `magmap/metrics8.py`, `scripts/evaluate.py` | authors' `test/A_test_unet.py` and `utils/utils_image.py` |
| settings with provenance tags | `config_official_unet.yaml` | |

Order of use (from `sar_rfi_baseline/`):

```
python -m pytest repro_magmap/tests -q
python repro_magmap/scripts/inspect_dataset.py  --data-dir ../Data --out-dir <dir>
python repro_magmap/scripts/verify_pipeline.py  --out-dir <dir>
python repro_magmap/scripts/overfit_one_sample.py --out-dir <dir>
python repro_magmap/scripts/compare_amp.py      --out-dir <dir>              # GPU
python repro_magmap/scripts/train.py            --out-dir <run> --device cuda [--precision fp32]
python repro_magmap/scripts/evaluate.py         --run <run> --device cuda [--image-domain]
python repro_magmap/scripts/evaluate.py         --run <run> --checkpoint best_val --device cuda
```

The default config is `config_official_unet_sir20_amp.yaml` (paper Section 5.2: train on −20 dB only,
test on 0/−10/−20/−30 dB). `train.py` writes `protocol_audit.json` and stops when the laboratory set
is not 640 slices and 8 interference patterns. `--allow-incomplete-data` is only for a smoke test.
`config_official_unet.yaml` is the earlier all-SIR setting and is not the paper protocol.

```
python repro_magmap/scripts/audit_protocol.py --out-dir repro_magmap/protocol_audit
```

Results on RFI-1 alone are a proxy. The loader picks up every `RFI-*/*.mat` once the rest of the
authors' archive is unpacked under `Data/RFI-Patch`. The eight interference parameter sets are not
in the paper, so they are not synthesized.

## Variant SIR20 + AMP (2026-11-23)

`config_official_unet_sir20_amp.yaml` = the same settings with the approved changes: training on −20 dB only
(paper Sec. 5.2) with validation and test on all four SIR levels (`data.train_levels / val_levels / test_levels`),
FP16 autocast + GradScaler (as in the authors' code), `channels_last` only if `scripts/benchmark_t4.py` shows a gain,
validation every 5 epochs and at the last epoch, `best_val.pt` by clipped 8-bit PSNR, `epoch_100.pt` reported.
Test inference is FP32 by default (`evaluate.py`), as in the authors' `A_test_unet.py`.
Configs without the new keys behave exactly as before. Colab: `scripts/package_colab_sir20_amp.py` builds
`BR_repro_magmap_sir20_amp_colab.zip` and `notebooks/Repro_MagMap_UNet_SIR20_AMP_Colab.ipynb`
(runs go to `MyDrive/repro_magmap_runs_sir20_amp/`). Pre-change code: `../_backup_pre_sir20_amp_2026-11-23/`.
