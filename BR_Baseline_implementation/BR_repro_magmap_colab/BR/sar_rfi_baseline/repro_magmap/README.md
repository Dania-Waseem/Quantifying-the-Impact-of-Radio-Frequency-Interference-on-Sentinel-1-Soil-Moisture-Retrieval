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

Every script refuses to write into a non-empty output directory. Results on RFI-1 alone are a proxy;
the paper's 640-slice protocol needs all RFI-Patch folders (they are picked up automatically).
