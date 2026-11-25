"""Build BR_repro_magmap_colab.zip (code + RFI-Patch data) and the Colab notebook. Read-only on sources.

Run from sar_rfi_baseline/ (standard library only):
    python repro_magmap/scripts/package_colab.py
Zip layout: BR/sar_rfi_baseline/{repro_magmap,src,configs/splits}, BR/Data/RFI-Patch/RFI-*/*.mat
(the config's data path ../Data/RFI-Patch resolves inside the zip).
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT.parent
ZIP = WORK / "BR_repro_magmap_colab.zip"
NB = PROJECT / "notebooks" / "Repro_MagMap_UNet_Colab.ipynb"


def files():
    for sub in ("repro_magmap", "src"):
        for p in sorted((PROJECT / sub).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                yield p, Path("BR/sar_rfi_baseline") / p.relative_to(PROJECT)
    for p in sorted((PROJECT / "configs" / "splits").glob("*")):
        yield p, Path("BR/sar_rfi_baseline") / p.relative_to(PROJECT)
    authors_model = WORK / "SAR-RFI-DATASET-main" / "SAR-RFI-DATASET-main" / "extracted" / "Code" / "FDNet" / "model.py"
    yield authors_model, Path("BR") / authors_model.relative_to(WORK)     # for the weight-for-weight check 7b
    for p in sorted((WORK / "Data" / "RFI-Patch").rglob("*")):
        if p.is_file():
            yield p, Path("BR/Data") / p.relative_to(WORK / "Data")


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def notebook(zip_sha: str) -> dict:
    cells = [
        cell("""
# U-Net baseline, authors' magnitude-map representation (repro_magmap)

Reproduction track of the U-Net baseline in Fang et al., *Remote Sens.* 2025, 17, 3688, following the authors'
released code: 1-channel 512 x 512 |STFT| maps per range line, the authors' U-Net (9,502,561 parameters),
AdamW 2e-4, wd 0.02, Charbonnier 1e-3, batch 4, 100 epochs, StepLR(50, 0.5) with the authors' initial step,
seed 1234, and the authors' 8-bit map-domain PSNR / SSIM.

**Results on RFI-1 only are a proxy**, not a reproduction of Fig. 9 (the paper uses 640 slices).
Settings that are inferred or chosen (FFT size 512, hop 1, padding, fftshift, linear magnitude, per-map
normalization scope, 2048/1024 map sampling) are listed in `REPRODUCTION_STATUS.md`.

1. Upload `BR_repro_magmap_colab.zip` to `MyDrive/`. 2. Runtime -> GPU. 3. Run the cells in order.
Run outputs go to `MyDrive/repro_magmap_runs/` so a disconnect loses at most one epoch (re-run the training cell: it resumes).
""", "markdown"),
        cell("!nvidia-smi"),
        cell(f"""
from google.colab import drive
drive.mount('/content/drive')
ZIP = '/content/drive/MyDrive/BR_repro_magmap_colab.zip'
EXPECTED_SHA256 = '{zip_sha}'
import hashlib, os
h = hashlib.sha256(open(ZIP, 'rb').read()).hexdigest()
print('zip sha256', h, 'OK' if h == EXPECTED_SHA256 else 'MISMATCH - re-upload the zip')
!rm -rf /content/BR && unzip -q "$ZIP" -d /content && ls /content/BR /content/BR/Data/RFI-Patch
RUNS = '/content/drive/MyDrive/repro_magmap_runs'
os.makedirs(RUNS, exist_ok=True)
%cd /content/BR/sar_rfi_baseline
"""),
        cell("!pip -q install pyyaml scikit-image\n!python -m pytest repro_magmap/tests -q"),
        cell("""
# checks 1-7 (data, pairing, dims, STFT/ISTFT, normalization, U-Net = authors' class)
import os
if not os.path.exists(f'{RUNS}/verification_colab'):
    !python repro_magmap/scripts/verify_pipeline.py --out-dir {RUNS}/verification_colab
"""),
        cell("""
# checks 8-9 with a longer budget than the CPU run (one -20 dB training map, fp32, authors' optimizer settings)
if not os.path.exists(f'{RUNS}/overfit_gpu'):
    !python repro_magmap/scripts/overfit_one_sample.py --out-dir {RUNS}/overfit_gpu --steps 3000 --eval-every 100 --device cuda --max-minutes 30
"""),
        cell("""
# FP32 vs AMP before the full run (same init, same 300 batches); writes the decision
import json, os
if not os.path.exists(f'{RUNS}/amp_check/amp_comparison.json'):
    !python repro_magmap/scripts/compare_amp.py --out-dir {RUNS}/amp_check --steps 300
PRECISION = json.load(open(f'{RUNS}/amp_check/amp_comparison.json'))['summary']['recommendation']
print('precision for the full run:', PRECISION)
"""),
        cell("""
# full training (100 epochs); re-run this cell after a disconnect: it resumes from last.pt
import json, os
assert json.load(open(f'{RUNS}/overfit_gpu/overfit_result.json'))['pass'], 'one-sample overfit did not pass: do not start the full run'
RUN = f'{RUNS}/unet_official_{PRECISION}'
resume = '--resume' if os.path.exists(f'{RUN}/checkpoints/last.pt') else ''
!python repro_magmap/scripts/train.py --out-dir {RUN} --device cuda --precision {PRECISION} {resume}
"""),
        cell("""
# test evaluation: reported checkpoint (epoch 100) with supplementary image-domain metrics, then best-val
!python repro_magmap/scripts/evaluate.py --run {RUN} --device cuda --image-domain
!python repro_magmap/scripts/evaluate.py --run {RUN} --checkpoint best_val --device cuda
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> int:
    if ZIP.exists():
        print(f"refusing to overwrite {ZIP}")
        return 2
    manifest = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for src, arc in files():   # .mat files are already zlib-compressed inside: store them as they are
            z.write(src, arc.as_posix(), compress_type=zipfile.ZIP_STORED if src.suffix == ".mat" else zipfile.ZIP_DEFLATED)
            manifest.append(arc.as_posix())
    sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    NB.write_text(json.dumps(notebook(sha), indent=1), encoding="utf-8")
    print(f"{ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB, {len(manifest)} files) sha256 {sha}\n{NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
