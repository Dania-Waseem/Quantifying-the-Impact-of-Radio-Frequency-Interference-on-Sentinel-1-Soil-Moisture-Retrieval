"""Build the SIR20/AMP Colab package (v2): a small CODE-ONLY overlay zip + its notebook.

The data are not packed again: the notebook first unzips the existing BR_repro_magmap_colab.zip
(code + RFI-Patch data, SHA-256 checked, already on Drive), then unzips
BR_repro_magmap_sir20_amp_code.zip over it (updated repro_magmap/, src/, configs/splits/, and the
authors' model.py for check 7b; same paths, so the new code replaces the old copy inside /content only).
Nothing in the existing zip, notebook or Drive runs is changed.

Run from sar_rfi_baseline/:  python repro_magmap/scripts/package_colab_sir20_amp.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_colab as base  # noqa: E402

ZIP = base.WORK / "BR_repro_magmap_sir20_amp_code_v2.zip"     # v1 (_code.zip) kept unchanged
BASE_ZIP_NAME = "BR_repro_magmap_colab.zip"
BASE_ZIP_SHA256 = "34702b057a25f05a18f10a834150ccf49b19fa06d58be062a4f8361fbeb7db6b"
NB = base.PROJECT / "notebooks" / "Repro_MagMap_UNet_SIR20_AMP_Colab_v2.ipynb"
CONFIG = "repro_magmap/config_official_unet_sir20_amp.yaml"
cell = base.cell


def code_files():
    for src, arc in base.files():
        if "/Data/" not in arc.as_posix():
            yield src, arc


def notebook(zip_sha: str) -> dict:
    cells = [
        cell("""
# U-Net baseline (authors' representation), variant SIR20 + AMP — v2: precision decision by noise floor

Same experiment as v1 (`config_official_unet_sir20_amp.yaml`: training on -20 dB only, validation/test on all
four SIR levels; FP16 + GradScaler as in the authors' code; validation every 5 epochs and at epoch 100;
`best_val.pt` by clipped PSNR; `epoch_100.pt` reported; FP32 test inference). v2 adds:

* **validation in FP32 inference** during training (same precision as the test), config `train.val_precision: fp32`;
* `compare_amp.py --noise-floor`: FP32 vs FP32 and AMP vs AMP (rerun with the same batch order and a run with a
  different order) to measure run-to-run variation, then the decision rule: AMP only if it is not worse than FP32
  by more than max(noise floor, fixed tolerance) in clipped PSNR, clipped SSIM and loss, with no non-finite loss;
* `benchmark_t4.py --repeats 3`: AMP with and without channels_last, alternating, medians.

Upload `BR_repro_magmap_sir20_amp_code_v2.zip` to `MyDrive/`. Results go to new folders in
`MyDrive/repro_magmap_runs_sir20_amp/`; the v1 results there and the FP32 run in `MyDrive/repro_magmap_runs/` are
not touched. **Run the cells up to the STOP cell only and send the printed results for review.**
""", "markdown"),
        cell("!nvidia-smi"),
        cell(f"""
from google.colab import drive
drive.mount('/content/drive')
import hashlib, os, json
BASE = '/content/drive/MyDrive/{BASE_ZIP_NAME}'          # code + data, uploaded earlier
CODE = '/content/drive/MyDrive/{ZIP.name}'               # new code only (upload this one)
for f, want in ((BASE, '{BASE_ZIP_SHA256}'), (CODE, '{zip_sha}')):
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    print(os.path.basename(f), 'OK' if h == want else 'MISMATCH ' + h)
    assert h == want, 'checksum mismatch: re-upload ' + os.path.basename(f)
!rm -rf /content/BR && unzip -q "$BASE" -d /content && unzip -q -o "$CODE" -d /content
!grep -n "val_precision" /content/BR/sar_rfi_baseline/repro_magmap/config_official_unet_sir20_amp.yaml
RUNS = '/content/drive/MyDrive/repro_magmap_runs_sir20_amp'
CONFIG = '{CONFIG}'
os.makedirs(RUNS, exist_ok=True)
%cd /content/BR/sar_rfi_baseline
"""),
        cell("!pip -q install pyyaml scikit-image\n!python -m pytest repro_magmap/tests -q"),
        cell("""
# GPU smoke test of the v2 code path (FP32 validation inference); Colab VM disk, not Drive; not a result (~1-2 min)
!rm -rf /content/smoke
!python repro_magmap/scripts/train.py --config {CONFIG} --out-dir /content/smoke/train --device cuda --epochs 5 --max-steps-per-epoch 2 --train-maps 8 --val-maps 4 --channels-last true
!python repro_magmap/scripts/evaluate.py --run /content/smoke/train --checkpoint epoch_005 --allow-incomplete --device cuda --max-maps-per-group 1
"""),
        cell("""
# channels_last re-check: AMP only, 3 alternating repeats (median), + one FP32-inference validation pass (~8-10 min)
if not os.path.exists(f'{RUNS}/benchmark_t4_amp_repeats/benchmark.json'):
    !python repro_magmap/scripts/benchmark_t4.py --config {CONFIG} --out-dir {RUNS}/benchmark_t4_amp_repeats --precisions amp --repeats 3
B2 = json.load(open(f'{RUNS}/benchmark_t4_amp_repeats/benchmark.json'))
print(json.dumps({k: B2[k] for k in ('gpu_step', 'validation_pass_seconds', 'validation_inference', 'projection',
                                     'recommendation')}, indent=1))
"""),
        cell("""
# precision decision with noise floor: 6 runs x 300 steps (FP32 A, A-rerun, B; AMP A, A-rerun, B) (~15-20 min)
if not os.path.exists(f'{RUNS}/amp_check_noise_floor/amp_comparison.json'):
    !python repro_magmap/scripts/compare_amp.py --config {CONFIG} --out-dir {RUNS}/amp_check_noise_floor --steps 300 --noise-floor
C2 = json.load(open(f'{RUNS}/amp_check_noise_floor/amp_comparison.json'))['summary']
print(json.dumps({k: C2[k] for k in ('noise_floor', 'precision_effect_amp_minus_fp32', 'within_precision_differences',
                                     'output_noise_floor_mean_abs_diff_over_label_max',
                                     'output_precision_effect_mean_abs_diff_over_label_max', 'checks',
                                     'amp_nonfinite_losses', 'amp_skipped_steps', 'speedup_amp', 'recommendation')}, indent=1))
"""),
        cell("""
# >>> STOP: send the two printed blocks above for review before running anything below. <<<
# full training (100 epochs). Re-run after a disconnect: it resumes from last.pt.
APPROVED_BY_USER = False
assert APPROVED_BY_USER, 'full training not approved yet'
PRECISION = json.load(open(f'{RUNS}/amp_check_noise_floor/amp_comparison.json'))['summary']['recommendation']
CL = 'true' if (PRECISION == 'amp' and json.load(open(f'{RUNS}/benchmark_t4_amp_repeats/benchmark.json'))['recommendation']['channels_last']) else 'false'
print('precision', PRECISION, 'channels_last', CL)
RUN = f'{RUNS}/unet_sir20_{PRECISION}'
resume = '--resume' if os.path.exists(f'{RUN}/checkpoints/last.pt') else ''
!python repro_magmap/scripts/train.py --config {CONFIG} --out-dir {RUN} --device cuda --precision {PRECISION} --channels-last {CL} {resume}
"""),
        cell("""
# test evaluation (FP32 inference as in the authors' test): reported epoch_100 (+ image-domain), then best_val
!python repro_magmap/scripts/evaluate.py --run {RUN} --device cuda --image-domain
!python repro_magmap/scripts/evaluate.py --run {RUN} --checkpoint best_val --device cuda
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1e9, help="stop after this long; re-run to continue")
    args = ap.parse_args()
    if ZIP.exists():
        print(f"refusing to overwrite {ZIP}")
        return 2
    part = ZIP.with_suffix(".zip.part")
    t0 = time.time()
    with zipfile.ZipFile(part, "a") as z:
        have = set(z.namelist())
        for src, arc in code_files():
            name = arc.as_posix()
            if name in have:
                continue
            if time.time() - t0 > args.seconds:
                print(f"partial: {len(have)} files written; re-run to continue")
                return 3
            z.write(src, name, compress_type=zipfile.ZIP_DEFLATED)
            have.add(name)
    part.rename(ZIP)
    sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    NB.write_text(json.dumps(notebook(sha), indent=1), encoding="utf-8")
    print(f"DONE {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB, {len(have)} files) sha256 {sha}\n{NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
