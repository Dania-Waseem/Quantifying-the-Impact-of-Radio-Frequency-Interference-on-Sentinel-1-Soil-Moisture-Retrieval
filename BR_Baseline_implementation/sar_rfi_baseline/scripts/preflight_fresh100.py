"""Preflight for the fresh 100-epoch UNet run (RFI-Patch RFI-1, documented -20 dB = ss[3]).

Checks only: CUDA, GPU, VRAM, disk, frozen split, dataset pairing, the exact train/val data path,
the original 5-epoch run (when present) and the fresh output directory. It NEVER trains.

Run from the workspace root (the folder that holds Data/ and sar_rfi_baseline/):
    python sar_rfi_baseline/scripts/preflight_fresh100.py
Optional: set FRESH100_OUT_DIR to put the run directory elsewhere (e.g. on Google Drive).
Exit code 0 = all checks passed.
"""

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]           # sar_rfi_baseline/
WORKSPACE = REPO.parent                               # folder holding Data/ and sar_rfi_baseline/
sys.path.insert(0, str(REPO / "src"))

DATA_ROOT = WORKSPACE / "Data" / "RFI-Patch" / "RFI-1"
FROZEN_SPLIT = REPO / "configs" / "splits" / "rfipatch_rfi1_seed42_frozen.json"
DATA_MANIFEST = REPO / "configs" / "splits" / "rfipatch_rfi1_data.sha256"
ORIG_DIR = REPO / "outputs" / "rfi_patch_proxy_unet"
NEW_DIR = Path(os.environ.get("FRESH100_OUT_DIR",
                              REPO / "outputs" / "unet_rfipatch_snr-20_fresh100_gpu_amp"))

SPLIT_SHA = "15a7f51184cd4291322085d1f8025ca61307ab3fdb969af484b8b1165556d95f"
ORIG_SHA = {
    "last.pt": "bdffb884eac15b91576becb1cc8f21a4ebb11e54dfbf7d3caf90b6edf7befc1b",
    "best.pt": "6f440351ab9eab23cc0be5047d54891d86524d784982fc8684bb2a16660caaf3",
    "metrics.csv": "c8706606b1d38678a8c9eaf662cea9033e652d6f77690ee52ecd5953708bf725",
    "config.json": "4554b5d33e8aadd322c48d1ea65fed970c06898589898abb6e0f09a12a96d87e",
    "splits.json": SPLIT_SHA,
}
EXPECTED = {"train": ["fifth", "first", "fourth", "s1", "s10", "s2", "s4", "s5", "s6", "s9", "second", "sixth"],
            "val": ["s7", "s8"], "test": ["s3", "third"]}
MIN_FREE_GB = 2.0    # 2 x 114 MB checkpoints + .tmp copy during save + logs, with margin
MIN_VRAM_GB = 4.0

TRAIN_ARGV = [
    "--model", "unet", "--source", "rfi_patch", "--root", str(DATA_ROOT),
    "--split-file", str(FROZEN_SPLIT), "--snr-db", "-20",
    "--epochs", "100", "--batch-size", "4", "--rows-per-image", "32", "--val-row-chunk", "128",
    "--lr", "2e-4", "--weight-decay", "0.01", "--loss", "mse", "--lr-step", "0", "--grad-clip", "0",
    "--seed", "42", "--num-workers", "2", "--device", "cuda", "--amp", "--no-tensorboard",
    "--out-dir", str(NEW_DIR),
]

failures = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("== 1. CUDA / GPU / VRAM ==")
import torch
print(f"        torch {torch.__version__}, CUDA build {torch.version.cuda}")
check(torch.cuda.is_available(), "CUDA available")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"        GPU 0: {torch.cuda.get_device_name(0)}  (compute capability {torch.cuda.get_device_capability(0)})")
    print(f"        VRAM: {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    check(free / 2**30 >= MIN_VRAM_GB, f"at least {MIN_VRAM_GB:.0f} GiB VRAM free")

print("== 2. Disk ==")
NEW_DIR.parent.mkdir(parents=True, exist_ok=True)
du = shutil.disk_usage(NEW_DIR.parent)
print(f"        {NEW_DIR.parent}: {du.free / 2**30:.2f} GiB free of {du.total / 2**30:.2f} GiB")
check(du.free / 2**30 >= MIN_FREE_GB, f"at least {MIN_FREE_GB:.0f} GiB free for checkpoints and logs")

print("== 3. Fresh output directory ==")
print(f"        {NEW_DIR}")
existing = [n for n in ("last.pt", "best.pt", "metrics.csv", "config.json") if (NEW_DIR / n).exists()]
check(not existing, f"no previous checkpoints/logs in it {existing if existing else ''}".rstrip())
check(NEW_DIR.resolve() != ORIG_DIR.resolve(), "differs from the 5-epoch directory rfi_patch_proxy_unet/")

print("== 4. Original 5-epoch run ==")
if not ORIG_DIR.exists():
    print("  SKIP  not present on this machine (expected on Colab; it stays on your PC)")
else:
    for name, expected in ORIG_SHA.items():
        p = ORIG_DIR / name
        check(p.exists() and sha(p) == expected, f"{name} unchanged (hash matches audit)")

print("== 5. Frozen split ==")
check(FROZEN_SPLIT.exists() and sha(FROZEN_SPLIT) == SPLIT_SHA,
      "frozen split is a byte copy of the original splits.json (sha256 15a7f511...)")
from data.dataset import load_split_file, load_rfi_patch, RFI_PATCH_CLEAN_INDEX, RFI_PATCH_SNR_DB
split = load_split_file(FROZEN_SPLIT)
by = {k: sorted(s for s, v in split.items() if v == k) for k in ("train", "val", "test")}
print(f"        train ({len(by['train'])}): {', '.join(by['train'])}")
print(f"        val: {', '.join(by['val'])}   test: {', '.join(by['test'])}")
check(by == EXPECTED, "split = 12 train / val s7, s8 / test s3, third")

print("== 6. Dataset pairing ==")
import numpy as np
check(RFI_PATCH_CLEAN_INDEX == 0, "clean reference = stack[0]")
check(RFI_PATCH_SNR_DB.get(3) == -20.0, "stack[3] = documented -20 dB")
files = sorted(DATA_ROOT.glob("*.mat"))
check(len(files) == 16 and {f.stem for f in files} == set(split), f"{len(files)} RFI-1 .mat files, one per split entry")
want = dict(line.split()[::-1] for line in DATA_MANIFEST.read_text().splitlines() if line.strip())
mism = [f.name for f in files if want.get(f.name) != sha(f)]
check(not mism and len(want) == 16, f"all 16 .mat files match the PC copies byte-for-byte (sha256) {mism if mism else ''}".rstrip())
bad, r3 = [], []
for f in files:
    ss = load_rfi_patch(f)["ss"]
    c = ss[0]
    r = [10 * np.log10(np.sum(np.abs(c) ** 2) / np.sum(np.abs(ss[k] - c) ** 2)) for k in range(1, 5)]
    r3.append(r[2])
    if not (ss.shape == (5, 512, 1500) and np.iscomplexobj(ss) and r[0] > r[1] > r[2] > r[3]
            and np.isfinite(ss).all()):
        bad.append(f.name)
check(not bad, f"all stacks complex (5,512,1500), finite, interference rising ss[1]->ss[4] {bad if bad else ''}".rstrip())
print(f"        measured clean/(stack[3]-clean) power ratio: {min(r3):.2f} to {max(r3):.2f} dB "
      "(documented label -20 dB; label, not measured SIR)")

print("== 7. Exact training data path (datasets built, no training) ==")
_spec = importlib.util.spec_from_file_location("train_entry", REPO / "src" / "train" / "train.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
args = T.parse_args(TRAIN_ARGV)
with tempfile.TemporaryDirectory() as tmp:
    tr, va = T.build_datasets(args, Path(tmp))
recs = tr.records + va.records
check(len(tr.records) == 12 and len(va.records) == 2, f"{len(tr.records)} train / {len(va.records)} val slices")
check(all(r.ss_index == 3 and r.snr_db_documented == -20.0 for r in recs), "every train/val input is stack[3] (-20 dB)")
check(not ({r.scene_id for r in recs} & set(EXPECTED["test"])), "test scenes s3, third excluded from train and val")
check(args.epochs == 100 and args.resume is None and args.model == "unet", "UNet, 100 epochs, fresh start (no resume)")

print("\n== Launch command (NOT executed) ==")
print("python sar_rfi_baseline/src/train/train.py " + " ".join(f'"{a}"' if " " in a else a for a in TRAIN_ARGV))
print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
