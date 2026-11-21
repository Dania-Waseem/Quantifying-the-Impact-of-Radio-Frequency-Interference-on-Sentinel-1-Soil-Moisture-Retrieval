"""Check CLI argument compatibility between notebook and scripts."""

import sys
from pathlib import Path

ROOT = Path("sar_rfi_baseline")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train.train import parse_args as parse_train_args
from evaluate_model import parse_args as parse_eval_args

# Test 1: Train command for all 4 models and both data sources
models = ["unet", "dncnn", "pisnet", "fdnet"]
for m in models:
    # RFI-Patch call
    argv_patch = [
        "--model",
        m,
        "--source",
        "rfi_patch",
        "--root",
        "Data/RFI-Patch/RFI-1",
        "--epochs",
        "100",
        "--batch-size",
        "4",
        "--rows-per-image",
        "32",
        "--lr",
        "0.0002",
        "--weight-decay",
        "0.01",
        "--device",
        "cuda",
        "--seed",
        "42",
        "--out-dir",
        f"outputs/test_{m}",
        "--no-tensorboard",
        "--snr-db",
        "-20.0",
        "--amp",
        "--resume",
        "auto",
    ]
    args = parse_train_args(argv_patch)
    assert args.model == m
    assert args.amp is True
    assert args.resume == "auto"

    # Lab MAT call
    argv_mat = [
        "--model",
        m,
        "--source",
        "mat",
        "--root",
        "data/lab_mat",
        "--epochs",
        "100",
        "--batch-size",
        "4",
        "--rows-per-image",
        "32",
        "--lr",
        "0.0002",
        "--weight-decay",
        "0.01",
        "--device",
        "cuda",
        "--seed",
        "42",
        "--out-dir",
        f"outputs/test_{m}",
        "--no-tensorboard",
        "--sir",
        "-20.0",
        "--amp",
        "--resume",
        "auto",
    ]
    args2 = parse_train_args(argv_mat)
    assert args2.source == "mat"
    assert args2.sir == [-20.0]

print("Train CLI Argument Parsing: PASS for all 4 models and both data sources!")

# Test 2: Eval command
eval_argv = [
    "--checkpoint",
    "outputs/test_run/best.pt",
    "--source",
    "rfi_patch",
    "--root",
    "Data/RFI-Patch/RFI-1",
    "--split-file",
    "outputs/test_run/splits.json",
    "--device",
    "cuda",
    "--row-batch",
    "128",
    "--out-dir",
    "outputs/test_run/test_evaluation",
]
eval_args = parse_eval_args(eval_argv)
assert eval_args.device == "cuda"
assert eval_args.row_batch == 128
print("Eval CLI Argument Parsing: PASS!")
