"""Setup script for 100-epoch UNet baseline experiment on RFI-Patch (SNR -20 dB).

Creates outputs/runs/unet_rfipatch_snr-20_100epochs/, copies the zero-leakage splits.json,
writes the reference paper configuration (100 epochs, AdamW 2e-4, batch 4),
and packages the codebase for Colab GPU execution.
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def main():
    exp_dir = WORKSPACE_ROOT / "outputs" / "runs" / "unet_rfipatch_snr-20_100epochs"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy exact scene split file
    src_splits = WORKSPACE_ROOT / "outputs" / "runs" / "unet_rfipatch_snr-20" / "splits.json"
    dest_splits = exp_dir / "splits.json"
    shutil.copy2(src_splits, dest_splits)
    print(f"Copied zero-leakage splits: {dest_splits}")

    # 2. Write reference paper configuration (100 epochs)
    config = {
        "model": "unet",
        "source": "rfi_patch",
        "root": "Data/RFI-Patch/RFI-1",
        "split_file": str(dest_splits),
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "sir": None,
        "snr_db": [-20.0],
        "rfi_params": "default",
        "n_repeats": 1,
        "limit_train": None,
        "limit_val": None,
        "epochs": 100,
        "batch_size": 4,
        "rows_per_image": 32,
        "val_row_chunk": 128,
        "lr": 0.0002,
        "weight_decay": 0.01,
        "loss": "mse",
        "lr_step": 0,
        "lr_gamma": 0.5,
        "fd_eps": 0.001,
        "fd_lambda1": 0.0,
        "fd_lambda2": 1e-05,
        "fd_cos": "paper",
        "grad_clip": 0.0,
        "amp": True,
        "device": "auto",
        "num_workers": 2,
        "seed": 42,
        "out_dir": str(exp_dir),
        "resume": None,
        "tensorboard": True,
        "log_every": 50,
        "smoke": False,
        "smoke_scenes": 20,
        "smoke_shape": [64, 256],
        "n_train": 12,
        "n_val": 2,
        "parameters": 9503170,
        "paper_reference": "Fang et al., Remote Sensing 2025, 17, 3688 (Sec 4.3, Sec 5.2)",
    }

    config_path = exp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Written 100-epoch paper-matched configuration: {config_path}")

    # 3. Create junction inside sar_rfi_baseline/outputs/runs for consistency
    repo_runs = REPO_ROOT / "outputs" / "runs"
    repo_runs.mkdir(parents=True, exist_ok=True)
    repo_exp_dir = repo_runs / "unet_rfipatch_snr-20_100epochs"
    if not repo_exp_dir.exists():
        import subprocess
        cmd = f'cmd /c mklink /J "{repo_exp_dir}" "{exp_dir}"'
        subprocess.run(cmd, shell=True, check=False)
        print(f"Created junction: {repo_exp_dir} -> {exp_dir}")

    # 4. Refresh sar_rfi_baseline_colab.zip
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import package_for_colab
    package_for_colab.main()


if __name__ == "__main__":
    main()
