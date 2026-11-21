"""Script to generate SAR_RFI_Colab_Training.ipynb."""

import json
from pathlib import Path

cells = []


def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    })


def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    })


# --- CELL 1: Header ---
add_md(
    """# SAR-RFI Baseline: Full Model Training on Google Colab GPU

**Reference Paper:**  
Fang et al., *Standardized Dataset and Image-Subspace-Based Method for Strip-Mode Synthetic Aperture Radar Radio Frequency Interference Suppression*, Remote Sensing, 2025, 17, 3688.

**Purpose:**  
Train and evaluate deep learning SAR radio frequency interference (RFI) mitigation baselines on GPU:
* **DNCNN** (17-layer residual CNN)
* **PISNet** (EM attention unit baseline with sigmoid removed per Fang et al.)
* **UNet** (Official 4-level U-Net with residual ConvBlocks)
* **FDNet** (Feature Decomposition Network / FDUformer)

> **Important Experimental Status & Integrity Notice:**  
> * **Dataset Context:** If training on the local 16-scene subset, results represent an **RFI-Patch Proxy Benchmark** (RFI-1 pattern only). Full replication of the paper's 640-slice benchmark requires downloading the complete Baidu Netdisk dataset (`data/lab_mat`).
> * **Evaluation Isolation:** The test split is held strictly separate from model selection. The best checkpoint is chosen purely from validation loss/PSNR during training. Test evaluation is executed separately after training finishes.
"""
)

# --- CELL 2: GPU Detection ---
add_md(
    """## 1. Hardware & CUDA Verification
Ensure Google Colab is running with a GPU runtime:
* Go to **Runtime** > **Change runtime type**
* Hardware accelerator: select **GPU** (e.g. T4, V100, or A100)
"""
)

add_code(
    """# Check GPU allocation and CUDA availability
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Allocated Device: {device_name}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    raise SystemError("CRITICAL: GPU acceleration is not available. Please change your runtime to GPU in Runtime -> Change runtime type.")
"""
)

# --- CELL 3: Mount Google Drive or Extract Code ---
add_md(
    """## 2. Project Workspace Setup

Choose one of the following options to set up the project files in Colab:
* **Option A (Recommended for Persistence):** Mount your Google Drive where `sar_rfi_baseline` is stored. Checkpoints will be saved directly to your Drive so they survive session timeouts.
* **Option B:** Upload `sar_rfi_baseline.zip` directly to the Colab session files and extract it.
"""
)

add_code(
    """import os
import sys
from pathlib import Path

# --- OPTION A: Mount Google Drive ---
MOUNT_DRIVE = True  # Set to True if using Google Drive

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Adjust this path to where your project directory is located in Google Drive:
    # e.g., '/content/drive/MyDrive/SAR_RFI_Project/sar_rfi_baseline'
    PROJECT_ROOT = Path('/content/drive/MyDrive/sar_rfi_baseline')
    
    if not PROJECT_ROOT.exists():
        print(f"Drive path {PROJECT_ROOT} not found. Defaulting to local /content/sar_rfi_baseline")
        PROJECT_ROOT = Path('/content/sar_rfi_baseline')
else:
    PROJECT_ROOT = Path('/content/sar_rfi_baseline')

print(f"Project root directory set to: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)
"""
)

# --- CELL 4: Install Dependencies ---
add_md(
    """## 3. Environment Dependencies
Install required packages and verify imports:
"""
)

add_code(
    """!pip install -q pyyaml scipy scikit-image h5py tqdm matplotlib pandas

import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import yaml

print("Core scientific libraries loaded successfully.")
"""
)

# --- CELL 5: Dataset Verification ---
add_md(
    """## 4. Dataset Path & Integrity Verification
Specify the location of your dataset:
* **For RFI-Patch Proxy:** Path to `Data/RFI-Patch/RFI-1` (or `Data/RFI-Patch`).
* **For Full Lab Benchmark:** Path to the unzipped Baidu Netdisk dataset `data/lab_mat`.
"""
)

add_code(
    """DATA_SOURCE = "rfi_patch"  # Choices: "rfi_patch" or "mat"

if DATA_SOURCE == "rfi_patch":
    default_dir = Path("/content/Data/RFI-Patch/RFI-1")
    fallbacks = [
        default_dir,
        Path("/content/Data/RFI-Patch"),
        PROJECT_ROOT / "Data" / "RFI-Patch" / "RFI-1",
        PROJECT_ROOT / "Data" / "RFI-Patch",
        PROJECT_ROOT.parent / "Data" / "RFI-Patch" / "RFI-1",
    ]
else:
    default_dir = Path("/content/data/lab_mat")
    fallbacks = [
        default_dir,
        PROJECT_ROOT / "data" / "lab_mat",
        PROJECT_ROOT.parent / "data" / "lab_mat",
    ]

DATA_ROOT = default_dir
for candidate in fallbacks:
    if candidate.exists():
        DATA_ROOT = candidate
        break

print(f"Verifying dataset at: {DATA_ROOT}")
assert DATA_ROOT.exists(), f"ERROR: DATA_ROOT {DATA_ROOT} does not exist. Please upload or link the dataset."

from data.dataset import discover_rfi_patch_records, discover_mat_records, load_rfi_patch

if DATA_SOURCE == "rfi_patch":
    records = discover_rfi_patch_records(DATA_ROOT)
    print(f"Discovered {len(records)} slice records across {len(records)//4} scenes.")
    # Check first record
    sample = load_rfi_patch(records[0].path)
    print(f"Sample array shape: {sample['ss'].shape}, dtype: {sample['ss'].dtype}")
    assert not np.isnan(sample['ss']).any(), "NaN values detected in dataset!"
    assert not np.isinf(sample['ss']).any(), "Inf values detected in dataset!"
    print("Dataset verification: 100% Valid (zero NaNs, zero Infs, complex64 confirmed).")
else:
    records = discover_mat_records(DATA_ROOT)
    print(f"Discovered {len(records)} lab slice records.")
"""
)

# --- CELL 6: Training Configuration Form ---
add_md(
    """## 5. Experiment Configuration (Independent Model Training)
Configure training parameters. You can train **DNCNN**, **PISNet**, **UNet**, or **FDNet** independently.
All values adhere to the reference paper (Section 4.3):
* **Batch Size:** 4
* **Epochs:** 100
* **Optimizer:** AdamW
* **Initial Learning Rate:** `0.0002` (FDNet standard: `0.0002` per paper Section 4.3; optional `0.002` per official 2024 code)
* **Training SIR / SNR Level:** `-20.0 dB` (Paper Section 5.2 protocol)
* **Automatic Mixed Precision (AMP):** Enabled for GPU acceleration
"""
)

add_code(
    """# @title Training Hyperparameters { run: "auto" }
MODEL_NAME = "unet"  # @param ["unet", "dncnn", "pisnet", "fdnet"]
EPOCHS = 100  # @param {type:"integer"}
BATCH_SIZE = 4  # @param {type:"integer"}
LEARNING_RATE = 0.0002  # @param {type:"number"}
WEIGHT_DECAY = 0.01  # @param {type:"number"}
ROWS_PER_IMAGE = 32  # @param {type:"integer"}  # 32 rows per slice sampled per step
TRAIN_LEVEL = -20.0  # @param {type:"number"}  # Train at -20 dB SIR/SNR per Section 5.2
USE_AMP = True  # @param {type:"boolean"}
SEED = 42  # @param {type:"integer"}
AUTO_RESUME = True  # @param {type:"boolean"}

# Define run output directory
RUN_DIR = PROJECT_ROOT / "outputs" / "colab_runs" / f"{MODEL_NAME}_{DATA_SOURCE}_level{int(TRAIN_LEVEL)}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run directory configured at: {RUN_DIR}")
"""
)

# --- CELL 7: GPU Pre-Flight Sanity Check ---
add_md(
    """## 6. Pre-Flight GPU Sanity Check
Verifies forward pass, backward pass, and memory allocation on the active GPU before launching the 100-epoch loop:
"""
)

add_code(
    """from models import get_model
from train.losses import Criterion

print(f"Testing {MODEL_NAME.upper()} on GPU...")
device = torch.device("cuda")
model = get_model(MODEL_NAME).to(device)
model.train()

# Input dimension: (B=4, C=2, F=187, T=33) matching Algorithm 1 STFT spectrograms
x = torch.randn(BATCH_SIZE, 2, 187, 33, device=device)
target = torch.randn(BATCH_SIZE, 2, 187, 33, device=device)

torch.cuda.reset_peak_memory_stats()
loss_mode = "fdnet" if MODEL_NAME == "fdnet" else "mse"
crit = Criterion(loss_mode)

outputs = crit.model_forward(model, x)
pred, total_loss, parts = crit.loss(outputs, target)
total_loss.backward()

allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
print("Forward + Backward pass successful!")
print(f"Peak VRAM used by single batch: {allocated_mb:.2f} MB")
print(f"Total loss: {total_loss.item():.4f} | Loss breakdown: {parts}")

del x, target, model, outputs, pred, total_loss
torch.cuda.empty_cache()
"""
)

# --- CELL 8: Full 100-Epoch Training Loop ---
add_md(
    """## 7. Full 100-Epoch Model Training
Launches the training loop.  
* Tracks training and validation loss every epoch.
* Evaluates validation PSNR on held-out scenes.
* Automatically updates `best.pt` whenever validation PSNR improves.
* Saves `last.pt` each epoch so training can be resumed seamlessly if disconnected.
"""
)

add_code(
    """# Build the training command line arguments
train_cmd = [
    "python", "src/train/train.py",
    "--model", MODEL_NAME,
    "--source", DATA_SOURCE,
    "--root", str(DATA_ROOT),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--rows-per-image", str(ROWS_PER_IMAGE),
    "--lr", str(LEARNING_RATE),
    "--weight-decay", str(WEIGHT_DECAY),
    "--device", "cuda",
    "--seed", str(SEED),
    "--out-dir", str(RUN_DIR),
    "--no-tensorboard"
]

if DATA_SOURCE == "rfi_patch":
    train_cmd += ["--snr-db", str(TRAIN_LEVEL)]
else:
    train_cmd += ["--sir", str(TRAIN_LEVEL)]

if USE_AMP:
    train_cmd += ["--amp"]
else:
    train_cmd += ["--no-amp"]

if AUTO_RESUME and (RUN_DIR / "last.pt").exists():
    train_cmd += ["--resume", "auto"]
    print(f"Found existing checkpoint {RUN_DIR / 'last.pt'}. Resuming training...")

print("Executing command:")
print(" ".join(train_cmd))

# Execute training
!{" ".join(train_cmd)}
"""
)

# --- CELL 9: Plot Training Curves ---
add_md(
    """## 8. Training & Validation Analytics
Plot loss convergence and validation PSNR progression across epochs:
"""
)

add_code(
    """import pandas as pd

metrics_file = RUN_DIR / "metrics.csv"
assert metrics_file.exists(), f"Metrics file not found: {metrics_file}"

df_metrics = pd.read_csv(metrics_file)
print(f"Loaded training history: {len(df_metrics)} epochs completed.")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axs[0].plot(df_metrics["epoch"], df_metrics["train_loss"], label="Train Loss", color="tab:blue", lw=2)
axs[0].plot(df_metrics["epoch"], df_metrics["val_loss"], label="Val Loss", color="tab:orange", lw=2, linestyle="--")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_title(f"{MODEL_NAME.upper()} - Training & Validation Loss")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# PSNR plot
axs[1].plot(df_metrics["epoch"], df_metrics["val_psnr"], label="Restored Val PSNR", color="tab:green", lw=2)
axs[1].plot(df_metrics["epoch"], df_metrics["input_psnr"], label="Polluted Input PSNR", color="tab:red", lw=1.5, linestyle=":")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("PSNR (dB)")
axs[1].set_title(f"{MODEL_NAME.upper()} - Validation PSNR Progression")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / "training_curves.png", dpi=150)
plt.show()
"""
)

# --- CELL 10: Held-Out Test Evaluation ---
add_md(
    """## 9. Independent Test-Set Evaluation (Experiment 5B)

> **Evaluation Isolation:**  
> This step loads **`best.pt`** and evaluates strictly on the held-out test scenes across all four interference levels:
> **0 dB, -10 dB, -20 dB, -30 dB**.  
> The test split was never seen during training or validation model selection.
"""
)

add_code(
    """eval_cmd = [
    "python", "scripts/evaluate_model.py",
    "--checkpoint", str(RUN_DIR / "best.pt"),
    "--source", DATA_SOURCE,
    "--root", str(DATA_ROOT),
    "--split-file", str(RUN_DIR / "splits.json"),
    "--device", "cuda",
    "--row-batch", "128",
    "--out-dir", str(RUN_DIR / "test_evaluation")
]

print("Executing test evaluation:")
print(" ".join(eval_cmd))

!{" ".join(eval_cmd)}
"""
)

# --- CELL 11: Display Test Results & Visuals ---
add_md(
    """## 10. Test Metrics Summary & Visual Inspection
Review the final quantitative table and visual outputs generated on the held-out test set:
"""
)

add_code(
    """summary_csv = RUN_DIR / "test_evaluation" / "test_metrics_summary.csv"
if summary_csv.exists():
    df_summary = pd.read_csv(summary_csv)
    print("=" * 80)
    print(f"=== FINAL HELD-OUT TEST EVALUATION: {MODEL_NAME.upper()} ===")
    print("=" * 80)
    display(df_summary)

# Display sample visual output
fig_dir = RUN_DIR / "test_evaluation" / "figures"
sample_figs = list(fig_dir.glob("*_sar.png"))
if sample_figs:
    print(f"Displaying sample restoration visual: {sample_figs[0].name}")
    from PIL import Image
    display(Image.open(sample_figs[0]))
"""
)

# --- CELL 12: Archive & Download Results ---
add_md(
    """## 11. Archive & Download Results
Compress all checkpoints, metric logs, evaluation tables, and visual outputs into a zip file for download to your local machine:
"""
)

add_code(
    """import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_name = f"{MODEL_NAME}_{DATA_SOURCE}_results_{timestamp}"
zip_path = PROJECT_ROOT / "outputs" / zip_name

print(f"Archiving results from {RUN_DIR}...")
shutil.make_archive(str(zip_path), "zip", str(RUN_DIR))
print(f"Archive ready at: {zip_path}.zip")

# In Colab, you can trigger download directly:
try:
    from google.colab import files
    files.download(f"{zip_path}.zip")
    print("Download initiated.")
except Exception as e:
    print("Note: To download, locate the file in the Colab file browser and right-click -> Download.")
"""
)

nb_data = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_json_str = json.dumps(nb_data, indent=2)

path1 = Path("sar_rfi_baseline/notebooks/SAR_RFI_Colab_Training.ipynb")
path2 = Path("SAR_RFI_Colab_Training.ipynb")

path1.parent.mkdir(parents=True, exist_ok=True)
path1.write_text(nb_json_str, encoding="utf-8")
path2.write_text(nb_json_str, encoding="utf-8")

print("Notebook successfully written to:")
print("  -", path1.resolve())
print("  -", path2.resolve())
