"""Test-only evaluation script for UNet RFI-Patch SNR -20 dB training run.

Loads the best checkpoint from outputs/runs/unet_rfipatch_snr-20/best.pt,
identifies saved configuration and splits, loads the held-out test split (s3, third),
runs inference via Algorithm 1 (suppress_rfi), computes PSNR and SSIM for:
  - Noisy input vs clean reference
  - UNet output vs clean reference
Calculates improvement / degradation, and saves visual comparisons with consistent scales.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data.dataset import RFI_PATCH_SNR_DB, load_rfi_patch, load_split_file
from eval.metrics import psnr, ssim
from eval.pipeline import load_network, suppress_rfi
from eval.experiments import most_interfered_row
from utils.tf import stft_row


def main():
    workspace_root = REPO_ROOT.parent
    run_dir = workspace_root / "outputs" / "runs" / "unet_rfipatch_snr-20"
    if not run_dir.exists():
        run_dir = REPO_ROOT / "outputs" / "runs" / "unet_rfipatch_snr-20"
    if not run_dir.exists():
        run_dir = REPO_ROOT / "outputs" / "rfi_patch_proxy_unet"

    print(f"=== Inspecting Run Directory: {run_dir} ===")
    
    ckpt_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    splits_path = run_dir / "splits.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Loaded config: model={config.get('model')}, source={config.get('source')}, "
          f"snr_db={config.get('snr_db')}, epochs={config.get('epochs')}")

    # 2. Load splits
    split_info = load_split_file(splits_path)
    test_scenes = sorted([s for s, sp in split_info.items() if sp == "test"])
    train_scenes = sorted([s for s, sp in split_info.items() if sp == "train"])
    val_scenes = sorted([s for s, sp in split_info.items() if sp == "val"])
    print(f"Scene splits: Train={len(train_scenes)} scenes, Val={len(val_scenes)} scenes, Test={len(test_scenes)} scenes")
    print(f"Held-out test scenes (no data leakage): {test_scenes}")
    assert len(test_scenes) == 2, f"Expected exactly 2 test scenes, got {len(test_scenes)}"
    assert not (set(test_scenes) & set(train_scenes)), "Leakage detected between test and train!"
    assert not (set(test_scenes) & set(val_scenes)), "Leakage detected between test and val!"

    # 3. Load checkpoint
    device = torch.device("cpu")
    print(f"Loading best checkpoint: {ckpt_path} on {device}...")
    model, model_name = load_network(ckpt_path, device=device)
    model.eval()

    ckpt_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_epoch = ckpt_raw.get("epoch")
    best_val_psnr = ckpt_raw.get("best_val_psnr")
    print(f"Model loaded: {model_name} (Best Epoch: {best_epoch}, Best Val PSNR: {best_val_psnr:.4f} dB)")

    # 4. Prepare output directories
    out_dir = run_dir / "test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Dataset directory
    data_root = workspace_root / config.get("root", "Data/RFI-Patch/RFI-1")
    if not data_root.exists():
        data_root = REPO_ROOT / config.get("root", "Data/RFI-Patch/RFI-1")

    # Evaluate across target SNR -20 dB (primary test) and all levels
    levels = [0.0, -10.0, -20.0, -30.0]
    records = []

    for sc in test_scenes:
        mat_path = data_root / f"{sc}.mat"
        if not mat_path.exists():
            matches = list(data_root.rglob(f"{sc}.mat"))
            if not matches:
                raise FileNotFoundError(f"Could not locate {sc}.mat in {data_root}")
            mat_path = matches[0]

        loaded = load_rfi_patch(mat_path)
        stack = loaded["ss"]  # shape (5, 512, 1500), complex64
        clean = stack[0]

        for lvl in levels:
            # Match SNR level index
            idx_matches = [i for i, snr in RFI_PATCH_SNR_DB.items() if np.isclose(snr, lvl, atol=1e-3)]
            if not idx_matches:
                continue
            idx = idx_matches[0]
            polluted = stack[idx]

            print(f"Evaluating scene '{sc}' at SNR {lvl:g} dB...")
            t0 = time.time()
            restored = suppress_rfi(polluted, model, row_batch=128, device=device).cpu().numpy()
            elapsed = time.time() - t0

            # Quantized 8-bit metrics (official evaluation benchmark standard)
            psnr_in_u8 = psnr(clean, polluted, quantize_8bit=True)
            ssim_in_u8 = ssim(clean, polluted, quantize_8bit=True)
            psnr_out_u8 = psnr(clean, restored, quantize_8bit=True)
            ssim_out_u8 = ssim(clean, restored, quantize_8bit=True)

            delta_psnr_u8 = psnr_out_u8 - psnr_in_u8
            delta_ssim_u8 = ssim_out_u8 - ssim_in_u8

            # Continuous float64 metrics
            psnr_in_f64 = psnr(clean, polluted, quantize_8bit=False)
            ssim_in_f64 = ssim(clean, polluted, quantize_8bit=False)
            psnr_out_f64 = psnr(clean, restored, quantize_8bit=False)
            ssim_out_f64 = ssim(clean, restored, quantize_8bit=False)

            delta_psnr_f64 = psnr_out_f64 - psnr_in_f64
            delta_ssim_f64 = ssim_out_f64 - ssim_in_f64

            rec = {
                "scene": sc,
                "snr_db": lvl,
                "input_psnr_u8": psnr_in_u8,
                "output_psnr_u8": psnr_out_u8,
                "delta_psnr_u8": delta_psnr_u8,
                "input_ssim_u8": ssim_in_u8,
                "output_ssim_u8": ssim_out_u8,
                "delta_ssim_u8": delta_ssim_u8,
                "input_psnr_f64": psnr_in_f64,
                "output_psnr_f64": psnr_out_f64,
                "delta_psnr_f64": delta_psnr_f64,
                "input_ssim_f64": ssim_in_f64,
                "output_ssim_f64": ssim_out_f64,
                "delta_ssim_f64": delta_ssim_f64,
                "inference_time_s": elapsed,
            }
            records.append(rec)

            print(f"  [SNR {lvl:5.1f} dB] Input PSNR: {psnr_in_u8:6.2f} dB -> Output PSNR: {psnr_out_u8:6.2f} dB "
                  f"(Δ: {delta_psnr_u8:+6.2f} dB) | Input SSIM: {ssim_in_u8:.4f} -> Output SSIM: {ssim_out_u8:.4f} "
                  f"(Δ: {delta_ssim_u8:+.4f}) [{elapsed:.1f}s]")

            # Generate visual comparison figures with CONSISTENT visualization scales
            # Power law / sqrt of magnitude as per standard SAR display
            p_clean = np.sqrt(np.abs(clean))
            p_noisy = np.sqrt(np.abs(polluted))
            p_out = np.sqrt(np.abs(restored))

            # Unified visualization bounds across all three identical patches
            v_min = 0.0
            v_max = float(max(np.percentile(p_clean, 99.8), np.percentile(p_noisy, 99.8), np.percentile(p_out, 99.8)))

            # 1. Full scene comparison (Noisy Input | UNet Output | Clean Reference)
            fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
            im0 = axs[0].imshow(p_noisy, cmap="gray", aspect="auto", vmin=v_min, vmax=v_max)
            axs[0].set_title(f"Noisy Input (SNR: {lvl:g} dB)\nPSNR: {psnr_in_u8:.2f} dB | SSIM: {ssim_in_u8:.4f}", fontsize=11)
            axs[0].set_xlabel("Range Samples")
            axs[0].set_ylabel("Azimuth Rows")

            im1 = axs[1].imshow(p_out, cmap="gray", aspect="auto", vmin=v_min, vmax=v_max)
            axs[1].set_title(f"UNet Output\nPSNR: {psnr_out_u8:.2f} dB ({delta_psnr_u8:+.2f} dB) | SSIM: {ssim_out_u8:.4f} ({delta_ssim_u8:+.4f})", fontsize=11)
            axs[1].set_xlabel("Range Samples")
            axs[1].set_ylabel("Azimuth Rows")

            im2 = axs[2].imshow(p_clean, cmap="gray", aspect="auto", vmin=v_min, vmax=v_max)
            axs[2].set_title(f"Clean Reference\nGround Truth Target", fontsize=11)
            axs[2].set_xlabel("Range Samples")
            axs[2].set_ylabel("Azimuth Rows")

            cbar = fig.colorbar(im2, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
            cbar.set_label("Amplitude Scale (Shared)", fontsize=10)

            fig_full_name = f"{sc}_snr{int(lvl)}dB_full_comparison.png"
            fig.savefig(fig_dir / fig_full_name, dpi=180)
            plt.close(fig)

            # 2. Zoomed patch comparison (256x256 crop around center/features)
            # Center coordinates
            r_start, r_end = 128, 384
            c_start, c_end = 600, 856
            crop_noisy = p_noisy[r_start:r_end, c_start:c_end]
            crop_out = p_out[r_start:r_end, c_start:c_end]
            crop_clean = p_clean[r_start:r_end, c_start:c_end]

            crop_max = float(max(np.percentile(crop_clean, 99.8), np.percentile(crop_noisy, 99.8), np.percentile(crop_out, 99.8)))

            fig_zoom, axs_z = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            axs_z[0].imshow(crop_noisy, cmap="gray", vmin=v_min, vmax=crop_max)
            axs_z[0].set_title(f"Noisy Input Patch [Crop 256x256]\nSNR: {lvl:g} dB", fontsize=11)
            axs_z[0].axis("off")

            axs_z[1].imshow(crop_out, cmap="gray", vmin=v_min, vmax=crop_max)
            axs_z[1].set_title(f"UNet Output Patch\nΔPSNR: {delta_psnr_u8:+.2f} dB | ΔSSIM: {delta_ssim_u8:+.4f}", fontsize=11)
            axs_z[1].axis("off")

            im_z = axs_z[2].imshow(crop_clean, cmap="gray", vmin=v_min, vmax=crop_max)
            axs_z[2].set_title(f"Clean Reference Patch\nGround Truth", fontsize=11)
            axs_z[2].axis("off")

            cbar_z = fig_zoom.colorbar(im_z, ax=axs_z, orientation="vertical", fraction=0.02, pad=0.02)
            cbar_z.set_label("Amplitude Scale (Shared Crop)", fontsize=10)

            fig_zoom_name = f"{sc}_snr{int(lvl)}dB_patch_zoom.png"
            fig_zoom.savefig(fig_dir / fig_zoom_name, dpi=180)
            plt.close(fig_zoom)

            # 3. Time-Frequency spectrogram of most interfered row
            row_idx = most_interfered_row(polluted)
            win_len = clean.shape[1] // 8
            tf_clean = np.abs(stft_row(torch.as_tensor(clean[row_idx]), win_len).numpy())
            tf_noisy = np.abs(stft_row(torch.as_tensor(polluted[row_idx]), win_len).numpy())
            tf_out = np.abs(stft_row(torch.as_tensor(restored[row_idx]), win_len).numpy())

            tf_vmax = float(max(np.percentile(tf_clean, 99.8), np.percentile(tf_noisy, 99.8), np.percentile(tf_out, 99.8)))

            fig_tf, axs_tf = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
            axs_tf[0].imshow(tf_noisy, cmap="inferno", aspect="auto", origin="lower", vmin=0, vmax=tf_vmax)
            axs_tf[0].set_title(f"Noisy STFT (Azimuth Row {row_idx})\nSNR: {lvl:g} dB", fontsize=11)
            axs_tf[0].set_xlabel("Time Bins")
            axs_tf[0].set_ylabel("Frequency Bins")

            axs_tf[1].imshow(tf_out, cmap="inferno", aspect="auto", origin="lower", vmin=0, vmax=tf_vmax)
            axs_tf[1].set_title(f"UNet Filtered STFT\nSuppressed Spectrogram", fontsize=11)
            axs_tf[1].set_xlabel("Time Bins")
            axs_tf[1].set_ylabel("Frequency Bins")

            im_tf = axs_tf[2].imshow(tf_clean, cmap="inferno", aspect="auto", origin="lower", vmin=0, vmax=tf_vmax)
            axs_tf[2].set_title(f"Clean STFT\nGround Truth Spectrogram", fontsize=11)
            axs_tf[2].set_xlabel("Time Bins")
            axs_tf[2].set_ylabel("Frequency Bins")

            cbar_tf = fig_tf.colorbar(im_tf, ax=axs_tf, orientation="vertical", fraction=0.02, pad=0.02)
            cbar_tf.set_label("STFT Magnitude (Shared)", fontsize=10)

            fig_tf_name = f"{sc}_snr{int(lvl)}dB_spectrogram.png"
            fig_tf.savefig(fig_dir / fig_tf_name, dpi=180)
            plt.close(fig_tf)

    # Convert to DataFrame
    df_all = pd.DataFrame(records)
    df_all.to_csv(out_dir / "test_metrics_all_levels.csv", index=False)

    # Filter target SNR -20 dB (primary evaluation matching unet_rfipatch_snr-20 run)
    df_snr20 = df_all[df_all["snr_db"] == -20.0].copy()
    df_snr20.to_csv(out_dir / "test_metrics_snr-20.csv", index=False)

    # Summary table across levels
    summary = (
        df_all.groupby("snr_db")
        .agg(
            {
                "input_psnr_u8": "mean",
                "output_psnr_u8": "mean",
                "delta_psnr_u8": "mean",
                "input_ssim_u8": "mean",
                "output_ssim_u8": "mean",
                "delta_ssim_u8": "mean",
                "input_psnr_f64": "mean",
                "output_psnr_f64": "mean",
                "delta_psnr_f64": "mean",
                "input_ssim_f64": "mean",
                "output_ssim_f64": "mean",
                "delta_ssim_f64": "mean",
                "inference_time_s": "mean",
            }
        )
        .reset_index()
        .sort_values("snr_db", ascending=False)
    )
    summary.to_csv(out_dir / "test_metrics_summary.csv", index=False)

    # Save JSON summary
    summary_dict = {
        "checkpoint": str(ckpt_path),
        "best_epoch": int(best_epoch),
        "best_val_psnr": float(best_val_psnr),
        "test_scenes": test_scenes,
        "test_sample_count_target_snr": len(df_snr20),
        "test_sample_count_all_levels": len(df_all),
        "snr_-20dB_results": df_snr20.to_dict(orient="records"),
        "average_metrics_snr_-20dB": {
            "input_psnr_u8": float(df_snr20["input_psnr_u8"].mean()),
            "output_psnr_u8": float(df_snr20["output_psnr_u8"].mean()),
            "delta_psnr_u8": float(df_snr20["delta_psnr_u8"].mean()),
            "input_ssim_u8": float(df_snr20["input_ssim_u8"].mean()),
            "output_ssim_u8": float(df_snr20["output_ssim_u8"].mean()),
            "delta_ssim_u8": float(df_snr20["delta_ssim_u8"].mean()),
            "input_psnr_f64": float(df_snr20["input_psnr_f64"].mean()),
            "output_psnr_f64": float(df_snr20["output_psnr_f64"].mean()),
            "delta_psnr_f64": float(df_snr20["delta_psnr_f64"].mean()),
            "input_ssim_f64": float(df_snr20["input_ssim_f64"].mean()),
            "output_ssim_f64": float(df_snr20["output_ssim_f64"].mean()),
            "delta_ssim_f64": float(df_snr20["delta_ssim_f64"].mean()),
        },
    }

    with open(out_dir / "test_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("=== TEST EVALUATION SUMMARY: TARGET RUN (SNR -20 dB) ===")
    print("=" * 80)
    for r in summary_dict["snr_-20dB_results"]:
        print(f"Scene: {r['scene']:6s} | Input PSNR: {r['input_psnr_u8']:5.2f} dB -> Output PSNR: {r['output_psnr_u8']:5.2f} dB ({r['delta_psnr_u8']:+5.2f} dB) | "
              f"Input SSIM: {r['input_ssim_u8']:.4f} -> Output SSIM: {r['output_ssim_u8']:.4f} ({r['delta_ssim_u8']:+.4f})")
    
    avg20 = summary_dict["average_metrics_snr_-20dB"]
    print("-" * 80)
    print(f"AVERAGE (SNR -20 dB) | Input PSNR: {avg20['input_psnr_u8']:5.2f} dB -> Output PSNR: {avg20['output_psnr_u8']:5.2f} dB ({avg20['delta_psnr_u8']:+5.2f} dB) | "
          f"Input SSIM: {avg20['input_ssim_u8']:.4f} -> Output SSIM: {avg20['output_ssim_u8']:.4f} ({avg20['delta_ssim_u8']:+.4f})")
    print("=" * 80)
    print(f"Evaluation artifacts saved successfully inside: {out_dir}")


if __name__ == "__main__":
    main()
