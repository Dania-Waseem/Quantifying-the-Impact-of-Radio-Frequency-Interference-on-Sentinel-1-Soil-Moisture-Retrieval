"""Evaluation script for UNet trained on RFI-Patch proxy benchmark.

Evaluates on held-out test scenes (s3, third) at all four documented SNR levels
(0 dB, -10 dB, -20 dB, -30 dB).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data.dataset import RFI_PATCH_SNR_DB, load_rfi_patch
from eval.metrics import psnr, ssim
from eval.pipeline import load_network, suppress_rfi
from eval.experiments import most_interfered_row, magnitude_db


def main():
    ckpt_path = ROOT / "outputs" / "rfi_patch_proxy_unet" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    model, meta = load_network(ckpt_path, device=device)
    model.eval()

    test_scenes = ["s3", "third"]
    snr_levels = [(1, 0.0), (2, -10.0), (3, -20.0), (4, -30.0)]
    out_dir = ROOT / "outputs" / "rfi_patch_proxy_unet" / "eval_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"=== Evaluating UNet Checkpoint on Held-Out Test Scenes: {test_scenes} ===")

    for sc in test_scenes:
        mat_path = ROOT.parent / "Data" / "RFI-Patch" / "RFI-1" / f"{sc}.mat"
        loaded = load_rfi_patch(mat_path)
        stack = loaded["ss"]  # shape (5, 512, 1500)
        clean = stack[0]

        for idx, snr_val in snr_levels:
            polluted = stack[idx]
            t0 = time.time()
            restored = suppress_rfi(polluted, model, row_batch=128, device=device).numpy()
            el_time = time.time() - t0

            # 8-bit quantized metrics (matching author's official evaluation script)
            psnr_pol_u8 = psnr(clean, polluted, quantize_8bit=True)
            ssim_pol_u8 = ssim(clean, polluted, quantize_8bit=True)
            psnr_res_u8 = psnr(clean, restored, quantize_8bit=True)
            ssim_res_u8 = ssim(clean, restored, quantize_8bit=True)

            # Continuous float64 metrics
            psnr_pol_f64 = psnr(clean, polluted, quantize_8bit=False)
            ssim_pol_f64 = ssim(clean, polluted, quantize_8bit=False)
            psnr_res_f64 = psnr(clean, restored, quantize_8bit=False)
            ssim_res_f64 = ssim(clean, restored, quantize_8bit=False)

            delta_psnr = psnr_res_u8 - psnr_pol_u8
            delta_ssim = ssim_res_u8 - ssim_pol_u8

            row = {
                "scene": sc,
                "snr_db": snr_val,
                "pol_psnr_u8": psnr_pol_u8,
                "res_psnr_u8": psnr_res_u8,
                "delta_psnr_u8": delta_psnr,
                "pol_ssim_u8": ssim_pol_u8,
                "res_ssim_u8": ssim_res_u8,
                "delta_ssim_u8": delta_ssim,
                "pol_psnr_f64": psnr_pol_f64,
                "res_psnr_f64": psnr_res_f64,
                "pol_ssim_f64": ssim_pol_f64,
                "res_ssim_f64": ssim_res_f64,
                "seconds": el_time,
            }
            results.append(row)
            print(
                f"Scene: {sc:5s} | SNR: {snr_val:5.1f} dB | Input PSNR: {psnr_pol_u8:6.2f} dB -> "
                f"Restored: {psnr_res_u8:6.2f} dB ({delta_psnr:+6.2f} dB) | "
                f"SSIM: {ssim_pol_u8:.4f} -> {ssim_res_u8:.4f} ({delta_ssim:+.4f}) [{el_time:.1f}s]"
            )

            # Visualizations for representative slices
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            p_clean = np.sqrt(np.abs(clean))
            p_pol = np.sqrt(np.abs(polluted))
            p_res = np.sqrt(np.abs(restored))

            axs[0].imshow(p_pol, cmap="gray", aspect="auto")
            axs[0].set_title(f"Polluted ({snr_val:g} dB)\nPSNR: {psnr_pol_u8:.2f} dB | SSIM: {ssim_pol_u8:.4f}")
            axs[1].imshow(p_res, cmap="gray", aspect="auto")
            axs[1].set_title(
                f"UNet Restored\nPSNR: {psnr_res_u8:.2f} dB ({delta_psnr:+.2f} dB) | SSIM: {ssim_res_u8:.4f}"
            )
            axs[2].imshow(p_clean, cmap="gray", aspect="auto")
            axs[2].set_title("Clean Reference Label")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(out_dir / f"{sc}_snr{int(snr_val)}dB_sar.png", dpi=150)
            plt.close(fig)

            # Time-Frequency plot of the most interfered range line
            row_idx = most_interfered_row(polluted)
            from utils.tf import stft_row
            win_len = clean.shape[1] // 8
            clean_t = torch.as_tensor(clean[row_idx])
            pol_t = torch.as_tensor(polluted[row_idx])
            res_t = torch.as_tensor(restored[row_idx])
            tf_clean = np.abs(stft_row(clean_t, win_len).numpy())
            tf_pol = np.abs(stft_row(pol_t, win_len).numpy())
            tf_res = np.abs(stft_row(res_t, win_len).numpy())

            fig_tf, axs_tf = plt.subplots(1, 3, figsize=(15, 4))
            axs_tf[0].imshow(tf_pol, cmap="inferno", aspect="auto", origin="lower")
            axs_tf[0].set_title(f"Polluted STFT (Row {row_idx})")
            axs_tf[1].imshow(tf_res, cmap="inferno", aspect="auto", origin="lower")
            axs_tf[1].set_title("Restored STFT")
            axs_tf[2].imshow(tf_clean, cmap="inferno", aspect="auto", origin="lower")
            axs_tf[2].set_title("Clean STFT")
            for ax in axs_tf:
                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Frequency Bins")
            plt.tight_layout()
            fig_tf.savefig(out_dir / f"{sc}_snr{int(snr_val)}dB_tf.png", dpi=150)
            plt.close(fig_tf)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "eval_results.csv", index=False)

    print("\n" + "=" * 80)
    print("=== SUMMARY METRICS TABLE ACROSS INTERFERENCE LEVELS (RFI-PATCH PROXY BENCHMARK) ===")
    print("=" * 80)
    summary = (
        df.groupby("snr_db")
        .agg(
            {
                "pol_psnr_u8": "mean",
                "res_psnr_u8": "mean",
                "delta_psnr_u8": "mean",
                "pol_ssim_u8": "mean",
                "res_ssim_u8": "mean",
                "delta_ssim_u8": "mean",
                "seconds": "mean",
            }
        )
        .reset_index()
        .sort_values("snr_db", ascending=False)
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
