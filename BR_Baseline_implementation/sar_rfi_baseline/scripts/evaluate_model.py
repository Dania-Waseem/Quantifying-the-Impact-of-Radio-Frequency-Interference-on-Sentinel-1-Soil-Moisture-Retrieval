"""General Evaluation Script for Trained SAR RFI Baselines.

Evaluates any trained model checkpoint (UNet, DNCNN, PISNet, FDNet) on the
strictly held-out test split across all 4 interference levels (0, -10, -20, -30 dB).
Supports both RFI-Patch stacks and laboratory .mat slices.
"""

from __future__ import annotations

import argparse
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

from data.dataset import RFI_PATCH_SNR_DB, load_rfi_patch, load_mat_slice, load_split_file
from eval.metrics import psnr, ssim
from eval.pipeline import load_network, suppress_rfi
from eval.experiments import most_interfered_row
from utils.tf import stft_row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or checkpoint")
    p.add_argument("--source", choices=("rfi_patch", "mat"), default="rfi_patch")
    p.add_argument("--root", type=Path, required=True, help="Dataset directory")
    p.add_argument("--split-file", type=Path, required=True, help="Path to splits.json with 'splits' mapping")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p.add_argument("--row-batch", type=int, default=128, help="Range lines per batch during inference")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for results and figures")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0, -10.0, -20.0, -30.0])
    p.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def evaluate(args):
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    )
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    model, meta = load_network(args.checkpoint, device=device)
    model.eval()

    split_map = load_split_file(args.split_file)
    test_scenes = sorted([s for s, sp in split_map.items() if sp == "test"])
    if not test_scenes:
        raise ValueError(f"No scenes assigned to 'test' in {args.split_file}")
    print(f"Held-out test scenes ({len(test_scenes)}): {test_scenes}")

    out_dir = args.out_dir or args.checkpoint.parent / "test_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    if args.save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    results = []
    levels = sorted([float(v) for v in args.levels], reverse=True)

    for sc in test_scenes:
        if args.source == "rfi_patch":
            # Search for the .mat file matching scene_id
            matches = list(args.root.rglob(f"{sc}.mat"))
            if not matches:
                print(f"Warning: file {sc}.mat not found under {args.root}, skipping.")
                continue
            loaded = load_rfi_patch(matches[0])
            stack = loaded["ss"]  # (5, 512, 1500)
            clean = stack[0]

            for level in levels:
                # Find matching index in RFI_PATCH_SNR_DB
                idx_matches = [i for i, snr in RFI_PATCH_SNR_DB.items() if np.isclose(snr, level, atol=1e-3)]
                if not idx_matches:
                    continue
                idx = idx_matches[0]
                polluted = stack[idx]

                t0 = time.time()
                restored = suppress_rfi(polluted, model, row_batch=args.row_batch, device=device).cpu().numpy()
                el_time = time.time() - t0

                # Quantized 8-bit metrics (official evaluation standard)
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
                    "level_db": level,
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
                    f"Scene: {sc:6s} | Level: {level:5.1f} dB | Input: {psnr_pol_u8:5.2f} dB -> "
                    f"Restored: {psnr_res_u8:5.2f} dB ({delta_psnr:+5.2f} dB) | "
                    f"SSIM: {ssim_pol_u8:.4f} -> {ssim_res_u8:.4f} ({delta_ssim:+.4f}) [{el_time:.2f}s]"
                )

                if args.save_figures:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    p_clean = np.sqrt(np.abs(clean))
                    p_pol = np.sqrt(np.abs(polluted))
                    p_res = np.sqrt(np.abs(restored))

                    axs[0].imshow(p_pol, cmap="gray", aspect="auto")
                    axs[0].set_title(f"Polluted ({level:g} dB)\nPSNR: {psnr_pol_u8:.2f} dB | SSIM: {ssim_pol_u8:.4f}")
                    axs[1].imshow(p_res, cmap="gray", aspect="auto")
                    axs[1].set_title(f"Restored\nPSNR: {psnr_res_u8:.2f} dB ({delta_psnr:+.2f} dB) | SSIM: {ssim_res_u8:.4f}")
                    axs[2].imshow(p_clean, cmap="gray", aspect="auto")
                    axs[2].set_title("Clean Reference")
                    for ax in axs:
                        ax.axis("off")
                    plt.tight_layout()
                    fig.savefig(fig_dir / f"{sc}_level{int(level)}dB_sar.png", dpi=150)
                    plt.close(fig)

                    # Range line spectrogram
                    row_idx = most_interfered_row(polluted)
                    win_len = clean.shape[1] // 8
                    tf_clean = np.abs(stft_row(torch.as_tensor(clean[row_idx]), win_len).numpy())
                    tf_pol = np.abs(stft_row(torch.as_tensor(polluted[row_idx]), win_len).numpy())
                    tf_res = np.abs(stft_row(torch.as_tensor(restored[row_idx]), win_len).numpy())

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
                    fig_tf.savefig(fig_dir / f"{sc}_level{int(level)}dB_tf.png", dpi=150)
                    plt.close(fig_tf)

    if not results:
        print("No test slices were evaluated.")
        return

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "test_metrics_per_slice.csv", index=False)

    summary = (
        df.groupby("level_db")
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
        .sort_values("level_db", ascending=False)
    )
    summary.to_csv(out_dir / "test_metrics_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("=== FINAL HELD-OUT TEST EVALUATION SUMMARY ===")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {out_dir}")


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
