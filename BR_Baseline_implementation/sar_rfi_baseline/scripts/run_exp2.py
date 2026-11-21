"""Experiment 2 (Section 5.2): same distribution. Train at -20 dB, test at 0/-10/-20/-30 dB.

Reports PSNR and SSIM (Eqs. 23-24) on the test-split scenes of the
constructed data for the RFI-polluted input and for every model, and saves
one line plot per metric (Figure 9).

    python scripts/run_exp2.py --lab-root data/lab_mat
    python scripts/run_exp2.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.experiments import (
    MODEL_STYLE, SIR_LEVELS, add_common_args, evaluate_labeled, load_models, mat_cases, plot_metric_vs_sir,
    resolve_device, results_prefix, runs_dir_for, smoke_cleans, summarize, synthetic_cases, test_scene_ids,
    train_models, write_csv,
)


def print_table(summary) -> None:
    sirs = sorted({r["sir_db"] for r in summary}, reverse=True)
    print("\n" + "method".ljust(14) + "".join(f"{s:>16g} dB" for s in sirs))
    for method in MODEL_STYLE:
        cells = {r["sir_db"]: r for r in summary if r["method"] == method}
        if cells:
            print(MODEL_STYLE[method]["label"].ljust(14) + "".join(
                f"{cells[s]['psnr_mean']:9.2f} / {cells[s]['ssim_mean']:.3f}" if s in cells else " " * 19 for s in sirs))
    print("(PSNR dB / SSIM)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lab-root", type=Path, help="Constructed (laboratory) .mat slices")
    p.add_argument("--split-file", type=Path, help="Scene split (default: made by train.py)")
    p.add_argument("--train-sir", type=float, default=-20.0)
    p.add_argument("--test-sir", type=float, nargs="+", default=list(SIR_LEVELS))
    add_common_args(p)
    args = p.parse_args(argv)

    runs = runs_dir_for(args, "exp2")
    prefix = results_prefix(args, "exp2")
    if args.smoke:
        train_argv = ["--smoke", "--sir", str(args.train_sir)]
    else:
        if args.lab_root is None:
            p.error("--lab-root is required (or use --smoke)")
        train_argv = ["--source", "mat", "--root", str(args.lab_root), "--sir", str(args.train_sir)]
        if args.split_file:
            train_argv += ["--split-file", str(args.split_file)]

    checkpoints = train_models(args.models, runs, train_argv, args)
    device = resolve_device(args.device)
    models = load_models(checkpoints, device)
    scenes = test_scene_ids(runs / args.models[0])
    cases = (synthetic_cases(smoke_cleans(), scenes, "default", args.test_sir) if args.smoke
             else mat_cases(args.lab_root, scenes, args.test_sir))
    print(f"\nscoring {len(scenes)} test scenes at SIRs {args.test_sir}", flush=True)
    rows = evaluate_labeled(cases, models, args.row_batch, device, extra={"train_sir_db": args.train_sir})

    write_csv(args.results_dir / f"{prefix}_per_slice.csv", rows)
    summary = summarize(rows, ["sir_db", "method"], ["psnr", "ssim"])
    write_csv(args.results_dir / f"{prefix}_summary.csv", summary)
    for metric in ("psnr", "ssim"):
        plot_metric_vs_sir(summary, metric, args.results_dir / f"{prefix}_{metric}.png",
                           f"Experiment 2: same distribution (trained at {args.train_sir:g} dB)")
    print_table(summary)


if __name__ == "__main__":
    main()
