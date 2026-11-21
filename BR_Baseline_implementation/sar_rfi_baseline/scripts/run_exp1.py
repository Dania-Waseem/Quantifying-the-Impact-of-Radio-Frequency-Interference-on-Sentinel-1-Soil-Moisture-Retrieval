"""Experiment 1 (Section 5.1): train on constructed data, test on real Sentinel-1 slices.

The 30 Sentinel-1 slices have no clean label, so only mean-entropy (Eq. 25)
is reported, for the polluted input and for every model (Table 3). Before and
after SAR images and time-frequency plots of the most interfered range line
are saved for every slice.

Training uses every constructed (laboratory) slice at all four SIRs by
default, since the paper trains on "the data constructed by our laboratory";
pass ``--train-sir -20`` to train at -20 dB only.

    python scripts/run_exp1.py --lab-root data/lab_mat --s1-root data/sentinel1
    python scripts/run_exp1.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from data.dataset import synthesize_pair
from eval.experiments import (
    MODEL_STYLE, POLLUTED, SIR_LEVELS, _timed_restore, add_common_args, load_models, mean_entropy_row,
    most_interfered_row, resolve_device, results_prefix, runs_dir_for, save_image_panel, save_tf_panel,
    smoke_cleans, summarize, test_scene_ids, train_models, unlabeled_cases, write_csv,
)


def smoke_unlabeled(run_dir: Path):
    """Stand-in for Sentinel-1: test-split smoke scenes polluted at -20 and -30 dB, labels dropped."""
    test = set(test_scene_ids(run_dir))
    for item in smoke_cleans():
        if item["scene_id"] not in test:
            continue
        for k, sir in enumerate((-20.0, -30.0)):
            yield {"case": f"{item['scene_id']}_{sir:g}dB", "scene_type": item["scene_type"],
                   "polluted": synthesize_pair(item["image"], sir, "chirp_wideband", 900 + k)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lab-root", type=Path, help="Constructed (laboratory) .mat slices for training")
    p.add_argument("--split-file", type=Path, help="Scene split for training (default: made by train.py)")
    p.add_argument("--s1-root", type=Path, help="The 30 open-source Sentinel-1 .mat slices")
    p.add_argument("--train-sir", type=float, nargs="+", default=list(SIR_LEVELS))
    p.add_argument("--max-figures", type=int, default=None, help="Save figures for at most this many slices")
    add_common_args(p)
    args = p.parse_args(argv)

    runs = runs_dir_for(args, "exp1")
    prefix = results_prefix(args, "exp1")
    if args.smoke:
        train_argv = ["--smoke", "--sir", *map(str, args.train_sir)]
    else:
        if args.lab_root is None or args.s1_root is None:
            p.error("--lab-root and --s1-root are required (or use --smoke)")
        train_argv = ["--source", "mat", "--root", str(args.lab_root), "--sir", *map(str, args.train_sir)]
        if args.split_file:
            train_argv += ["--split-file", str(args.split_file)]

    checkpoints = train_models(args.models, runs, train_argv, args)
    device = resolve_device(args.device)
    models = load_models(checkpoints, device)
    cases = smoke_unlabeled(runs / args.models[0]) if args.smoke else unlabeled_cases(args.s1_root)

    fig_dir = args.results_dir / f"{prefix}_figures"
    rows = []
    for n, case in enumerate(cases, start=1):
        S = np.asarray(case["polluted"])
        outputs = {POLLUTED: (S, 0.0)}
        for name, model in models.items():
            S_star, seconds = _timed_restore(S, model, args.row_batch, device)
            outputs[name] = (S_star.numpy(), seconds)
        rows += [mean_entropy_row(case, method, image, seconds) for method, (image, seconds) in outputs.items()]
        if args.max_figures is None or n <= args.max_figures:
            labels = [(MODEL_STYLE[m]["label"], image) for m, (image, _) in outputs.items()]
            save_image_panel(labels, fig_dir / f"{case['case']}_sar.png", f"{case['case']}: SAR magnitude")
            row = most_interfered_row(S)
            save_tf_panel([(label, image[row]) for label, image in labels], fig_dir / f"{case['case']}_tf.png",
                          f"{case['case']}: |STFT| of azimuth row {row}")
        print(f"  {case['case']}: " + ", ".join(f"{r['method']} ME {r['me']:.4f}" for r in rows[-len(outputs):]), flush=True)

    if not rows:
        raise SystemExit("no Sentinel-1 slices were found")
    write_csv(args.results_dir / f"{prefix}_per_slice.csv", rows)
    summary = summarize(rows, ["method"], ["me", "seconds"])
    write_csv(args.results_dir / f"{prefix}_me.csv", summary)
    print("\nMean-entropy (lower is better; paper Table 3 lists polluted 0.314):")
    for entry in summary:
        print(f"  {MODEL_STYLE[entry['method']]['label']:13s} {entry['me_mean']:.4f}  (n={entry['n']})")
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
