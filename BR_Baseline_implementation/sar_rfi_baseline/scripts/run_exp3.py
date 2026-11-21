"""Experiment 3 (Section 5.3): domain shift. Train on one interference set, test on a disjoint one.

Training pollutes the clean training scenes with ``exp3_train`` (chirp
wideband, low chirp rates) at -20 dB. Testing pollutes the clean test scenes
with ``exp3_test`` (higher, non-overlapping chirp rates, plus narrowband and
sinusoidal modulation never seen in training) at all four SIRs. The sets are
defined and checked in ``src/data/param_sets.py``. By default the test scenes
are also scored on fresh ``exp3_train`` draws, the in-distribution reference.

Clean scenes are the RFI-free images of the released slices. With
``--lab-root``, one clean image per scene is written to ``--clean-root`` as
``{scene_id}__{scene_type}.npy`` the first time (about 6 MB per scene).

    python scripts/run_exp3.py --lab-root data/lab_mat --clean-root data/clean_scenes
    python scripts/run_exp3.py --smoke
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from data.dataset import discover_mat_records, load_clean_directory, load_mat_slice
from data.param_sets import check_disjoint
from eval.experiments import (
    MODEL_STYLE, SIR_LEVELS, add_common_args, evaluate_labeled, load_models, plot_metric_vs_sir, resolve_device,
    results_prefix, runs_dir_for, smoke_cleans, summarize, synthetic_cases, test_scene_ids, train_models,
    write_csv,
)

TRAIN_SET, TEST_SET = "exp3_train", "exp3_test"


def extract_clean_scenes(lab_root: Path, clean_root: Path) -> None:
    """One RFI-free image per scene from the released slices."""
    clean_root.mkdir(parents=True, exist_ok=True)
    done = set()
    for record in discover_mat_records(lab_root):
        if record.scene_id in done:
            continue
        loaded = load_mat_slice(record.path)
        safe_id = record.scene_id.replace("__", "_")
        np.save(clean_root / f"{safe_id}__{record.scene_type}.npy", loaded["clean"].astype(np.complex64))
        done.add(record.scene_id)
    print(f"wrote {len(done)} clean scenes to {clean_root}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clean-root", type=Path, default=ROOT / "data" / "clean_scenes",
                   help="Clean {scene_id}__{scene_type}.npy scenes")
    p.add_argument("--lab-root", type=Path, help="Released .mat slices to extract clean scenes from")
    p.add_argument("--train-sir", type=float, default=-20.0)
    p.add_argument("--test-sir", type=float, nargs="+", default=list(SIR_LEVELS))
    p.add_argument("--n-repeats", type=int, default=1, help="Interference draws per scene, preset and SIR")
    p.add_argument("--no-in-distribution", action="store_true", help="Skip the exp3_train reference test")
    add_common_args(p)
    args = p.parse_args(argv)
    check_disjoint(TRAIN_SET, TEST_SET)

    runs = runs_dir_for(args, "exp3")
    prefix = results_prefix(args, "exp3")
    if args.smoke:
        train_argv = ["--smoke", "--rfi-params", TRAIN_SET, "--sir", str(args.train_sir)]
    else:
        if not any(args.clean_root.glob("*.npy")):
            if args.lab_root is None:
                p.error(f"no clean scenes in {args.clean_root}; pass --lab-root to extract them")
            extract_clean_scenes(args.lab_root, args.clean_root)
        train_argv = ["--source", "synthetic", "--root", str(args.clean_root), "--rfi-params", TRAIN_SET,
                      "--sir", str(args.train_sir), "--n-repeats", str(args.n_repeats)]

    checkpoints = train_models(args.models, runs, train_argv, args)
    device = resolve_device(args.device)
    models = load_models(checkpoints, device)
    scenes = test_scene_ids(runs / args.models[0])
    cleans = smoke_cleans() if args.smoke else load_clean_directory(args.clean_root)

    conditions = [("shifted", TEST_SET)] + ([] if args.no_in_distribution else [("in-distribution", TRAIN_SET)])
    rows = []
    for condition, param_set in conditions:
        print(f"\nscoring {len(scenes)} test scenes, {condition} ({param_set}) at SIRs {args.test_sir}", flush=True)
        cases = synthetic_cases(cleans, scenes, param_set, args.test_sir, n_repeats=args.n_repeats)
        rows += evaluate_labeled(cases, models, args.row_batch, device,
                                 extra={"condition": condition, "param_set": param_set, "train_sir_db": args.train_sir})

    write_csv(args.results_dir / f"{prefix}_per_slice.csv", rows)
    summary = summarize(rows, ["condition", "sir_db", "method"], ["psnr", "ssim"])
    write_csv(args.results_dir / f"{prefix}_summary.csv", summary)
    by_preset = summarize([r for r in rows if r["condition"] == "shifted"], ["preset", "sir_db", "method"], ["psnr", "ssim"])
    write_csv(args.results_dir / f"{prefix}_shifted_by_modulation.csv", by_preset)
    for metric in ("psnr", "ssim"):
        plot_metric_vs_sir(summary, metric, args.results_dir / f"{prefix}_{metric}.png",
                           "Experiment 3: domain shift (dashed: in-distribution)",
                           condition_key="condition", dashed_condition="in-distribution")

    print("\nshifted condition, PSNR dB / SSIM:")
    for method, sir in itertools.product(MODEL_STYLE, sorted(set(args.test_sir), reverse=True)):
        for r in summary:
            if r["condition"] == "shifted" and r["method"] == method and r["sir_db"] == sir:
                print(f"  {MODEL_STYLE[method]['label']:13s} {sir:5g} dB  {r['psnr_mean']:7.2f} / {r['ssim_mean']:.3f}")


if __name__ == "__main__":
    main()
