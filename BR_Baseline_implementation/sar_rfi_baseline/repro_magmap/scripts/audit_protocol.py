"""Check the files on disk against Fang et al. 2025, Sections 3.1 and 5.2.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/audit_protocol.py --out-dir repro_magmap/protocol_audit

Writes protocol_audit.json, comparison_table.csv, and diagnostic images of the clean image,
the interfered image, and the additive interference. No restored image is written: that would
require a model trained on the paper's set, which is not on disk.

The script refuses to invent interference parameters or to start training.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.protocol import assess, count_residual_clusters, normalized_residual  # noqa: E402
from magmap.stft import stft_lines  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--previous-summary", default="",
                    help="eval summary.json from the earlier all-SIR RFI-1 run, if it is on disk")
    return ap.parse_args(argv)


def _audit(cfg, idx, parts):
    residuals = []
    blocks = []
    n_polluted = 0
    for r in idx["records"]:
        ss = D.load_ss(r.path)
        residuals.append(normalized_residual(ss, 3))
        pairing = idx["pairing"][r.file_id]
        b = pairing["levels"][1]
        blocks.append((tuple(b["affected_azimuth"] or ()), tuple(b["affected_range"] or ())))
        n_polluted += sum(1 for lv in pairing["levels"].values() if lv["n_affected_lines"] > 0)
        del ss
    leaked = sorted(set(s[0] for s in parts["test"]) & ({s[0] for s in parts["train"]} | {s[0] for s in parts["val"]}))
    report = assess(
        n_files=len(idx["records"]),
        n_polluted_slices=n_polluted,
        n_scenes=len(idx["split"]["scenes"]),
        folders=sorted({r.folder for r in idx["records"]}),
        unique_blocks=sorted(set(blocks)),
        n_residual_clusters=count_residual_clusters(residuals),
        train_sirs_db=[D.LEVELS[k] for k in P.split_levels(cfg, "train")],
        test_sirs_db=[D.LEVELS[k] for k in P.split_levels(cfg, "test")],
        leaked_files=leaked,
    )
    report["split_scenes"] = idx["split"]["scenes"]
    report["n_maps"] = {p: len(v) for p, v in parts.items()}
    report["train_levels_only"] = sorted({k for _, k, _ in parts["train"]})
    report["test_files_disjoint"] = not leaked
    return report


def _comparison_rows(previous: dict | None) -> list[dict]:
    """Paper Figure 9 has no table. The read-off values already stored in the previous summary
    are plot readings, not published numbers. The corrected column is empty on purpose."""
    rows = []
    levels = {0.0: (30.7, 0.965), -10.0: (30.8, 0.948), -20.0: (31.5, 0.953), -30.0: (25.4, 0.852)}
    prev = {}
    if previous:
        for item in previous.get("per_level", []):
            prev[float(item["sir_db"])] = item
            if "paper_fig9_read_off" in item:
                levels[float(item["sir_db"])] = (
                    item["paper_fig9_read_off"]["unet_psnr"],
                    item["paper_fig9_read_off"]["unet_ssim"],
                )
    for sir, (p_psnr, p_ssim) in levels.items():
        got = prev.get(sir, {})
        rows.append({
            "sir_db": sir,
            "paper_fig9_unet_psnr_plot_reading": p_psnr,
            "paper_fig9_unet_ssim_plot_reading": p_ssim,
            "previous_rfi1_all_sir_train_psnr": got.get("output_psnr_mean", ""),
            "previous_rfi1_all_sir_train_ssim": got.get("output_ssim_mean", ""),
            "corrected_sec52_psnr": "",
            "corrected_sec52_ssim": "",
            "corrected_status": "not trained; laboratory set is incomplete",
        })
    return rows


def _diagnostics(idx, mcfg, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A test-scene file if the split has one, otherwise the first file.
    test_ids = [fid for fid, part in idx["split"]["files"].items() if part == "test"]
    fid = sorted(test_ids)[0] if test_ids else idx["records"][0].file_id
    ss = D.load_ss(idx["paths"][fid])
    clean = np.abs(ss[0])
    polluted = np.abs(ss[3])          # −20 dB
    residual = np.abs(ss[3] - ss[0])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for a, img, title in (
        (ax[0], clean, "RFI-free |image|"),
        (ax[1], polluted, "Polluted |image|, −20 dB"),
        (ax[2], residual, "Additive |interference|"),
    ):
        a.imshow(img, aspect="auto", cmap="gray")
        a.set_title(title)
        a.set_xlabel("azimuth")
        a.set_ylabel("range")
    fig.suptitle(f"{fid}: one copied interference block, not a restored result")
    fig.tight_layout()
    fig.savefig(out / "image_clean_polluted_residual.png", dpi=130)
    plt.close(fig)

    cols = D.affected_lines(ss, 3)
    line = int(cols[len(cols) // 2])
    specs = np.abs(stft_lines(np.stack([ss[3][:, line], ss[0][:, line]]), mcfg))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    ax[0].imshow(specs[0], aspect="auto", cmap="magma")
    ax[0].set_title(f"polluted spectrogram, azimuth {line}")
    ax[1].imshow(specs[1], aspect="auto", cmap="magma")
    ax[1].set_title("clean spectrogram (label)")
    for a in ax:
        a.set_xlabel("time frame")
        a.set_ylabel("frequency bin")
    fig.suptitle("No restored spectrogram: the Section 5.2 model was not trained")
    fig.tight_layout()
    fig.savefig(out / "spectrogram_clean_vs_polluted.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = P.load_config(args.config)
    idx = P.index_dataset(cfg, hash_files=False, log=print)
    parts = {p: P.samples(cfg, idx, p) for p in ("train", "val", "test")}
    report = _audit(cfg, idx, parts)
    P.write_json(out / "protocol_audit.json", report)
    previous = None
    if args.previous_summary:
        import json
        with open(args.previous_summary, encoding="utf-8") as f:
            previous = json.load(f)
    rows = _comparison_rows(previous)
    with open(out / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _diagnostics(idx, P.map_config(cfg), out)
    print(f"[protocol] training_blocked={report['training_blocked']} "
          f"files={report['n_files']} polluted_slices={report['n_polluted_slices']} "
          f"residual_clusters={report['n_residual_clusters']}")
    print("[protocol] " + report["how_to_obtain_dataset"])
    return 3 if report["training_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
