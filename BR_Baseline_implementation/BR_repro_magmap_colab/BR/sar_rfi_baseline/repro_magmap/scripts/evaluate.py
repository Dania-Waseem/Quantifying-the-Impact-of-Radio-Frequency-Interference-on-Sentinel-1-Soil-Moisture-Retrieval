"""Test-split evaluation in the authors' protocol (8-bit map PSNR / SSIM, test/A_test_unet.py).

Primary: per 512 x 512 map, label-max-scaled uint8 PSNR and SSIM of (a) the RFI-polluted input and
(b) the U-Net output against the clean label; per scene, per SIR level, mean and std over maps.
Supplementary (--image-domain): every range line of each test slice goes through the network, the
predicted magnitude gets the polluted phase, ISTFT, and image-domain PSNR / SSIM / ME are computed
with the project's existing eval.metrics. Image-domain numbers are NOT comparable with Fig. 9.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/evaluate.py --run outputs/repro_magmap_runs/unet_official --device cuda
    python repro_magmap/scripts/evaluate.py --run <run> --checkpoint best_val --device cuda
    python repro_magmap/scripts/evaluate.py --run <run> --max-maps-per-group 4 --device cpu   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.metrics8 import map_metrics  # noqa: E402
from magmap.stft import MapConfig, lines_from_magnitude, stft_lines  # noqa: E402

# Paper Fig. 9 (same-distribution experiment), values READ OFF THE PLOT (+-0.5 dB / +-0.01); see
# outputs/reproduction_audit_2026-11-23/evidence/paper_fig9_values_read_off_plot.csv
PAPER_FIG9_READ_OFF = {0.0: {"unet_psnr": 30.7, "unet_ssim": 0.965, "polluted_psnr": 22.3},
                       -10.0: {"unet_psnr": 30.8, "unet_ssim": 0.948, "polluted_psnr": 14.8},
                       -20.0: {"unet_psnr": 31.5, "unet_ssim": 0.953, "polluted_psnr": 11.7},
                       -30.0: {"unet_psnr": 25.4, "unet_ssim": 0.852, "polluted_psnr": 10.1}}
METRICS = ("psnr", "ssim", "psnr_clip", "ssim_clip", "frac_out_of_range")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--checkpoint", default=None, help="epoch_100 (default: train.reported_checkpoint) | best_val | last | path")
    ap.add_argument("--out-dir", help="default <run>/eval_<checkpoint>")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", choices=["amp", "fp32"], help="default: the precision the run trained with")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-maps-per-group", type=int, help="smoke test: first N maps of each (scene, level)")
    ap.add_argument("--allow-incomplete", action="store_true", help="evaluate a checkpoint before the final epoch")
    ap.add_argument("--image-domain", action="store_true", help="also run the supplementary image-domain evaluation")
    ap.add_argument("--image-lines", choices=["all", "affected"], default="all")
    ap.add_argument("--image-max-lines", type=int, help="smoke test: process only the first N selected lines per slice")
    ap.add_argument("--n-visuals", type=int, default=3)
    return ap.parse_args(argv)


def _ckpt_path(run: Path, which: str) -> Path:
    p = Path(which)
    if p.suffix == ".pt" and p.exists():
        return p
    return run / "checkpoints" / f"{which}.pt"


def summarize(rows: list[dict], keys) -> dict:
    out = {}
    for who in ("input", "output"):
        for m in METRICS:
            v = np.array([r[f"{who}_{m}"] for r in rows], dtype=float)
            fin = v[np.isfinite(v)]
            out[f"{who}_{m}_mean"] = float(fin.mean()) if fin.size else None
            out[f"{who}_{m}_std"] = float(fin.std(ddof=0)) if fin.size else None
            out[f"{who}_{m}_n_nonfinite"] = int(v.size - fin.size)
    out["n_maps"] = len(rows)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from magmap import model as M
    from magmap import trainer as T

    run = P.resolve(args.run)
    rc = json.load(open(run / "config_resolved.json"))
    cfg = rc["config"]
    which = args.checkpoint or cfg["train"]["reported_checkpoint"]
    ck_path = _ckpt_path(run, which)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if which == cfg["train"]["reported_checkpoint"] and ck["epoch"] != cfg["train"]["epochs"] and not args.allow_incomplete:
        print(f"{ck_path.name} is from epoch {ck['epoch']}, not {cfg['train']['epochs']}; use --allow-incomplete")
        return 2
    out = Path(args.out_dir) if args.out_dir else run / f"eval_{Path(which).stem}"
    out = P.resolve(out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    mcfg = MapConfig.from_dict(ck["map_config"])
    norm = ck["normalization"]
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    prec = T.Precision(args.precision or ck.get("precision_used", "fp32"), device)
    model = M.build_unet(cfg["model"]["dim"]).to(device).eval()
    model.load_state_dict(ck["state_dict"], strict=True)

    idx = P.index_dataset(cfg)
    split = json.load(open(run / "split.json"))
    if split["files"] != idx["split"]["files"]:
        raise AssertionError("dataset split differs from the one the run was trained with")
    test = P.samples(cfg, idx, "test")
    if [list(s) for s in test] != split["samples"]["test"]:
        raise AssertionError("test sample list differs from the run's split.json")
    trained_files = {s[0] for s in split["samples"]["train"]} | {s[0] for s in split["samples"]["val"]}
    if trained_files & {s[0] for s in test}:
        raise AssertionError("test files overlap training/validation files")
    if args.max_maps_per_group:
        seen: dict = {}
        test = [s for s in test if seen.setdefault((s[0], s[1]), []).append(s) or len(seen[(s[0], s[1])]) <= args.max_maps_per_group]
    pairs = P.MapPairs(idx, test, mcfg, norm)
    print(f"[eval] {ck_path.name} (epoch {ck['epoch']}), {len(test)} test maps, precision {prec.mode}, device {device}")

    rows, visuals, t0 = [], {}, time.time()
    for b0 in range(0, len(pairs), args.batch_size):
        batch = [pairs[i] for i in range(b0, min(b0 + args.batch_size, len(pairs)))]
        x = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None].to(device)
        o = T.predict(model, x, prec)[:, 0].cpu().numpy()
        for (xi, yi, scale, (fid, k, line)), oi in zip(batch, o):
            mi, mo = map_metrics(xi, yi), map_metrics(oi, yi)
            row = {"file_id": fid, "scene": fid.split("/")[-1], "level_index": k, "sir_db": D.LEVELS[k], "line": line}
            row.update({f"input_{m}": mi[m] for m in METRICS})
            row.update({f"output_{m}": mo[m] for m in METRICS})
            rows.append(row)
            lst = visuals.setdefault((fid, k), [])          # a few lines >= 200 apart per (scene, level)
            if len(lst) < args.n_visuals and (not lst or line - lst[-1][0] >= 200):
                lst.append((line, xi, oi, yi, mi["psnr"], mo["psnr"]))
        if (b0 // args.batch_size) % 200 == 0:
            print(f"  {len(rows)}/{len(pairs)} maps, {time.time() - t0:.0f}s", flush=True)

    with open(out / "per_map_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["scene"], r["sir_db"]), []).append(r)
    per_scene_level = [{"scene": s, "sir_db": l, **summarize(g, METRICS)} for (s, l), g in sorted(groups.items())]
    per_level = []
    for lvl in sorted({r["sir_db"] for r in rows}, reverse=True):
        g = [r for r in rows if r["sir_db"] == lvl]
        per_level.append({"sir_db": lvl, **summarize(g, METRICS), "paper_fig9_read_off": PAPER_FIG9_READ_OFF.get(lvl)})
    with open(out / "per_scene_level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_scene_level[0]))
        w.writeheader()
        w.writerows(per_scene_level)
    summary = {"checkpoint": str(ck_path), "epoch": ck["epoch"], "precision": prec.mode, "n_maps": len(rows),
               "smoke_test": bool(args.max_maps_per_group), "dataset_root": idx["root"],
               "dataset_folders": sorted({r.folder for r in idx["records"]}),
               "label": ("PROXY on the available RFI-Patch folders; the paper's 640-slice protocol is not "
                         "reproduced unless all its folders are present"),
               "protocol": "authors' A_test_unet.py: uint8(round(x / max(label) * 255)), PSNR / Gaussian SSIM, mean over maps",
               "per_level": per_level, "per_scene_level": per_scene_level, "overall": summarize(rows, METRICS),
               "seconds": time.time() - t0}
    P.write_json(out / "summary.json", summary)
    _figures(out, per_level, visuals)
    print("\n  SIR    input PSNR -> output PSNR | input SSIM -> output SSIM   (paper U-Net PSNR/SSIM read off Fig. 9)")
    for r in per_level:
        pf = r["paper_fig9_read_off"] or {}
        print(f"  {r['sir_db']:>5g}  {r['input_psnr_mean']:.2f}+-{r['input_psnr_std']:.2f} -> {r['output_psnr_mean']:.2f}+-"
              f"{r['output_psnr_std']:.2f} | {r['input_ssim_mean']:.4f} -> {r['output_ssim_mean']:.4f}   "
              f"({pf.get('unet_psnr')}/{pf.get('unet_ssim')})")
    if args.image_domain:
        image_domain(out, idx, split, model, prec, device, mcfg, norm, args)
    print(f"[eval] wrote {out}")
    return 0


def image_domain(out, idx, split, model, prec, device, mcfg, norm, args) -> None:
    import torch

    sys.path.insert(0, str(P.PROJECT / "src"))
    from eval.metrics import mean_entropy, psnr, ssim

    from magmap import trainer as T

    store = D.SliceStore(idx["paths"])
    rows = []
    test_files = sorted(f for f, p in split["files"].items() if p == "test")
    for fid in test_files:
        ss = store.get(fid)
        for k in sorted(D.LEVELS):
            lines = np.arange(D.N_AZIMUTH) if args.image_lines == "all" else D.affected_lines(ss, k)
            if args.image_max_lines:
                lines = lines[: args.image_max_lines]
            rec = ss[k].astype(np.complex128).copy()            # (range, azimuth)
            for b0 in range(0, len(lines), args.batch_size):
                js = lines[b0:b0 + args.batch_size]
                noisy = ss[k][:, js].T                           # (B, 512) range lines
                mags = np.abs(stft_lines(noisy, mcfg))
                scales = mags.reshape(len(js), -1).max(1) if norm == "map" else (
                    np.full(len(js), np.abs(ss[k]).max()) if norm == "slice_image" else np.ones(len(js)))
                x = torch.from_numpy((mags / scales[:, None, None]).astype(np.float32))[:, None].to(device)
                o = T.predict(model, x, prec)[:, 0].cpu().numpy().astype(np.float64) * scales[:, None, None]
                rec[:, js] = lines_from_magnitude(np.maximum(o, 0), noisy, mcfg).T
            clean = ss[0]
            row = {"file_id": fid, "sir_db": D.LEVELS[k], "lines": args.image_lines, "n_lines_processed": len(lines),
                   "smoke_test": bool(args.image_max_lines)}
            for who, img in (("input", ss[k]), ("output", rec)):
                row[f"{who}_psnr_f64"] = psnr(clean, img)
                row[f"{who}_ssim_f64"] = ssim(clean, img)
                row[f"{who}_psnr_u8"] = psnr(clean, img, quantize_8bit=True)
                row[f"{who}_ssim_u8"] = ssim(clean, img, quantize_8bit=True)
                row[f"{who}_me"] = mean_entropy(img)
            rows.append(row)
            print(f"  [image] {fid} {D.LEVELS[k]:>5g} dB PSNR {row['input_psnr_f64']:.2f} -> {row['output_psnr_f64']:.2f}")
            np.save(out / f"image_{fid.replace('/', '_')}_L{k}.npy", rec.astype(np.complex64))
    with open(out / "image_domain_supplementary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    P.write_json(out / "image_domain_supplementary.json", {
        "note": ("SUPPLEMENTARY. Phase of the RFI-polluted STFT is reused (assumption); metrics are image-domain "
                 "and must not be compared with the paper's Fig. 9 map-domain values."), "rows": rows})


def _figures(out: Path, per_level: list[dict], visuals: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lv = [r["sir_db"] for r in per_level]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m, lab in ((ax[0], "psnr", "8-bit map PSNR (dB)"), (ax[1], "ssim", "8-bit map SSIM")):
        a.errorbar(lv, [r[f"input_{m}_mean"] for r in per_level], [r[f"input_{m}_std"] for r in per_level],
                   marker="o", capsize=3, label="RFI-polluted (ours)")
        a.errorbar(lv, [r[f"output_{m}_mean"] for r in per_level], [r[f"output_{m}_std"] for r in per_level],
                   marker="s", capsize=3, label="U-Net (ours)")
        ref = [PAPER_FIG9_READ_OFF[l][f"unet_{m}"] for l in lv if l in PAPER_FIG9_READ_OFF]
        if len(ref) == len(lv):
            a.plot(lv, ref, "k--", marker="x", label="U-Net, paper Fig. 9 (read off plot)")
        if m == "psnr" and all(l in PAPER_FIG9_READ_OFF for l in lv):
            a.plot(lv, [PAPER_FIG9_READ_OFF[l]["polluted_psnr"] for l in lv], ":", color="gray", marker="x",
                   label="polluted, paper Fig. 9 (read off)")
        a.set_xlabel("SIR (dB)")
        a.set_ylabel(lab)
        a.invert_xaxis()
        a.legend(fontsize=7)
    fig.suptitle("Test maps, authors' protocol (mean +- std over maps)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "metrics_vs_sir.png", dpi=130)
    plt.close(fig)
    for (fid, k), items in visuals.items():
        fig, ax = plt.subplots(len(items), 3, figsize=(10, 3.2 * len(items)), squeeze=False)
        for r, (line, xi, oi, yi, pi, po) in enumerate(items):
            vmax = yi.max()
            for c, (img, title) in enumerate(((xi, f"polluted (PSNR {pi:.1f})"), (oi, f"U-Net (PSNR {po:.1f})"),
                                              (yi, "label"))):
                ax[r, c].imshow(np.clip(img / vmax, 0, 1), cmap="gray", aspect="auto", origin="lower")
                ax[r, c].set_title(f"{title}\nline {line}", fontsize=8)
                ax[r, c].axis("off")
        fig.suptitle(f"{fid}  SIR {D.LEVELS[k]:g} dB  (display clipped to label max)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"visual_{fid.replace('/', '_')}_L{k}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
