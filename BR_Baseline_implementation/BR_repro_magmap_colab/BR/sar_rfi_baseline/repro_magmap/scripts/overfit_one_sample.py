"""Checks 8-9: can the 1-channel U-Net learn ONE 512 x 512 magnitude-map pair?

Diagnostic only (not a baseline result). One training map, repeated every step (batch of 1),
the authors' optimizer settings (AdamW 2e-4, betas 0.9/0.999, wd 0.02, Charbonnier eps 1e-3), fp32.
PASS when the output's 8-bit map PSNR beats the polluted input's by >= --min-gain-db and the
Charbonnier loss falls below --max-loss-ratio x its initial value. The PSNR used for the decision is
--criterion-metric: "psnr_clip" (default; values above the label maximum are clipped to 255) or "psnr"
(the authors' literal np.uint8 expression, which wraps values above the label maximum to small numbers
and so swings by several dB when an output pixel slightly exceeds the label peak). Both are logged.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/overfit_one_sample.py --out-dir outputs/repro_magmap_2026-11-23/overfit_one_sample
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.metrics8 import map_metrics  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-minutes", type=float, default=75.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--level", type=int, default=3, help="ss index of the polluted version (3 = -20 dB)")
    ap.add_argument("--min-gain-db", type=float, default=10.0)
    ap.add_argument("--max-loss-ratio", type=float, default=0.5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--criterion-metric", default="psnr_clip", choices=["psnr_clip", "psnr"])
    args = ap.parse_args(argv)
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)

    import torch

    from magmap import model as M
    from magmap import trainer as T

    cfg = P.load_config(args.config)
    t = cfg["train"]
    T.seed_everything(t["seed"], False)
    idx = P.index_dataset(cfg)
    train = [s for s in P.samples(cfg, idx, "train") if s[1] == args.level]
    sample = train[len(train) // 2]
    pairs = P.MapPairs(idx, [sample], P.map_config(cfg), cfg["data"]["normalization"])
    x, y, scale, _ = pairs[0]
    device = torch.device(args.device)
    X = torch.from_numpy(x)[None, None].to(device)
    Y = torch.from_numpy(y)[None, None].to(device)
    model = M.build_unet(cfg["model"]["dim"]).to(device)
    opt = T.build_optimizer(model, t)
    crit = T.CharbonnierLoss(t["charbonnier_eps"])
    m_in = map_metrics(x, y)
    print(f"[overfit] sample {sample} ({D.LEVELS[args.level]:g} dB), input PSNR {m_in['psnr']:.2f} (clipped {m_in['psnr_clip']:.2f}) "
          f"SSIM {m_in['ssim']:.4f}; "
          f"torch threads {torch.get_num_threads()}", flush=True)
    rows, t0, loss0 = [], time.time(), None
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(X), Y)
        loss.backward()
        opt.step()
        loss0 = loss0 if loss0 is not None else loss.item()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                o = model(X)[0, 0].cpu().numpy()
            m = map_metrics(o, y)
            rows.append({"step": step, "loss": loss.item(), "output_psnr": m["psnr"], "output_ssim": m["ssim"],
                         "output_psnr_clip": m["psnr_clip"], "frac_out_of_range": m["frac_out_of_range"],
                         "seconds": time.time() - t0})
            print(f"  step {step:4d} loss {loss.item():.6f} PSNR {m['psnr']:.2f} clipped {m['psnr_clip']:.2f} SSIM {m['ssim']:.4f} "
                  f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
            with open(out / "overfit_curve.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"  stopping at step {step}: time limit {args.max_minutes} min")
            break
    model.eval()
    with torch.no_grad():
        o = model(X)[0, 0].cpu().numpy()
    m_out = map_metrics(o, y)
    final_loss = rows[-1]["loss"]
    cm = args.criterion_metric
    gain = m_out[cm] - m_in[cm]
    passed = gain >= args.min_gain_db and final_loss <= args.max_loss_ratio * loss0
    res = {"sample": list(sample), "sir_db": D.LEVELS[args.level], "steps_run": rows[-1]["step"],
           "precision": "fp32", "device": str(device), "input": m_in, "output": m_out, "psnr_gain_db": gain,
           "initial_loss": loss0, "final_loss": final_loss, "criteria": {"metric": cm, "min_gain_db": args.min_gain_db,
                                                                         "max_loss_ratio": args.max_loss_ratio},
           "pass": bool(passed), "seconds": time.time() - t0, "seconds_per_step": (time.time() - t0) / rows[-1]["step"],
           "note": "diagnostic on one training map; not a baseline result"}
    P.write_json(out / "overfit_result.json", res)
    torch.save({"state_dict": model.state_dict(), "sample": sample, "config": cfg}, out / "overfit_unet.pt")
    _fig(out, x, o, y, m_in, m_out, rows)
    print(f"[overfit] {'PASS' if passed else 'FAIL'} ({cm}): {m_in[cm]:.2f} -> {m_out[cm]:.2f} dB ({gain:+.2f}); "
          f"authors' expression {m_in['psnr']:.2f} -> {m_out['psnr']:.2f}; "
          f"loss {loss0:.5f} -> {final_loss:.5f}")
    return 0 if passed else 1


def _fig(out, x, o, y, m_in, m_out, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, img, title in ((ax[0], x, f"polluted input  PSNR {m_in['psnr']:.1f}"), (ax[1], o, f"U-Net output  PSNR {m_out['psnr']:.1f}"),
                          (ax[2], y, "label")):
        a.imshow(np.clip(img / y.max(), 0, 1), cmap="gray", origin="lower", aspect="auto")
        a.set_title(title, fontsize=9)
    ax[3].plot([r["step"] for r in rows], [r["output_psnr"] for r in rows], marker=".", label="output, authors' uint8")
    ax[3].plot([r["step"] for r in rows], [r["output_psnr_clip"] for r in rows], marker=".", label="output, clipped")
    ax[3].axhline(m_in["psnr"], color="gray", ls=":", label="input, authors' uint8")
    ax[3].axhline(m_in["psnr_clip"], color="k", ls="--", lw=0.8, label="input, clipped")
    ax[3].set_xlabel("step")
    ax[3].set_ylabel("8-bit map PSNR (dB)")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "overfit_one_sample.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
