"""FP32 vs mixed precision (the authors' torch.cuda.amp + NativeScaler) before the full run. GPU only.

Both runs start from the same initial weights and see the same batches in the same order for
--steps optimizer steps; then both are scored on the first --val-maps validation maps.
Recommendation rule (decided before running): use AMP only if it produced no non-finite loss, its
final validation 8-bit PSNR is within --tol-db of FP32, and its last-50-step mean loss is within
--tol-loss (relative) of FP32's. Otherwise use FP32.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/compare_amp.py --out-dir outputs/repro_magmap_runs/amp_check --steps 300
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import pipeline as P  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-maps", type=int, default=128)
    ap.add_argument("--tol-db", type=float, default=0.2)
    ap.add_argument("--tol-loss", type=float, default=0.05)
    args = ap.parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    from magmap import model as M
    from magmap import trainer as T

    if not torch.cuda.is_available():
        print("compare_amp.py needs a CUDA GPU (AMP is CUDA-only in the authors' code)")
        return 2
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)
    cfg = P.load_config(args.config)
    t = cfg["train"]
    dev = torch.device("cuda")
    T.seed_everything(t["seed"], t["cudnn_benchmark"])
    idx = P.index_dataset(cfg)
    mcfg, norm = P.map_config(cfg), cfg["data"]["normalization"]
    train = P.TorchPairs(P.MapPairs(idx, P.samples(cfg, idx, "train"), mcfg, norm))
    val = P.TorchPairs(P.MapPairs(idx, P.samples(cfg, idx, "val")[: args.val_maps], mcfg, norm))
    g = torch.Generator()
    g.manual_seed(t["seed"])
    order = torch.randperm(len(train), generator=g).tolist()
    need = args.steps * t["batch_size"]
    order = (order * (need // len(order) + 1))[:need]
    batches = [order[i:i + t["batch_size"]] for i in range(0, need, t["batch_size"])]
    init = M.build_unet(cfg["model"]["dim"]).state_dict()
    crit = T.CharbonnierLoss(t["charbonnier_eps"])
    val_loader = DataLoader(val, batch_size=t["batch_size"], shuffle=False, num_workers=2)
    cache = {}

    def batch(ids):
        xs, ys = zip(*[cache.setdefault(i, train[i][:2]) for i in ids])
        return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

    res = {}
    for mode in ("fp32", "amp"):
        model = M.build_unet(cfg["model"]["dim"]).to(dev)
        model.load_state_dict(copy.deepcopy(init))
        opt = T.build_optimizer(model, t)
        prec = T.Precision(mode, dev)
        losses, nonfinite = [], 0
        torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        for ids in batches:
            x, y = batch(ids)
            opt.zero_grad()
            with prec.autocast():
                loss = crit(model(x), y)
            nonfinite += int(not torch.isfinite(loss))
            prec.scaler.scale(loss).backward()
            prec.scaler.step(opt)
            prec.scaler.update()
            losses.append(loss.item())
        torch.cuda.synchronize()
        sec = time.time() - t0
        v = T.evaluate_loader(model, val_loader, crit, prec, dev, with_ssim=True)
        res[mode] = {"losses": losses, "nonfinite_losses": nonfinite, "seconds": sec, "steps_per_s": len(batches) / sec,
                     "last50_loss_mean": float(np.mean(losses[-50:])), "val": v,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        torch.cuda.reset_peak_memory_stats()
        print(f"[{mode}] {sec:.0f}s ({len(batches) / sec:.2f} steps/s), last-50 loss {res[mode]['last50_loss_mean']:.6f}, "
              f"val PSNR {v['val_psnr']:.3f} SSIM {v['val_ssim']:.4f}, non-finite {nonfinite}", flush=True)
    a, f = res["amp"], res["fp32"]
    dpsnr = a["val"]["val_psnr"] - f["val"]["val_psnr"]
    dloss = abs(a["last50_loss_mean"] - f["last50_loss_mean"]) / f["last50_loss_mean"]
    use_amp = a["nonfinite_losses"] == 0 and abs(dpsnr) <= args.tol_db and dloss <= args.tol_loss
    summary = {"steps": args.steps, "val_maps": len(val), "delta_val_psnr_amp_minus_fp32": dpsnr,
               "relative_last50_loss_diff": dloss, "speedup_amp": a["steps_per_s"] / f["steps_per_s"],
               "rule": {"tol_db": args.tol_db, "tol_loss": args.tol_loss, "nonfinite_allowed": 0},
               "recommendation": "amp" if use_amp else "fp32", "gpu": torch.cuda.get_device_name(0)}
    P.write_json(out / "amp_comparison.json", {"summary": summary, "runs": res})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ("fp32", "amp"):
        l = np.array(res[m]["losses"])
        ax.plot(np.convolve(l, np.ones(10) / 10, mode="valid"), label=m)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("Charbonnier loss (10-step mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "amp_vs_fp32_loss.png", dpi=130)
    print(f"[decision] {summary['recommendation'].upper()}  (dPSNR {dpsnr:+.3f} dB, loss diff {dloss:.1%}, "
          f"speed-up x{summary['speedup_amp']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
