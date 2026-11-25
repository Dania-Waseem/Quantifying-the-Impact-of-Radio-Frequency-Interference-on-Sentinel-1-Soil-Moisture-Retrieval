"""FP32 vs mixed precision (the authors' torch.cuda.amp + NativeScaler) before the full run.

Every run starts from the same initial weights and trains --steps optimizer steps with the real
configuration; validation always uses FP32 inference (as the authors' test does). Reported per run:
loss curve, non-finite losses, GradScaler-skipped steps, throughput, peak memory, and validation 8-bit
map PSNR / SSIM, literal (authors' np.uint8 expression) and clipped at the label maximum.

Two protocols:
  default        one FP32 run and one AMP run on the same batch order; decision by fixed tolerances
                 (--tol-db on clipped PSNR, --tol-loss on the last-50-step loss, two-sided).
  --noise-floor  three runs per precision: A (batch order seed S), A' (same order, rerun: cuDNN
                 non-determinism) and B (order seed S+1: ordinary training randomness). The noise floor of a
                 metric is the largest within-precision difference (A-A', A-B, for FP32 and for AMP); the
                 precision effect is the FP32-A vs AMP-A difference. Decision rule, fixed before running:
                 AMP is acceptable if it had no non-finite loss, and it is NOT WORSE than FP32 by more than
                 max(noise floor, fixed tolerance) in clipped val PSNR, clipped val SSIM and last-50 loss.
                 Differences in AMP's favour do not count against it. All pairwise output differences are
                 reported, so the cross-precision output difference can be compared with the within-precision one.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/compare_amp.py --config repro_magmap/config_official_unet_sir20_amp.yaml \
        --out-dir <runs>/amp_check_noise_floor --steps 300 --noise-floor
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import pipeline as P  # noqa: E402
from magmap.metrics8 import to_u8  # noqa: E402

QUALITY = ("val_psnr_clip", "val_ssim_clip")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-maps", type=int, default=128, help="N validation maps spread over all SIR levels")
    ap.add_argument("--tol-db", type=float, default=0.2)
    ap.add_argument("--tol-ssim", type=float, default=0.005)
    ap.add_argument("--tol-loss", type=float, default=0.05)
    ap.add_argument("--noise-floor", action="store_true", help="3 runs per precision (see docstring)")
    ap.add_argument("--smoke", action="store_true", help="allow CPU (AMP is then inactive): code-path check only")
    args = ap.parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    from magmap import model as M
    from magmap import trainer as T

    if not torch.cuda.is_available() and not args.smoke:
        print("compare_amp.py needs a CUDA GPU (AMP is CUDA-only in the authors' code); --smoke for a CPU code check")
        return 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to overwrite {out}")
        return 2
    out.mkdir(parents=True)
    cfg = P.load_config(args.config)
    t = cfg["train"]
    T.seed_everything(t["seed"], t["cudnn_benchmark"])
    idx = P.index_dataset(cfg)
    mcfg, norm = P.map_config(cfg), cfg["data"]["normalization"]
    train = P.TorchPairs(P.MapPairs(idx, P.samples(cfg, idx, "train"), mcfg, norm))
    val_all = P.samples(cfg, idx, "val")
    val_s = [val_all[i] for i in np.linspace(0, len(val_all) - 1, min(args.val_maps, len(val_all))).astype(int)]
    val = P.TorchPairs(P.MapPairs(idx, val_s, mcfg, norm))
    val_loader = DataLoader(val, batch_size=t["batch_size"], shuffle=False, num_workers=0)
    init = M.build_unet(cfg["model"]["dim"]).state_dict()
    crit = T.CharbonnierLoss(t["charbonnier_eps"])
    cache: dict = {}
    fp32p = T.Precision("fp32", dev)

    def batches_for(order_seed: int):
        g = torch.Generator()
        g.manual_seed(order_seed)
        order = torch.randperm(len(train), generator=g).tolist()
        need = args.steps * t["batch_size"]
        order = (order * (need // len(order) + 1))[:need]
        return [order[i:i + t["batch_size"]] for i in range(0, need, t["batch_size"])]

    def batch(ids):
        xs, ys = zip(*[cache.setdefault(i, train[i][:2]) for i in ids])
        return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

    def run(mode: str, order_seed: int):
        model = M.build_unet(cfg["model"]["dim"]).to(dev)
        model.load_state_dict(copy.deepcopy(init))
        opt = T.build_optimizer(model, t)
        prec = T.Precision(mode, dev)
        losses, nonfinite, skipped = [], 0, 0
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        model.train()
        for ids in batches_for(order_seed):
            x, y = batch(ids)
            opt.zero_grad()
            with prec.autocast():
                loss = crit(model(x), y)
            nonfinite += int(not torch.isfinite(loss))
            before = prec.scaler.get_scale() if prec.enabled else None
            prec.scaler.scale(loss).backward()
            prec.scaler.step(opt)
            prec.scaler.update()
            if prec.enabled:
                skipped += int(prec.scaler.get_scale() < before)
            losses.append(loss.item())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        v = T.evaluate_loader(model, val_loader, crit, fp32p, dev, with_ssim=True)
        r = {"mode": mode, "order_seed": order_seed, "losses": losses, "nonfinite_losses": nonfinite,
             "skipped_steps": skipped, "seconds": sec, "steps_per_s": len(losses) / sec,
             "last50_loss_mean": float(np.mean(losses[-50:])), "val_fp32_inference": v,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None}
        return r, model.eval()

    s0 = t["seed"]
    plan = ([("fp32", "A", s0), ("fp32", "A_rerun", s0), ("fp32", "B", s0 + 1),
             ("amp", "A", s0), ("amp", "A_rerun", s0), ("amp", "B", s0 + 1)] if args.noise_floor
            else [("fp32", "A", s0), ("amp", "A", s0)])
    runs, models = {}, {}
    for mode, tag, seed in plan:
        key = f"{mode}_{tag}"
        r, m = run(mode, seed)
        runs[key], models[key] = r, m
        v = r["val_fp32_inference"]
        print(f"[{key}] {r['seconds']:.0f}s ({r['steps_per_s']:.2f} steps/s), last-50 loss {r['last50_loss_mean']:.6f}, "
              f"non-finite {r['nonfinite_losses']}, skipped {r['skipped_steps']}; val (fp32 inference) PSNR literal "
              f"{v['val_psnr']:.3f} clipped {v['val_psnr_clip']:.3f}, SSIM clipped {v['val_ssim_clip']:.4f}", flush=True)

    # outputs of every model on the same validation maps (fp32 inference); AMP models also with fp16 inference
    outs, labels, amp_inf = {k: [] for k in models}, [], []
    ampp = T.Precision("amp", dev)
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(dev)
            for k, m in models.items():
                outs[k].append(T.predict(m, x, fp32p).cpu().numpy())
            for k, m in models.items():
                if k.startswith("amp"):
                    o16 = T.predict(m, x, ampp).cpu().numpy()
                    o32 = outs[k][-1]
                    amp_inf.append(float((np.abs(o16 - o32).reshape(len(o16), -1).max(1)
                                          / y.numpy().reshape(len(o16), -1).max(1)).max()))
            labels.append(y.numpy())
    outs = {k: np.concatenate(v) for k, v in outs.items()}
    labels = np.concatenate(labels)

    def out_diff(a, b):
        A, B = outs[a], outs[b]
        mx = labels.reshape(len(labels), -1).max(1)
        d = np.abs(A - B)
        diff8 = np.mean([np.mean(to_u8(A[i], mx[i], "clip")[0] != to_u8(B[i], mx[i], "clip")[0]) for i in range(len(A))])
        return {"mean_abs_diff_over_label_max": float(np.mean(d.reshape(len(d), -1).mean(1) / mx)),
                "max_abs_diff": float(d.max()), "frac_pixels_different_clipped_8bit": float(diff8)}

    pair_diffs = {f"{a}__vs__{b}": out_diff(a, b) for a, b in itertools.combinations(models, 2)}

    def metric(key, m):
        r = runs[key]
        return r["last50_loss_mean"] if m == "loss" else r["val_fp32_inference"][m]

    def delta(a, b, m):
        va, vb = metric(a, m), metric(b, m)
        return (va - vb) / vb if m == "loss" else va - vb

    metrics = ("val_psnr_clip", "val_ssim_clip", "val_psnr", "loss")
    tol = {"val_psnr_clip": args.tol_db, "val_ssim_clip": args.tol_ssim, "loss": args.tol_loss}
    summary = {"config": str(Path(args.config).resolve()), "steps": args.steps, "val_maps": len(val),
               "protocol": "noise_floor" if args.noise_floor else "single_pair", "validation_inference": "fp32",
               "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "smoke_test": bool(args.smoke),
               "speedup_amp": runs["amp_A"]["steps_per_s"] / runs["fp32_A"]["steps_per_s"],
               "amp_nonfinite_losses": sum(r["nonfinite_losses"] for k, r in runs.items() if k.startswith("amp")),
               "amp_skipped_steps": {k: r["skipped_steps"] for k, r in runs.items() if k.startswith("amp")},
               "precision_effect_amp_minus_fp32": {m: delta("amp_A", "fp32_A", m) for m in metrics},
               "amp_model_fp16_vs_fp32_inference_max_abs_diff_over_label_max": max(amp_inf) if amp_inf else None,
               "output_differences": pair_diffs}
    if args.noise_floor:
        within = [("fp32_A", "fp32_A_rerun"), ("fp32_A", "fp32_B"), ("amp_A", "amp_A_rerun"), ("amp_A", "amp_B")]
        noise = {m: max(abs(delta(a, b, m)) for a, b in within) for m in metrics}
        summary["within_precision_differences"] = {f"{a}__vs__{b}": {m: delta(a, b, m) for m in metrics} for a, b in within}
        summary["noise_floor"] = noise
        summary["output_noise_floor_mean_abs_diff_over_label_max"] = max(
            pair_diffs[f"{a}__vs__{b}"]["mean_abs_diff_over_label_max"] for a, b in within)
        summary["output_precision_effect_mean_abs_diff_over_label_max"] = \
            pair_diffs["fp32_A__vs__amp_A"]["mean_abs_diff_over_label_max"]
        allowed = {m: max(noise[m], tol[m]) for m in tol}
        eff = summary["precision_effect_amp_minus_fp32"]
        worse = {"val_psnr_clip": -eff["val_psnr_clip"], "val_ssim_clip": -eff["val_ssim_clip"], "loss": eff["loss"]}
        checks = {m: {"amp_worse_by": worse[m], "allowed": allowed[m], "pass": worse[m] <= allowed[m]} for m in tol}
        ok = summary["amp_nonfinite_losses"] == 0 and all(c["pass"] for c in checks.values())
        summary["rule"] = ("AMP acceptable if no non-finite loss and not worse than FP32 by more than "
                           "max(noise floor, fixed tolerance) in clipped val PSNR, clipped val SSIM and last-50 loss")
        summary["checks"] = checks
    else:
        eff = summary["precision_effect_amp_minus_fp32"]
        ok = (summary["amp_nonfinite_losses"] == 0 and abs(eff["val_psnr_clip"]) <= args.tol_db
              and abs(eff["loss"]) <= args.tol_loss)
        summary["rule"] = "two-sided fixed tolerances on clipped val PSNR and last-50 loss"
    summary["tolerances"] = tol
    summary["recommendation"] = "amp" if ok else "fp32"
    P.write_json(out / "amp_comparison.json", {"summary": summary, "runs": runs})
    _plot(out, runs)
    print("\n" + ("noise floor (largest within-precision difference): " + ", ".join(
        f"{m} {v:.4g}" for m, v in summary["noise_floor"].items()) if args.noise_floor else ""))
    print("precision effect (AMP_A - FP32_A): " + ", ".join(f"{m} {v:+.4g}" for m, v in summary["precision_effect_amp_minus_fp32"].items()))
    if args.noise_floor:
        print(f"output difference / label max: precision effect {summary['output_precision_effect_mean_abs_diff_over_label_max']:.4f} "
              f"vs within-precision max {summary['output_noise_floor_mean_abs_diff_over_label_max']:.4f}")
        for m, c in summary["checks"].items():
            print(f"  {m}: AMP worse by {c['amp_worse_by']:+.4g}, allowed {c['allowed']:.4g} -> {'pass' if c['pass'] else 'FAIL'}")
    print(f"[decision] {summary['recommendation'].upper()}  (speed-up x{summary['speedup_amp']:.2f})")
    return 0


def _plot(out: Path, runs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, r in runs.items():
        lo = np.array(r["losses"])
        n = min(10, len(lo))
        ax.plot(np.convolve(lo, np.ones(n) / n, mode="valid"), label=k, ls="-" if k.startswith("fp32") else "--")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("Charbonnier loss (10-step mean)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "amp_vs_fp32_loss.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
