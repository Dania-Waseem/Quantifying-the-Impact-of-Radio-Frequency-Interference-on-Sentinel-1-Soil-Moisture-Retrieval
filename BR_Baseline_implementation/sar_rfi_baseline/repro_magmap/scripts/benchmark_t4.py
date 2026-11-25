"""Speed benchmark on the Colab GPU before the full run (no training run, nothing overwritten).

Measures, with the real configuration (U-Net 1->1, batch 4, 512 x 512 maps, AdamW, Charbonnier):
  1. optimizer steps per second for FP32 / AMP x default (NCHW) / channels_last memory layout,
     each from the same initial weights on the same cached GPU batches (data loading excluded),
     after --warmup steps, timed over --steps steps with CUDA synchronisation;
  2. data-loader throughput alone (on-the-fly STFT on the CPU workers, as in training);
  3. one full validation pass (all validation maps, loss + 8-bit PSNR/SSIM) in the chosen setting;
  4. the forward-output difference of the same weights in NCHW vs channels_last (layout only).
Projection: epoch = steps_per_epoch x max(GPU step, loader batch); total = epochs x epoch +
(number of validations) x validation pass. channels_last is recommended only if it is at least
--min-gain faster than NCHW for the chosen precision.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/benchmark_t4.py --config repro_magmap/config_official_unet_sir20_amp.yaml \
        --out-dir <runs>/benchmark_t4
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--distinct-batches", type=int, default=25)
    ap.add_argument("--loader-batches", type=int, default=100)
    ap.add_argument("--min-gain", type=float, default=0.03, help="relative speed-up needed to enable channels_last")
    ap.add_argument("--repeats", type=int, default=1, help="repeat each timing, alternating layouts; medians are used")
    ap.add_argument("--precisions", default="fp32,amp", help="comma list of precisions to time")
    ap.add_argument("--skip-val", action="store_true", help="skip the validation-pass timing")
    ap.add_argument("--smoke", action="store_true", help="allow CPU with tiny sizes: code-path check only")
    args = ap.parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    from magmap import model as M
    from magmap import trainer as T

    if not torch.cuda.is_available() and not args.smoke:
        print("benchmark_t4.py needs the Colab GPU (use --smoke for a CPU code-path check)")
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
    train_s, val_s = P.samples(cfg, idx, "train"), P.samples(cfg, idx, "val")
    train = P.TorchPairs(P.MapPairs(idx, train_s, mcfg, norm))
    bs = t["batch_size"]
    nb = args.distinct_batches
    batches = []
    for b in range(nb):
        xs, ys = zip(*[train[(b * bs + i) % len(train)][:2] for i in range(bs)])
        batches.append((torch.stack(xs).to(dev), torch.stack(ys).to(dev)))
    init = M.build_unet(cfg["model"]["dim"]).state_dict()
    crit = T.CharbonnierLoss(t["charbonnier_eps"])
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    # 1. GPU step time ------------------------------------------------------------------------------
    combos = {}
    precs = [p_.strip() for p_ in args.precisions.split(",") if p_.strip()]
    order = [(p_, cl, r) for r in range(args.repeats) for p_ in precs
             for cl in ((False, True) if r % 2 == 0 else (True, False))]       # alternate layout order per repeat
    for prec_name, cl, rep in order:
        model = M.build_unet(cfg["model"]["dim"]).to(dev)
        model.load_state_dict(copy.deepcopy(init))
        if cl:
            model = model.to(memory_format=torch.channels_last)
        opt = T.build_optimizer(model, t)
        prec = T.Precision(prec_name, dev)
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        nonfinite = 0
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                sync()
                t0 = time.time()
            x, y = batches[i % nb]
            x = T.as_input(x, dev, cl)
            opt.zero_grad()
            with prec.autocast():
                loss = crit(model(x), y)
            nonfinite += int(not torch.isfinite(loss))
            prec.scaler.scale(loss).backward()
            prec.scaler.step(opt)
            prec.scaler.update()
        sync()
        sec = (time.time() - t0) / args.steps
        key = f"{prec_name}_{'channels_last' if cl else 'nchw'}"
        c = combos.setdefault(key, {"repeat_seconds_per_step": [], "nonfinite_losses": 0, "peak_mem_gb": None})
        c["repeat_seconds_per_step"].append(sec)
        c["nonfinite_losses"] += nonfinite
        c["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
        c["seconds_per_step"] = float(np.median(c["repeat_seconds_per_step"]))
        c["steps_per_s"] = 1 / c["seconds_per_step"]
        print(f"[gpu] repeat {rep + 1} {key:20s} {sec * 1000:7.1f} ms/step  ({1 / sec:.2f} steps/s)", flush=True)
        del model, opt

    # layout-only numerical check: same weights, NCHW vs channels_last, fp32 forward
    m1 = M.build_unet(cfg["model"]["dim"]).to(dev).eval()
    m1.load_state_dict(copy.deepcopy(init))
    m2 = copy.deepcopy(m1).to(memory_format=torch.channels_last)
    with torch.no_grad():
        x = batches[0][0]
        layout_diff = float((m1(x) - m2(x.contiguous(memory_format=torch.channels_last))).abs().max())
    del m1, m2

    # 2. loader throughput --------------------------------------------------------------------------
    nw = t["num_workers"]
    loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=dev.type == "cuda",
                        persistent_workers=False)
    it = iter(loader)
    next(it)
    t0 = time.time()
    n = 0
    for _ in range(args.loader_batches):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    loader_s = (time.time() - t0) / max(n, 1)
    print(f"[loader] {loader_s * 1000:.1f} ms/batch with {nw} workers ({n} batches)", flush=True)
    del it, loader

    # choose precision (config) and layout (benchmark) ----------------------------------------------
    p = t["precision"] if (dev.type == "cuda" and f"{t['precision']}_nchw" in combos) else precs[0]
    nchw, clk = combos[f"{p}_nchw"]["seconds_per_step"], combos[f"{p}_channels_last"]["seconds_per_step"]
    per_rep = [(a - b) / a for a, b in zip(combos[f"{p}_nchw"]["repeat_seconds_per_step"],
                                           combos[f"{p}_channels_last"]["repeat_seconds_per_step"])]
    use_cl = (nchw - clk) / nchw >= args.min_gain and all(g > 0 for g in per_rep)

    # 3. validation pass ----------------------------------------------------------------------------
    val = P.TorchPairs(P.MapPairs(idx, val_s if not args.smoke else val_s[:8], mcfg, norm))
    vloader = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=dev.type == "cuda")
    model = M.build_unet(cfg["model"]["dim"]).to(dev)
    model.load_state_dict(copy.deepcopy(init))
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    val_prec = t.get("val_precision") or p
    if args.skip_val:
        val_s_time = float("nan")
    else:
        sync()
        t0 = time.time()
        T.evaluate_loader(model, vloader, crit, T.Precision(val_prec, dev), dev, with_ssim=True, channels_last=use_cl)
        sync()
        val_s_time = time.time() - t0
        print(f"[val] one validation pass over {len(val)} maps ({val_prec} inference): {val_s_time:.1f} s", flush=True)

    # 4. projection ---------------------------------------------------------------------------------
    n_train = len(train_s)
    spe = int(np.ceil(n_train / bs))
    epochs = t["epochs"]
    n_val = len([e for e in range(1, epochs + 1) if e % t["val_every"] == 0 or e == epochs])
    scale_val = (len(val_s) / len(val)) if args.smoke else 1.0

    def project(step_s, validations):
        ep = spe * max(step_s, loader_s)
        return {"train_epoch_min": ep / 60, "total_hours": (epochs * ep + validations * val_s_time * scale_val) / 3600}

    proj = {k: project(v["seconds_per_step"], n_val) for k, v in combos.items()}
    if "fp32_nchw" in combos:
        proj["fp32_nchw_validate_every_epoch"] = project(combos["fp32_nchw"]["seconds_per_step"], epochs)
    rec = {"precision": p, "channels_last": bool(use_cl), "channels_last_gain": (nchw - clk) / nchw,
           "channels_last_gain_per_repeat": per_rep,
           "rule": f"channels_last if the median step is >= {args.min_gain:.0%} faster and it is faster in every repeat"}
    res = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           "config": str(Path(args.config).resolve()), "batch_size": bs, "train_maps": n_train,
           "steps_per_epoch": spe, "epochs": epochs, "validations": n_val, "gpu_step": combos,
           "loader_seconds_per_batch": loader_s, "loader_workers": nw, "loader_bound": bool(loader_s > min(
               v["seconds_per_step"] for v in combos.values())), "validation_pass_seconds": val_s_time * scale_val,
           "layout_only_max_abs_output_diff_fp32": layout_diff, "projection": proj, "recommendation": rec,
           "smoke_test": bool(args.smoke), "warmup": args.warmup, "timed_steps": args.steps,
           "repeats": args.repeats, "validation_inference": val_prec}
    P.write_json(out / "benchmark.json", res)
    print("\nprojection (100 epochs incl. validation):")
    for k, v in proj.items():
        print(f"  {k:32s} {v['train_epoch_min']:.2f} min/epoch training, {v['total_hours']:.2f} h total")
    print(f"[recommendation] precision {p}, channels_last {use_cl} (gain {rec['channels_last_gain']:+.1%}); "
          f"layout-only output diff {layout_diff:.2e}; loader-bound: {res['loader_bound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
