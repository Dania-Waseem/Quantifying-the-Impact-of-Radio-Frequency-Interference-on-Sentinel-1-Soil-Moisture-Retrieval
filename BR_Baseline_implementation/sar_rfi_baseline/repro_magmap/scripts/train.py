"""Train the authors'-configuration U-Net on 512 x 512 magnitude maps.

Run from sar_rfi_baseline/:
    python repro_magmap/scripts/train.py --out-dir outputs/repro_magmap_runs/unet_official --dry-run
    python repro_magmap/scripts/train.py --out-dir outputs/repro_magmap_runs/unet_official --device cuda
    python repro_magmap/scripts/train.py --out-dir <same dir> --device cuda --resume      # continue

Writes into --out-dir (refuses a non-empty directory unless --resume):
  config_resolved.json, split.json, data_check.json, environment.json, code_manifest.json,
  commands.jsonl, history.csv (per epoch: lr, train loss, val loss / PSNR / SSIM, time),
  train.log, checkpoints/{last,best_val,epoch_020,...,epoch_100}.pt, loss_curves.png, lr_history.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from magmap import data as D  # noqa: E402
from magmap import pipeline as P  # noqa: E402
from magmap.protocol import assess, count_residual_clusters, normalized_residual  # noqa: E402

HIST_FIELDS = ["epoch", "lr", "train_loss_mean", "train_loss_sum", "train_steps", "nonfinite_losses", "skipped_steps",
               "val_loss", "val_psnr", "val_psnr_clip", "val_ssim", "val_ssim_clip", "epoch_seconds"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(P.DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", choices=["amp", "fp32"], help="override train.precision")
    ap.add_argument("--channels-last", choices=["true", "false"], help="override train.channels_last (memory layout only)")
    ap.add_argument("--epochs", type=int, help="override train.epochs (smoke tests only)")
    ap.add_argument("--max-steps-per-epoch", type=int, help="smoke tests only")
    ap.add_argument("--max-val-batches", type=int, help="smoke tests only")
    ap.add_argument("--train-maps", help="override data.train_maps (int or 'all')")
    ap.add_argument("--val-maps", help="override data.val_maps")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="index data, write provenance, build model, no training")
    ap.add_argument("--allow-incomplete-data", action="store_true",
                    help="train even when the laboratory set is not the paper's 640-slice, 8-pattern set")
    return ap.parse_args(argv)


def _maybe_int(v):
    return v if v in (None, "all") else int(v)


def main(argv=None) -> int:
    args = parse_args(argv)
    ov = {}
    if args.precision:
        ov["train.precision"] = args.precision
    if args.channels_last:
        ov["train.channels_last"] = args.channels_last == "true"
    if args.epochs:
        ov["train.epochs"] = args.epochs
    if args.train_maps:
        ov["data.train_maps"] = _maybe_int(args.train_maps)
    if args.val_maps:
        ov["data.val_maps"] = _maybe_int(args.val_maps)
    if args.num_workers is not None:
        ov["train.num_workers"] = args.num_workers
    cfg = P.load_config(args.config, ov)
    out = P.resolve(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        print(f"refusing to write into non-empty {out} (use --resume to continue a run)")
        return 2
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    with open(out / "commands.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": str(Path.cwd()),
                            "argv": [sys.executable] + sys.argv,
                            "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)}) + "\n")

    import torch

    from magmap import model as M
    from magmap import trainer as T

    t = cfg["train"]
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type != args.device:
        log(f"[warn] CUDA unavailable, using {device}")
    T.seed_everything(t["seed"], t["cudnn_benchmark"])

    idx = P.index_dataset(cfg, hash_files=True, log=log)
    parts = {p: P.samples(cfg, idx, p) for p in ("train", "val", "test")}
    mcfg = P.map_config(cfg)
    norm = cfg["data"]["normalization"]
    if not args.resume:
        P.write_json(out / "config_resolved.json", {"config_file": str(Path(args.config).resolve()), "config": cfg,
                                                   "map_shape": mcfg.map_shape(), "cli": vars(args)})
        P.write_json(out / "split.json", P.split_record(idx, parts))
        P.write_json(out / "data_check.json", {"root": idx["root"], "file_sha256": idx["file_sha256"],
                                              "pairing": idx["pairing"]})
        P.write_json(out / "environment.json", P.environment())
        P.write_json(out / "code_manifest.json", P.code_manifest())
    test_files = {s[0] for s in parts["test"]}
    leak = test_files & ({s[0] for s in parts["train"]} | {s[0] for s in parts["val"]})
    if leak:
        raise AssertionError(f"test files used for training/validation: {leak}")
    log(f"[data] maps: train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])} "
        f"(test is not loaded during training); map {mcfg.map_shape()}, normalization={norm}")
    log("[data] SIR levels (dB): " + "; ".join(
        f"{p} {[D.LEVELS[k] for k in P.split_levels(cfg, p)]}"
        for p in ("train", "val", "test")))

    audit = _protocol_audit(cfg, idx, parts)
    P.write_json(out / "protocol_audit.json", audit)
    for g in audit["gaps"]:
        log(f"[protocol] {g['key']}: found {g['found']}; required {g['required']} ({g['source']})")
    require = bool((cfg.get("protocol") or {}).get("require_complete_laboratory_dataset"))
    if require and audit["training_blocked"] and not args.allow_incomplete_data:
        log("[protocol] training stopped. This disk is not the paper's laboratory set, so a run "
            "here would not be Figure 9. " + audit["how_to_obtain_dataset"])
        log("[protocol] " + audit["not_synthesized_because"])
        return 3
    if audit["training_blocked"] and args.allow_incomplete_data:
        log("[protocol] incomplete laboratory set; training continues only because "
            "--allow-incomplete-data was set. Do not compare the scores with Figure 9.")

    model = M.build_unet(cfg["model"]["dim"]).to(device)
    cl = bool(t.get("channels_last", False))
    if cl:
        model = model.to(memory_format=torch.channels_last)
    best_metric = t.get("best_val_metric", "val_psnr")
    if best_metric not in ("val_psnr", "val_psnr_clip"):
        raise ValueError("train.best_val_metric must be val_psnr or val_psnr_clip")
    n_params = M.count_parameters(model)
    log(f"[model] UNet 1->1, {n_params:,} parameters (authors' log: {M.AUTHORS_PARAMS:,})")
    optimizer = T.build_optimizer(model, t)
    scheduler = T.build_scheduler(optimizer, t)
    criterion = T.CharbonnierLoss(t["charbonnier_eps"])
    prec = T.Precision(t["precision"], device)
    vprec = T.Precision(t.get("val_precision") or t["precision"], device)   # inference precision for validation
    log(f"[train] precision requested={prec.requested} used={prec.mode}; epochs={t['epochs']}; "
        f"batch={t['batch_size']}; lr={t['lr']} AdamW wd={t['weight_decay']}; StepLR({t['lr_step']},{t['lr_gamma']}) "
        f"initial_step={t['scheduler_initial_step']}; seed={t['seed']}; channels_last={cl}; "
        f"val every {t['val_every']} epochs + last ({vprec.mode} inference); best_val by {best_metric}")

    from torch.utils.data import DataLoader

    train_ds = P.TorchPairs(P.MapPairs(idx, parts["train"], mcfg, norm))
    val_ds = P.TorchPairs(P.MapPairs(idx, parts["val"], mcfg, norm))
    gen = torch.Generator()
    gen.manual_seed(t["seed"])
    nw = t["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=t["shuffle"], drop_last=t["drop_last"],
                              num_workers=nw, generator=gen, pin_memory=device.type == "cuda",
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=nw,
                            pin_memory=device.type == "cuda", persistent_workers=nw > 0)

    x0, y0, _ = train_ds[0]
    log(f"[check] sample tensors {tuple(x0.shape)} {x0.dtype}; input max {x0.max():.4f}; label max {y0.max():.4f}")
    with torch.no_grad():
        o0 = T.predict(model, x0[None].to(device), T.Precision("fp32", device), cl)
    assert o0.shape == (1,) + tuple(x0.shape), o0.shape

    start_epoch, best_psnr, history = 1, -np.inf, []
    ck = out / "checkpoints"
    if args.resume and (ck / "last.pt").exists():
        s = torch.load(ck / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(s["state_dict"])
        optimizer.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"])
        prec.scaler.load_state_dict(s["scaler"])
        T.set_rng_state(s["rng"])
        gen.set_state(s["loader_generator"])
        start_epoch, best_psnr, history = s["epoch"] + 1, s["best_val_psnr"], s["history"]
        log(f"[resume] from epoch {s['epoch']} (lr now {optimizer.param_groups[0]['lr']:.3e})")
    if args.dry_run:
        preview = T.lr_schedule_preview(t, t["epochs"])
        P.write_json(out / "lr_schedule_preview.json", {"epoch_lr": dict(enumerate(preview, 1))})
        log(f"[dry-run] steps/epoch {len(train_loader)}; lr epoch1 {preview[0]:.1e}, epoch49 {preview[48]:.1e}, "
            f"epoch50 {preview[49]:.1e}, epoch99 {preview[98]:.1e}, epoch100 {preview[99]:.1e}. No training done.")
        return 0

    def save(name, epoch, extra=None):
        obj = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": prec.scaler.state_dict(), "config": cfg,
               "map_config": mcfg.to_dict(), "normalization": norm, "precision_used": prec.mode,
               "best_val_psnr": best_psnr, "history": history, "rng": T.rng_state(),
               "loader_generator": gen.get_state(), "channels_last": cl, "best_val_metric": best_metric,
               "val_inference_precision": vprec.mode}
        obj.update(extra or {})
        tmp = ck / (name + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(ck / name)

    for epoch in range(start_epoch, t["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr = T.train_one_epoch(model, train_loader, optimizer, criterion, prec, device, args.max_steps_per_epoch,
                               log_every=100, log=log, channels_last=cl)
        row = {"epoch": epoch, "lr": lr, "train_loss_mean": tr["loss_mean"], "train_loss_sum": tr["loss_sum"],
               "train_steps": tr["steps"], "nonfinite_losses": tr["nonfinite_losses"],
               "skipped_steps": tr["skipped_steps"], "val_loss": None, "val_psnr": None, "val_psnr_clip": None,
               "val_ssim": None, "val_ssim_clip": None}
        if t["val_every"] and (epoch % t["val_every"] == 0 or epoch == t["epochs"]):
            with_ssim = epoch % t["val_ssim_every"] == 0 or epoch == t["epochs"]
            v = T.evaluate_loader(model, val_loader, criterion, vprec, device, with_ssim, args.max_val_batches,
                                  channels_last=cl)
            row.update({k: v[k] for k in ("val_loss", "val_psnr", "val_psnr_clip", "val_ssim", "val_ssim_clip")})
            if v[best_metric] is not None and v[best_metric] > best_psnr:
                best_psnr = v[best_metric]
                save("best_val.pt", epoch, {"selected_by": f"{best_metric} (8-bit map PSNR; "
                                                           f"{'clipped at the label maximum' if best_metric.endswith('clip') else 'authors literal uint8'})"})
        scheduler.step()
        row["epoch_seconds"] = time.time() - t0
        history.append(row)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HIST_FIELDS)
            w.writeheader()
            w.writerows(history)
        log(f"Epoch {epoch}\tlr {lr:.3e}\ttrain {tr['loss_mean']:.6f} (sum {tr['loss_sum']:.4f}; non-finite "
            f"{tr['nonfinite_losses']}, skipped {tr['skipped_steps']})\tval loss {row['val_loss']}\tval PSNR "
            f"{row['val_psnr']} (clipped {row['val_psnr_clip']})\tval SSIM {row['val_ssim']}\t{row['epoch_seconds']:.1f}s")
        save("last.pt", epoch)
        if epoch % t["checkpoint_every"] == 0 or epoch == t["epochs"]:
            save(f"epoch_{epoch:03d}.pt", epoch)
    _plots(out, history)
    final = f"epoch_{t['epochs']:03d}"
    note = "" if final == t["reported_checkpoint"] else f" (config's reported checkpoint is {t['reported_checkpoint']})"
    log(f"[done] final checkpoint: checkpoints/{final}.pt{note}; best {best_metric} {best_psnr:.3f} (checkpoints/best_val.pt)")
    return 0


def _protocol_audit(cfg: dict, idx: dict, parts: dict) -> dict:
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
    return assess(
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


def _plots(out: Path, history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss_mean"] for h in history], label="train (mean per step)")
    vv = [(h["epoch"], h["val_loss"]) for h in history if h["val_loss"] is not None]
    if vv:
        ax[0].plot(*zip(*vv), label="validation")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Charbonnier loss")
    ax[0].legend()
    for key, lab in (("val_psnr", "authors' uint8"), ("val_psnr_clip", "clipped")):
        vp = [(h["epoch"], h.get(key)) for h in history if h.get(key) is not None]
        if vp:
            ax[1].plot(*zip(*vp), marker="o", label=lab)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val 8-bit map PSNR (dB)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.step(ep, [h["lr"] for h in history], where="post")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "lr_history.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
