"""Validate the FDNet port before full training, and write one report.

Runs, in order, each in isolation so one failure does not hide the others:

1. environment: Python, torch, CUDA, timm, einops, official repository
2. the full pytest suite (``pytest tests -q``), with a JUnit XML of every test
3. official equivalence: official ``FDUformer`` weights loaded into our
   ``FDNet`` with ``strict=True``; the output and every matching intermediate
   module output (input projection, each encoder/decoder stage and LeWin
   block, attention, LeFF, both decomposition branches, output projection)
   compared at 256x256 (and 512x512 with ``--full``)
4. FDNet unit checks: 187x33 (2-channel) and 512x512 (1-channel) forward,
   decomposition features, full loss, backward, finite gradients, one AdamW
   step, checkpoint save/load, optimizer-state resume
5. padding: does the output inside the 187x33 crop depend on how far the
   zero padding extends (256x128 vs 384x256)? With a checkpoint and data,
   also the restoration error of edge versus interior bins and frames
6. a 2-epoch smoke training run (``train.py --smoke --model fdnet``)
7. inference on one real slice (``--sample``), with PSNR/SSIM if it is labeled

Writes ``outputs/fdnet_validation/report.md`` and ``report.json``.

    git clone https://github.com/fangfuping/FDNet third_party/FDNet
    pip install timm einops
    python scripts/validate_fdnet.py --official-dir third_party/FDNet --sample path/to/slice.mat
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: dict[str, dict] = {}


def section(name):
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n=== {name}", flush=True)
            tic = time.time()
            try:
                info = fn(*args, **kwargs) or {}
                status = info.pop("status", "pass")
            except SkipSection as exc:
                status, info = "skipped", {"reason": str(exc)}
            except Exception as exc:  # report and continue
                status, info = "fail", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            info["seconds"] = round(time.time() - tic, 1)
            RESULTS[name] = {"status": status, **info}
            print(f"--> {status.upper()}  {json.dumps({k: v for k, v in info.items() if k != 'traceback'}, default=str)[:600]}")
            return status
        return run
    return wrap


class SkipSection(Exception):
    pass


# --------------------------------------------------------------------------- 1


@section("1 environment")
def check_environment(args):
    info = {"python": platform.python_version(), "platform": platform.platform()}
    import torch

    info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for name in ("timm", "einops", "skimage", "scipy", "h5py", "matplotlib"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "installed")
        except ImportError:
            info[name] = "MISSING"
    info["official_repo"] = str(args.official_dir) if (args.official_dir / "model.py").exists() else "MISSING"
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) < (1, 13):
        info["status"] = "fail"
        info["error"] = "torch >= 1.13 is required (meshgrid indexing=, torch.load weights_only=)"
    return info


# --------------------------------------------------------------------------- 2


@section("2 pytest suite")
def run_pytest(args, out_dir):
    junit = out_dir / "pytest_junit.xml"
    env_cmd = [sys.executable, "-m", "pytest", "tests", "-q", "-rA", f"--junitxml={junit}"]
    import os

    env = dict(os.environ, FDNET_OFFICIAL_DIR=str(args.official_dir.resolve()))
    proc = subprocess.run(env_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (out_dir / "pytest_output.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    cases = {"passed": [], "failed": [], "skipped": [], "error": []}
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            name = f"{case.get('classname')}::{case.get('name')}"
            if case.find("failure") is not None:
                cases["failed"].append(name)
            elif case.find("error") is not None:
                cases["error"].append(name)
            elif case.find("skipped") is not None:
                cases["skipped"].append(f"{name} ({case.find('skipped').get('message', '')[:120]})")
            else:
                cases["passed"].append(name)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-400:]
    return {"status": "pass" if proc.returncode == 0 else "fail", "summary": tail, "returncode": proc.returncode,
            **{k: v for k, v in cases.items()}}


# --------------------------------------------------------------------------- 3


def _load_official(official_dir: Path):
    if not (official_dir / "model.py").exists():
        raise SkipSection(f"official repository not found at {official_dir}")
    for name in ("timm", "einops"):
        if importlib.util.find_spec(name) is None:
            raise SkipSection(f"'{name}' is not installed (pip install timm einops)")
    spec = importlib.util.spec_from_file_location("fdnet_official_model", official_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@section("3 official equivalence")
def check_official(args):
    import torch
    from torch import nn

    from models.fdnet import FDNet

    official_module = _load_official(args.official_dir)
    sizes = [256, 512] if args.full else [256]
    report = {}
    worst = 0.0
    for size in sizes:
        torch.manual_seed(0)
        official = official_module.FDUformer(img_size=size, embed_dim=16, win_size=8, token_projection="linear",
                                             token_mlp="leff", modulator=True).eval()
        ours = FDNet(in_channels=1).eval()
        missing, unexpected = ours.load_state_dict(official.state_dict(), strict=True)
        n_official = sum(p.numel() for p in official.parameters())
        n_ours = sum(p.numel() for p in ours.parameters())

        captured = {"official": {}, "ours": {}}
        names_official = dict(official.named_modules())
        hooks = []
        for name, module in ours.named_modules():
            if not name or name not in names_official or isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            for tag, mod in (("ours", module), ("official", names_official[name])):
                def hook(_m, _inp, out, tag=tag, name=name):
                    if torch.is_tensor(out):
                        captured[tag][name] = out.detach()
                hooks.append(mod.register_forward_hook(hook))
        x = torch.randn(2 if size == 256 else 1, 1, size, size)
        with torch.no_grad():
            ref_out, ref_signal, ref_jam = official(x)
            out, signal, interference = ours(x, return_features=True)
        for h in hooks:
            h.remove()

        layer_diffs = {}
        for name, ref in captured["official"].items():
            mine = captured["ours"].get(name)
            if mine is None or mine.shape != ref.shape:
                layer_diffs[name] = f"shape {None if mine is None else tuple(mine.shape)} vs {tuple(ref.shape)}"
                continue
            layer_diffs[name] = float((mine - ref).abs().max())
        numeric = [v for v in layer_diffs.values() if isinstance(v, float)]
        mismatched = {k: v for k, v in layer_diffs.items() if not isinstance(v, float) or v > 1e-4}
        diffs = {
            "output": float((out - ref_out).abs().max()),
            "signal_features": float((signal.reshape(1, -1) - ref_signal).abs().max()),
            "interference_features": float((interference.reshape(1, -1) - ref_jam).abs().max()),
            "intermediate_modules_compared": len(numeric),
            "intermediate_max_abs_diff": max(numeric) if numeric else None,
        }
        worst = max([worst, diffs["output"], diffs["signal_features"], diffs["interference_features"], *numeric])
        report[f"{size}x{size}"] = {
            "params_official": n_official, "params_ours": n_ours, "strict_load": not missing and not unexpected,
            **diffs, "first_mismatches": dict(list(mismatched.items())[:10]),
        }
    status = "pass" if worst <= 1e-4 and all(r["params_official"] == r["params_ours"] for r in report.values()) else "fail"
    return {"status": status, "max_abs_diff": worst, **report}


# --------------------------------------------------------------------------- 4


@section("4 FDNet unit checks")
def check_units(args, out_dir):
    import torch

    from models import get_model
    from models.fdnet import FDNet
    from train.losses import Criterion

    info = {}
    torch.manual_seed(0)
    model = get_model("fdnet")
    x = torch.randn(2, 2, 187, 33)
    y = torch.randn(2, 2, 187, 33)
    model.eval()
    with torch.no_grad():
        out = model(x)
        _, s, n = model(x, return_features=True)
    info["shape_187x33"] = list(out.shape)
    info["features_187x33"] = [list(s.shape), list(n.shape)]
    assert out.shape == x.shape and torch.isfinite(out).all()

    big = FDNet(in_channels=1).eval()
    with torch.no_grad():
        out512 = big(torch.randn(1, 1, 512, 512))
    info["shape_512x512"] = list(out512.shape)
    assert out512.shape == (1, 1, 512, 512) and torch.isfinite(out512).all()

    model.train()
    criterion = Criterion("fdnet", eps=1e-3, lambda1=0.25, lambda2=1e-5, cosine_mode="paper")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    pred, loss, parts = criterion.loss(criterion.model_forward(model, x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    info["loss"] = float(loss)
    info["loss_parts"] = {k: float(v) for k, v in parts.items()}
    info["params_with_grad"] = sum(g is not None for g in grads)
    info["params_total"] = len(grads)
    info["grads_finite"] = all(g is None or torch.isfinite(g).all().item() for g in grads)
    info["noise_branch_grad_norm"] = float(sum(p.grad.norm() ** 2 for p in model.noise.parameters()) ** 0.5)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    info["optimizer_step_changed_params"] = sum(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert info["grads_finite"] and info["params_with_grad"] == info["params_total"]
    assert info["optimizer_step_changed_params"] > 0

    path = out_dir / "unit_ckpt.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    clone = get_model("fdnet")
    clone.load_state_dict(state["model"])
    opt2 = torch.optim.AdamW(clone.parameters(), lr=2e-3, weight_decay=0.02)
    opt2.load_state_dict(state["optimizer"])
    model.eval(), clone.eval()
    with torch.no_grad():
        info["checkpoint_output_equal"] = bool(torch.equal(model(x), clone(x)))
    # One more identical step on both must keep them identical (optimizer moments restored).
    for m, opt in ((model, optimizer), (clone, opt2)):
        m.train()
        torch.manual_seed(1)
        opt.zero_grad()
        criterion.loss(criterion.model_forward(m, x), y)[1].backward()
        opt.step()
    info["resume_step_identical"] = all(torch.allclose(a, b, atol=1e-7) for a, b in zip(model.parameters(), clone.parameters()))
    path.unlink(missing_ok=True)
    assert info["checkpoint_output_equal"] and info["resume_step_identical"]
    return info


# --------------------------------------------------------------------------- 5


@section("5 padding analysis")
def check_padding(args):
    import torch
    import torch.nn.functional as F

    from eval.pipeline import load_network
    from models import get_model

    info = {}
    if args.checkpoint:
        model, _ = load_network(args.checkpoint, "cpu")
        info["model"] = f"checkpoint {args.checkpoint}"
    else:
        torch.manual_seed(0)
        model = get_model("fdnet").eval()
        info["model"] = "random init (pass --checkpoint for a trained model)"
    torch.manual_seed(2)
    x = torch.randn(2, 2, 187, 33) * 0.05
    with torch.no_grad():
        native = model(x)                                   # padded to 256x128 internally
        wide = model(F.pad(x, (0, 256 - 33, 0, 384 - 187)))[..., :187, :33]
    diff = (native - wide).abs()
    scale = native.abs().max().item()
    info["pad_256x128_vs_384x256_max_abs"] = diff.max().item()
    info["relative_to_output_peak"] = diff.max().item() / scale
    # Where does the dependence on padding live? Last 8 bins / frames versus the rest.
    edge = torch.zeros(187, 33, dtype=torch.bool)
    edge[-8:, :] = True
    edge[:, -8:] = True
    d = diff.amax(dim=(0, 1))
    info["max_diff_edge_band"] = d[edge].max().item()
    info["max_diff_interior"] = d[~edge].max().item() if (~edge).any() else None

    if args.checkpoint and args.data_root:
        from data.dataset import RFIDataset

        ds = RFIDataset("mat", root=args.data_root)
        ds.records = ds.records[: args.n_profile]
        per_bin = per_frame = base_bin = base_frame = 0.0
        for i in range(len(ds)):
            item = ds[i]
            xs, ys = item["polluted"], item["clean"]
            with torch.no_grad():
                pred = torch.cat([model(xs[k:k + 64]) for k in range(0, xs.shape[0], 64)])
            err = ((pred - ys) ** 2).sum(dim=1)              # rows, F, T
            base = ((xs - ys) ** 2).sum(dim=1)
            per_bin += err.mean(dim=(0, 2)); per_frame += err.mean(dim=(0, 1))
            base_bin += base.mean(dim=(0, 2)); base_frame += base.mean(dim=(0, 1))
        ratio_bin = (per_bin / base_bin.clamp_min(1e-12))
        ratio_frame = (per_frame / base_frame.clamp_min(1e-12))
        info["error_ratio_last8_bins_vs_rest"] = (ratio_bin[-8:].mean() / ratio_bin[:-8].mean()).item()
        info["error_ratio_last8_frames_vs_rest"] = (ratio_frame[-8:].mean() / ratio_frame[:-8].mean()).item()
        info["note"] = "ratios near 1 mean the edge next to the zero padding is restored as well as the interior"
    else:
        info["edge_error_profile"] = "skipped: needs --checkpoint and --data-root"
    return info


# --------------------------------------------------------------------------- 6


@section("6 smoke training (2 epochs)")
def smoke_train(args, out_dir):
    import csv

    import torch

    from train.train import main as train_main

    run = out_dir / "smoke_run"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_main(["--smoke", "--model", "fdnet", "--out-dir", str(run), "--no-tensorboard",
                         "--device", device, "--rows-per-image", str(args.smoke_rows)])
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ok = len(rows) == 2 and all(float(r["train_loss"]) == float(r["train_loss"]) for r in rows)
    return {"status": "pass" if ok else "fail", "device": device, "epochs": rows,
            "best_val_psnr": result["best_val_psnr"], "checkpoint": str(run / "best.pt")}


# --------------------------------------------------------------------------- 7


@section("7 real-sample inference")
def real_inference(args, out_dir):
    import numpy as np
    import torch

    from data.dataset import load_mat_slice, load_unlabeled_slice
    from eval.metrics import psnr, ssim
    from eval.pipeline import load_network, suppress_rfi

    if args.sample is None:
        raise SkipSection("no --sample given (a released .mat slice)")
    ckpt = args.checkpoint or out_dir / "smoke_run" / "best.pt"
    if not Path(ckpt).exists():
        raise SkipSection(f"no checkpoint at {ckpt}")
    try:
        loaded = load_mat_slice(args.sample)
        clean = loaded["clean"]
    except ValueError:
        loaded = load_unlabeled_slice(args.sample)
        clean = None
    S = loaded["polluted"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_network(ckpt, device)
    tic = time.time()
    S_star = suppress_rfi(S, model, row_batch=args.row_batch, device=device).cpu().numpy()
    info = {"checkpoint": str(ckpt), "shape": list(S.shape), "seconds": round(time.time() - tic, 2),
            "finite": bool(np.isfinite(S_star).all())}
    if clean is not None:
        info.update(psnr_polluted=psnr(clean, S), psnr_restored=psnr(clean, S_star),
                    ssim_polluted=ssim(clean, S), ssim_restored=ssim(clean, S_star))
    np.save(out_dir / "real_sample_restored.npy", S_star)
    info["note"] = ("plumbing check only: a 2-epoch smoke model is not expected to restore well"
                    if args.checkpoint is None else "trained checkpoint")
    if not info["finite"]:
        info["status"] = "fail"
    return info


# --------------------------------------------------------------------------- report


def write_report(out_dir: Path) -> None:
    (out_dir / "report.json").write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
    lines = ["# FDNet validation report", ""]
    for name, res in RESULTS.items():
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}.get(res["status"], res["status"])
        lines.append(f"## {name}: {mark}")
        for key, value in res.items():
            if key in ("status", "traceback"):
                continue
            if isinstance(value, list) and len(value) > 12:
                value = value[:12] + [f"... {len(value) - 12} more"]
            lines.append(f"- **{key}**: {json.dumps(value, default=str)}")
        if "traceback" in res:
            lines += ["", "```", res["traceback"][-3000:], "```"]
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {out_dir / 'report.md'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--official-dir", type=Path, default=ROOT / "third_party" / "FDNet")
    p.add_argument("--sample", type=Path, help="One released .mat slice for real-data inference")
    p.add_argument("--checkpoint", type=Path, help="Trained FDNet checkpoint (default: the smoke run's best.pt)")
    p.add_argument("--data-root", type=Path, help="Directory of .mat slices for the edge-error profile")
    p.add_argument("--n-profile", type=int, default=8, help="Slices used for the edge-error profile")
    p.add_argument("--full", action="store_true", help="Also compare with the official model at 512x512")
    p.add_argument("--skip-pytest", action="store_true")
    p.add_argument("--smoke-rows", type=int, default=4)
    p.add_argument("--row-batch", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "fdnet_validation")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if check_environment(args) == "fail" and "torch" not in RESULTS["1 environment"]:
        write_report(args.out_dir)
        return 1
    if not args.skip_pytest:
        run_pytest(args, args.out_dir)
    check_official(args)
    check_units(args, args.out_dir)
    check_padding(args)
    smoke_train(args, args.out_dir)
    real_inference(args, args.out_dir)
    write_report(args.out_dir)
    return 0 if all(r["status"] != "fail" for r in RESULTS.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
