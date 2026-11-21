"""Shared pieces of the three experiments in Section 5 of Fang et al.

The scripts in ``scripts/run_exp{1,2,3}.py`` are thin wrappers around this
module. Each one:

1. trains every requested model through :func:`train.train.main` (or reuses
   ``best.pt`` with ``--skip-train``; an unfinished run resumes),
2. builds test cases (polluted image, optional clean label, SIR, scene),
3. runs Algorithm 1 (:func:`eval.pipeline.suppress_rfi`) with each model,
4. writes per-slice and summary CSVs and matplotlib figures to ``results/``.

Test scenes are the ``test`` entries of ``splits.json`` written by the
training run, so a scene used for training or model selection is never
scored. ``--smoke`` runs everything on the 20 small synthetic scenes of
``train.py --smoke``; its 4 test scenes stand in for the real test split.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from data.dataset import (
    _stable_seed,
    discover_mat_records,
    load_mat_slice,
    load_unlabeled_slice,
    synthesize_pair,
)
from data.param_sets import get_param_set
from eval.metrics import mean_entropy, psnr, ssim
from eval.pipeline import load_network, suppress_rfi
from models import IMPLEMENTED_MODELS
from utils.tf import stft_row

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIR_LEVELS = (0.0, -10.0, -20.0, -30.0)
POLLUTED = "polluted"

# Fixed categorical order (one hue per model, never reassigned) and display names.
MODEL_STYLE = {
    "dncnn": {"color": "#2a78d6", "marker": "o", "label": "DnCNN"},
    "dcnet": {"color": "#41b6c4", "marker": "v", "label": "DCNet"},
    "unet": {"color": "#eb6834", "marker": "s", "label": "U-Net"},
    "official_unet": {"color": "#eb6834", "marker": "s", "label": "Official U-Net"},
    "ronneberger_unet": {"color": "#fa9fb5", "marker": "p", "label": "Ronneberger U-Net"},
    "pisnet": {"color": "#1baf7a", "marker": "^", "label": "PISNet"},
    "imn": {"color": "#a1d99b", "marker": "<", "label": "IMN"},
    "fdnet": {"color": "#eda100", "marker": "D", "label": "FDNet"},
    POLLUTED: {"color": "#8a8a85", "marker": "x", "label": "RFI-polluted"},
}

# The smoke scenes must match train.py --smoke exactly.
SMOKE_SCENES, SMOKE_SHAPE, SMOKE_SEED = 20, (64, 256), 42


# --------------------------------------------------------------------------- common CLI


def add_common_args(parser) -> None:
    parser.add_argument("--models", nargs="+", default=list(IMPLEMENTED_MODELS), choices=list(IMPLEMENTED_MODELS))
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default 100, or 2 with --smoke)")
    parser.add_argument("--skip-train", action="store_true", help="Use existing best.pt checkpoints only")
    parser.add_argument("--smoke", action="store_true", help="Tiny synthetic run to check the script end to end")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:N (default: cuda if available)")
    parser.add_argument("--row-batch", type=int, default=128, help="Rows per network call at test time")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Default outputs/runs/<exp>[_smoke]")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--train-arg", action="append", default=[], metavar="ARG",
                        help="Extra argument passed to train.py, e.g. --train-arg=--num-workers=4")


def runs_dir_for(args, experiment: str) -> Path:
    return Path(args.runs_dir or PROJECT_ROOT / "outputs" / "runs" / f"{experiment}{'_smoke' if args.smoke else ''}")


def results_prefix(args, experiment: str) -> str:
    return f"{experiment}{'_smoke' if args.smoke else ''}"


def resolve_device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


# --------------------------------------------------------------------------- training


def train_models(models: Sequence[str], runs_dir: Path, train_argv: Sequence[str], args) -> dict[str, Path]:
    """Train (or resume, or reuse) each model and return its ``best.pt``."""
    from train.train import main as train_main

    checkpoints = {}
    for name in models:
        run_dir = runs_dir / name
        best = run_dir / "best.pt"
        if args.skip_train:
            if not best.exists():
                raise SystemExit(f"--skip-train but {best} does not exist")
        else:
            argv = ["--model", name, "--out-dir", str(run_dir), "--resume", "auto", *train_argv, *args.train_arg]
            if args.epochs is not None:
                argv += ["--epochs", str(args.epochs)]
            if args.device:
                argv += ["--device", args.device]
            print(f"\n=== training {name}: train.py {' '.join(argv)}", flush=True)
            train_main(argv)
        checkpoints[name] = best
    return checkpoints


def load_models(checkpoints: dict[str, Path], device) -> dict[str, torch.nn.Module]:
    return {name: load_network(path, device)[0] for name, path in checkpoints.items()}


def test_scene_ids(run_dir: Path) -> list[str]:
    """Scenes marked ``test`` in the split the training run used."""
    path = run_dir / "splits.json"
    splits = json.loads(path.read_text(encoding="utf-8"))["splits"]
    scenes = sorted(scene for scene, split in splits.items() if split == "test")
    if not scenes:
        raise SystemExit(f"{path} has no test scenes; train with a nonzero --test-ratio")
    return scenes


def smoke_cleans() -> list[dict]:
    from train.train import make_smoke_scenes

    return make_smoke_scenes(SMOKE_SCENES, SMOKE_SHAPE, SMOKE_SEED)


# --------------------------------------------------------------------------- test cases


def mat_cases(root, scene_ids: Iterable[str], sirs: Sequence[float] = SIR_LEVELS) -> Iterator[dict]:
    """Labeled released slices of the given scenes at the given SIRs."""
    scene_ids = set(scene_ids)
    records = [r for r in discover_mat_records(root)
               if r.scene_id in scene_ids and any(math.isclose(r.sir_db, s, abs_tol=1e-6) for s in sirs)]
    if not records:
        raise SystemExit(f"no .mat slices under {root} for test scenes {sorted(scene_ids)[:5]} at SIRs {list(sirs)}")
    for record in records:
        loaded = load_mat_slice(record.path)
        yield {
            "case": Path(record.path).stem,
            "scene_id": record.scene_id,
            "scene_type": record.scene_type,
            "sir_db": record.sir_db,
            "preset": "released",
            "polluted": loaded["polluted"],
            "clean": loaded["clean"],
        }


def synthetic_cases(cleans: Sequence[dict], scene_ids: Iterable[str], param_set: str,
                    sirs: Sequence[float] = SIR_LEVELS, n_repeats: int = 1, seed: int = 1234) -> Iterator[dict]:
    """Labeled slices made by polluting clean test scenes with a named parameter set.

    ``seed`` differs from the training seed, so even an in-distribution test
    case is a fresh interference draw.
    """
    rfi = get_param_set(param_set)
    scene_ids = set(scene_ids)
    chosen = [c for c in cleans if str(c["scene_id"]) in scene_ids]
    if not chosen:
        raise SystemExit(f"none of the test scenes {sorted(scene_ids)[:5]} are among the clean images")
    for item in chosen:
        clean = np.asarray(item["image"])
        for sir in sirs:
            for preset in rfi["presets"]:
                for repeat in range(n_repeats):
                    case_seed = _stable_seed(item["scene_id"], sir, preset, repeat, param_set, base=seed)
                    yield {
                        "case": f"{item['scene_id']}_{preset}_{sir:g}dB_r{repeat}",
                        "scene_id": str(item["scene_id"]),
                        "scene_type": item["scene_type"],
                        "sir_db": float(sir),
                        "preset": preset,
                        "polluted": synthesize_pair(clean, sir, preset, case_seed, rfi["ranges"]),
                        "clean": clean,
                    }


def unlabeled_cases(root) -> Iterator[dict]:
    paths = sorted(Path(root).rglob("*.mat"))
    if not paths:
        raise SystemExit(f"no .mat slices under {root}")
    for path in paths:
        loaded = load_unlabeled_slice(path)
        yield {"case": loaded["name"], "scene_type": loaded["scene_type"], "polluted": loaded["polluted"]}


# --------------------------------------------------------------------------- evaluation


def _timed_restore(S, model, row_batch, device) -> tuple[torch.Tensor, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    S_star = suppress_rfi(S, model, row_batch=row_batch, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return S_star.cpu(), time.perf_counter() - tic


def evaluate_labeled(cases: Iterable[dict], models: dict, row_batch: int, device, extra: dict | None = None,
                     on_case=None) -> list[dict]:
    """PSNR and SSIM of the polluted input and of every model's S*, per test slice."""
    rows = []
    extra = extra or {}
    for n, case in enumerate(cases, start=1):
        clean = case["clean"]
        outputs = {POLLUTED: (np.asarray(case["polluted"]), 0.0)}
        for name, model in models.items():
            S_star, seconds = _timed_restore(case["polluted"], model, row_batch, device)
            outputs[name] = (S_star.numpy(), seconds)
        for method, (image, seconds) in outputs.items():
            rows.append({
                **extra,
                "case": case["case"], "scene_id": case["scene_id"], "scene_type": case["scene_type"],
                "sir_db": case["sir_db"], "preset": case["preset"], "method": method,
                "psnr": psnr(clean, image), "ssim": ssim(clean, image), "seconds": round(seconds, 4),
            })
        if on_case is not None:
            on_case(case, outputs)
        if n % 10 == 0:
            print(f"  scored {n} slices", flush=True)
    return rows


def summarize(rows: Sequence[dict], keys: Sequence[str], metrics: Sequence[str]) -> list[dict]:
    """Mean, standard deviation and count of each metric per group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, members in groups.items():
        entry = dict(zip(keys, key))
        entry["n"] = len(members)
        for metric in metrics:
            values = np.array([float(m[metric]) for m in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            entry[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            entry[f"{metric}_std"] = float(finite.std()) if finite.size else float("nan")
        out.append(entry)
    order = list(MODEL_STYLE)
    out.sort(key=lambda e: tuple(
        (order.index(e[k]) if k == "method" and e[k] in order else (-e[k] if k == "sir_db" else e[k])) for k in keys))
    return out


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {path}")


# --------------------------------------------------------------------------- plots


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e4e3de", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a85",
        "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
        "legend.frameon": False, "font.size": 10,
    })
    return plt


def plot_metric_vs_sir(summary: Sequence[dict], metric: str, path: Path, title: str,
                       condition_key: str | None = None, dashed_condition: str | None = None) -> None:
    """One line per method over SIR. With ``condition_key``, ``dashed_condition`` is drawn faint and dashed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    conditions = sorted({row.get(condition_key) for row in summary}) if condition_key else [None]
    for condition in conditions:
        faint = condition_key is not None and condition == dashed_condition
        for method in MODEL_STYLE:
            points = sorted(((r["sir_db"], r[f"{metric}_mean"]) for r in summary
                             if r["method"] == method and (condition_key is None or r[condition_key] == condition)),
                            reverse=True)
            if not points:
                continue
            style = MODEL_STYLE[method]
            x, y = zip(*points)
            label = style["label"] if condition_key is None else f"{style['label']} ({condition})"
            ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=6, linewidth=2,
                    linestyle="--" if (method == POLLUTED or faint) else "-", alpha=0.45 if faint else 1.0,
                    label=label)
    ax.set_xlabel("Signal-to-interference ratio (dB)")
    ax.set_ylabel({"psnr": "PSNR (dB)", "ssim": "SSIM"}.get(metric, metric.upper()))
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(sorted({r["sir_db"] for r in summary}, reverse=True))
    ax.invert_xaxis()  # 0 dB (weak interference) on the left, -30 dB on the right
    ax.legend(fontsize=8, ncol=2 if condition_key else 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def magnitude_db(S, floor_db: float = -40.0) -> np.ndarray:
    mag = np.abs(np.asarray(S, dtype=np.complex128))
    peak = mag.max() if mag.max() > 0 else 1.0
    return np.maximum(20 * np.log10(np.maximum(mag / peak, 1e-12)), floor_db)


def save_image_panel(images: Sequence[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    """SAR magnitude in dB: azimuth horizontal, range vertical, as in the paper's figures."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(images), figsize=(2.6 * len(images), 4.2), constrained_layout=True, squeeze=False)
    for ax, (label, image) in zip(axes[0], images):
        im = ax.imshow(magnitude_db(image).T, cmap="gray", aspect="auto", vmin=-40, vmax=0)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Azimuth")
        ax.grid(False)
    axes[0][0].set_ylabel("Range")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="dB")
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def most_interfered_row(S) -> int:
    return int(np.argmax(np.sum(np.abs(np.asarray(S)) ** 2, axis=1)))


def save_tf_panel(rows: Sequence[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    """|STFT| in dB of one range line per method, frequency centered."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3.4), constrained_layout=True, squeeze=False)
    specs = []
    for _, line in rows:
        spec = stft_row(torch.as_tensor(np.ascontiguousarray(line).astype(np.complex64))).abs().numpy()
        specs.append(np.fft.fftshift(spec, axes=0))
    peak = max(float(s.max()) for s in specs) or 1.0
    for ax, (label, _), spec in zip(axes[0], rows, specs):
        db = np.maximum(20 * np.log10(np.maximum(spec / peak, 1e-12)), -60.0)
        im = ax.imshow(db, cmap="viridis", aspect="auto", origin="lower", vmin=-60, vmax=0)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time frame")
        ax.grid(False)
    axes[0][0].set_ylabel("Frequency bin")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="dB")
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def mean_entropy_row(case: dict, method: str, image, seconds: float) -> dict:
    return {"case": case["case"], "scene_type": case.get("scene_type", "unknown"), "method": method,
            "me": mean_entropy(image), "seconds": round(seconds, 4)}
