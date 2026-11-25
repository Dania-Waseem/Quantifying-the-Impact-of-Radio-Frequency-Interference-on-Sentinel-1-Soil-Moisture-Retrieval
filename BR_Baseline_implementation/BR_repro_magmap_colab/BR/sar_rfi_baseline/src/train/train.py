"""Train one spectrogram baseline for block-type RFI suppression.

Section 4.3 of Fang et al. fixes the optimizer and schedule: AdamW, initial
learning rate 2e-4, batch size 4, 100 epochs. The loss is the MSE between the
predicted and clean time-frequency maps, over the real and imaginary channels.

Experiment 2 (Section 5.2) trains and validates at one SIR, -20 dB, so both
splits share a distribution. ``--sir`` picks that level and the loop refuses
to start if either split holds any other SIR. ``--sir`` takes several levels
for experiment 1, which trains on the whole constructed set.

FDNet uses the recipe of its own paper unless overridden: AdamW with lr
2e-3 and weight decay 0.02, and the Eq. 24 loss (Charbonnier
reconstruction, output-norm term, cosine decomposition term; see
:mod:`train.losses`). From the official code it also takes StepLR (halve
every 50 epochs), which the paper does not state. For the shared Fang 2025
setting instead, pass ``--loss mse --lr 2e-4 --weight-decay 0.01 --lr-step 0``.

``--source rfi_patch`` reads the downloaded RFI-Patch files (``ss`` stacks;
see :func:`data.dataset.discover_rfi_patch_records`). Their levels are the
dataset's documented SNR labels, not measured SIRs, so they are chosen with
``--snr-db`` (default -20) and ``--sir`` is refused.

``--rfi-params`` picks a named interference parameter set from
:mod:`data.param_sets` for the synthetic source (experiment 3 trains on
``exp3_train``).

Orientation
-----------
Stored slices are ``512 range x 1500 azimuth``: axis 0 is range
(``--range-axis 0``, the default; see ``docs/AXIS_ORIENTATION.md``). The
dataset transposes each image once into ``1500`` range lines of ``512``
samples and applies the STFT with a 64-sample Hamming window and a 16-sample
hop (``--stft-win-len`` / ``--stft-hop`` override; defaults are ``L // 8`` and
``win // 4``). The resolved settings are written to ``config.json`` and to
every checkpoint. ``--range-axis 1`` reproduces runs made before the
2026-11-23 correction (STFT along the 1500-sample azimuth axis).

Batching
--------
A dataset item is one SAR slice: ``M`` azimuth positions, each a range line
with a ``(2, F, T)`` spectrogram (``1500 x (2, 64, 33)`` for RFI-Patch). The
network sees one range line at a time (Algorithm 1, Step III). The paper's
batch size of 4 is read here as 4 slices per step. Training draws
``--rows-per-image`` random range lines from each slice, so a step holds
``batch_size * rows_per_image`` spectrograms. The paper does not say how
lines were grouped; set ``--rows-per-image 0`` to use every line.

Validation runs every range line of every validation slice, inverts the STFT,
undoes the peak normalization, transposes back to the stored layout, and
scores the SAR image with Eq. 23 PSNR on magnitudes. The checkpoint with the
best mean validation PSNR is kept.

Examples
--------
Smoke test (20 synthetic slices, 2 epochs, CPU is fine)::

    python src/train/train.py --smoke --model dncnn

Experiment 2 on the released .mat slices::

    python src/train/train.py --model unet --source mat --root data/lab_mat \
        --split-file data/splits.json --sir -20

Resume an interrupted run::

    python src/train/train.py --model unet --source mat --root data/lab_mat \
        --split-file data/splits.json --resume auto
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.dataset import (
    SCENE_TYPES,
    RFIDataset,
    assign_scene_splits,
    channels_to_complex,
    denormalize,
    load_split_file,
    write_split_file,
)
from data.param_sets import PARAM_SETS, get_param_set
from eval.metrics import psnr
from train.losses import Criterion
from models import IMPLEMENTED_MODELS, MODEL_NAMES, get_model, parameter_count
from utils.orientation import RANGE_AXIS, check_range_axis, checkpoint_range_axis
from utils.seed import set_seed
from utils.tf import istft_row

PROJECT_ROOT = SRC.parent
CSV_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_psnr",
    "input_psnr",
    "best_val_psnr",
    "train_recon",
    "train_fd_cos",
    "lr",
    "epoch_time_s",
)


# --------------------------------------------------------------------------- data


class RowSampler(Dataset):
    """Wrap an :class:`RFIDataset` and return ``rows`` random range lines (azimuth positions) per slice."""

    def __init__(self, base: Dataset, rows: int):
        self.base = base
        self.rows = int(rows)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        x, y = item["polluted"], item["clean"]
        if 0 < self.rows < x.shape[0]:
            keep = torch.randperm(x.shape[0])[: self.rows]
            x, y = x[keep], y[keep]
        return x, y


def collate_rows(batch):
    """Stack the rows of every slice in the batch into one ``(B*rows, 2, F, T)`` tensor."""
    xs, ys = zip(*batch)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def collate_full(batch):
    """Validation keeps one slice per batch with its scale and record index."""
    if len(batch) != 1:
        raise ValueError("validation loader must use batch_size=1")
    return batch[0]


class IndexedDataset(Dataset):
    def __init__(self, base: RFIDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        item["index"] = index
        return item


def make_smoke_scenes(n_scenes: int, shape: tuple[int, int], seed: int) -> list[dict]:
    """Small synthetic complex scenes: speckle over a smooth map plus bright points.

    ``shape`` is the stored layout, like a real slice: ``(range, azimuth)``
    when ``--range-axis 0``. The default ``(256, 64)`` gives 64 range lines of
    256 samples, the same line count and window as the pre-correction smoke run.
    """
    rng = np.random.default_rng(seed)
    n0, n1 = shape
    az = np.linspace(0.0, 1.0, n0)[:, None]
    rg = np.linspace(0.0, 1.0, n1)[None, :]
    scenes = []
    for index in range(n_scenes):
        fx, fy = rng.uniform(1.0, 4.0, size=2)
        reflectivity = 0.6 + 0.4 * np.sin(2 * np.pi * fx * az + rng.uniform(0, 2 * np.pi)) * np.cos(
            2 * np.pi * fy * rg
        )
        speckle = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
        image = reflectivity * speckle
        for _ in range(int(rng.integers(3, 9))):
            i, j = int(rng.integers(0, n0)), int(rng.integers(0, n1))
            image[i, j] += rng.uniform(5.0, 15.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        scenes.append(
            {
                "image": image.astype(np.complex64),
                "scene_id": f"smoke_{index:03d}",
                "scene_type": SCENE_TYPES[index % len(SCENE_TYPES)],
            }
        )
    return scenes


def _subset(dataset: RFIDataset, records) -> RFIDataset:
    view = copy.copy(dataset)
    view.records = list(records)
    return view


def _limit(records, limit: int | None, seed: int):
    if not limit or limit >= len(records):
        return list(records)
    order = np.random.default_rng(seed).permutation(len(records))[:limit]
    return [records[int(i)] for i in sorted(order)]


def build_datasets(args, run_dir: Path):
    """Return train and validation views that hold only scenes of their split at ``args.sir``."""
    rfi = get_param_set(args.rfi_params)
    if args.smoke:
        cleans = make_smoke_scenes(args.smoke_scenes, tuple(args.smoke_shape), args.seed)
        full = RFIDataset(
            "synthetic",
            clean_images=cleans,
            sir_levels=tuple(args.sir),
            presets=rfi["presets"],
            n_repeats=1,
            seed=args.seed,
            param_ranges=rfi["ranges"],
            range_axis=args.range_axis, win_len=args.stft_win_len, hop=args.stft_hop,
        )
    elif args.source == "synthetic":
        if args.root is None:
            raise SystemExit("--source synthetic needs --root with clean {scene_id}__{scene_type}.npy images")
        full = RFIDataset(
            "synthetic",
            root=args.root,
            sir_levels=tuple(args.sir),
            presets=rfi["presets"],
            n_repeats=args.n_repeats,
            seed=args.seed,
            param_ranges=rfi["ranges"],
            range_axis=args.range_axis, win_len=args.stft_win_len, hop=args.stft_hop,
        )
    elif args.source == "rfi_patch":
        if args.rfi_params != "default":
            raise SystemExit("--rfi-params applies to synthetic data; RFI-Patch files have fixed interference")
        if args.root is None:
            raise SystemExit("--source rfi_patch needs --root, the directory of RFI-Patch .mat files")
        full = RFIDataset("rfi_patch", root=args.root, range_axis=args.range_axis, win_len=args.stft_win_len, hop=args.stft_hop)
    else:
        if args.rfi_params != "default":
            raise SystemExit("--rfi-params applies to synthetic data; the .mat slices have fixed interference")
        if args.root is None:
            raise SystemExit("--source mat needs --root, the directory of released .mat slices")
        full = RFIDataset("mat", root=args.root, range_axis=args.range_axis, win_len=args.stft_win_len, hop=args.stft_hop)

    # Experiment 2: only the chosen level, in both splits. RFI-Patch levels are documented SNR labels.
    patch = args.source == "rfi_patch"
    level_name = "documented SNR" if patch else "SIR"
    level = (lambda r: r.snr_db_documented) if patch else (lambda r: r.sir_db)
    wanted = [float(v) for v in (args.snr_db if patch else args.sir)]
    records = [r for r in full.records if any(math.isclose(level(r), v, abs_tol=1e-6) for v in wanted)]
    if not records:
        found = sorted({level(r) for r in full.records})
        raise SystemExit(f"no slices at {level_name} {wanted} dB; the data holds {found}")

    scene_ids = list(dict.fromkeys(r.scene_id for r in records))
    if args.split_file is not None:
        split_map = load_split_file(args.split_file)
        missing = sorted(set(scene_ids) - set(split_map))
        if missing:
            raise SystemExit(f"split file has no entry for scenes {missing[:10]}")
    else:
        split_map = assign_scene_splits(scene_ids, args.val_ratio, args.test_ratio, args.seed)
    write_split_file(run_dir / "splits.json", {s: split_map[s] for s in scene_ids}, args.val_ratio, args.test_ratio, args.seed)

    train_records = [r for r in records if split_map[r.scene_id] == "train"]
    val_records = [r for r in records if split_map[r.scene_id] == "val"]
    train_records = _limit(train_records, args.limit_train, args.seed)
    val_records = _limit(val_records, args.limit_val, args.seed + 1)
    if not train_records or not val_records:
        raise SystemExit(
            f"empty split: {len(train_records)} train and {len(val_records)} val slices; "
            "add scenes or change --val-ratio"
        )

    train_sirs = {level(r) for r in train_records}
    val_sirs = {level(r) for r in val_records}
    if len(wanted) == 1 and not (train_sirs == val_sirs == {wanted[0]}):
        raise SystemExit(f"train {level_name}s {train_sirs} and val {level_name}s {val_sirs} "
                         f"must both be exactly {wanted[0]} dB")
    if not (train_sirs | val_sirs) <= set(wanted):
        raise SystemExit(f"slices outside the requested {level_name}s {wanted}: "
                         f"{sorted((train_sirs | val_sirs) - set(wanted))}")
    overlap = {r.scene_id for r in train_records} & {r.scene_id for r in val_records}
    if overlap:
        raise SystemExit(f"scenes leak across train and val: {sorted(overlap)[:10]}")

    return _subset(full, train_records), _subset(full, val_records)


# --------------------------------------------------------------------------- amp


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    return torch.cuda.amp.autocast()


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- loops


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, grad_clip, log_every, writer, global_step,
                    criterion: Criterion | None = None):
    """One pass over the training loader. Returns mean loss, global step, and mean loss parts."""
    criterion = criterion or Criterion("mse")
    model.train()
    total, count = 0.0, 0
    part_sums: dict[str, float] = {}
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = criterion.model_forward(model, x)
        # Losses in float32, outside autocast.
        _, loss, parts = criterion.loss(outputs, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch = x.shape[0]
        total += loss.item() * batch
        count += batch
        for key, value in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + float(value) * batch
        global_step += 1
        if writer is not None:
            writer.add_scalar("loss/train_step", loss.item(), global_step)
        if log_every and step % log_every == 0:
            print(f"    step {step}/{len(loader)}  loss {loss.item():.6f}", flush=True)
    parts_mean = {key: value / max(count, 1) for key, value in part_sums.items()}
    return total / max(count, 1), global_step, parts_mean


@torch.no_grad()
def validate(model, loader, dataset: RFIDataset, device, use_amp, row_chunk):
    """Mean TF-domain MSE plus mean image-domain PSNR of output and input.

    Each prediction is inverted with the item's own window and hop, back to
    ``(n_lines, n_range)``, then transposed once to the stored layout before
    scoring.
    """
    model.eval()
    sq_err, n_elem = 0.0, 0
    out_psnrs, in_psnrs = [], []
    for item in loader:
        x, y, scale = item["polluted"], item["clean"], item["scale"]
        preds = []
        for start in range(0, x.shape[0], row_chunk):
            xb = x[start : start + row_chunk].to(device, non_blocking=True)
            yb = y[start : start + row_chunk].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                pb = model(xb)
            pb = pb.float()
            sq_err += F.mse_loss(pb, yb.float(), reduction="sum").item()
            n_elem += yb.numel()
            preds.append(pb.cpu())
        pred = torch.cat(preds, dim=0)

        meta = item["metadata"]
        win_len, hop, length = int(meta["win_len"]), int(meta["hop"]), int(meta["n_range"])
        if win_len != x.shape[-2]:
            raise RuntimeError(f"item window {win_len} does not match spectrogram height {x.shape[-2]}")
        to_image = lambda spec: dataset.to_image(
            istft_row(denormalize(channels_to_complex(spec), scale), win_len, length, hop))
        clean_img = to_image(y)
        if tuple(clean_img.shape) != tuple(meta["image_shape"]):
            raise RuntimeError(f"reconstruction {tuple(clean_img.shape)} != stored {tuple(meta['image_shape'])}")
        out_psnrs.append(psnr(clean_img, to_image(pred)))
        in_psnrs.append(psnr(clean_img, to_image(x)))
    return sq_err / max(n_elem, 1), float(np.mean(out_psnrs)), float(np.mean(in_psnrs))


# --------------------------------------------------------------------------- io


def _read_csv_rows(path: Path, upto_epoch: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if int(row["epoch"]) <= upto_epoch]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(path: Path, **state) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _make_writer(run_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # tensorboard is an optional install
        warnings.warn(f"TensorBoard logging is off ({exc}); install it with 'pip install tensorboard'")
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


# --------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a block-type RFI suppression baseline (Fang et al. 2025).")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--source", choices=("mat", "synthetic", "rfi_patch"), default="mat")
    p.add_argument("--root", type=Path, help="Released .mat slices, clean .npy scenes for --source synthetic, "
                                             "or RFI-Patch .mat files for --source rfi_patch")
    p.add_argument("--split-file", type=Path, help="scene_id -> split JSON from scripts/split_scenes.py")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--sir", type=float, nargs="+", default=None,
                   help="SIR level(s) in dB for both train and val (experiment 2: -20; the default)")
    p.add_argument("--snr-db", type=float, nargs="+", default=None,
                   help="--source rfi_patch only: documented SNR level(s) 0, -10, -20, -30 (default -20)")
    p.add_argument("--rfi-params", choices=sorted(PARAM_SETS), default="default",
                   help="Interference parameter set for synthetic data (experiment 3: exp3_train)")
    p.add_argument("--n-repeats", type=int, default=1, help="RFI draws per scene and preset for --source synthetic")
    p.add_argument("--limit-train", type=int, default=None, help="Cap on training slices")
    p.add_argument("--limit-val", type=int, default=None, help="Cap on validation slices")

    p.add_argument("--epochs", type=int, default=None, help="Default 100 (2 with --smoke)")
    p.add_argument("--batch-size", type=int, default=4, help="Slices per optimizer step")
    p.add_argument("--rows-per-image", type=int, default=None, help="Random rows per slice in training; 0 = all (default 32, 8 for fdnet, 16 with --smoke)")
    p.add_argument("--val-row-chunk", type=int, default=128, help="Rows per forward pass during validation")
    p.add_argument("--lr", type=float, default=None, help="Default 2e-4 (Fang 2025); 2e-3 for fdnet (FDNet paper)")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Default 1e-2 (AdamW default; Fang 2025 does not state it); 0.02 for fdnet (FDNet paper)")
    p.add_argument("--loss", choices=("auto", "mse", "fdnet"), default="auto",
                   help="auto: fdnet loss for --model fdnet, MSE otherwise")
    p.add_argument("--lr-step", type=int, default=None,
                   help="StepLR period in epochs, 0 = constant. Default 0; 50 for fdnet (official FDNet code)")
    p.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR factor (official FDNet code: 0.5)")
    p.add_argument("--fd-eps", type=float, default=1e-3, help="Charbonnier eps (official code 1e-3)")
    p.add_argument("--fd-lambda1", type=float, default=0.0,
                   help="Output-norm weight; 0 in the released script, 0.25 in its commented-out code")
    p.add_argument("--fd-lambda2", type=float, default=1e-5, help="Decomposition-loss weight (official code 1e-5)")
    p.add_argument("--fd-cos", choices=("paper", "official"), default="paper",
                   help="Eq. 26 per-sample cosine (paper) or one cosine over the flattened batch (official script)")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm, 0 disables")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision on CUDA")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--num-workers", type=int, default=None, help="Default 2 (0 with --smoke)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-dir", type=Path, default=None, help="Default outputs/runs/<model>_sir<SIR>[_smoke]")
    p.add_argument("--resume", default=None, help="Path to a checkpoint, or 'auto' for <out-dir>/last.pt")
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=50, help="Print every N steps, 0 disables")

    p.add_argument("--smoke", action="store_true", help="20 synthetic slices, 2 epochs, to check the loop runs")
    p.add_argument("--smoke-scenes", type=int, default=20)
    p.add_argument("--smoke-shape", type=int, nargs=2, default=(256, 64), metavar=("DIM0", "DIM1"),
                   help="Stored layout of the smoke scenes; with --range-axis 0 that is RANGE AZIMUTH")

    p.add_argument("--range-axis", type=int, choices=(0, 1), default=RANGE_AXIS,
                   help="Axis of the stored image that is range. 0 (default): RFI-Patch 512 range x 1500 "
                        "azimuth, STFT along the 512 axis. 1: the pre-2026-11-23 behaviour (STFT along axis 1)")
    p.add_argument("--stft-win-len", type=int, default=None,
                   help="Hamming window length; default L // 8 of the range length (64 for RFI-Patch)")
    p.add_argument("--stft-hop", type=int, default=None,
                   help="STFT hop; default win // 4 (16 for RFI-Patch)")
    args = p.parse_args(argv)

    if args.smoke:
        args.source = "synthetic"
        args.epochs = 2 if args.epochs is None else args.epochs
        args.rows_per_image = 16 if args.rows_per_image is None else args.rows_per_image
        args.num_workers = 0 if args.num_workers is None else args.num_workers
        args.val_ratio = 0.2
        args.test_ratio = 0.2  # keeps 4 scenes back for the experiment scripts' smoke runs
        # 20 samples in all: 16 train, 4 val.
        total = args.smoke_scenes
        args.limit_train = args.limit_train or total - total // 5
        args.limit_val = args.limit_val or total // 5
        args.log_every = 0 if args.log_every == 50 else args.log_every
    # RFI-Patch levels are documented SNR labels and are never passed off as SIR.
    if args.source == "rfi_patch":
        if args.sir is not None:
            raise SystemExit("--sir does not apply to --source rfi_patch: its levels are documented SNR "
                             "labels, not measured SIR; use --snr-db")
        args.snr_db = [-20.0] if args.snr_db is None else args.snr_db
    else:
        if args.snr_db is not None:
            raise SystemExit("--snr-db applies only to --source rfi_patch")
        args.sir = [-20.0] if args.sir is None else args.sir
    args.epochs = 100 if args.epochs is None else args.epochs
    # FDNet trains with its own paper's recipe unless overridden; every other model keeps the shared one.
    fd = args.model == "fdnet"
    args.loss = ("fdnet" if fd else "mse") if args.loss == "auto" else args.loss
    args.lr = (2e-3 if fd else 2e-4) if args.lr is None else args.lr
    args.weight_decay = (0.02 if fd else 1e-2) if args.weight_decay is None else args.weight_decay
    args.lr_step = (50 if fd else 0) if args.lr_step is None else args.lr_step
    if args.loss == "fdnet" and args.model != "fdnet":
        raise SystemExit("--loss fdnet needs the decomposition features, which only --model fdnet returns")
    # Rows per slice: 32 for most baselines, 8 for FDNet. The FDNet value was chosen when a
    # line map was 187x33 (padded to 256x128): 4 slices x 8 lines then matched the official
    # 4 x 512 x 512 pixels per step. With --range-axis 0 a line map is 64x33 (padded to
    # 128x128), so that equivalence no longer holds; revisit before training FDNet.
    args.rows_per_image = (8 if fd else 32) if args.rows_per_image is None else args.rows_per_image
    args.num_workers = 2 if args.num_workers is None else args.num_workers
    args.range_axis = check_range_axis(args.range_axis)
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.model not in IMPLEMENTED_MODELS:
        raise SystemExit(f"--model {args.model} is not implemented; choose one of {', '.join(IMPLEMENTED_MODELS)}.")
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    use_amp = bool(args.amp and device.type == "cuda")

    if args.source == "rfi_patch":
        level_tag = "rfipatch_snr" + "_".join(f"{v:g}" for v in args.snr_db)
        level_text = f"documented SNR {args.snr_db} dB"
    else:
        level_tag = "sir" + "_".join(f"{v:g}" for v in args.sir)
        level_text = f"SIR {args.sir} dB"
    run_dir = args.out_dir or PROJECT_ROOT / "outputs" / "runs" / f"{args.model}_{level_tag}{'_smoke' if args.smoke else ''}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(args, run_dir)
    # Resolved orientation and STFT settings; every slice of both splits must share them.
    stft = train_set.stft_settings(0)
    for view in (train_set, val_set):
        for i in range(len(view)):
            other = view.stft_settings(i)
            if (other["n_range"], other["win_len"], other["hop"]) != (stft["n_range"], stft["win_len"], stft["hop"]):
                raise SystemExit(f"slices disagree on range length / window / hop: {stft} vs {other}")
    stft = {key: (list(value) if isinstance(value, tuple) else value) for key, value in stft.items()}
    loader_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(
        RowSampler(train_set, args.rows_per_image),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_rows,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kw,
    )
    val_loader = DataLoader(IndexedDataset(val_set), batch_size=1, shuffle=False, collate_fn=collate_full, **loader_kw)

    model = get_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
                 if args.lr_step > 0 else None)
    scaler = _grad_scaler(use_amp)
    criterion = Criterion(args.loss, eps=args.fd_eps, lambda1=args.fd_lambda1, lambda2=args.fd_lambda2,
                          cosine_mode=args.fd_cos) if args.loss == "fdnet" else Criterion("mse")

    start_epoch, best_psnr, global_step = 1, -math.inf, 0
    resume_path = None
    if args.resume:
        resume_path = run_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            if args.resume == "auto":
                print(f"no checkpoint at {resume_path}; starting fresh")
                resume_path = None
            else:
                raise SystemExit(f"checkpoint {resume_path} not found")
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["model_name"] != args.model:
            raise SystemExit(f"checkpoint is for {state['model_name']}, not {args.model}")
        saved_args = state.get("args") or {}
        saved_axis = checkpoint_range_axis(saved_args)
        if saved_axis != args.range_axis:
            raise SystemExit(f"checkpoint was trained with range axis {saved_axis}, this run uses "
                             f"--range-axis {args.range_axis}; resuming would mix two STFT orientations")
        saved_stft = saved_args.get("stft") or {}
        for key in ("win_len", "hop"):
            if key in saved_stft and int(saved_stft[key]) != int(stft[key]):
                raise SystemExit(f"checkpoint STFT {key}={saved_stft[key]} differs from this run's {stft[key]}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") and use_amp:
            scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_psnr = float(state["best_val_psnr"])
        global_step = int(state.get("global_step", 0))
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed from {resume_path} at epoch {start_epoch} (best val PSNR {best_psnr:.3f} dB)")

    csv_path = run_dir / "metrics.csv"
    rows = _read_csv_rows(csv_path, start_epoch - 1)
    _write_csv(csv_path, rows)
    writer = _make_writer(run_dir, args.tensorboard)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(stft=stft)
    config.update(device=str(device), amp=use_amp, n_train=len(train_set), n_val=len(val_set),
                  parameters=parameter_count(model))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")

    print(
        f"{args.model}: {parameter_count(model):,} parameters on {device} (amp={use_amp}); "
        f"{len(train_set)} train / {len(val_set)} val slices at {level_text}; "
        f"{args.batch_size} slices x {args.rows_per_image or 'all'} rows per step; loss {args.loss}, AdamW lr "
        f"{args.lr:g} wd {args.weight_decay:g}{f', StepLR {args.lr_step}x{args.lr_gamma:g}' if args.lr_step else ''}; "
        f"run dir {run_dir}",
        flush=True,
    )
    print(
        f"orientation: range axis {stft['range_axis']} of stored {tuple(stft['image_shape'])}; "
        f"{stft['n_lines']} range lines x {stft['n_range']} samples; STFT Hamming win {stft['win_len']} "
        f"hop {stft['hop']} -> item {tuple(stft['spectrogram'])}",
        flush=True,
    )

    last_val_psnr = None
    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        lr = optimizer.param_groups[0]["lr"]  # the rate this epoch trains with
        train_loss, global_step, parts = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, args.grad_clip, args.log_every, writer,
            global_step, criterion,
        )
        if scheduler is not None:
            scheduler.step()
        val_loss, val_psnr, input_psnr = validate(model, val_loader, val_set, device, use_amp, args.val_row_chunk)
        elapsed = time.time() - tic
        improved = val_psnr > best_psnr
        best_psnr = max(best_psnr, val_psnr)
        last_val_psnr = val_psnr

        row = dict(epoch=epoch, train_loss=f"{train_loss:.8f}", val_loss=f"{val_loss:.8f}",
                   val_psnr=f"{val_psnr:.4f}", input_psnr=f"{input_psnr:.4f}",
                   best_val_psnr=f"{best_psnr:.4f}", train_recon=f"{parts.get('recon', float('nan')):.8f}",
                   train_fd_cos=f"{parts['fd_cos']:.6f}" if "fd_cos" in parts else "",
                   lr=f"{lr:.3g}", epoch_time_s=f"{elapsed:.1f}")
        rows.append(row)
        _write_csv(csv_path, rows)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("psnr/val", val_psnr, epoch)
            writer.add_scalar("psnr/val_input", input_psnr, epoch)
            writer.add_scalar("lr", lr, epoch)
            for key, value in parts.items():
                writer.add_scalar(f"loss_parts/{key}", value, epoch)
            writer.flush()

        state = dict(
            epoch=epoch, model_name=args.model, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict() if use_amp else None,
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            best_val_psnr=best_psnr, val_psnr=val_psnr,
            global_step=global_step, torch_rng=torch.get_rng_state(), args=config,
        )
        _save_checkpoint(run_dir / "last.pt", **state)
        if improved:
            _save_checkpoint(run_dir / "best.pt", **state)
        print(
            f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.6f}  val {val_loss:.6f}  "
            f"PSNR {val_psnr:.3f} dB (input {input_psnr:.3f})  best {best_psnr:.3f}"
            f"{'  *saved best*' if improved else ''}  {elapsed:.1f}s",
            flush=True,
        )

    if writer is not None:
        writer.close()
    return {"run_dir": run_dir, "best_val_psnr": best_psnr, "last_val_psnr": last_val_psnr, "epochs_logged": len(rows)}


if __name__ == "__main__":
    main()
