"""Training pieces that copy the authors' train/train_FDNet.py for the U-Net."""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch
from torch import nn

from .metrics8 import map_metrics


class CharbonnierLoss(nn.Module):
    """losses.py: mean(sqrt(diff*diff + eps*eps)), eps = 1e-3."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def seed_everything(seed: int, cudnn_benchmark: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def build_optimizer(model: nn.Module, t: dict) -> torch.optim.Optimizer:
    if t["optimizer"].lower() != "adamw":
        raise ValueError("the authors' U-Net configuration uses AdamW")
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]), eps=t["eps"],
                             weight_decay=t["weight_decay"])


def build_scheduler(opt: torch.optim.Optimizer, t: dict):
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=t["lr_step"], gamma=t["lr_gamma"])
    if t.get("scheduler_initial_step", True):
        with warnings.catch_warnings():   # "scheduler.step() before optimizer.step()" - intentional, as in the authors' code
            warnings.simplefilter("ignore", UserWarning)
            sched.step()
    return sched


def lr_schedule_preview(t: dict, epochs: int) -> list[float]:
    """Learning rate used in each epoch 1..epochs (what the authors' loop produces)."""
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=t["lr"])
    sched = build_scheduler(opt, t)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(epochs):
            out.append(opt.param_groups[0]["lr"])
            sched.step()
    return out


class Precision:
    def __init__(self, mode: str, device: torch.device):
        if mode not in ("amp", "fp32"):
            raise ValueError("precision must be amp or fp32")
        self.requested = mode
        self.enabled = mode == "amp" and device.type == "cuda"
        self.mode = "amp" if self.enabled else "fp32"
        self.device = device
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.enabled)


def train_one_epoch(model, loader, optimizer, criterion, prec: Precision, device, max_steps: int | None = None,
                    log_every: int = 0, log=print) -> dict:
    model.train()
    losses, nonfinite = [], 0
    for step, (x, y, _) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with prec.autocast():
            out = model(x)
            loss = criterion(out, y)
        if not torch.isfinite(loss):
            nonfinite += 1
        prec.scaler.scale(loss).backward()
        prec.scaler.step(optimizer)
        prec.scaler.update()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            log(f"    step {step + 1}/{len(loader)} loss {np.mean(losses[-log_every:]):.6f}")
    return {"loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_sum": float(np.sum(losses)), "steps": len(losses), "nonfinite_losses": nonfinite}


@torch.no_grad()
def predict(model, x: torch.Tensor, prec: Precision) -> torch.Tensor:
    with prec.autocast():
        return model(x).float()


@torch.no_grad()
def evaluate_loader(model, loader, criterion, prec: Precision, device, with_ssim: bool,
                    max_batches: int | None = None) -> dict:
    """Validation loss and the authors' 8-bit map metrics (output vs label)."""
    model.eval()
    losses, psnr, ssim = [], [], []
    for b, (x, y, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = predict(model, x, prec)
        losses.append(criterion(out, y).item())
        for o, t in zip(out.cpu().numpy(), y.cpu().numpy()):
            m = map_metrics(o, t) if with_ssim else None
            if m is None:
                from .metrics8 import psnr_u8, to_u8

                mx = float(t.max())
                psnr.append(psnr_u8(to_u8(o, mx)[0], to_u8(t, mx)[0]))
            else:
                psnr.append(m["psnr"])
                ssim.append(m["ssim"])
    fin = [p for p in psnr if np.isfinite(p)]
    return {"val_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_psnr": float(np.mean(fin)) if fin else float("nan"),
            "val_ssim": float(np.mean(ssim)) if ssim else None, "n_maps": len(psnr)}


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
