"""Training losses.

``mse`` is the shared baseline loss (Fang et al. 2025, Section 4.3).

``fdnet`` is the loss of the FDNet paper (Remote Sensing 2024, Eqs. 24-26)::

    Loss = l(X_hat, X) + lambda1 * ||X_hat||_F + lambda2 * D_loss
    l(X_hat, X) = mean( sqrt((X_hat - X)^2 + eps^2) )          (Charbonnier)
    D_loss = (1/B) * | sum_n cos(s_n, n_n) |                   (Eq. 26)

where ``s_n`` and ``n_n`` are the flattened signal and interference
features of sample ``n`` from the decomposition block.

The paper leaves ``eps``, ``lambda1`` and ``lambda2`` unspecified. Defaults
follow the released training script (``train/train_FDNet.py``):

* ``eps = 1e-3``: the ``CharbonnierLoss`` default it uses.
* ``lambda2 = 1e-5``: it adds ``0.00001 * |cos(signal, jam)|``.
* ``lambda1 = 0``: the output-norm term is commented out there, so the
  released recipe does not use it. The commented code computes
  ``0.25 * torch.norm(restored, p=2) / 512 / 512``, the Frobenius norm of
  the output over the batch divided by ``H * W``. :func:`output_norm_penalty`
  uses that form, so ``--fd-lambda1 0.25`` reproduces it.

Cosine mode. Eq. 26 averages one cosine per sample (``mode="paper"``). The
released script flattens the whole batch into one vector and takes a
single cosine (``mode="official"``). The two agree for batch size 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CharbonnierLoss(nn.Module):
    """``mean(sqrt((x - y)^2 + eps^2))``, as the official ``losses.CharbonnierLoss``."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def decomposition_loss(signal: torch.Tensor, interference: torch.Tensor, mode: str = "paper") -> torch.Tensor:
    """Absolute cosine similarity between signal and interference features (Eq. 26)."""
    if signal.shape != interference.shape:
        raise ValueError(f"feature shapes differ: {tuple(signal.shape)} vs {tuple(interference.shape)}")
    s = signal.float()
    n = interference.float()
    if mode == "paper":
        cos = F.cosine_similarity(s.flatten(1), n.flatten(1), dim=1, eps=1e-8)
        return cos.mean().abs()
    if mode == "official":
        return F.cosine_similarity(s.reshape(1, -1), n.reshape(1, -1), dim=1, eps=1e-8).abs().squeeze(0)
    raise ValueError(f"mode must be 'paper' or 'official', got {mode!r}")


def output_norm_penalty(output: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the output over the batch, divided by ``H * W`` (official commented form)."""
    H, W = output.shape[-2:]
    return torch.linalg.vector_norm(output.float()) / (H * W)


class FDNetLoss(nn.Module):
    """Eq. 24: Charbonnier reconstruction + lambda1 * output norm + lambda2 * decomposition loss."""

    def __init__(self, eps: float = 1e-3, lambda1: float = 0.0, lambda2: float = 1e-5, cosine_mode: str = "paper"):
        super().__init__()
        self.reconstruction = CharbonnierLoss(eps)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.cosine_mode = cosine_mode

    def forward(self, output, target, signal, interference) -> tuple[torch.Tensor, dict]:
        recon = self.reconstruction(output.float(), target.float())
        norm = output_norm_penalty(output)
        dloss = decomposition_loss(signal, interference, self.cosine_mode)
        total = recon + self.lambda1 * norm + self.lambda2 * dloss
        return total, {"recon": recon.detach(), "output_norm": norm.detach(), "fd_cos": dloss.detach()}


class Criterion:
    """Runs the model and returns ``(prediction, loss, parts)`` for one batch."""

    def __init__(self, kind: str, **fd_kwargs):
        if kind not in {"mse", "fdnet"}:
            raise ValueError(f"loss must be 'mse' or 'fdnet', got {kind!r}")
        self.kind = kind
        self.fd_loss = FDNetLoss(**fd_kwargs) if kind == "fdnet" else None

    def model_forward(self, model: nn.Module, x: torch.Tensor):
        if self.kind == "fdnet":
            return model(x, return_features=True)
        return model(x)

    def loss(self, outputs, target: torch.Tensor):
        if self.kind == "fdnet":
            pred, signal, interference = outputs
            total, parts = self.fd_loss(pred, target, signal, interference)
            return pred, total, parts
        pred = outputs
        total = F.mse_loss(pred.float(), target.float())
        return pred, total, {"recon": total.detach()}
