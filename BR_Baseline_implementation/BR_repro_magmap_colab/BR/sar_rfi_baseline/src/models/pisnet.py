"""PISNet and IMN baselines from Shen et al. / Fang et al.

This module provides two baselines originating from the Shen et al. pipeline:
1. ``PISNet``: A time-frequency network mapping spectrograms through a stride-2
   encoder, residual blocks, an expectation-maximization attention unit (EMAU)
   that rebuilds features on a compact basis set, and a transposed conv decoder.
   Fang et al. drop the final sigmoid activation because signed complex
   spectrograms have real and imaginary components that should not be squashed
   to (0, 1).
2. ``IMN`` (Interference Mitigation Network): The 19-layer residual CNN
   from ``extracted/Code/FDNet/network.py:203-360`` and ``PISNet+IMN/src/network.py``.
   Consists of an initial Conv+ReLU, 16 residual blocks with BatchNorm, an
   aggregation layer with a global skip connection from layer 1, and an output Conv.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, height, width


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _EMReconstruction(nn.Module):
    """Rebuild features as a combination of ``bases`` learned bases.

    The assignment and the bases are updated for ``stage_num`` iterations
    without tracking gradients, matching the EM attention unit PISNet uses
    for low-rank reorganization. A residual path around the unit keeps a
    gradient route into the encoder. ``mu`` is the initial basis set; the
    original training loop replaces it with a moving average.
    """

    def __init__(self, channels: int, bases: int = 128, stage_num: int = 16):
        super().__init__()
        if stage_num < 1:
            raise ValueError(f"stage_num must be at least 1, got {stage_num}")
        self.stage_num = stage_num
        mu = torch.randn(1, channels, bases) * math.sqrt(2.0 / bases)
        mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-6)
        self.register_buffer("mu", mu)
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.rebuild = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        identity = features
        features = self.project(features)
        batch, channels, height, width = features.shape
        flat = features.reshape(batch, channels, height * width)
        # The EM loop estimates bases from the current features. Both the
        # bases and the assignment are treated as constants, as in the
        # published attention unit, so the gradient goes through ``identity``.
        with torch.no_grad():
            bases = self.mu.expand(batch, -1, -1).clone()
            assignment = None
            for _ in range(self.stage_num):
                logits = torch.bmm(flat.transpose(1, 2), bases)
                assignment = torch.softmax(logits, dim=2)
                weights = assignment / (assignment.sum(dim=1, keepdim=True) + 1e-6)
                bases = torch.bmm(flat, weights)
                bases = bases / (bases.norm(dim=1, keepdim=True) + 1e-6)
        restored = torch.bmm(bases, assignment.transpose(1, 2))
        restored = restored.reshape(batch, channels, height, width)
        restored = self.rebuild(F.relu(restored))
        return F.relu(identity + restored)


class PISNet(nn.Module):
    """PISNet with the final sigmoid removed (the Fang et al. modification)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        residual_blocks: int = 8,
        bases: int = 128,
        stage_num: int = 16,
    ):
        super().__init__()
        if residual_blocks < 1:
            raise ValueError(f"residual_blocks must be at least 1, got {residual_blocks}")
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(*[_ResidualBlock(256) for _ in range(residual_blocks)])
        self.reconstruct = _EMReconstruction(256, bases=bases, stage_num=stage_num)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, F, T), got {tuple(x.shape)}")
        x, height, width = _pad_to_multiple(x, 4)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.reconstruct(x)
        x = self.decoder(x)
        return x[..., :height, :width]


class IMN(nn.Module):
    """Interference Mitigation Network (IMN) baseline from Fang et al. / Shen et al.

    A 19-layer residual convolutional network:
    - Layer 1: Conv2d(in_channels, dim, 3, 1, 1) + ReLU
    - Blocks 1..16: 16 residual blocks with Conv2d + BatchNorm2d + ReLU + Conv2d + BatchNorm2d
    - Layer 18: Conv2d(dim, dim, 3, 1, 1) + BatchNorm2d, combined with global skip:
                x18 = x1 + layer18(x17)
    - Layer 19: Conv2d(dim, out_channels, 3, 1, 1)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int | None = None,
    ):
        super().__init__()
        channel_dim = dim if dim is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = channel_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        for i in range(1, 17):
            block = nn.Sequential(
                nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channel_dim, eps=1e-5, momentum=0.1),
            )
            setattr(self, f"blocks{i}", block)

        self.layer18 = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_dim, eps=1e-5, momentum=0.1),
        )
        self.layer19 = nn.Sequential(
            nn.Conv2d(channel_dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, F, T), got {tuple(x.shape)}")
        x1 = self.layer1(x)

        cur = x1
        for i in range(1, 17):
            block = getattr(self, f"blocks{i}")
            cur = cur + block(cur)

        x18 = x1 + self.layer18(cur)
        out = self.layer19(x18)
        return out
