"""DnCNN and DCNet baselines for spectrogram restoration.

This module provides two non-multiscale or non-skip convolutional baselines:
1. ``DnCNN`` (Zhang et al.): Canonical 17-layer residual CNN with BatchNorm
   and global subtraction of the estimated noise residual: ``x - residual``.
2. ``DCNet`` (Fang et al., official baseline from ``third_party/FDNet/model.py:206-277``):
   A 4-level encoder-decoder using the same ``ConvBlock`` modules as the official
   U-Net, but with skip concatenations and the global residual connection removed
   (plain encoder-decoder without skip connections).
"""

from __future__ import annotations

import torch
from torch import nn

from .unet import ConvBlock, _pad_to_multiple


class DnCNN(nn.Module):
    """Residual DnCNN. ``depth`` counts convolution layers, not activations."""

    def __init__(self, in_channels: int = 2, features: int = 64, depth: int = 17):
        super().__init__()
        if depth < 3:
            raise ValueError(f"depth must be at least 3, got {depth}")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                (
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, F, T), got {tuple(x.shape)}")
        residual = self.body(x)
        return x - residual


class DCNet(nn.Module):
    """Official DCNet baseline from Fang et al. (model.py:206-277).

    An encoder-decoder structured identically to the paper's U-Net but without
    skip connections and without the global residual connection.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 32,
        block: type[nn.Module] = ConvBlock,
    ):
        super().__init__()
        self.dim = dim
        self.levels = 4

        self.ConvBlock1 = ConvBlock(in_channels, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim * 16, dim * 8, kernel_size=2, stride=2)
        self.ConvBlock6 = block(dim * 8, dim * 8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.ConvBlock7 = block(dim * 4, dim * 4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.ConvBlock8 = block(dim * 2, dim * 2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.ConvBlock9 = block(dim * 1, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, F, T), got {tuple(x.shape)}")
        x_pad, height, width = _pad_to_multiple(x, 16)

        conv1 = self.ConvBlock1(x_pad)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        return conv10[..., :height, :width]
