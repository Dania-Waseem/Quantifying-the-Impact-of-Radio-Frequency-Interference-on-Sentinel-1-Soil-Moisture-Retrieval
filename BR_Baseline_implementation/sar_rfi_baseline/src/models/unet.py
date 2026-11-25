"""Official U-Net baseline from Fang et al. (Remote Sensing 2024 / 2025).

The paper's official U-Net architecture (from ``third_party/FDNet/model.py``
and ``extracted/Code/FDNet/model.py``, lines 107-178) is a 4-level encoder-decoder
with:
  - Base feature dimension = 32
  - Residual convolutional blocks (``ConvBlock``: two 3x3 convolutions with
    LeakyReLU activations and a 1x1 convolution shortcut, without BatchNorm)
  - Downsampling via 4x4 stride-2 convolutions with padding 1 (not maxpooling)
  - Upsampling via 2x2 stride-2 transposed convolutions (``ConvTranspose2d``)
  - Skip connections concatenating encoder features with upsampled decoder features
  - A global residual connection: ``out = x + conv10(conv9)``

The previous canonical Ronneberger U-Net (base-64, BatchNorm, MaxPool2d, no
residual skip) is preserved as ``RonnebergerUNet`` for comparison.

Input is padded to a multiple of 16 so odd spectrogram dimensions survive four
downsampling stages, then cropped back to original dimensions.
"""

from __future__ import annotations

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


class ConvBlock(nn.Module):
    """Official residual convolutional block from FDNet/UNet (model.py:83-101).

    Consists of two 3x3 convs with LeakyReLU, combined with a 1x1 conv shortcut:
        out = Conv3x3(LReLU(Conv3x3(x))) + Conv1x1(x)
    Zero BatchNorm is used.
    """

    def __init__(self, in_channel: int, out_channel: int, strides: int = 1, negative_slope: float = 0.01):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.conv11(x)


class UNet(nn.Module):
    """The paper's actual official U-Net architecture (model.py:107-178).

    - Base dimension: 32 (embed_dim)
    - 4 downsampling stages (4x4 stride-2 conv)
    - 4 upsampling stages (2x2 stride-2 transposed conv)
    - Skip connections: channel concatenation
    - Global residual: output = input + conv10
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
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1)

        # Views for test compatibility and inspectability without duplicating state_dict keys
        self.encoders = [self.ConvBlock1, self.ConvBlock2, self.ConvBlock3, self.ConvBlock4]
        self.decoders = [self.ConvBlock6, self.ConvBlock7, self.ConvBlock8, self.ConvBlock9]

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
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x_pad + conv10
        return out[..., :height, :width]


# Alias for explicit clarity
OfficialUNet = UNet


# --------------------------------------------------------------------------- Ronneberger reference
class _RonnebergerConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RonnebergerUNet(nn.Module):
    """Canonical Ronneberger U-Net (base-64, BatchNorm, MaxPool2d, no global skip).

    Kept as an alternate baseline for comparison against the paper's official UNet.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        levels: int = 4,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        self.levels = levels
        self.encoders = nn.ModuleList()
        channels = in_channels
        skip_channels: list[int] = []
        for level in range(levels):
            out_ch = base_channels * (2**level)
            self.encoders.append(_RonnebergerConvBlock(channels, out_ch))
            skip_channels.append(out_ch)
            channels = out_ch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _RonnebergerConvBlock(channels, base_channels * (2**levels))
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = base_channels * (2**levels)
        for skip_ch in reversed(skip_channels):
            self.ups.append(nn.ConvTranspose2d(channels, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(_RonnebergerConvBlock(skip_ch * 2, skip_ch))
            channels = skip_ch
        self.head = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, F, T), got {tuple(x.shape)}")
        x, height, width = _pad_to_multiple(x, 2**self.levels)
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = x[..., : skip.shape[-2], : skip.shape[-1]]
            x = decoder(torch.cat((skip, x), dim=1))
        x = self.head(x)
        return x[..., :height, :width]
