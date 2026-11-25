"""Baselines that map a spectrogram ``(B, 2, F, T)`` onto the same shape."""

from __future__ import annotations

import torch
from torch import nn

from .dncnn import DCNet, DnCNN
from .fdnet import FDNet
from .pisnet import IMN, PISNet
from .unet import RonnebergerUNet, UNet

OfficialUNet = UNet

# Names accepted by get_model; all are runnable.
MODEL_NAMES = (
    "dncnn",
    "dcnet",
    "unet",
    "official_unet",
    "ronneberger_unet",
    "pisnet",
    "imn",
    "fdnet",
)

IMPLEMENTED_MODELS = (
    "dncnn",
    "dcnet",
    "unet",
    "ronneberger_unet",
    "pisnet",
    "imn",
    "fdnet",
)

_BUILDERS = {
    "dncnn": DnCNN,
    "dcnet": DCNet,
    "unet": UNet,
    "official_unet": UNet,
    "ronneberger_unet": RonnebergerUNet,
    "pisnet": PISNet,
    "imn": IMN,
    "fdnet": FDNet,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Build one baseline by name.

    - ``unet`` / ``official_unet``: Official 4-level U-Net from Fang et al.
      (base dim 32, residual ConvBlocks, 4x4 stride-2 convs, ConvTranspose2d,
      skip concatenations, global residual ``x + conv10``, zero BatchNorm).
    - ``ronneberger_unet``: Canonical Ronneberger U-Net with BatchNorm,
      MaxPool2d downsampling, and bilinear upsampling.
    - ``dcnet``: Official DCNet from model.py:206-277 (encoder-decoder matching
      official U-Net but without skip connections and without global residual).
    - ``dncnn``: Canonical 17-layer residual CNN with BatchNorm.
    - ``pisnet``: Fang et al. variant with the final sigmoid removed.
    - ``imn``: 19-layer residual CNN baseline from Fang et al. / Shen et al.
    - ``fdnet``: FDUformer port in :mod:`models.fdnet` (C=16, 8x8 windows,
      2 input channels for the real/imaginary spectrogram).
    """
    key = name.strip().lower()
    try:
        builder = _BUILDERS[key]
    except KeyError as exc:
        known = ", ".join(MODEL_NAMES)
        raise KeyError(f"unknown model {name!r}; expected one of {known}") from exc
    return builder(**kwargs)


def parameter_count(model: nn.Module) -> int:
    """Number of trainable and frozen parameters, excluding buffers."""
    return sum(parameter.numel() for parameter in model.parameters())


def format_parameter_counts(device: str | torch.device = "cpu") -> str:
    """One line per runnable baseline: name, parameter count, and a forward shape check."""
    probe = torch.randn(1, 2, 17, 13, device=device)
    lines = []
    for name in IMPLEMENTED_MODELS:
        model = get_model(name).to(device)
        model.eval()
        with torch.no_grad():
            output = model(probe)
        if output.shape != probe.shape:
            raise RuntimeError(f"{name} changed shape from {tuple(probe.shape)} to {tuple(output.shape)}")
        lines.append(f"{name:16s} {parameter_count(model):10,d} parameters")
    return "\n".join(lines)


__all__ = [
    "DCNet",
    "DnCNN",
    "FDNet",
    "IMPLEMENTED_MODELS",
    "IMN",
    "MODEL_NAMES",
    "OfficialUNet",
    "PISNet",
    "RonnebergerUNet",
    "UNet",
    "format_parameter_counts",
    "get_model",
    "parameter_count",
]
