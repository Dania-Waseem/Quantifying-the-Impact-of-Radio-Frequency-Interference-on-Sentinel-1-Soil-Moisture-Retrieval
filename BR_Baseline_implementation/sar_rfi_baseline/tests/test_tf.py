"""Round-trip tests for the complex row-wise STFT."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.tf import istft_row, stft_row


def test_stft_istft_roundtrip():
    torch.manual_seed(0)
    length = 1500
    x = torch.randn(length) + 1j * torch.randn(length)
    win_len = length // 8
    spec = stft_row(x, win_len)
    recon = istft_row(spec, win_len, length)
    error = torch.max(torch.abs(recon - x)).item()
    assert recon.shape == x.shape
    assert error < 1e-5, f"reconstruction error {error} is not below 1e-5"


def test_stft_istft_image_rows():
    """A complex image (M, N) is processed row-wise in one call."""
    torch.manual_seed(1)
    rows, cols = 6, 512
    image = torch.randn(rows, cols) + 1j * torch.randn(rows, cols)
    spec = stft_row(image)
    recon = istft_row(spec, cols // 8, cols)
    error = torch.max(torch.abs(recon - image)).item()
    assert spec.shape[0] == rows
    assert recon.shape == image.shape
    assert error < 1e-5, f"image reconstruction error {error} is not below 1e-5"


if __name__ == "__main__":
    test_stft_istft_roundtrip()
    test_stft_istft_image_rows()
    print("test_tf: passed")
