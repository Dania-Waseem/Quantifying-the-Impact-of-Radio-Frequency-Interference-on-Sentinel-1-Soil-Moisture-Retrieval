"""Row-wise STFT and inverse STFT for complex SAR range lines.

Algorithm 1 treats each azimuth row of a complex SAR image as a 1-D range
signal. The forward transform uses a Hamming window of length L/8 and hop
L/32 (win_len // 4), which satisfies the nonzero-overlap-add condition used
by ``torch.istft``.
"""

from __future__ import annotations

import torch


def _resolve_win_len(length: int, win_len: int | None) -> int:
    if win_len is None:
        win_len = length // 8
    win_len = int(win_len)
    if win_len < 4:
        raise ValueError(
            f"win_len must be >= 4 so hop = win_len // 4 is at least 1, got {win_len}"
        )
    return win_len


def _hamming(win_len: int, ref: torch.Tensor) -> torch.Tensor:
    """Periodic Hamming window matching the real dtype and device of ``ref``."""
    if ref.dtype == torch.complex128:
        dtype = torch.float64
    elif ref.is_complex():
        dtype = torch.float32
    else:
        dtype = ref.dtype
    return torch.hamming_window(
        win_len, periodic=True, dtype=dtype, device=ref.device
    )


def stft_row(x: torch.Tensor, win_len: int | None = None) -> torch.Tensor:
    """STFT of a complex 1-D signal, or of every row of a complex image.

    Parameters
    ----------
    x:
        Complex tensor of shape ``(L,)`` or ``(..., L)``. An image of shape
        ``(M, N)`` is transformed row by row in a single call.
    win_len:
        Hamming window length. Defaults to ``L // 8``. The hop is
        ``win_len // 4``.

    Returns
    -------
    Complex spectrogram of shape ``(..., win_len, n_frames)``.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    length = x.shape[-1]
    win_len = _resolve_win_len(length, win_len)
    hop = win_len // 4
    window = _hamming(win_len, x)
    return torch.stft(
        x,
        n_fft=win_len,
        hop_length=hop,
        win_length=win_len,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=False,
        return_complex=True,
    )


def istft_row(X: torch.Tensor, win_len: int, length: int) -> torch.Tensor:
    """Inverse of :func:`stft_row` for one line or a batch of lines.

    Parameters
    ----------
    X:
        Complex spectrogram of shape ``(..., win_len, n_frames)``.
    win_len:
        Hamming window length used by the forward STFT.
    length:
        Length ``L`` of each reconstructed range line.

    Returns
    -------
    Complex tensor of shape ``(..., length)``.
    """
    win_len = _resolve_win_len(length, win_len)
    hop = win_len // 4
    window = _hamming(win_len, X)
    return torch.istft(
        X,
        n_fft=win_len,
        hop_length=hop,
        win_length=win_len,
        window=window,
        center=True,
        normalized=False,
        onesided=False,
        length=int(length),
        return_complex=True,
    )
