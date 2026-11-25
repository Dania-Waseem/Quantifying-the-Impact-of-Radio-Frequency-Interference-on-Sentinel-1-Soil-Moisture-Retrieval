"""STFT and inverse STFT of complex 1-D range signals.

Algorithm 1 transforms each 1-D *range* signal of a complex SAR image. These
functions always transform the **last** axis, so callers must first arrange
the image as ``(n_lines, n_range)`` with :func:`utils.orientation.to_range_lines`
(the stored RFI-Patch layout is range x azimuth, so that is one transpose).

The forward transform uses a periodic Hamming window of length ``L // 8``
(paper Sec. 4.1) and, by default, a hop of ``win_len // 4``. For a 512-sample
range line that is a 64-sample window and a 16-sample hop. The hop satisfies
the nonzero-overlap-add condition used by ``torch.istft``.
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


def _resolve_hop(win_len: int, hop: int | None) -> int:
    hop = win_len // 4 if hop is None else int(hop)
    if not 1 <= hop <= win_len:
        raise ValueError(f"hop must be in [1, win_len={win_len}], got {hop}")
    return hop


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


def stft_row(x: torch.Tensor, win_len: int | None = None, hop: int | None = None) -> torch.Tensor:
    """STFT of a complex 1-D range signal, or of every row of ``(..., L)``.

    Parameters
    ----------
    x:
        Complex tensor of shape ``(L,)`` or ``(..., L)``; the last axis must be
        range. Pass ``to_range_lines(image)`` for a stored image.
    win_len:
        Hamming window length. Defaults to ``L // 8``.
    hop:
        Hop length. Defaults to ``win_len // 4``.

    Returns
    -------
    Complex spectrogram of shape ``(..., win_len, n_frames)``.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    length = x.shape[-1]
    win_len = _resolve_win_len(length, win_len)
    hop = _resolve_hop(win_len, hop)
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


def istft_row(X: torch.Tensor, win_len: int, length: int, hop: int | None = None) -> torch.Tensor:
    """Inverse of :func:`stft_row` for one line or a batch of lines.

    Parameters
    ----------
    X:
        Complex spectrogram of shape ``(..., win_len, n_frames)``.
    win_len:
        Hamming window length used by the forward STFT.
    length:
        Length ``L`` of each reconstructed range line.
    hop:
        Hop used by the forward STFT. Defaults to ``win_len // 4``.

    Returns
    -------
    Complex tensor of shape ``(..., length)``.
    """
    win_len = _resolve_win_len(length, win_len)
    hop = _resolve_hop(win_len, hop)
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
