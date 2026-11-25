"""Range / azimuth orientation of stored SAR slices, in one place.

Stored layout
-------------
RFI-Patch files hold ``ss`` with MATLAB dims ``(5, 512, 1500)``. ``ss[k]`` is a
``512 x 1500`` complex image whose **axis 0 (512 samples) is range** and whose
**axis 1 (1500 samples) is azimuth**. Evidence (see ``docs/AXIS_ORIENTATION.md``):

* ``Data/RFI-Patch/explaination.docx``: "Datasize: 5*512*1500 % range=512, azimuth=1500".
* Fang et al. 2025, Figs. 3-4: vertical axis 0-512 is labelled range, horizontal 0-1500 azimuth.
* Figs. 7a/10a: the RFI is a constant-frequency line at the centre bin of the
  time-frequency map. Only an STFT along axis 0 reproduces that on this data;
  along axis 1 the RFI shows up as slanted chirps.

So :data:`RANGE_AXIS` is 0. Algorithm 1 (``X = S(i,:)``) is written for an
*azimuth x range* matrix, which is the transpose of the stored layout.

The one transpose
-----------------
:func:`to_range_lines` turns a stored image into ``(n_lines, n_range)``: one
1-D range signal per row, the layout ``utils.tf.stft_row`` and the networks
expect (the STFT always runs along the last axis). :func:`from_range_lines` is
its exact inverse. Code that goes image -> lines -> image must call each of
them exactly once.

STFT settings
-------------
Window length ``L // 8`` (paper Sec. 4.1, L = range samples) and hop
``win // 4`` (a documented choice; the paper gives no hop). For L = 512 that
is a 64-sample Hamming window and a 16-sample hop.

Legacy runs
-----------
Checkpoints written before this correction have no ``range_axis`` in their
saved arguments. They were trained with the STFT along axis 1 (the 1500-sample
azimuth axis), so :func:`checkpoint_range_axis` returns
:data:`LEGACY_RANGE_AXIS` (1) for them and evaluation stays faithful to how
they were trained.

This module only uses ``swapaxes`` and shapes, so it works on NumPy arrays and
torch tensors alike and has no torch dependency.
"""

from __future__ import annotations

from typing import Mapping

# Stored RFI-Patch / laboratory slice layout: (range, azimuth).
STORED_SHAPE = (512, 1500)
RANGE_AXIS = 0
AZIMUTH_AXIS = 1
# Every checkpoint trained before the correction transformed axis 1.
LEGACY_RANGE_AXIS = 1
VALID_RANGE_AXES = (0, 1)

# Paper Sec. 4.1: window = L / 8. Hop = window / 4 is this project's choice.
WINDOW_DIVISOR = 8
HOP_DIVISOR = 4


def check_range_axis(range_axis) -> int:
    """Return ``range_axis`` as an int, or raise if it is not 0 or 1."""
    try:
        axis = int(range_axis)
    except (TypeError, ValueError):
        raise ValueError(f"range_axis must be 0 or 1, got {range_axis!r}") from None
    if axis not in VALID_RANGE_AXES or axis != range_axis:
        raise ValueError(f"range_axis must be 0 or 1, got {range_axis!r}")
    return axis


def azimuth_axis(range_axis: int = RANGE_AXIS) -> int:
    return 1 - check_range_axis(range_axis)


def range_length(shape, range_axis: int = RANGE_AXIS) -> int:
    """Number of range samples ``L`` of a stored 2-D image of ``shape``."""
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D image shape, got {shape}")
    return shape[check_range_axis(range_axis)]


def n_range_lines(shape, range_axis: int = RANGE_AXIS) -> int:
    """Number of 1-D range signals (azimuth positions) in a stored image."""
    shape = tuple(int(s) for s in shape)
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D image shape, got {shape}")
    return shape[azimuth_axis(range_axis)]


def lines_shape(shape, range_axis: int = RANGE_AXIS) -> tuple[int, int]:
    """``(n_lines, n_range)``: the shape :func:`to_range_lines` returns."""
    return n_range_lines(shape, range_axis), range_length(shape, range_axis)


def to_range_lines(image, range_axis: int = RANGE_AXIS):
    """Stored 2-D image -> ``(n_lines, n_range)``, one range signal per row.

    With ``range_axis=0`` this is the single transpose of the correction. It
    returns a view (NumPy) or a non-contiguous tensor (torch); call
    ``np.ascontiguousarray`` / ``.contiguous()`` if you need a copy.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {tuple(image.shape)}")
    return image.swapaxes(0, 1) if check_range_axis(range_axis) == 0 else image


def from_range_lines(lines, range_axis: int = RANGE_AXIS):
    """Inverse of :func:`to_range_lines`: ``(n_lines, n_range)`` -> stored layout."""
    if lines.ndim != 2:
        raise ValueError(f"expected 2-D range lines, got shape {tuple(lines.shape)}")
    return lines.swapaxes(0, 1) if check_range_axis(range_axis) == 0 else lines


def range_line(image, index: int, range_axis: int = RANGE_AXIS):
    """The 1-D range signal at azimuth position ``index`` of a stored image."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {tuple(image.shape)}")
    return image[:, index] if check_range_axis(range_axis) == 0 else image[index, :]


def most_interfered_line(image, range_axis: int = RANGE_AXIS) -> int:
    """Azimuth index of the range line with the most energy (the RFI block's strongest line)."""
    import numpy as np

    magnitude = np.abs(np.asarray(image))
    if magnitude.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {magnitude.shape}")
    energy = np.sum(magnitude**2, axis=check_range_axis(range_axis))
    return int(np.argmax(energy))


def resolve_stft(n_range: int, win_len: int | None = None, hop: int | None = None) -> tuple[int, int]:
    """Window and hop for range lines of length ``n_range``.

    Defaults: ``win = n_range // 8`` and ``hop = win // 4`` (64 and 16 for
    ``n_range = 512``). Raises if the window is too short or longer than the
    line, or if the hop is not in ``[1, win]``.
    """
    n_range = int(n_range)
    win = n_range // WINDOW_DIVISOR if win_len is None else int(win_len)
    if win < 4:
        raise ValueError(f"STFT window must be >= 4 samples, got {win} for range length {n_range}")
    if win > n_range:
        raise ValueError(f"STFT window {win} is longer than the range line ({n_range})")
    step = win // HOP_DIVISOR if hop is None else int(hop)
    if not 1 <= step <= win:
        raise ValueError(f"STFT hop must be in [1, {win}], got {step}")
    return win, step


def spectrogram_shape(n_range: int, win_len: int | None = None, hop: int | None = None) -> tuple[int, int]:
    """``(frequency bins, frames)`` of ``utils.tf.stft_row`` on a line of ``n_range`` samples.

    Two-sided FFT of size ``win`` and a centred window with ``win // 2``
    samples of reflect padding on each side, as in ``torch.stft(center=True)``.
    """
    win, step = resolve_stft(n_range, win_len, hop)
    padded = int(n_range) + 2 * (win // 2)
    return win, 1 + (padded - win) // step


def checkpoint_range_axis(saved_args: Mapping | None) -> int:
    """Range axis a checkpoint was trained with; :data:`LEGACY_RANGE_AXIS` if it predates the setting."""
    if not saved_args or "range_axis" not in saved_args or saved_args["range_axis"] is None:
        return LEGACY_RANGE_AXIS
    return check_range_axis(saved_args["range_axis"])


def image_axis_labels(range_axis: int = RANGE_AXIS) -> tuple[str, str]:
    """``(vertical, horizontal)`` axis names when a stored image is drawn with ``imshow(image)``."""
    return ("Range", "Azimuth") if check_range_axis(range_axis) == 0 else ("Azimuth", "Range")


def display_image(image, range_axis: int = RANGE_AXIS):
    """Stored image arranged as the paper draws it: range vertical, azimuth horizontal."""
    return image if check_range_axis(range_axis) == 0 else image.swapaxes(0, 1)
