"""512 x 512 magnitude time-frequency maps of single range lines (authors'-style representation).

What is documented
------------------
* [paper, Sec. 4.1 Step II] Hamming window, length L/8 = 64 for an L = 512-sample range line.
* [paper, Algorithm 1] one 1-D range signal per azimuth position goes through STFT -> network -> ISTFT.
* [authors' README + dataset/mydataset.py] network samples are single-channel [N, 512, 512] real maps;
  the loader crops every stored map to [0:512, 0:512].
* [authors' A_test_unet.py] metrics are computed directly on those maps (non-negative, scaled by
  the label maximum), which implies magnitude-type maps.

What is NOT documented (implementation choices, each a field of MapConfig)
---------------------------------------------------------------------------
* FFT size and hop. The only setting that gives a 512 x 512 map from a 512-sample line without any
  resizing is n_fft = 512 (512 frequency rows) and hop = 1 (one frame per range sample). This is an
  inference from the map size, not a confirmed parameter.
* Centring/padding: frames are centred on each sample with zero padding of win_len/2 on both sides
  (513 frames), cropped to the first 512 exactly as the authors' loader crops [0:512].
* Symmetric Hamming (MATLAB ``hamming(64)``), fftshift-ed (centred) frequency axis, linear |STFT|.
* The MATLAB Time-Frequency Toolbox ``tfrstft`` would also give an N x N map, but it requires an
  odd window length (63 or 65) and does not fftshift; see the reproduction status report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MapConfig:
    win_len: int = 64              # [paper] L/8
    window: str = "hamming_symmetric"   # [assumption] "hamming_symmetric" | "hamming_periodic"
    n_fft: int = 512               # [inferred from 512 x 512 maps]
    hop: int = 1                   # [inferred from 512 x 512 maps]
    center: bool = True            # [assumption] frame t centred on sample t
    pad_mode: str = "constant"     # [assumption] "constant" (zeros) | "reflect"
    fftshift: bool = True          # [assumption] zero frequency in the middle row
    n_frames: int = 512            # [code] mydataset.py crops to [0:512]
    magnitude: str = "linear"      # [assumption] |STFT|, not log, not power

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MapConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown map settings: {sorted(unknown)}")
        return cls(**d)

    def validate(self, length: int) -> None:
        if self.win_len < 2 or self.win_len > self.n_fft:
            raise ValueError("need 2 <= win_len <= n_fft")
        if self.hop < 1:
            raise ValueError("hop must be >= 1")
        if self.n_frames > self.total_frames(length):
            raise ValueError(f"n_frames={self.n_frames} exceeds {self.total_frames(length)} available frames")
        if self.magnitude != "linear":
            raise ValueError("only linear magnitude is implemented")

    def pad(self) -> int:
        return self.win_len // 2 if self.center else 0

    def total_frames(self, length: int) -> int:
        return 1 + (length + 2 * self.pad() - self.win_len) // self.hop

    def map_shape(self) -> tuple[int, int]:
        return (self.n_fft, self.n_frames)


def window_array(cfg: MapConfig) -> np.ndarray:
    n = cfg.win_len
    if cfg.window == "hamming_symmetric":
        return np.hamming(n)                     # 0.54 - 0.46 cos(2 pi k / (n - 1))
    if cfg.window == "hamming_periodic":
        k = np.arange(n)
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    raise ValueError(f"unknown window {cfg.window!r}")


def _frame_index(cfg: MapConfig, length: int) -> np.ndarray:
    t = np.arange(cfg.total_frames(length))[:, None] * cfg.hop
    return t + np.arange(cfg.win_len)[None, :]  # (T, win) indices into the padded signal


def stft_lines(lines: np.ndarray, cfg: MapConfig = MapConfig()) -> np.ndarray:
    """Complex STFT of a batch of 1-D range lines.

    lines: (B, L) complex. Returns (B, n_fft, n_frames) complex128, rows = frequency, cols = time.
    """
    x = np.atleast_2d(np.asarray(lines))
    if x.ndim != 2:
        raise ValueError(f"expected (B, L) lines, got {x.shape}")
    length = x.shape[1]
    cfg.validate(length)
    p = cfg.pad()
    xp = np.pad(x.astype(np.complex128), ((0, 0), (p, p)), mode=cfg.pad_mode) if p else x.astype(np.complex128)
    frames = xp[:, _frame_index(cfg, length)] * window_array(cfg)[None, None, :]   # (B, T, win)
    spec = np.fft.fft(frames, n=cfg.n_fft, axis=-1)                                 # (B, T, n_fft)
    spec = np.swapaxes(spec, 1, 2)[:, :, : cfg.n_frames]                             # (B, n_fft, n_frames)
    if cfg.fftshift:
        spec = np.fft.fftshift(spec, axes=1)
    return spec


def istft_lines(spec: np.ndarray, length: int, cfg: MapConfig = MapConfig()) -> np.ndarray:
    """Weighted overlap-add inverse of :func:`stft_lines` (exact for an unmodified STFT).

    spec: (B, n_fft, n_frames) complex. Returns (B, length) complex128.
    """
    s = np.asarray(spec)
    if s.ndim == 2:
        s = s[None]
    cfg.validate(length)
    if s.shape[1:] != cfg.map_shape():
        raise ValueError(f"expected (B, {cfg.n_fft}, {cfg.n_frames}), got {s.shape}")
    if cfg.fftshift:
        s = np.fft.ifftshift(s, axes=1)
    seg = np.fft.ifft(np.swapaxes(s, 1, 2), axis=-1)[:, :, : cfg.win_len]            # (B, T, win)
    w = window_array(cfg)
    p = cfg.pad()
    n_pad = length + 2 * p
    out = np.zeros((s.shape[0], n_pad), dtype=np.complex128)
    wsum = np.zeros(n_pad)
    starts = np.arange(cfg.n_frames) * cfg.hop
    for k in range(cfg.win_len):
        out[:, starts + k] += seg[:, :, k] * w[k]
        wsum[starts + k] += w[k] ** 2
    wsum = wsum[p : p + length]
    if np.any(wsum <= 1e-12):
        raise ValueError("window overlap does not cover every sample; ISTFT undefined")
    return out[:, p : p + length] / wsum[None, :]


def magnitude_maps(lines: np.ndarray, cfg: MapConfig = MapConfig()) -> np.ndarray:
    """(B, n_fft, n_frames) float64 |STFT| maps."""
    return np.abs(stft_lines(lines, cfg))


def lines_from_magnitude(mag: np.ndarray, phase_source_lines: np.ndarray, cfg: MapConfig = MapConfig()) -> np.ndarray:
    """Image-domain reconstruction used only for the supplementary evaluation.

    The network predicts magnitude only, so the phase is taken from the STFT of the RFI-polluted
    line (an implementation assumption; the paper does not say how phase is handled).
    """
    ref = stft_lines(phase_source_lines, cfg)
    return istft_lines(np.asarray(mag) * np.exp(1j * np.angle(ref)), np.asarray(phase_source_lines).shape[-1], cfg)
