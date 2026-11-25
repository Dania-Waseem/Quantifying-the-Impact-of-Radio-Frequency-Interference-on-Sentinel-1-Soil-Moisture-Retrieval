"""Map-domain 8-bit PSNR / SSIM exactly as in the authors' test script (test/A_test_unet.py).

Authors' code, per sample (one 512 x 512 map):
    max1    = np.max(img_H)                                  # label maximum
    img_net = np.uint8((img_net / max1 * 255.0).round())
    img_H   = np.uint8((img_H   / max1 * 255.0).round())
    psnr = util.calculate_psnr(img_net, img_H, border=0)     # utils/utils_image.py
    ssim = util.calculate_ssim(img_net, img_H, border=0)     # Gaussian 11x11, sigma 1.5, valid
then the plain mean over samples.

``np.uint8`` of a float outside [0, 255] is not defined by NumPy (it wraps on common platforms),
so :func:`to_u8` reports how many pixels fell outside that range, and a clipped variant is
reported next to the authors' expression. They are identical whenever nothing is out of range.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_u8(x: np.ndarray, ref_max: float, mode: str = "authors") -> tuple[np.ndarray, int]:
    v = (np.asarray(x, dtype=np.float64) / ref_max * 255.0).round()
    n_out = int(np.count_nonzero((v < 0) | (v > 255)))
    if mode == "authors":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.uint8(v), n_out
    if mode == "clip":
        return np.uint8(np.clip(v, 0, 255)), n_out
    raise ValueError(mode)


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    """utils_image.calculate_psnr: 20 log10(255 / sqrt(MSE)) on uint8 images (inf if identical)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def _gauss11() -> np.ndarray:
    # cv2.getGaussianKernel(11, 1.5)
    i = np.arange(11) - 5
    g = np.exp(-(i ** 2) / (2 * 1.5 ** 2))
    return g / g.sum()


def _filter_valid(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    # == cv2.filter2D(img, -1, outer(g, g))[5:-5, 5:-5]: the cropped border never touches padding
    t = sliding_window_view(img, g.size, axis=1) @ g          # (H, W-10)
    return sliding_window_view(t, g.size, axis=0) @ g         # (H-10, W-10)


def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    """utils_image.ssim (MATLAB-equivalent SSIM, C1=(0.01*255)^2, C2=(0.03*255)^2)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    g = _gauss11()
    mu1, mu2 = _filter_valid(a, g), _filter_valid(b, g)
    s11 = _filter_valid(a * a, g) - mu1 ** 2
    s22 = _filter_valid(b * b, g) - mu2 ** 2
    s12 = _filter_valid(a * b, g) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def map_metrics(estimate: np.ndarray, label: np.ndarray) -> dict:
    """Authors' protocol for one map, plus the clipped variant and the out-of-range count."""
    est = np.squeeze(np.asarray(estimate, dtype=np.float64))
    lab = np.squeeze(np.asarray(label, dtype=np.float64))
    if est.shape != lab.shape or est.ndim != 2:
        raise ValueError(f"need two equal 2-D maps, got {est.shape} and {lab.shape}")
    m = float(lab.max())
    if not m > 0:
        raise ValueError("label maximum must be positive")
    h8, _ = to_u8(lab, m, "authors")
    e8, n_out = to_u8(est, m, "authors")
    e8c, _ = to_u8(est, m, "clip")
    return {"psnr": psnr_u8(e8, h8), "ssim": ssim_u8(e8, h8),
            "psnr_clip": psnr_u8(e8c, h8), "ssim_clip": ssim_u8(e8c, h8),
            "frac_out_of_range": n_out / est.size}
