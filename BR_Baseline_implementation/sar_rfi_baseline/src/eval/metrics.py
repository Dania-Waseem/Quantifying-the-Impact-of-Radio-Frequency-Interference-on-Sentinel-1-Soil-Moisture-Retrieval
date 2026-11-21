"""Image-quality metrics from Section 4.2 of Fang et al. (Remote Sensing 2025).

All three metrics work on magnitude images. A complex SAR image ``X`` is
reduced to ``|X|`` first; a real input is taken as an amplitude image
already. Inputs may be NumPy arrays or torch tensors; torch is only touched
when a tensor is passed in.

PSNR, Eq. 23
    ``PSNR = 20 * log10(Max_X / sqrt(MSE))``, with
    ``MSE = 1/(H*W) * sum |X - X_hat|^2``.
    The paper's text says "MSE denotes the root-mean-square". That is
    inconsistent with Eq. 23 itself, which already takes the square root of
    MSE: if MSE were a root-mean-square the formula would take a square root
    twice. This module uses the standard mean squared error, which matches the
    equation and every common PSNR implementation.

    ``Max_X`` choice: the paper writes ``Max_X`` without defining it. Here it
    is the peak of the ground-truth magnitude image ``max|X|``, the label and
    never the network output, so every method is scored against the same
    ceiling. Pass ``data_range`` to fix it instead, e.g. ``1.0`` for images
    already divided by a common peak.

SSIM, Eq. 24
    :func:`ssim` is scikit-image's ``structural_similarity`` on the two
    magnitude images (7x7 uniform window, ``K1=0.01``, ``K2=0.03``), with
    ``data_range`` equal to the peak-to-trough of the ground truth. Eq. 24 is
    printed in its global form, one mean and variance per image; that
    version is :func:`ssim_global`, for reference only.

Mean-entropy, Eq. 25
    ``ME(X_hat) = Mean(X_hat) * Ent(X_hat)`` on the normalized magnitude
    image ``A = |X_hat| / peak``, where ``peak`` is ``max|X_hat|`` unless a
    ``reference_peak`` is given. ``Mean`` is the average of ``A``. The paper
    takes ``Ent`` from its reference [14] without defining it. The default is
    the SAR image entropy used for focus quality,
    ``-sum p * ln p`` with ``p = A^2 / sum A^2``; it is low for a focused or
    clean image, which fits "smaller entropy indicates a clearer image".
    ``kind="histogram"`` gives the Shannon entropy of a 256-bin histogram of
    ``A`` in bits instead. Neither is guaranteed to reproduce the absolute
    values in Table 3, so compare ME across methods, not against the paper.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity

_SSIM_C = (0.01, 0.03)


def _magnitude(image) -> np.ndarray:
    """``|X|`` as a float64 NumPy array, from NumPy or torch input."""
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if np.iscomplexobj(array):
        return np.abs(array).astype(np.float64)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"expected a real or complex image, got dtype {array.dtype}")
    return np.abs(array.astype(np.float64))


def _pair(reference, estimate) -> tuple[np.ndarray, np.ndarray]:
    ref = _magnitude(reference)
    est = _magnitude(estimate)
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {est.shape}")
    if ref.ndim != 2:
        raise ValueError(f"expected 2-D images, got shape {ref.shape}")
    return ref, est


def mse(reference, estimate) -> float:
    """Standard mean squared error of the magnitude images (not a root-mean-square)."""
    ref, est = _pair(reference, estimate)
    return float(np.mean((ref - est) ** 2))


def quantize_to_uint8(image, peak: float | None = None) -> np.ndarray:
    """Scale magnitude image by peak, multiply by 255.0, round, and cast to uint8.

    Matches the official test evaluation pipeline in ``A_test_*.py``:
        img_uint8 = np.uint8((|image| / peak * 255.0).round())
    If ``peak`` is None, the maximum of ``|image|`` is used.
    """
    mag = _magnitude(image)
    if peak is None:
        peak = float(mag.max())
    if peak <= 0.0:
        raise ValueError("cannot quantize an image with non-positive peak")
    scaled = np.clip(mag / peak * 255.0, 0.0, 255.0)
    return np.uint8(np.round(scaled))


def psnr(
    reference,
    estimate,
    data_range: float | None = None,
    quantize_8bit: bool = False,
) -> float:
    """Eq. 23 PSNR in dB on magnitude images.

    ``reference`` is the ground truth ``X``; ``estimate`` is ``X_hat``.
    ``Max_X`` is ``max|X|`` of the ground truth unless ``data_range`` is set.
    Returns ``inf`` for identical images.

    If ``quantize_8bit=True``, both reference and estimate are scaled by the
    reference peak, mapped to ``[0, 255]``, rounded, and cast to uint8 before
    computing PSNR with ``peak = 255.0``, faithfully matching the official
    evaluation code (``A_test_*.py``).
    """
    ref, est = _pair(reference, estimate)
    if quantize_8bit:
        peak = float(ref.max()) if data_range is None else float(data_range)
        if peak <= 0.0:
            raise ValueError("PSNR is undefined for a reference with no signal")
        ref_u8 = quantize_to_uint8(ref, peak=peak).astype(np.float64)
        est_u8 = quantize_to_uint8(est, peak=peak).astype(np.float64)
        error = float(np.mean((ref_u8 - est_u8) ** 2))
        if error == 0.0:
            return math.inf
        return 20.0 * math.log10(255.0 / math.sqrt(error))

    error = float(np.mean((ref - est) ** 2))
    peak = float(ref.max()) if data_range is None else float(data_range)
    if peak <= 0.0:
        raise ValueError("PSNR is undefined for a reference with no signal")
    if error == 0.0:
        return math.inf
    return 20.0 * math.log10(peak / math.sqrt(error))


def ssim(
    reference,
    estimate,
    data_range: float | None = None,
    win_size: int = 7,
    quantize_8bit: bool = False,
) -> float:
    """SSIM of the magnitude images via :func:`skimage.metrics.structural_similarity`.

    ``data_range`` defaults to ``max|X| - min|X|`` of the ground truth, so
    every method is scored on the same range.

    If ``quantize_8bit=True``, both images are converted to 8-bit uint8 via
    ``quantize_to_uint8`` with ``data_range=255.0``, matching official test scripts.
    """
    ref, est = _pair(reference, estimate)
    if quantize_8bit:
        peak = float(ref.max()) if data_range is None else float(data_range)
        if peak <= 0.0:
            raise ValueError("SSIM needs a reference with a nonzero data range")
        ref_u8 = quantize_to_uint8(ref, peak=peak)
        est_u8 = quantize_to_uint8(est, peak=peak)
        return float(structural_similarity(ref_u8, est_u8, data_range=255.0, win_size=win_size))

    if data_range is None:
        data_range = float(ref.max() - ref.min())
    if data_range <= 0.0:
        raise ValueError("SSIM needs a reference with a nonzero data range")
    return float(structural_similarity(ref, est, data_range=data_range, win_size=win_size))


def ssim_global(reference, estimate, data_range: float | None = None) -> float:
    """Eq. 24 exactly as printed: one mean, variance and covariance per image."""
    ref, est = _pair(reference, estimate)
    if data_range is None:
        data_range = float(ref.max() - ref.min())
    c1 = (_SSIM_C[0] * data_range) ** 2
    c2 = (_SSIM_C[1] * data_range) ** 2
    mu_x, mu_y = ref.mean(), est.mean()
    var_x, var_y = ref.var(), est.var()
    cov = np.mean((ref - mu_x) * (est - mu_y))
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)))


def normalized_magnitude(image, reference_peak: float | None = None) -> np.ndarray:
    """``|X| / peak`` with ``peak = max|X|`` unless ``reference_peak`` is given."""
    mag = _magnitude(image)
    peak = float(mag.max()) if reference_peak is None else float(reference_peak)
    if peak <= 0.0:
        raise ValueError("cannot normalize an image with no signal")
    return mag / peak


def entropy(image, kind: str = "image", bins: int = 256) -> float:
    """Entropy of a normalized magnitude image.

    ``kind="image"``: ``-sum p ln p`` with ``p = A^2 / sum A^2`` (nats).
    ``kind="histogram"``: Shannon entropy in bits of a ``bins``-bin histogram of ``A`` on ``[0, 1]``.
    """
    amplitude = _magnitude(image)
    if kind == "image":
        power = amplitude**2
        total = power.sum()
        if total <= 0.0:
            raise ValueError("entropy is undefined for an image with no signal")
        p = power[power > 0] / total
        return float(-np.sum(p * np.log(p)))
    if kind == "histogram":
        counts, _ = np.histogram(np.clip(amplitude, 0.0, 1.0), bins=int(bins), range=(0.0, 1.0))
        p = counts[counts > 0] / counts.sum()
        return float(-np.sum(p * np.log2(p)))
    raise ValueError(f"kind must be 'image' or 'histogram', got {kind!r}")


def mean_entropy(image, reference_peak: float | None = None, kind: str = "image", bins: int = 256) -> float:
    """Eq. 25: ``ME = Mean(A) * Ent(A)`` with ``A`` the normalized magnitude image. Smaller is better."""
    amplitude = normalized_magnitude(image, reference_peak)
    return float(amplitude.mean()) * entropy(amplitude, kind=kind, bins=bins)


def image_metrics(
    reference,
    estimate,
    data_range: float | None = None,
    quantize_8bit: bool = False,
) -> dict:
    """PSNR, SSIM and ME of one restored image against its label."""
    return {
        "psnr": psnr(reference, estimate, data_range, quantize_8bit=quantize_8bit),
        "ssim": ssim(reference, estimate, data_range, quantize_8bit=quantize_8bit),
        "me": mean_entropy(estimate),
    }
