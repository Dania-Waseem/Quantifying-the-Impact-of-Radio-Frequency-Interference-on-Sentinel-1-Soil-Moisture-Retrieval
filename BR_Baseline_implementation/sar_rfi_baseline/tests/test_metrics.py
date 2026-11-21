"""PSNR, SSIM and mean-entropy on inputs whose values can be worked out by hand."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.metrics import entropy, image_metrics, mean_entropy, mse, psnr, ssim, ssim_global


def _peaked():
    ref = np.ones((4, 4))
    ref[0, 0] = 2.0  # Max_X = 2
    return ref


def test_mse_is_mean_squared_not_root_mean_square():
    assert mse(np.zeros((3, 3)), np.full((3, 3), 2.0)) == pytest.approx(4.0)


def test_psnr_known_value_uses_ground_truth_peak():
    ref = _peaked()
    est = ref + 0.1  # MSE = 0.01
    assert psnr(ref, est) == pytest.approx(20 * math.log10(2.0 / 0.1))  # 26.0206 dB
    assert psnr(ref, est, data_range=1.0) == pytest.approx(20.0)
    # Max_X comes from the first argument, the label: swapping them changes it to 2.1.
    assert psnr(est, ref) == pytest.approx(20 * math.log10(2.1 / 0.1))
    assert psnr(ref, ref) == math.inf


def test_psnr_compares_complex_images_by_magnitude():
    ref = _peaked()
    rng = np.random.default_rng(0)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, ref.shape))
    assert psnr(ref * phase, (ref + 0.1) * np.conj(phase)) == pytest.approx(26.0206, abs=1e-4)


def test_ssim_global_matches_hand_calculation():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    # mu 1.5/2.5, var 1.25 each, cov 1.25, L = 3, C1 = 0.0009, C2 = 0.0081
    expected = (7.5 + 0.0009) / (8.5 + 0.0009)
    assert ssim_global(x, x + 1.0) == pytest.approx(expected)
    assert ssim_global(x, x) == pytest.approx(1.0)


def test_skimage_ssim_known_values():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 1, (64, 64))
    assert ssim(ref, ref) == pytest.approx(1.0)
    noisy = ref + rng.normal(0, 0.2, ref.shape)
    assert 0.0 < ssim(ref, noisy) < 0.9
    # Two flat images differ only in luminance: SSIM = (2*a*b + C1) / (a^2 + b^2 + C1).
    a, b, c1 = 0.5, 0.7, (0.01 * 1.0) ** 2
    flat = ssim(np.full((16, 16), a), np.full((16, 16), b), data_range=1.0)
    assert flat == pytest.approx((2 * a * b + c1) / (a**2 + b**2 + c1))


def test_mean_entropy_known_values():
    n = 100
    uniform = np.ones((10, 10))
    # A = 1 everywhere: Mean = 1, image entropy = ln(N).
    assert mean_entropy(uniform) == pytest.approx(math.log(n))
    # One bright pixel: entropy 0, so ME = 0 (the clearest possible image).
    point = np.zeros((10, 10))
    point[3, 4] = 5.0
    assert mean_entropy(point) == pytest.approx(0.0)
    # A shared reference peak halves A: Mean = 0.5, entropy unchanged.
    assert mean_entropy(uniform, reference_peak=2.0) == pytest.approx(0.5 * math.log(n))
    # Normalizing by its own peak makes ME independent of the image scale.
    rng = np.random.default_rng(2)
    image = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    assert mean_entropy(1000.0 * image) == pytest.approx(mean_entropy(image))


def test_histogram_entropy_known_values():
    assert entropy(np.ones((8, 8)), kind="histogram") == pytest.approx(0.0)
    half = np.ones((8, 8))
    half[:4] = 0.5  # two equally likely bins: 1 bit
    assert entropy(half, kind="histogram") == pytest.approx(1.0)
    assert mean_entropy(half, kind="histogram") == pytest.approx(0.75)


def test_less_interference_gives_lower_me():
    rng = np.random.default_rng(3)
    clean = np.zeros((64, 64))
    clean[rng.integers(0, 64, 20), rng.integers(0, 64, 20)] = rng.uniform(1, 3, 20)
    polluted = clean.copy()
    polluted[10:40, 5:60] += 2.0 * np.abs(rng.normal(size=(30, 55)))
    assert mean_entropy(clean) < mean_entropy(polluted)
    scores = image_metrics(clean, clean)
    assert scores["psnr"] == math.inf and scores["ssim"] == pytest.approx(1.0)


def test_torch_tensors_are_accepted():
    torch = pytest.importorskip("torch")
    ref = torch.tensor(_peaked(), dtype=torch.complex64)
    assert psnr(ref, ref + 0.1) == pytest.approx(26.0206, abs=1e-3)


def test_quantize_to_uint8_matches_official_formula():
    from eval.metrics import quantize_to_uint8

    img = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float64)
    q = quantize_to_uint8(img, peak=1.0)
    assert q.dtype == np.uint8
    assert np.array_equal(q, np.array([[0, 128], [191, 255]], dtype=np.uint8))


def test_quantize_8bit_in_psnr_and_ssim():
    ref = np.linspace(0, 1, 64 * 64, dtype=np.float64).reshape(64, 64)
    est = ref + 0.01
    val_unquant = psnr(ref, est, quantize_8bit=False)
    val_quant = psnr(ref, est, quantize_8bit=True)
    assert isinstance(val_quant, float)
    assert math.isfinite(val_quant)
    assert abs(val_unquant - val_quant) < 2.0

    ssim_unquant = ssim(ref, est, quantize_8bit=False)
    ssim_quant = ssim(ref, est, quantize_8bit=True)
    assert 0.0 <= ssim_quant <= 1.0
    assert abs(ssim_unquant - ssim_quant) < 0.1

    metrics_8bit = image_metrics(ref, est, quantize_8bit=True)
    assert "psnr" in metrics_8bit and "ssim" in metrics_8bit and "me" in metrics_8bit
