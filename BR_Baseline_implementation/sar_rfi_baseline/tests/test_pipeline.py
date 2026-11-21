"""Algorithm 1 pipeline: identity round trip, batching, and the normalization inverse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.pipeline import (
    check_identity,
    identity_network,
    load_image,
    suppress_rfi,
    suppress_rfi_rowwise,
)
from models import get_model


def _image(shape, seed=0, amplitude=1.0):
    rng = np.random.default_rng(seed)
    image = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    image /= np.abs(image).max()
    return (amplitude * image).astype(np.complex64)


def test_identity_network_returns_s_on_the_paper_slice_size():
    S = _image((512, 1500))
    errors = check_identity(S, tol=1e-5)
    # A unit-peak image: the absolute error is held to 1e-5 as well.
    assert errors["max_abs_error"] <= 1e-5


def test_identity_holds_at_sar_amplitudes_and_in_float64():
    S = _image((64, 400), seed=1, amplitude=3.0e3)
    check_identity(S, tol=1e-5)
    S64 = S.astype(np.complex128)
    S_star = suppress_rfi(S64, identity_network, dtype=torch.complex128)
    assert torch.allclose(S_star, torch.from_numpy(S64), atol=1e-9, rtol=0)


def test_load_image_accepts_snr_s4_mat_without_a_clean_reference(tmp_path):
    image = _image((12, 1536), seed=8, amplitude=120.0)
    path = tmp_path / "snr5.mat"
    scipy.io.savemat(path, {"s4": image})

    loaded, clean = load_image(path)

    assert loaded.shape == image.shape
    assert loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, image)
    assert clean is None


def test_row_batch_size_does_not_change_the_output():
    S = _image((37, 256), seed=2)  # 37 rows: last batch is ragged
    torch.manual_seed(0)
    model = get_model("dncnn").eval()
    batched = suppress_rfi(S, model, row_batch=8)
    whole = suppress_rfi(S, model, row_batch=37)
    rowwise = suppress_rfi_rowwise(S, model)
    assert batched.shape == S.shape
    assert torch.allclose(batched, whole, atol=1e-5)
    assert torch.allclose(batched, rowwise, atol=1e-5)


def test_normalization_is_undone_so_output_is_in_input_units():
    S = _image((16, 256), seed=3, amplitude=750.0)
    halve = lambda x: 0.5 * x  # linear H: S* must be exactly S / 2 in original units
    S_star = suppress_rfi(S, halve, row_batch=5)
    expected = torch.from_numpy(S) / 2
    assert torch.allclose(S_star, expected, atol=1e-5 * 750.0, rtol=0)
    # H itself must see the training scale: peak magnitude 1 over the slice.
    seen = []

    def spy(x):
        seen.append(torch.linalg.vector_norm(x, dim=1).max().item())
        return x

    suppress_rfi(S, spy, row_batch=4)
    assert abs(max(seen) - 1.0) < 1e-5
