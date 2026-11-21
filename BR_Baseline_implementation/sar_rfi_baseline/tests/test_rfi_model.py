"""Checks for the unified RFI model, the image block, and SIR scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.rfi_model import (
    PARAM_RANGES,
    interference_mask,
    make_block_rfi,
    sample_rfi_params,
    scale_to_sir,
    unified_rfi,
)


def test_bessel_expansion_matches_sinusoid():
    beta = 2.5
    f_mod = 1.0e6
    fc = 3.0e6
    tau = np.linspace(-8e-6, 8e-6, 4000)
    direct = np.exp(1j * (2.0 * np.pi * fc * tau + beta * np.sin(2.0 * np.pi * f_mod * tau)))
    params = {
        "preset": "sinusoidal",
        "amplitudes": np.array([1.0]),
        "fc_hz": np.array([fc]),
        "k_rfi_hz_s": np.array([0.0]),
        "beta": np.array([beta]),
        "f_mod_hz": np.array([f_mod]),
        "bessel_m_max": 24,
        "t_rfi_s": 1.0,
    }
    recon = unified_rfi(tau, params, apply_rect=False)
    # The series is the sum of narrowband tones, so every chirp rate is zero.
    error = np.max(np.abs(recon - direct))
    assert error < 1e-6, error
    orders = np.arange(-4, 5)
    assert np.allclose(jv(orders, beta), jv(orders, beta))


def test_presets_differ_in_chirp_rate_and_form_a_block():
    rng = np.random.default_rng(7)
    shape = (64, 256)
    narrow = sample_rfi_params(rng, "narrowband", shape=shape)
    wide = sample_rfi_params(rng, "chirp_wideband", shape=shape)
    sine = sample_rfi_params(rng, "sinusoidal", shape=shape)
    assert np.max(np.abs(narrow["k_rfi_hz_s"])) < np.min(np.abs(wide["k_rfi_hz_s"]))
    assert np.all(np.abs(wide["bandwidth_hz"]) <= PARAM_RANGES["bandwidth_hz"][1])
    for params in (narrow, wide, sine):
        image = make_block_rfi(shape, params, rng)
        assert image.shape == shape
        assert np.iscomplexobj(image)
        a0, a1 = params["azimuth_start"], params["azimuth_end"]
        r0, r1 = params["range_start"], params["range_end"]
        assert np.max(np.abs(image[:a0])) == 0
        assert np.max(np.abs(image[a1:])) == 0
        assert np.max(np.abs(image[:, :r0])) == 0
        assert np.max(np.abs(image[:, r1:])) == 0
        assert np.max(np.abs(image[a0:a1, r0:r1])) > 0
        mask = interference_mask(image)
        assert np.array_equal(mask[a0:a1, r0:r1], np.ones((a1 - a0, r1 - r0), dtype=bool))


def test_azimuth_quadratic_phase_and_flat_block_magnitude():
    params = {
        "preset": "narrowband",
        "n_components": 1,
        "amplitudes": np.array([1.3]),
        "fc_hz": np.array([1.0e6]),
        "k_rfi_hz_s": np.array([1.0e8]),
        "bandwidth_hz": np.array([1.0e5]),
        "t_rfi_s": 20e-6,
        "beta": np.array([1.0]),
        "f_mod_hz": np.array([1.0e6]),
        "bessel_m_max": 8,
        "azimuth_start": 10,
        "azimuth_end": 40,
        "range_start": 20,
        "range_end": 80,
        "k_delta_a": 0.25,
        "phase0": 0.4,
    }
    image = make_block_rfi((48, 96), params, np.random.default_rng(0))
    block = image[10:40, 20:80]
    assert np.allclose(np.abs(block), np.abs(block[0, 0]), rtol=1e-4, atol=1e-5)
    assert np.all(image[:10] == 0)
    assert np.all(image[40:] == 0)
    mid = 50
    column = image[10:40, mid]
    center_row = column.shape[0] // 2
    relative = column / column[center_row]
    index = np.arange(10, 40)
    center = 0.5 * (10 + 40 - 1)
    half = 0.5 * (40 - 10)
    eta = (index - center) / half * 0.5
    eta0 = eta[center_row]
    expected = np.exp(-1j * np.pi * (eta**2 - eta0**2) / params["k_delta_a"])
    assert np.max(np.abs(relative - expected)) < 1e-4


def test_scale_to_sir_hits_table2_grid():
    rng = np.random.default_rng(3)
    shape = (32, 128)
    clean = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    params = sample_rfi_params(rng, "chirp_wideband", shape=shape)
    rfi = make_block_rfi(shape, params, rng)
    mask = interference_mask(rfi)
    for sir_db in PARAM_RANGES["sir_db"]:
        polluted = scale_to_sir(clean, rfi, sir_db)
        added = polluted - clean
        target = np.mean(np.abs(clean[mask]) ** 2)
        interference = np.mean(np.abs(added[mask]) ** 2)
        got = 10.0 * np.log10(target / interference)
        assert abs(got - sir_db) < 0.05, (sir_db, got)
        assert np.allclose(polluted[~mask], clean[~mask])
