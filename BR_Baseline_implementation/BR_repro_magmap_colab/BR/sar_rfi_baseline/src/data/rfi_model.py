"""Block-type RFI synthesizer from Fang et al., Remote Sensing 2025.

The echo-domain model (Eq. 5) unifies narrowband, chirp-modulated wideband, and
sinusoidal-modulated wideband interference:

    s_rfi(tau) = sum_n D_n * exp(j*2*pi*fc_n*tau + j*pi*k_rfi_n*tau^2)

inside a rectangular pulse of duration t_rfi (Eq. 6). Sinusoidal modulation
(Eq. 3) is expanded with the Jacobi-Anger identity, which is the complex form
of the Bessel series in Eq. 4, into a sum of narrowband tones (k_rfi = 0):

    exp(j*beta*sin(2*pi*f_m*tau)) = sum_m J_m(beta) * exp(j*2*pi*m*f_m*tau)

After imaging, each azimuth row is that range signal times the complex
amplitude in Eq. 19:

    A(eta) = rect(eta / T_az) * exp(-j*phase0) * exp(-j*pi*eta^2 / k_delta_a)

with phase0 standing in for the constant 2*pi*fc*R1/c. The product of the two
rect windows is the block.

Layout: the generator works on the *range-line view* ``(azimuth, range)``,
one range signal per row. A stored RFI-Patch slice is the transpose of that
(``512 range x 1500 azimuth``), so the default shape here is ``(1500, 512)``.
:func:`data.dataset.synthesize_pair` converts between the two with
:mod:`utils.orientation`.

``PARAM_RANGES`` is the single config dict for every randomly drawn parameter.
Table 2 currently fixes only the bandwidth (0-100 MHz) and the SIR grid
{-30, -20, -10, 0} dB. The other intervals are documented placeholders and
can be replaced when the remaining Table 2 values are filled in.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.special import jv

PRESETS = ("narrowband", "chirp_wideband", "sinusoidal")

# Inclusive (low, high) intervals unless the note says the entry is a discrete grid.
# Frequencies are in Hz and chirp rates in Hz/s. Fractions are relative to the
# image axis they apply to.
PARAM_RANGES: dict[str, tuple] = {
    # Table 2.
    "bandwidth_hz": (0.0, 100e6),
    "sir_db": (0.0, -10.0, -20.0, -30.0),
    # Placeholder until the rest of Table 2 is filled in.
    "n_components": (1, 4),
    "amplitude": (0.5, 1.5),
    "fc_hz": (-20e6, 20e6),
    "t_rfi_s": (5e-6, 100e-6),
    "narrow_bandwidth_hz": (0.0, 1e6),
    "wide_bandwidth_hz": (20e6, 100e6),
    "beta": (0.5, 4.0),
    "f_mod_hz": (0.2e6, 5e6),
    "bessel_m_max": (8, 20),
    "azimuth_start_frac": (0.20, 0.45),
    "azimuth_width_frac": (0.15, 0.40),
    "range_start_frac": (0.05, 0.25),
    "range_width_frac": (0.55, 0.90),
    # Dimensionless k_delta_a. eta is normalized to [-0.5, 0.5] across the
    # azimuth rect, and the Eq. 19 phase is -pi * eta^2 / k_delta_a.
    "k_delta_a": (0.05, 1.0),
    "phase0_rad": (0.0, 2.0 * math.pi),
}


def _as_rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(rng)


def _window(rng: np.random.Generator, n: int, start_key: str, width_key: str, ranges: Mapping) -> tuple[int, int]:
    width = int(round(float(rng.uniform(*ranges[width_key])) * n))
    width = min(max(width, 4), n)
    start = int(round(float(rng.uniform(*ranges[start_key])) * n))
    start = min(max(start, 0), n - width)
    return start, start + width


def sample_rfi_params(
    rng=None,
    preset: str = "chirp_wideband",
    ranges: Mapping | None = None,
    shape: tuple[int, int] = (1500, 512),
) -> dict:
    """Draw one interference configuration from ``ranges``.

    The returned dict is a concrete parameter set, not an interval. Pass it
    to :func:`make_block_rfi`. Chirp rate is set from bandwidth and pulse
    length so a narrowband preset keeps a small ``k_rfi`` and a chirp
    wideband preset keeps a large one. Sinusoidal presets store the Eq. 3
    modulation parameters; the Bessel tones are built later.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    cfg = PARAM_RANGES if ranges is None else ranges
    rng = _as_rng(rng)
    n_az, n_rg = int(shape[0]), int(shape[1])
    n_comp = int(rng.integers(int(cfg["n_components"][0]), int(cfg["n_components"][1]) + 1))
    t_rfi = float(rng.uniform(*cfg["t_rfi_s"]))
    if preset == "narrowband":
        bandwidths = rng.uniform(*cfg["narrow_bandwidth_hz"], size=n_comp)
    elif preset == "chirp_wideband":
        bandwidths = rng.uniform(*cfg["wide_bandwidth_hz"], size=n_comp)
    else:
        bandwidths = np.zeros(n_comp, dtype=np.float64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_comp)
    k_rfi = signs * bandwidths / t_rfi
    a0, a1 = _window(rng, n_az, "azimuth_start_frac", "azimuth_width_frac", cfg)
    r0, r1 = _window(rng, n_rg, "range_start_frac", "range_width_frac", cfg)
    return {
        "preset": preset,
        "n_components": n_comp,
        "amplitudes": rng.uniform(*cfg["amplitude"], size=n_comp).astype(np.float64),
        "fc_hz": rng.uniform(*cfg["fc_hz"], size=n_comp).astype(np.float64),
        "k_rfi_hz_s": np.asarray(k_rfi, dtype=np.float64),
        "bandwidth_hz": np.asarray(np.abs(bandwidths), dtype=np.float64),
        "t_rfi_s": t_rfi,
        "beta": rng.uniform(*cfg["beta"], size=n_comp).astype(np.float64),
        "f_mod_hz": rng.uniform(*cfg["f_mod_hz"], size=n_comp).astype(np.float64),
        "bessel_m_max": int(rng.integers(int(cfg["bessel_m_max"][0]), int(cfg["bessel_m_max"][1]) + 1)),
        "azimuth_start": a0,
        "azimuth_end": a1,
        "range_start": r0,
        "range_end": r1,
        "k_delta_a": float(rng.uniform(*cfg["k_delta_a"])),
        "phase0": float(rng.uniform(*cfg["phase0_rad"])),
        "shape": (n_az, n_rg),
    }


def bessel_terms(amplitude: float, fc_hz: float, beta: float, f_mod_hz: float, m_max: int):
    """Narrowband tones (D, fc, k_rfi=0) from the Eq. 4 Bessel expansion."""
    orders = np.arange(-int(m_max), int(m_max) + 1)
    weights = np.real(jv(orders, beta))
    keep = np.abs(weights) >= 1e-8
    amplitudes = amplitude * weights[keep]
    carriers = fc_hz + orders[keep] * f_mod_hz
    chirp = np.zeros(amplitudes.shape, dtype=np.float64)
    return amplitudes, carriers, chirp


def unified_terms(params: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amplitudes, carriers, and chirp rates of the Eq. 5 sum."""
    preset = params.get("preset", "chirp_wideband")
    amplitudes = np.atleast_1d(np.asarray(params["amplitudes"], dtype=np.float64))
    carriers = np.atleast_1d(np.asarray(params["fc_hz"], dtype=np.float64))
    chirp = np.atleast_1d(np.asarray(params["k_rfi_hz_s"], dtype=np.float64))
    if amplitudes.shape != carriers.shape or amplitudes.shape != chirp.shape:
        raise ValueError(
            "amplitudes, fc_hz, and k_rfi_hz_s must have the same length "
            f"(got {amplitudes.shape}, {carriers.shape}, {chirp.shape})"
        )
    if preset != "sinusoidal":
        return amplitudes, carriers, chirp
    beta = np.atleast_1d(np.asarray(params["beta"], dtype=np.float64))
    f_mod = np.atleast_1d(np.asarray(params["f_mod_hz"], dtype=np.float64))
    if beta.shape != amplitudes.shape or f_mod.shape != amplitudes.shape:
        raise ValueError("beta and f_mod_hz must have one entry per component")
    m_max = int(params["bessel_m_max"])
    parts_d, parts_f, parts_k = [], [], []
    for amp, fc, b, fm in zip(amplitudes, carriers, beta, f_mod):
        # Keep enough orders for the modulation index actually drawn.
        order = max(m_max, int(math.ceil(abs(float(b)))) + 8)
        d, f, k = bessel_terms(float(amp), float(fc), float(b), float(fm), order)
        parts_d.append(d)
        parts_f.append(f)
        parts_k.append(k)
    return (
        np.concatenate(parts_d),
        np.concatenate(parts_f),
        np.concatenate(parts_k),
    )


def unified_rfi(tau: np.ndarray, params: Mapping, apply_rect: bool = True) -> np.ndarray:
    """Complex 1-D interference on a fast-time grid, in seconds.

    ``s(tau) = rect(tau / t_rfi) * sum_n D_n * exp(j*2*pi*fc*tau + j*pi*k*tau^2)``.
    The rect is centered at tau = 0 and lasts ``t_rfi`` seconds.
    """
    tau = np.asarray(tau, dtype=np.float64)
    amplitudes, carriers, chirp = unified_terms(params)
    phase = (
        2.0 * np.pi * carriers[:, None] * tau[None, :]
        + np.pi * chirp[:, None] * tau[None, :] ** 2
    )
    signal = np.sum(amplitudes[:, None] * np.exp(1j * phase), axis=0)
    if apply_rect:
        half = 0.5 * float(params["t_rfi_s"])
        signal = signal * (np.abs(tau) <= half)
    return signal


def _azimuth_amplitude(n_az: int, params: Mapping) -> np.ndarray:
    """Eq. 19 on the azimuth sample grid. eta is [-0.5, 0.5] inside the rect."""
    a0 = int(params["azimuth_start"])
    a1 = int(params["azimuth_end"])
    amplitude = np.zeros(n_az, dtype=np.complex128)
    if a1 <= a0:
        raise ValueError("azimuth rect is empty")
    index = np.arange(a0, a1)
    center = 0.5 * (a0 + a1 - 1)
    half = 0.5 * (a1 - a0)
    eta = (index - center) / half * 0.5
    k_delta_a = float(params["k_delta_a"])
    if k_delta_a == 0.0:
        raise ValueError("k_delta_a must be nonzero")
    phase = -float(params["phase0"]) - np.pi * eta**2 / k_delta_a
    amplitude[a0:a1] = np.exp(1j * phase)
    return amplitude


def make_block_rfi(shape=(1500, 512), params=None, rng=None) -> np.ndarray:
    """Complex block interference in the range-line view, shape (azimuth, range).

    Each row is a 1-D range signal. The default ``(1500, 512)`` is the
    transpose of a stored RFI-Patch slice; see :mod:`utils.orientation`. Rows inside the azimuth rect carry
    Eq. 19's amplitude (rect window times a quadratic phase); columns
    outside the range rect are zero. The result is one rectangular block.
    """
    shape = (int(shape[0]), int(shape[1]))
    rng = _as_rng(rng)
    if params is None:
        params = sample_rfi_params(rng, shape=shape)
    else:
        params = dict(params)
    n_az, n_rg = shape
    r0 = int(params["range_start"])
    r1 = int(params["range_end"])
    if not (0 <= r0 < r1 <= n_rg):
        raise ValueError(f"range rect [{r0}, {r1}) is outside length {n_rg}")
    if not (0 <= int(params["azimuth_start"]) < int(params["azimuth_end"]) <= n_az):
        raise ValueError("azimuth rect falls outside the image")

    center = 0.5 * (r0 + r1 - 1)
    half = 0.5 * (r1 - r0)
    # Fast time spans t_rfi across the range rect and is zero at its center,
    # so the Eq. 6 rect and the pixel window describe the same support.
    tau = (np.arange(n_rg) - center) / half * (0.5 * float(params["t_rfi_s"]))
    pulse = unified_rfi(tau, params, apply_rect=False)
    pulse[:r0] = 0.0
    pulse[r1:] = 0.0
    image = _azimuth_amplitude(n_az, params)[:, None] * pulse[None, :]
    return image.astype(np.complex64)


def interference_mask(rfi: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Rectangle covering every sample where the interference is nonzero.

    The bounding box keeps internal nulls of a multi-tone sum inside the
    interfered area used by the SIR definition.
    """
    magnitude = np.abs(np.asarray(rfi))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0.0:
        raise ValueError("RFI image has no energy")
    rows, cols = np.nonzero(magnitude > rel_tol * peak)
    mask = np.zeros(magnitude.shape, dtype=bool)
    mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1] = True
    return mask


def scale_to_sir(clean, rfi, sir_db: float):
    """Add ``rfi`` to ``clean`` at the requested signal-to-interference ratio.

    SIR is the ratio of average target power to average interference power on
    the interfered area (the rect support), in dB:

        SIR_dB = 10 * log10(mean(|clean|^2) / mean(|rfi_scaled|^2))

    Returns the polluted complex image, with the same array type as ``clean``.
    """
    clean_is_torch = False
    try:
        import torch

        clean_is_torch = isinstance(clean, torch.Tensor) or isinstance(rfi, torch.Tensor)
    except ImportError:
        torch = None  # type: ignore

    if clean_is_torch:
        clean_np = clean.detach().cpu().numpy() if isinstance(clean, torch.Tensor) else np.asarray(clean)
        rfi_np = rfi.detach().cpu().numpy() if isinstance(rfi, torch.Tensor) else np.asarray(rfi)
    else:
        clean_np = np.asarray(clean)
        rfi_np = np.asarray(rfi)
    if clean_np.shape != rfi_np.shape:
        raise ValueError(f"clean shape {clean_np.shape} != rfi shape {rfi_np.shape}")
    mask = interference_mask(rfi_np)
    target_power = float(np.mean(np.abs(clean_np[mask]) ** 2))
    interference_power = float(np.mean(np.abs(rfi_np[mask]) ** 2))
    if target_power <= 0.0 or interference_power <= 0.0:
        raise ValueError("SIR is undefined when target or interference power is zero")
    sir_linear = 10.0 ** (float(sir_db) / 10.0)
    alpha = math.sqrt(target_power / (interference_power * sir_linear))
    polluted = np.asarray(clean_np) + (alpha * np.asarray(rfi_np))
    if clean_is_torch:
        out = torch.as_tensor(polluted)
        ref = clean if isinstance(clean, torch.Tensor) else rfi
        return out.to(dtype=ref.dtype)
    return polluted.astype(np.asarray(clean).dtype, copy=False)
