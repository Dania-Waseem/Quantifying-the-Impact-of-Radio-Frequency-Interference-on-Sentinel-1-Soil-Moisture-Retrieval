"""Notebook-style check of the block-type RFI synthesizer.

Builds one complex scene, overlays one rectangular block, and scales that
block to SIR = 0, -10, -20, and -30 dB. The figure is the visual check
against Figure 4 of Fang et al.: the interfered region is a bright block,
scene structure is still visible at 0 dB, and the block washes the scene
out at -20 dB and -30 dB.

Rows of the complex image are azimuth samples. Columns are range samples.
The panels are transposed so the horizontal axis is azimuth and the vertical
axis is range, matching the paper figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.rfi_model import make_block_rfi, sample_rfi_params, scale_to_sir

SIR_LEVELS_DB = (0.0, -10.0, -20.0, -30.0)
SHAPE = (512, 1500)


def make_clean_scene(shape, rng) -> np.ndarray:
    """Speckle scene: dark water, a brighter land mass, and point scatterers."""
    azimuth = np.linspace(0.0, 1.0, shape[0])[:, None]
    range_ax = np.linspace(0.0, 1.0, shape[1])[None, :]
    coast = range_ax - 0.32 - 0.12 * np.sin(2.0 * np.pi * azimuth)
    land = 0.2 + 1.1 / (1.0 + np.exp(-coast / 0.02))
    land = land + 0.35 * np.exp(-((azimuth - 0.62) ** 2) / 0.01) * (range_ax > 0.4)
    speckle = (
        rng.normal(scale=1.0 / np.sqrt(2.0), size=shape)
        + 1j * rng.normal(scale=1.0 / np.sqrt(2.0), size=shape)
    )
    scene = land * speckle
    n_points = 70
    rows = rng.integers(0, shape[0], size=n_points)
    cols = rng.integers(0, shape[1], size=n_points)
    amps = rng.uniform(6.0, 16.0, size=n_points)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    scene[rows, cols] += amps * np.exp(1j * phases)
    return scene


def to_db(image: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(image) + 1e-8)


def measured_sir_db(clean, polluted, rfi) -> float:
    from data.rfi_model import interference_mask

    mask = interference_mask(rfi)
    added = polluted - clean
    target = np.mean(np.abs(clean[mask]) ** 2)
    interference = np.mean(np.abs(added[mask]) ** 2)
    return float(10.0 * np.log10(target / interference))


def main() -> None:
    rng = np.random.default_rng(0)
    clean = make_clean_scene(SHAPE, rng)
    params = sample_rfi_params(rng, preset="chirp_wideband", shape=SHAPE)
    # Fixed central block so the four SIR panels share one geometry.
    # Most of the range axis and a central azimuth slice: a vertical bar
    # once azimuth is drawn horizontally, as in Figure 4.
    params["azimuth_start"] = 180
    params["azimuth_end"] = 340
    params["range_start"] = 80
    params["range_end"] = 1420
    params["n_components"] = 1
    params["amplitudes"] = np.array([1.0])
    params["fc_hz"] = np.array([5.0e6])
    params["bandwidth_hz"] = np.array([50.0e6])
    params["k_rfi_hz_s"] = params["bandwidth_hz"] / params["t_rfi_s"]
    rfi = make_block_rfi(SHAPE, params, rng)

    clean_db = to_db(clean)
    vmin = float(np.percentile(clean_db, 5))
    vmax = float(np.percentile(clean_db, 99))

    fig, axes = plt.subplots(len(SIR_LEVELS_DB), 3, figsize=(12, 14), constrained_layout=True)
    for row, sir_db in enumerate(SIR_LEVELS_DB):
        polluted = scale_to_sir(clean, rfi, sir_db)
        added = polluted - clean
        got = measured_sir_db(clean, polluted, rfi)
        panels = (
            (clean, "Clean"),
            (added, "RFI"),
            (polluted, f"Polluted\nSIR {sir_db:.0f} dB (measured {got:.2f})"),
        )
        for col, (image, title) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(
                to_db(image).T,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                aspect="auto",
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Azimuth")
            ax.set_ylabel("Range")
        print(f"SIR target {sir_db:.1f} dB, measured {got:.3f} dB")

    fig.suptitle(
        "Block-type RFI at the Table 2 SIR grid (horizontal = azimuth, vertical = range)",
        fontsize=12,
    )
    out = ROOT / "outputs" / "block_rfi_sir_panels.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
