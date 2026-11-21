"""Shared figure style: palette, fonts, sizes and saving.

Every figure and diagram uses this module, so that the same variant always
has the same colour, and all text is readable at IEEE column width
(3.5 inch single column, 7.16 inch double column).

The palette is a five-colour subset of the Okabe-Ito colour-blind-safe
palette plus a neutral slate and grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import load_config, path

# Five main colours (Okabe-Ito) plus neutrals.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
SLATE = "#3B4B5C"
GREY = "#9AA3AD"
LIGHT = "#E9EDF1"

# Fixed colour per variant, used in every figure.
VARIANT_COLOURS = {
    "M0": SLATE,        # clean input
    "M1": VERMILLION,   # contaminated, no mitigation
    "M2": GREY,         # published suppression (not available)
    "M3": BLUE,         # detect-and-fill
    "M3b": ORANGE,      # detect-and-discard
    "M4": GREEN,        # detect-and-fill with RFI-aware training
    "M4_nofill": SKY,   # ablation: RFI-aware model without fill
}
VARIANT_LABELS = {
    "M0": "M0 clean", "M1": "M1 RFI", "M2": "M2 published", "M3": "M3 fill",
    "M3b": "M3b discard", "M4": "M4 fill + aware", "M4_nofill": "M4 without fill",
}
NETWORK_COLOURS = {"USCRN": BLUE, "OZNET": ORANGE}
# Feature sets are distinguished by line style and marker (colour is kept
# for the variants).
FSET_STYLES = {
    "F1": {"linestyle": "-", "marker": "o"},
    "F2": {"linestyle": "--", "marker": "s"},
    "F3": {"linestyle": ":", "marker": "^"},
    "VV": {"linestyle": "-.", "marker": "D"},
}
FSET_SHADES = {"F1": SLATE, "F2": GREY, "F3": SKY, "VV": ORANGE}
FONT = "Arial"


def apply() -> None:
    """Set the matplotlib defaults for all figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "axes.grid": True,
        "grid.color": LIGHT,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,       # embed TrueType fonts in PDF
    })


def width(kind: str = "single") -> float:
    """Figure width in inches for a single or double IEEE column."""
    p = load_config()["plots"]
    return p["single_column_in"] if kind == "single" else p["double_column_in"]


def save(fig: plt.Figure, name: str, folder: Path | None = None) -> list[Path]:
    """Save a figure as PNG (400 dpi) and as vector PDF, then close it."""
    folder = folder or path("figures")
    folder.mkdir(parents=True, exist_ok=True)
    dpi = load_config()["plots"]["dpi"]
    out = [folder / f"{name}.png", folder / f"{name}.pdf"]
    fig.savefig(out[0], dpi=dpi)
    # No creation date in the PDF metadata, so that reruns give identical files.
    fig.savefig(out[1], metadata={"CreationDate": None})
    plt.close(fig)
    return out


def panel_label(ax, text: str) -> None:
    """Write a panel label such as '(a)' in the upper left corner."""
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold")
