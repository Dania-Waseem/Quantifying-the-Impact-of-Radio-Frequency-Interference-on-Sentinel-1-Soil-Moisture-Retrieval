"""Research diagrams drawn with matplotlib (Stage 6).

Four diagrams are produced as high-resolution PNG files in
``figures/diagrams/``:

  1. conceptual (research) model with hypotheses H1 to H6;
  2. research-design flowchart;
  3. system architecture in three panels (A data and injection, B detection
     and filling, C retrieval and evaluation) with the variants as branches;
  4. case study: irrigation advice under RFI, using only numbers from
     ``results/aggregate/stage6_case_study.csv``.

All drawing uses a coordinate system in which one unit is one hundredth of
the figure width, so that sizes and gaps are consistent between diagrams.
Colours come from :mod:`src.plotstyle`. Thumbnails show analytical
quantities (for example the injected error as a function of SIR and extent)
or schematic series, never data that is not computed by the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from . import plotstyle as ps
from .config import load_config, path
from .rfi import eps_db

# Module colours: fill (light tint) and edge (full colour).
MODULES = {
    "rfi": ("#FBE3D6", ps.VERMILLION),
    "mediator": ("#FCEFD2", ps.ORANGE),
    "outcome": ("#E2E6EA", ps.SLATE),
    "mitigation": ("#D6E8F5", ps.BLUE),
    "aware": ("#D3EFE6", ps.GREEN),
    "moderator": ("#E3F2FB", ps.SKY),
    "data": ("#EEF0F3", ps.SLATE),
}
TEXT = "#1F2A36"
MUTED = "#5B6773"


# --------------------------------------------------------------------------
# Drawing toolkit
# --------------------------------------------------------------------------

class Canvas:
    """A figure with a 100-unit wide coordinate system and drawing helpers."""

    def __init__(self, height_units: float, width_in: float | None = None):
        ps.apply()
        self.w_in = width_in or ps.width("double")
        self.h = height_units
        self.fig = plt.figure(figsize=(self.w_in, self.w_in * height_units / 100))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, height_units)
        self.ax.axis("off")
        self.ax.grid(False)

    # Text sizes in points, chosen for a 7.16 inch wide figure.
    def text(self, x, y, s, size=7.5, weight="normal", color=TEXT, ha="center",
             va="center", style="normal", **kw):
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
                            va=va, fontstyle=style, linespacing=1.25, zorder=6, **kw)

    def panel(self, x, y, w, h, title, letter=None):
        """A light panel with a title bar."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                         fc="#F7F8FA", ec="#C9D0D7", lw=0.8, zorder=0))
        label = f"{letter}  {title}" if letter else title
        self.text(x + 1.6, y + h - 2.0, label, size=8.5, weight="bold", ha="left")

    def box(self, x, y, w, h, title, body=None, module="data", dashed=False, title_size=7.6,
            body_size=6.6, title_dx=0.0):
        """A rounded box with a bold title and optional body text (centre-aligned)."""
        fc, ec = MODULES[module]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                         fc=fc, ec=ec, lw=1.1, ls=(0, (3, 2)) if dashed else "-",
                                         zorder=3))
        if body:
            self.text(x + w / 2 + title_dx, y + h - 2.1, title, size=title_size, weight="bold")
            self.text(x + w / 2, y + h - 4.0, body, size=body_size, color=MUTED, va="top")
        else:
            self.text(x + w / 2, y + h / 2, title, size=title_size, weight="bold")
        return (x, y, w, h)

    def arrow(self, p, q, color=TEXT, dashed=False, lw=1.0, rad=0.0, head=6):
        """A straight or curved arrow from p to q."""
        a = FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head / 10},head_width={head / 20}",
                            color=color, lw=lw, ls=(0, (3, 2)) if dashed else "-",
                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                            mutation_scale=10, zorder=4)
        self.ax.add_patch(a)

    def tag(self, x, y, s, color=TEXT, size=6.8):
        """A pill-shaped label (used for hypotheses on arrows)."""
        self.ax.text(x, y, s, fontsize=size, fontweight="bold", color="white", ha="center",
                     va="center", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.8", fc=color, ec="none"))

    def number(self, x, y, n, color=ps.SLATE, r=1.35):
        """A numbered circle linking text to a block."""
        self.ax.add_patch(Circle((x, y), r, fc=color, ec="white", lw=0.8, zorder=8))
        t = self.text(x, y - 0.05, str(n), size=6.6, weight="bold", color="white")
        t.set_zorder(9)

    def heatmap(self, x, y, w, h, z, cmap="Oranges", label=None):
        """A small heatmap thumbnail inside the rectangle (x, y, w, h)."""
        ax = self.fig.add_axes([x / 100, y / self.h, w / 100, h / self.h])
        ax.imshow(z, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#C9D0D7")
            s.set_linewidth(0.6)
        if label:
            self.text(x + w / 2, y - 1.3, label, size=5.6, color=MUTED)
        return ax

    def sparkline(self, x, y, w, h, series, flags=None, color=ps.SLATE, label=None):
        """A small line thumbnail with optional flagged points."""
        ax = self.fig.add_axes([x / 100, y / self.h, w / 100, h / self.h])
        t = np.arange(len(series))
        ax.plot(t, series, color=color, lw=0.8)
        if flags is not None:
            ax.plot(t[flags], np.asarray(series)[flags], "o", ms=2.6, color=ps.VERMILLION)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor("#C9D0D7")
            s.set_linewidth(0.6)
        ax.set_facecolor("white")
        if label:
            self.text(x + w / 2, y - 1.3, label, size=5.6, color=MUTED)
        return ax

    def legend(self, x, y, items, title=None, ncol=None):
        """A row of legend items: (kind, label, module or colour)."""
        if title:
            self.text(x, y, title, size=6.8, weight="bold", ha="left")
            x += len(title) * 0.66 + 2
        for kind, label, spec in items:
            if kind == "box":
                fc, ec = MODULES[spec]
                self.ax.add_patch(FancyBboxPatch((x, y - 1.0), 3.2, 2.0,
                                                 boxstyle="round,pad=0,rounding_size=0.4",
                                                 fc=fc, ec=ec, lw=0.9, zorder=5))
            elif kind == "dashedbox":
                fc, ec = MODULES[spec]
                self.ax.add_patch(FancyBboxPatch((x, y - 1.0), 3.2, 2.0,
                                                 boxstyle="round,pad=0,rounding_size=0.4",
                                                 fc="white", ec=ec, lw=0.9, ls=(0, (3, 2)),
                                                 zorder=5))
            elif kind in ("arrow", "dashedarrow"):
                self.arrow((x, y), (x + 3.6, y), color=spec, dashed=kind == "dashedarrow")
            self.text(x + 4.4, y, label, size=6.4, ha="left", color=TEXT)
            x += 4.4 + len(label) * 0.62 + 2.4

    def save(self, name: str) -> str:
        """Save as PNG only, at the configured diagram resolution."""
        folder = path("diagrams")
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{name}.png"
        import matplotlib as mpl
        with mpl.rc_context({"savefig.bbox": "standard"}):
            self.fig.savefig(f, dpi=load_config()["plots"]["diagram_dpi"], facecolor="white")
        plt.close(self.fig)
        return str(f)


def mid(b, side):
    """Anchor point on a box (x, y, w, h): 'l', 'r', 't', 'b' or 'c'."""
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h),
            "b": (x + w / 2, y), "c": (x + w / 2, y + h / 2)}[side]


def eps_grid() -> tuple[np.ndarray, list, list]:
    """Injected error (dB, mean of VV and VH) on the injection grid (analytical)."""
    from .rfi import injection_grid

    g = injection_grid(verbose=False)
    z = np.array([[np.mean([float(eps_db(g["sir"][q][p], r)) for p in ("VV", "VH")])
                   for q in g["severity_q"]] for r in g["extent"]])
    return z, g["severity_q"], g["extent"]


# --------------------------------------------------------------------------
# Diagram 1: conceptual model
# --------------------------------------------------------------------------

def conceptual_model() -> str:
    """Conceptual model: RFI -> backscatter error -> soil moisture error."""
    c = Canvas(58)
    c.text(50, 55.2, "Conceptual model of RFI impact and mitigation", size=10, weight="bold")

    sev = c.box(4, 33, 21, 10, "RFI severity", "signal-to-interference\nratio SIR (dB)", "rfi")
    ext = c.box(4, 17, 21, 10, "RFI extent", "affected fraction ρ\nand probability pᶜ",
                "rfi")
    med = c.box(40, 25, 22, 11, "Backscatter error", "ε = 10 log10(1 + ρ 10^(−SIR/10))\nmediator (dB)", "mediator")
    out = c.box(76, 25, 21, 11, "Soil moisture error", "|retrieved − in situ|\n(m³/m³)",
                "outcome")
    mit = c.box(40, 43, 22, 8.5, "Mitigation (treatment)", "M3 detect-and-fill,\nM3b detect-and-discard",
                "mitigation")
    awa = c.box(76, 43, 21, 8.5, "RFI-aware training", "M4: model trained on\nclean + injected data",
                "aware")
    ndv = c.box(40, 6, 22, 7.5, "Vegetation (NDVI)", "terciles of NDVI", "moderator")
    pol = c.box(76, 6, 21, 7.5, "Polarisation", "VV or VH affected", "moderator")

    # Main causal chain.
    c.arrow(mid(sev, "r"), (40, 32.5), color=ps.VERMILLION, lw=1.3)
    c.arrow(mid(ext, "r"), (40, 28.5), color=ps.VERMILLION, lw=1.3)
    c.arrow(mid(med, "r"), mid(out, "l"), color=ps.ORANGE, lw=1.5)
    c.tag(31.5, 36.2, "H1", ps.VERMILLION)
    c.tag(31.5, 24.6, "H2", ps.VERMILLION)
    # Mitigation acts on the backscatter error (a path), and through it on
    # the soil moisture error (H3 mediation, H4 recovery).
    c.arrow(mid(mit, "b"), mid(med, "t"), color=ps.BLUE, lw=1.3)
    c.tag(53.6, 39.6, "H3", ps.BLUE)
    c.arrow((62, 47.2), (80, 36), color=ps.BLUE, lw=1.3, rad=-0.18)
    c.tag(70.8, 44.6, "H4", ps.BLUE)
    c.arrow(mid(awa, "b"), (88, 36), color=ps.GREEN, lw=1.3)
    c.tag(91.6, 39.6, "H5", ps.GREEN)
    # Moderators act on the link from backscatter error to soil moisture error.
    c.arrow((60, 13.5), (67, 30.3), color=ps.SKY, dashed=True, lw=1.2, rad=-0.15)
    c.arrow((78, 13.5), (71, 30.3), color=ps.SKY, dashed=True, lw=1.2, rad=0.15)
    c.tag(69, 20.5, "H6", ps.SKY)

    c.legend(4, 1.6, [("box", "RFI factors", "rfi"), ("box", "mediator", "mediator"),
                      ("box", "outcome", "outcome"), ("box", "treatment", "mitigation"),
                      ("box", "training", "aware"), ("box", "moderators", "moderator"),
                      ("arrow", "effect", TEXT), ("dashedarrow", "moderation", ps.SKY)])
    return c.save("diagram1_conceptual_model")


# --------------------------------------------------------------------------
# Diagram 2: research-design flowchart
# --------------------------------------------------------------------------

def research_design() -> str:
    """Research-design flowchart from data collection to statistics."""
    cfg = load_config()
    c = Canvas(50)
    c.text(50, 47.4, "Research design", size=10, weight="bold")
    steps = [
        ("Data collection", "Sentinel-1 GRD (VV, VH)\nSentinel-2 NDVI, ERA5-Land\nISMN USCRN and OZNET", "data"),
        ("Matchup", f"one orbit per station\nISMN within {cfg['matchup']['ismn_tolerance_minutes']} min\nMIN_N = {cfg['matchup']['min_n']}", "data"),
        ("Clean model", "RF, XGBoost, change\ndetection; F1, F2, F3\nstation-wise 5-fold CV", "outcome"),
        ("RFI injection", "σ̃ = σ (1 + ρ 10^(−SIR/10))\nOpenSAR severity, extent\n10 seeds per setting", "rfi"),
        ("Mitigation variants", "M0, M1, M3, M3b, M4\n(M2 reserved)", "mitigation"),
        ("Evaluation", "RMSE, ubRMSD, R², R\nD, G, RR\nUSCRN, OZNET", "mediator"),
        ("Statistics", f"cluster bootstrap ({cfg['stats']['n_bootstrap']:,})\nWilcoxon, Holm, mediation,\nmoderation, power", "aware"),
    ]
    w, h, gap = 20.5, 16.0, 4.0
    xs = [3 + i * (w + gap) for i in range(4)]
    # First row left to right; second row right to left, centred under it.
    pos = [(xs[i], 26.5) for i in range(4)] + [(xs[2 - i] + (w + gap) / 2, 6.0)
                                              for i in range(3)]
    boxes = []
    for i, ((t, b, m), (x, y)) in enumerate(zip(steps, pos)):
        bx = c.box(x, y, w, h, t, b, m)
        c.number(x + 1.9, y + h - 1.9, i + 1)
        boxes.append(bx)
    for i in range(3):
        c.arrow(mid(boxes[i], "r"), mid(boxes[i + 1], "l"), lw=1.2)
    # Turn from the end of the first row down to the second row.
    x3, y3, _, _ = boxes[3]
    x4, y4, w4, h4 = boxes[4]
    c.arrow((x3 + w / 2, y3), (x4 + w4 - 3, y4 + h4), lw=1.2, rad=-0.35)
    for i in range(4, 6):
        c.arrow(mid(boxes[i], "l"), mid(boxes[i + 1], "r"), lw=1.2)
    # Thumbnail: injected error on the provisional grid (analytical).
    z, _, _ = eps_grid()
    c.heatmap(xs[3] + 13.0, 27.6, 6.0, 4.6, z, label=None)
    c.text(xs[3] + 12.2, 29.9, "ε over severity × ρ", size=5.6, color=MUTED, ha="right")
    c.legend(3, 2.2, [("box", "data", "data"), ("box", "model", "outcome"),
                      ("box", "RFI", "rfi"), ("box", "mitigation", "mitigation"),
                      ("box", "evaluation", "mediator"), ("box", "inference", "aware")])
    return c.save("diagram2_research_design")


# --------------------------------------------------------------------------
# Diagram 3: system architecture
# --------------------------------------------------------------------------

def route(c: "Canvas", pts, color, lw=1.3):
    """A right-angled path through the points, with an arrow head at the end."""
    xs, ys = zip(*pts)
    c.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw,
              solid_capstyle="round", solid_joinstyle="round", zorder=4)
    c.arrow(pts[-2], pts[-1], color=color, lw=lw)


def architecture() -> str:
    """System architecture in three panels with the variants as branches.

    Each variant follows its own coloured lane into the retrieval models.
    The lanes run in the gutters between the panels and are ordered so that
    they do not cross. A strip at the bottom traces each variant through the
    numbered blocks.
    """
    cfg = load_config()
    det = cfg["detection"]
    col = ps.VARIANT_COLOURS
    c = Canvas(74)
    c.text(50, 71.6, "System architecture", size=10, weight="bold")
    c.panel(2, 25, 29, 41, "Data and injection", "A")
    c.panel(35, 25, 29, 41, "Detection and filling", "B")
    c.panel(70, 25, 28, 41, "Retrieval and evaluation", "C")

    # Panel A.
    a1 = c.box(5, 52, 23, 8.5, "Matchup table", "S1 VV, VH, angle; NDVI;\nprecipitation; site", "data")
    a2 = c.box(5, 40, 23, 8, "Clean test inputs", "station-wise test fold", "data")
    a3 = c.box(5, 27.5, 23, 9.5, "RFI injection", "σ̃ = σ (1 + ρ 10^(−SIR/10))\npᶜ; VV, VH or both", "rfi",
               dashed=True)
    c.arrow(mid(a1, "b"), mid(a2, "t"))
    c.arrow(mid(a2, "b"), mid(a3, "t"))

    # Panel B.
    b1 = c.box(38, 51, 23, 9.5, "Robust z-score", f"rolling median and MAD\nw = {det['window']}, τ = {det['tau']}, one-sided", "mitigation", dashed=True)
    b2 = c.box(38, 38, 11, 9, "Fill", f"median of\nprevious k = {cfg['fill']['k']}", "mitigation",
               dashed=True, title_dx=1.2)
    b3 = c.box(50, 38, 11, 9, "Discard", "drop flagged\nacquisitions", "mediator", dashed=True,
               title_dx=1.3)
    c.arrow((43.5, 51), (43.5, 47))
    c.arrow((55.5, 51), (55.5, 47))
    series = np.array([0, .2, -.1, .1, 0, 2.4, .1, -.2, .1, 2.0, 0, .1])
    flags = np.zeros(len(series), bool)
    flags[[5, 9]] = True
    c.sparkline(40, 26.4, 19, 5.0, series, flags)
    c.text(49.5, 32.6, "schematic series, flagged spikes", size=5.6, color=MUTED)

    # Panel C.
    c1 = c.box(73, 49, 23, 11, "Retrieval models", "RF and XGBoost (F1, F2, F3)\nM4: RFI-aware RF and XGBoost\ntrained on clean + injected data", "outcome")
    c3 = c.box(73, 38.5, 23, 7.5, "Change detection", "per-station references (VV)\nsame inputs as above", "outcome", dashed=True)
    c4 = c.box(73, 27.5, 23, 8, "Evaluation", "RMSE, ubRMSD, R², R\nD, G, RR per variant", "mediator")
    c.arrow(mid(c3, "b"), mid(c4, "t"))
    route(c, [(96, 54.5), (97.2, 54.5), (97.2, 31.5), (96, 31.5)], TEXT, lw=1.0)

    # Contaminated series from A to the detector (neutral colour).
    route(c, [(28, 32.2), (33.4, 32.2), (33.4, 55.8), (38, 55.8)], TEXT, lw=1.0)

    # Variant lanes into the retrieval models. Lane order (left to right) and
    # entry heights (top to bottom) are chosen so that no two lanes cross.
    route(c, [(28, 44), (32.2, 44), (32.2, 68.2), (69.2, 68.2), (69.2, 58.6), (73, 58.6)],
          col["M0"])
    route(c, [(61, 42.5), (65.0, 42.5), (65.0, 56.7), (73, 56.7)], col["M3b"])
    route(c, [(45.5, 38), (45.5, 36.2), (66.0, 36.2), (66.0, 54.8), (73, 54.8)], col["M3"])
    route(c, [(42.0, 38), (42.0, 34.6), (67.0, 34.6), (67.0, 52.9), (73, 52.9)], col["M4"])
    route(c, [(16.5, 27.5), (16.5, 23.4), (68.0, 23.4), (68.0, 51.0), (73, 51.0)], col["M1"])

    # Numbered circles on the blocks.
    blocks = [a1, a2, a3, b1, b2, b3, c1, c3, c4]
    for i, b in enumerate(blocks):
        c.number(b[0] + 1.8, b[1] + b[3] - 1.8, i + 1)

    # Variant branches strip.
    c.panel(2, 2, 96, 19, "Variant branches (numbers refer to the blocks above)")
    paths = [("M0", [2, 7, 9], "clean input, clean-trained model"),
             ("M1", [3, 7, 9], "contaminated input, no mitigation"),
             ("M3", [3, 4, 5, 7, 9], "detect-and-fill"),
             ("M3b", [3, 4, 6, 7, 9], "detect-and-discard (fewer rows)"),
             ("M4", [3, 4, 5, 7, 9], "detect-and-fill, RFI-aware model")]
    y = 15.6
    for v, seq, desc in paths:
        c.tag(7.5, y, v, col[v], size=6.6)
        x = 14
        for j, n in enumerate(seq):
            c.number(x, y, n, color=col[v], r=1.15)
            if j < len(seq) - 1:
                c.arrow((x + 1.4, y), (x + 3.0, y), color=MUTED, lw=0.8, head=4)
            x += 4.4
        c.text(37.5, y, desc, size=6.6, ha="left")
        y -= 2.9
    c.legend(64, 15.6, [("box", "trained component", "outcome")])
    c.legend(64, 12.7, [("dashedbox", "fixed rule (no training)", "mitigation")])
    c.legend(64, 9.8, [("arrow", "data flow", TEXT)])
    c.text(64, 6.6, "M2 (published suppression) is reserved\nand not part of this diagram.",
           size=6.2, color=MUTED, ha="left", va="center")
    return c.save("diagram3_system_architecture")


# --------------------------------------------------------------------------
# Diagram 4: case study
# --------------------------------------------------------------------------

def case_study_table() -> pd.DataFrame:
    """Compute the irrigation-advice case study from the Stage 4 row data.

    Advice is 'irrigate' when soil moisture is below the configured
    quantile of the in situ soil moisture of the evaluation. For each
    variant, the share of acquisitions whose advice differs from the in situ
    advice is reported for all rows and for the contaminated rows only,
    pooled over the injection seeds. Only aggregated numbers are saved.
    """
    cfg = load_config()
    cs = cfg["case_study"]
    rows = pd.read_csv(path("processed") / "stage4_rows.csv.gz")
    scen = pd.read_csv(path("results_aggregate") / "stage4_scenarios.csv")
    ref = int(scen[scen["roles"].str.contains("reference")]["scenario"].iloc[0])
    r = rows[(rows["scenario"] == ref) & (rows["evaluation"] == cs["evaluation"])]
    thr = float(np.quantile(r.loc[r["seed"] == 0, "sm"], cs["threshold_quantile"]))
    truth = r["sm"].to_numpy() < thr
    hit = (r["inj_VV"] | r["inj_VH"]).to_numpy(bool)
    out = []
    for v in cs["variants"]:
        p = r[f"{cs['feature_set']}|{cs['model']}|{v}"].to_numpy(float)
        adv = p < thr
        wrong = adv != truth
        out.append({"variant": v, "threshold_m3m3": thr, "n_rows": len(r),
                    "n_contaminated": int(hit.sum()),
                    "wrong_advice_all": float(wrong.mean()),
                    "wrong_advice_contaminated": float(wrong[hit].mean()),
                    "missed_irrigation_contaminated": float((truth & ~adv)[hit].mean()),
                    "unneeded_irrigation_contaminated": float((~truth & adv)[hit].mean()),
                    "mean_abs_error_contaminated": float(np.abs(p - r["sm"].to_numpy())[hit]
                                                         .mean())})
    t = pd.DataFrame(out)
    t.to_csv(path("results_aggregate") / "stage6_case_study.csv", index=False)
    return t


def case_study() -> str:
    """Irrigation-advice scenario with the numbers of the case-study table."""
    cfg = load_config()
    cs = cfg["case_study"]
    from .rfi import extent_value, injection_grid

    t = case_study_table().set_index("variant")
    thr = t["threshold_m3m3"].iloc[0]
    ref = cfg["stage4"]["reference"]
    g = injection_grid(verbose=False)
    rho = extent_value(g, ref["extent_q"])
    e_vv = float(eps_db(g["sir"][ref["severity_q"]]["VV"], rho))
    e_vh = float(eps_db(g["sir"][ref["severity_q"]]["VH"], rho))
    c = Canvas(54)
    c.text(50, 51.4, "Case study: irrigation advice from retrieved soil moisture", size=10,
           weight="bold")
    c.text(50, 48.0, f"USCRN, {cs['feature_set']} {cs['model'].upper()}, reference scenario "
           f"(OpenSAR median severity and extent: ρ = {rho:.3f}, pᶜ = {ref['p_c']}, "
           f"{ref['polarisation']} polarisations)", size=6.8, color=MUTED)

    s1 = c.box(3, 31, 20, 11, "Satellite overpass", "Sentinel-1 VV, VH\nat the field", "data")
    s2 = c.box(28, 31, 20, 11, "RFI event", f"pᶜ = {ref['p_c']} of overpasses\n"
               f"ε = {e_vv:.2f} dB (VV), {e_vh:.2f} dB (VH)", "rfi")
    s3 = c.box(53, 31, 20, 11, "Soil moisture estimate", "retrieval model\nwith or without\nmitigation", "outcome", title_dx=1.4)
    s4 = c.box(78, 31, 19, 11, "Advice rule", f"irrigate if\nSM < {thr:.3f} m³/m³\n"
               f"({int(cs['threshold_quantile'] * 100)}th percentile)", "mediator")
    for a, b in ((s1, s2), (s2, s3), (s3, s4)):
        c.arrow(mid(a, "r"), mid(b, "l"), lw=1.2)
    for i, b in enumerate((s1, s2, s3, s4)):
        c.number(b[0] + 1.9, b[1] + b[3] - 1.9, i + 1)

    # Result bars: share of wrong advice on contaminated overpasses.
    c.panel(3, 4, 94, 22, "Share of contaminated overpasses with wrong advice")
    names = {"M0": "M0 clean input", "M1": "M1 RFI, no mitigation", "M3": "M3 detect-and-fill",
             "M4": "M4 fill + RFI-aware"}
    vmax = max(t["wrong_advice_contaminated"].max(), 1e-9)
    y = 19.5
    for v in cs["variants"]:
        val = t.loc[v, "wrong_advice_contaminated"]
        c.text(5, y, names.get(v, v), size=7, ha="left")
        L = 50 * val / vmax
        c.ax.add_patch(Rectangle((27, y - 1.3), L, 2.6, fc=ps.VARIANT_COLOURS[v], ec="none",
                                 zorder=5))
        c.text(27 + L + 1.2, y, f"{val * 100:.1f}%  (MAE {t.loc[v, 'mean_abs_error_contaminated']:.3f})",
               size=6.8, ha="left")
        y -= 4.2
    c.text(95.5, 24.0, f"n = {int(t['n_contaminated'].iloc[0]):,} contaminated overpasses "
           f"(10 seeds)", size=6.0, color=MUTED, ha="right")
    return c.save("diagram4_case_study")


def make_all(with_case_study: bool = True) -> list[str]:
    """Draw all diagrams and return the file paths."""
    out = [conceptual_model(), research_design(), architecture()]
    if with_case_study:
        out.append(case_study())
    return out
