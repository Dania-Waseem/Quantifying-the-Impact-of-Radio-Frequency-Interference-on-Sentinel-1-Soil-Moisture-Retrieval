"""Result figures 5 to 14 (figures 1 to 4 are the diagrams in diagrams.py).

Each figure is drawn only from saved result files, so it can be regenerated
without rerunning any experiment. Figures that would show per-station or
per-row values are either built from aggregated tables (histogram counts)
or, for the station map, written to ``figures/private/``, which is not
tracked. Every figure is saved as PNG (400 dpi) and as vector PDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import plotstyle as ps
from .config import load_config, path
from .rfi import eps_db

FSETS = ["F1", "F2", "F3"]
EV_LABEL = {"USCRN_cv": "USCRN (CV)", "OZNET_cv": "OZNET (CV, small)",
            "OZNET_transfer": "OZNET (transfer)"}


def agg(name: str) -> pd.DataFrame:
    """Read an aggregated result table."""
    return pd.read_csv(path("results_aggregate") / f"{name}.csv")


def primary() -> tuple[str, str]:
    """Primary evaluation and model for the hypothesis figures."""
    s4 = load_config()["stage4"]
    return s4["primary_evaluation"], s4["primary_model"]


# --------------------------------------------------------------------------
# Aggregated inputs for the descriptive figures (built from private files)
# --------------------------------------------------------------------------

def build_descriptive_tables() -> None:
    """Write the aggregated histogram tables used by figures 6 and 7.

    Inputs are the local per-station and per-row files; outputs are counts
    per bin only (``results/aggregate/stage5_*.csv``). The station locations
    for the map are written to ``results/private/`` (not tracked).
    """
    from .data import load_matchups

    m = load_matchups()
    st = m.groupby(["network", "station_id"]).agg(lat=("lat", "first"), lon=("lon", "first"),
                                                  matchups=("sm", "size"), lc=("lc", "first"))
    st.reset_index().to_csv(path("results_private") / "station_locations.csv", index=False)

    bins = np.arange(40, 240, 10)
    rows = []
    for net, g in st.reset_index().groupby("network"):
        h, e = np.histogram(g["matchups"], bins=bins)
        rows += [{"network": net, "bin_low": a, "bin_high": b, "count": int(c)}
                 for a, b, c in zip(e[:-1], e[1:], h)]
    pd.DataFrame(rows).to_csv(path("results_aggregate") / "stage5_matchups_hist.csv", index=False)

    lc = st.reset_index().groupby(["network", "lc"]).size().rename("stations").reset_index()
    lc.to_csv(path("results_aggregate") / "stage5_landcover_counts.csv", index=False)

    nb = np.linspace(0, 1, 21)
    rows = []
    for net, g in m.groupby("network"):
        h, e = np.histogram(g["ndvi"], bins=nb)
        rows += [{"network": net, "bin_low": a, "bin_high": b, "count": int(c)}
                 for a, b, c in zip(e[:-1], e[1:], h)]
    pd.DataFrame(rows).to_csv(path("results_aggregate") / "stage5_ndvi_hist.csv", index=False)

    # Histogram counts of the OpenSAR patch statistics (figure 8).
    from .rfi import load_opensar_stats

    o = load_opensar_stats()
    if o is not None:
        rows = []
        for var, edges in (("severity_db_vv", np.arange(-2.0, 12.01, 0.5)),
                           ("severity_db_vh", np.arange(-2.0, 12.01, 0.5)),
                           ("extent", np.linspace(0, 1, 21))):
            h, e = np.histogram(o[var], bins=edges)
            rows += [{"variable": var, "bin_low": lo, "bin_high": hi, "count": int(c)}
                     for lo, hi, c in zip(e[:-1], e[1:], h)]
        pd.DataFrame(rows).to_csv(path("results_aggregate") / "stage5_opensar_hist.csv",
                                  index=False)

    # 2D histogram of predicted against in situ soil moisture (figure 7).
    p = pd.read_csv(path("processed") / "stage2_predictions.csv")
    p = p[p["setting"].isin(["USCRN_cv", "OZNET_transfer"]) & p["feature_set"].isin(FSETS)
          & (p["model"] == "rf")]
    edges = np.linspace(0, 0.55, 45)
    rows = []
    for (s, f), g in p.groupby(["setting", "feature_set"]):
        h, _, _ = np.histogram2d(g["sm"], g["pred"], bins=[edges, edges])
        ii, jj = np.nonzero(h)
        rows += [{"setting": s, "feature_set": f, "obs_low": edges[i], "obs_high": edges[i + 1],
                  "pred_low": edges[j], "pred_high": edges[j + 1], "count": int(h[i, j])}
                 for i, j in zip(ii, jj)]
    pd.DataFrame(rows).to_csv(path("results_aggregate") / "stage5_pred_obs_hist2d.csv",
                              index=False)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig05_station_map() -> list:
    """Station map in two panels (USA and Australia), coloured by network.

    Written to figures/private (per-station locations are not tracked).
    """
    ps.apply()
    st = pd.read_csv(path("results_private") / "station_locations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(ps.width("double"), 2.6),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    for ax, net, (xl, yl) in zip(axes, ["USCRN", "OZNET"],
                                 [((-170, -65), (17, 72)), ((140, 152), (-38, -26))]):
        g = st[st["network"] == net]
        ax.scatter(g["lon"], g["lat"], s=10, color=ps.NETWORK_COLOURS[net], edgecolor="white",
                   linewidth=0.4, label=f"{net} ({len(g)} stations)", zorder=3)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower left")
    ps.panel_label(axes[0], "(a)")
    ps.panel_label(axes[1], "(b)")
    folder = path("figures") / "private"
    return ps.save(fig, "fig05_station_map", folder)


def fig06_sample_profile() -> list:
    """Sample profile: stations, matchups per station, land cover, NDVI."""
    ps.apply()
    net = agg("screening_per_network")
    mh = agg("stage5_matchups_hist")
    lc = agg("stage5_landcover_counts")
    nd = agg("stage5_ndvi_hist")
    fig, ax = plt.subplots(1, 4, figsize=(ps.width("double"), 1.9))
    ax[0].bar(net["network"], net["stations"],
              color=[ps.NETWORK_COLOURS[n] for n in net["network"]], width=0.6)
    for i, v in enumerate(net["stations"]):
        ax[0].text(i, v + 1, str(v), ha="center", fontsize=7)
    ax[0].set_ylabel("Stations")
    for n, g in mh.groupby("network"):
        ax[1].bar(g["bin_low"] + 5, g["count"], width=9, alpha=0.75,
                  color=ps.NETWORK_COLOURS[n], label=n)
    ax[1].set_xlabel("Matchups per station")
    ax[1].set_ylabel("Stations")
    ax[1].legend()
    names = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop", 60: "Bare", 90: "Wetland"}
    classes = sorted(lc["lc"].unique())
    x = np.arange(len(classes))
    for k, n in enumerate(["USCRN", "OZNET"]):
        g = lc[lc["network"] == n].set_index("lc").reindex(classes)["stations"].fillna(0)
        ax[2].bar(x + (k - 0.5) * 0.38, g.to_numpy(), width=0.38, color=ps.NETWORK_COLOURS[n])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([names.get(c, str(c)) for c in classes], rotation=45, ha="right")
    ax[2].set_ylabel("Stations")
    for n, g in nd.groupby("network"):
        dens = g["count"] / g["count"].sum() / (g["bin_high"] - g["bin_low"])
        ax[3].step(g["bin_low"], dens, where="post", color=ps.NETWORK_COLOURS[n], label=n)
    ax[3].set_xlabel("NDVI (-)")
    ax[3].set_ylabel("Density")
    for a, lab in zip(ax, "abcd"):
        ps.panel_label(a, f"({lab})")
    fig.tight_layout(w_pad=1.2)
    return ps.save(fig, "fig06_sample_profile")


def fig07_clean_model() -> list:
    """Clean model: predicted against ISMN soil moisture (2D histograms)."""
    ps.apply()
    h = agg("stage5_pred_obs_hist2d")
    m = agg("stage2_metrics")
    fig, axes = plt.subplots(2, 3, figsize=(ps.width("double"), 4.4), sharex=True, sharey=True)
    for i, s in enumerate(["USCRN_cv", "OZNET_transfer"]):
        for j, f in enumerate(FSETS):
            ax = axes[i, j]
            g = h[(h["setting"] == s) & (h["feature_set"] == f)]
            edges = np.unique(np.r_[g["obs_low"], g["obs_high"]])
            ax.scatter((g["obs_low"] + g["obs_high"]) / 2, (g["pred_low"] + g["pred_high"]) / 2,
                       c=np.log10(g["count"]), s=6, marker="s", cmap="Blues", vmin=0,
                       linewidths=0)
            ax.plot([0, 0.55], [0, 0.55], color=ps.GREY, lw=0.8, ls="--")
            mm = m[(m["setting"] == s) & (m["feature_set"] == f) & (m["model"] == "rf")
                   & (m["scope"] == "all")].set_index("metric")["estimate"]
            ax.text(0.03, 0.97, f"RMSE {mm['rmse']:.3f}\nubRMSD {mm['ubrmsd']:.3f}\n"
                    f"R² {mm['r2']:.2f}\nstation R {mm['r_station_median']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=6.5)
            ax.set_xlim(0, 0.55)
            ax.set_ylim(0, 0.55)
            if i == 1:
                ax.set_xlabel("ISMN soil moisture (m³/m³)")
            if j == 0:
                ax.set_ylabel(f"{EV_LABEL[s]}\npredicted (m³/m³)")
            if i == 0:
                ax.text(0.5, 1.03, f"{f}, random forest", transform=ax.transAxes, ha="center",
                        fontsize=7.5)
    fig.tight_layout()
    return ps.save(fig, "fig07_clean_model")


def fig08_rfi_distributions() -> list:
    """RFI severity and extent from the OpenSAR patch statistics.

    (a) excess power per polarisation, (b) extent, both with the grid
    percentiles marked; (c, d) injected error eps on the grid for VV and VH.
    Falls back to a placeholder panel if the OpenSAR file is absent.
    """
    ps.apply()
    from .rfi import injection_grid

    grid = injection_grid(verbose=False)
    fig, ax = plt.subplots(1, 4, figsize=(ps.width("double"), 2.1),
                           gridspec_kw={"width_ratios": [1.25, 1.25, 1, 1]})
    hist_file = path("results_aggregate") / "stage5_opensar_hist.csv"
    if grid["source"] == "opensar" and hist_file.exists():
        h = pd.read_csv(hist_file)
        cmin = load_config()["injection"]["opensar"]["min_contrast_db"]
        for var, pol, c in (("severity_db_vv", "VV", ps.BLUE), ("severity_db_vh", "VH", ps.VERMILLION)):
            g = h[h.variable == var]
            ax[0].step(g["bin_low"], g["count"], where="post", color=c,
                       label=f"{pol} (excluded ≤ {cmin} dB: {grid['excluded'][pol]})")
            for q in grid["severity_q"]:
                ax[0].plot(grid["contrast_db"][q][pol], -0.6 if pol == "VV" else -1.4, "|",
                           color=c, ms=5, mew=1.2)
        ax[0].axvline(cmin, color=ps.GREY, lw=0.7, ls="--")
        ax[0].set_xlabel("Excess power C (dB)")
        ax[0].set_ylabel("Patches")
        ax[0].legend(loc="upper right", fontsize=5.8)
        g = h[h.variable == "extent"]
        ax[1].bar(g["bin_low"] + 0.025, g["count"], width=0.045, color=ps.ORANGE)
        for v in grid["extent"]:
            ax[1].axvline(v, color=ps.SLATE, lw=0.6, ls=":")
        ax[1].set_xlabel("Extent (masked fraction, -)")
        ax[1].set_ylabel("Patches")
        ax[1].text(0.97, 0.95, f"n = {grid['n_patches']}\nlines: P10 to P90", transform=ax[1].transAxes,
                   ha="right", va="top", fontsize=6)
    else:
        for a in ax[:2]:
            a.axis("off")
        ax[0].text(0.5, 0.5, "OpenSAR statistics not available\n(provisional grid)",
                   ha="center", va="center", transform=ax[0].transAxes, color=ps.VERMILLION)
    for a, pol in zip(ax[2:], ("VV", "VH")):
        z = np.array([[float(eps_db(grid["sir"][q][pol], r)) for q in grid["severity_q"]]
                      for r in grid["extent"]])
        im = a.imshow(z, origin="lower", cmap="Oranges", aspect="auto", vmin=0,
                      vmax=max(z.max(), 1e-6))
        a.set_xticks(range(len(grid["severity_q"])), [f"P{q}" for q in grid["severity_q"]],
                     rotation=45)
        a.set_yticks(range(len(grid["extent"])), [f"{v:.2f}" for v in grid["extent"]])
        a.set_xlabel(f"Severity level ({pol})")
        a.set_ylabel("Extent (-)")
        a.grid(False)
        for (i, j), v in np.ndenumerate(z):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.0,
                   color="white" if v > z.max() * 0.6 else ps.SLATE)
    for a, lab in zip(ax, "abcd"):
        ps.panel_label(a, f"({lab})")
    fig.tight_layout(w_pad=0.8)
    return ps.save(fig, "fig08_rfi_distributions")


def _fset_line(ax, x, y, lo, hi, fset, label):
    st = ps.FSET_STYLES[fset]
    c = ps.FSET_SHADES[fset] if fset != "F1" else ps.VERMILLION
    ax.plot(x, y, color=c, label=label, **st)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)


def fig09_degradation() -> list:
    """Degradation D against severity (SIR) and against extent (RQ1, RQ2)."""
    ps.apply()
    by_sir, by_ext = agg("stage3_summary_by_severity"), agg("stage3_summary_by_extent")
    gt = agg("stage3_grid")
    sev = gt[gt.factor == "severity"].set_index("level")
    by_sir = by_sir.assign(sev_db=by_sir["severity_q"].map(
        (sev["contrast_db_VV"] + sev["contrast_db_VH"]) / 2))
    evs = ["USCRN_cv", "OZNET_transfer"]
    fig, axes = plt.subplots(2, 2, figsize=(ps.width("double"), 4.0), sharey="row")
    for i, ev in enumerate(evs):
        for j, (tab, col, xl) in enumerate([(by_sir, "sev_db", "Severity: mean excess power of level (dB)"),
                                            (by_ext, "extent", "Extent ρ (-)")]):
            ax = axes[i, j]
            for f, mdl in [("F1", "rf"), ("F2", "rf"), ("F3", "rf"), ("VV", "cd")]:
                g = tab[(tab.evaluation == ev) & (tab.feature_set == f) & (tab.model == mdl)
                        & (tab.variant == "M3")].sort_values(col)
                if g.empty:
                    continue
                _fset_line(ax, g[col], g["D_mean"], g["D_ci_low"], g["D_ci_high"], f,
                           f"{f} RF" if mdl == "rf" else "change detection")
            ax.axhline(0, color=ps.GREY, lw=0.7)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"{EV_LABEL[ev]}\nD = RMSE$_{{M1}}$ − RMSE$_{{M0}}$ (m³/m³)")
            ps.panel_label(ax, f"({'abcd'[2 * i + j]})")
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return ps.save(fig, "fig09_degradation")


def _bars(ax, groups, variants, values, lo, hi, missing=()):
    """Grouped bars with error bars; missing variants drawn as hatched stubs."""
    n = len(variants)
    w = 0.8 / n
    for k, v in enumerate(variants):
        x = np.arange(len(groups)) + (k - (n - 1) / 2) * w
        if v in missing:
            # Legend entry only (no bar), and an 'n/a' mark at the bottom of the axis.
            ax.bar([np.nan], [np.nan], width=w, color="white", edgecolor=ps.GREY, hatch="///",
                   label=f"{ps.VARIANT_LABELS[v]} (n/a)")
            for xi in x:
                ax.text(xi, 0.02, "n/a", rotation=90, ha="center", va="bottom", fontsize=5.5,
                        color=ps.GREY, transform=ax.get_xaxis_transform())
            continue
        y, l, h = (np.array(a[v]) for a in (values, lo, hi))
        ax.bar(x, y, width=w, color=ps.VARIANT_COLOURS[v], label=ps.VARIANT_LABELS[v])
        ax.errorbar(x, y, yerr=[y - l, h - y], fmt="none", ecolor=ps.SLATE, elinewidth=0.7,
                    capsize=1.5)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def _headline(ev: str, model_map: dict, quantity: str, variants: list):
    h = agg("stage4_headline")
    h = h[(h["analysis"] == "reference") & (h["evaluation"] == ev) & (h["quantity"] == quantity)]
    vals, lo, hi = {}, {}, {}
    for v in variants:
        r = [h[(h.feature_set == f) & (h.model == m) & (h.variant == v)] for f, m in model_map]
        vals[v] = [x["estimate"].iloc[0] if len(x) else np.nan for x in r]
        lo[v] = [x["ci_low"].iloc[0] if len(x) else np.nan for x in r]
        hi[v] = [x["ci_high"].iloc[0] if len(x) else np.nan for x in r]
    return vals, lo, hi


def fig10_recovery() -> list:
    """Recovery G and recovery ratio RR by mitigation variant (RQ3)."""
    ps.apply()
    ev, _ = primary()
    groups = [("F1", "rf"), ("F2", "rf"), ("F3", "rf"), ("VV", "cd")]
    labels = ["F1 RF", "F2 RF", "F3 RF", "CD"]
    variants = ["M2", "M3", "M3b", "M4"]
    fig, ax = plt.subplots(1, 2, figsize=(ps.width("double"), 2.4))
    for a, q, yl in zip(ax, ["G", "RR"], ["Recovery G (m³/m³)", "Recovery ratio RR (-)"]):
        v, l, h = _headline(ev, groups, q, variants[1:])
        _bars(a, labels, variants, v, l, h, missing=("M2",))
        a.axhline(0, color=ps.GREY, lw=0.7)
        a.set_ylabel(yl)
        if q == "RR":
            # RR is undefined where the degradation D is below the minimum.
            for i in range(len(groups)):
                if all(not np.isfinite(v[x][i]) for x in variants[1:]):
                    a.text(i, 0.5, "RR undefined\n(D ≈ 0)", ha="center", va="bottom",
                           fontsize=6, color=ps.GREY, transform=a.get_xaxis_transform())
    ax[1].axhline(1, color=ps.GREY, lw=0.7, ls="--")
    hd, lb = ax[0].get_legend_handles_labels()
    fig.legend(hd, lb, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ps.panel_label(ax[0], "(a)")
    ps.panel_label(ax[1], "(b)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return ps.save(fig, "fig10_recovery")


def fig11_nested() -> list:
    """Nested comparison of RMSE for all variants with 95% intervals."""
    ps.apply()
    ev, _ = primary()
    groups = [("F1", "rf"), ("F2", "rf"), ("F3", "rf"), ("VV", "cd")]
    variants = ["M0", "M1", "M2", "M3", "M3b", "M4"]
    v, l, h = _headline(ev, groups, "rmse", [x for x in variants if x != "M2"])
    fig, ax = plt.subplots(figsize=(ps.width("double"), 3.0))
    _bars(ax, ["F1 RF", "F2 RF", "F3 RF", "CD"], variants, v, l, h, missing=("M2",))
    lo_y = np.nanmin([np.nanmin(x) for x in l.values()])
    ax.set_ylim(max(0, lo_y - 0.01), None)
    ax.set_ylabel("RMSE (m³/m³)")
    hd, lb = ax.get_legend_handles_labels()
    fig.legend(hd, lb, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return ps.save(fig, "fig11_nested")


def fig12_mediation() -> list:
    """Mediation path diagrams with estimates and bootstrap intervals."""
    ps.apply()
    ev, mdl = primary()
    med = agg("stage4_mediation")
    med = med[(med.evaluation == ev) & (med.model == mdl) & (med.mediation == "primary")]
    fig, axes = plt.subplots(1, 3, figsize=(ps.width("double"), 2.0))
    for ax, f in zip(axes, FSETS):
        t = med[med.feature_set == f].set_index("term")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        boxes = {"X": (0.2, 0.6, "Mitigation\n(M3 vs M1)", ps.BLUE),
                 "M": (3.6, 3.7, "Residual\nbackscatter\nerror (dB)", ps.ORANGE),
                 "Y": (7.0, 0.6, "Soil moisture\nerror", ps.SLATE)}
        for k, (x, y, s, c) in boxes.items():
            ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                                        fc="white", ec=c, lw=1.0))
            ax.text(x + 1.4, y + 0.75, s, ha="center", va="center", fontsize=6.2)

        def arr(p, q, txt, tx, ty):
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                         color=ps.SLATE))
            ax.text(tx, ty, txt, ha="center", va="center", fontsize=5.6)

        def fmt(term, scale=1.0, d=3):
            r = t.loc[term]
            return (f"{r.estimate * scale:.{d}f}\n[{r.ci_low * scale:.{d}f}, "
                    f"{r.ci_high * scale:.{d}f}]")
        arr((1.6, 2.1), (3.9, 3.7), "a = " + fmt("a", d=2), 1.5, 3.6)
        arr((6.2, 3.7), (8.4, 2.1), "b = " + fmt("b"), 8.5, 3.6)
        arr((3.0, 1.35), (7.0, 1.35), "c′ = " + fmt("c_prime", d=4), 5.0, 0.55)
        ind = t.loc["indirect_ab"]
        ax.text(5.0, 5.75, f"{f}: indirect a·b = {ind.estimate:.4f} "
                f"[{ind.ci_low:.4f}, {ind.ci_high:.4f}]", ha="center", fontsize=6.2,
                fontweight="bold")
    fig.tight_layout()
    return ps.save(fig, "fig12_mediation")


def fig13_learning_curve() -> list:
    """Station-count learning curve: interval width and recovery ratio."""
    ps.apply()
    lc = agg("stage4_learning_curve_summary")
    ev, mdl = primary()
    fig, ax = plt.subplots(1, 2, figsize=(ps.width("double"), 2.3))
    for f in FSETS:
        g = lc[(lc.evaluation == ev) & (lc.feature_set == f) & (lc.model == mdl)]
        g = g.sort_values("n_stations")
        st = ps.FSET_STYLES[f]
        c = ps.FSET_SHADES[f] if f != "F1" else ps.VERMILLION
        ax[0].plot(g["n_stations"], g["G_ci_width_mean"], color=c, label=f"{f} RF", **st)
        ax[1].errorbar(g["n_stations"], g["RR_mean"], yerr=g["RR_sd"].fillna(0), color=c,
                       capsize=1.5, elinewidth=0.6, label=f"{f} RF", **st)
    ax[0].set_xlabel("Number of stations")
    ax[0].set_ylabel("Width of 95% interval of G (m³/m³)")
    ax[1].set_xlabel("Number of stations")
    ax[1].set_ylabel("Recovery ratio RR (M3)")
    ax[1].axhline(0, color=ps.GREY, lw=0.7)
    ax[0].legend()
    ps.panel_label(ax[0], "(a)")
    ps.panel_label(ax[1], "(b)")
    fig.tight_layout()
    return ps.save(fig, "fig13_learning_curve")


def fig14_base_vs_extended() -> list:
    """Base (clean model) against extended pipeline on both samples."""
    ps.apply()
    variants = ["M0", "M1", "M3", "M4"]
    evs = ["USCRN_cv", "OZNET_cv", "OZNET_transfer"]
    fig, axes = plt.subplots(1, 3, figsize=(ps.width("double"), 2.4), sharey=False)
    for ax, ev, lab in zip(axes, evs, "abc"):
        v, l, h = _headline(ev, [(f, "rf") for f in FSETS], "rmse", variants)
        _bars(ax, FSETS, variants, v, l, h)
        ps.panel_label(ax, f"({lab})")
        ax.text(0.02, 1.02, EV_LABEL[ev], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7)
        lo_y = np.nanmin([np.nanmin(x) for x in l.values()])
        ax.set_ylim(max(0, lo_y - 0.01), None)
    axes[0].set_ylabel("RMSE, random forest (m³/m³)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return ps.save(fig, "fig14_base_vs_extended")


def make_all() -> list:
    """Build the aggregated inputs and draw figures 5 to 14."""
    build_descriptive_tables()
    out = []
    for fn in (fig05_station_map, fig06_sample_profile, fig07_clean_model,
               fig08_rfi_distributions, fig09_degradation, fig10_recovery, fig11_nested,
               fig12_mediation, fig13_learning_curve, fig14_base_vs_extended):
        out += fn()
    return out
