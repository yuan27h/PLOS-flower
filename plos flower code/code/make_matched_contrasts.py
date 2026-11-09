"""Matched feature-family contrasts vs compact ridge (manuscript Fig. 5).

Panel A: paired Delta R^2 for appeal (x) and interest (y), point estimates
         from results/paired_r2_differences.csv.
Panel B: paired Delta RMSE with 95% flower-level bootstrap intervals
         from results/paired_rmse_differences.csv.

Run from the package root:  python code/make_matched_contrasts.py
Writes figures/matched_contrasts.png (300 dpi) and .pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Okabe-Ito colourblind-safe palette (same key as the other journal figures)
C_RESNET = "#D55E00"
C_FUSION = "#E69F00"
C_OLS = "#009E73"
C_BOOST = "#CC79A7"
C_STACK = "#56B4E9"
C_APPEAL = "#0072B2"
C_INTEREST = "#E37E38"
INK = "#333333"
GREY = "#888888"

COMPS = [
    ("ridge_resnet50", "ResNet-50"),
    ("ridge_fusion", "Fusion"),
    ("ols_low_level", "OLS"),
    ("boost_low_level", "Boosting"),
    ("stack_low_level", "Stacking"),
]
MODEL_COLOR = {
    "ridge_resnet50": C_RESNET,
    "ridge_fusion": C_FUSION,
    "ols_low_level": C_OLS,
    "boost_low_level": C_BOOST,
    "stack_low_level": C_STACK,
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def despine_all(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def panel_letter(ax: plt.Axes, letter: str, dx: float = -0.13, dy: float = 1.05) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def parse_ci(text: str) -> tuple[float, float]:
    lo, hi = json.loads(text)
    return float(lo), float(hi)


def main() -> None:
    dr2 = pd.read_csv(RESULTS / "paired_r2_differences.csv")
    dr2 = dr2[dr2.target.isin(["appeal", "interest"])].copy()
    dr2["model"] = dr2["comparison"].str.replace(
        " - ridge_low_level", "", regex=False)

    paired = pd.read_csv(RESULTS / "paired_rmse_differences.csv")
    paired = paired[paired.comparison.str.endswith(" - ridge_low_level")].copy()
    paired["model"] = paired["comparison"].str.replace(
        " - ridge_low_level", "", regex=False)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(6.69, 2.7), gridspec_kw={"width_ratios": [1, 1.2]})

    # ---- Panel A: arrows from the origin, one per model -----------------
    for comp, lab in COMPS:
        da = float(dr2[(dr2.target == "appeal") & (dr2.model == comp)].r2_delta.iloc[0])
        di = float(dr2[(dr2.target == "interest") & (dr2.model == comp)].r2_delta.iloc[0])
        axA.annotate("", xy=(da, di), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=MODEL_COLOR[comp],
                                     lw=1.2, alpha=0.75))
        axA.scatter([da], [di], s=26, color=MODEL_COLOR[comp], zorder=3,
                    edgecolor="white", linewidth=0.5)
        axA.annotate(lab, (da, di), textcoords="offset points", xytext=(5, -2),
                     fontsize=6.5, color=MODEL_COLOR[comp])
    axA.axhline(0, color=INK, lw=0.8)
    axA.axvline(0, color=INK, lw=0.8)
    axA.set_xlabel(r"$\Delta R^2$ appeal")
    axA.set_ylabel(r"$\Delta R^2$ interest")
    axA.set_xlim(-0.20, 0.06)
    axA.set_ylim(-0.62, 0.05)
    despine_all(axA)
    axA.text(0.03, 0.05, "worse on both\ntargets", transform=axA.transAxes,
             fontsize=6.5, color=GREY, ha="left", va="bottom")
    panel_letter(axA, "A")

    # ---- Panel B: paired Delta RMSE intervals ---------------------------
    ypos = list(range(len(COMPS)))[::-1]
    for i, (comp, _lab) in enumerate(COMPS):
        for tkey, tcol in (("appeal", C_APPEAL), ("interest", C_INTEREST)):
            row = paired[(paired.target == tkey) & (paired.model == comp)].iloc[0]
            lo, hi = parse_ci(row.paired_bootstrap_95ci)
            yy = ypos[i] + (0.16 if tkey == "appeal" else -0.16)
            axB.plot([lo, hi], [yy, yy], color=tcol, lw=2.0, alpha=0.55,
                     solid_capstyle="round")
            axB.scatter([row.rmse_delta], [yy], s=20, color=tcol, zorder=3,
                        edgecolor="white", linewidth=0.4)
    axB.axvline(0, color=INK, lw=0.8)
    axB.set_yticks(ypos)
    axB.set_yticklabels([lab for _, lab in COMPS])
    axB.set_xlabel("Paired $\\Delta$RMSE vs compact ridge\n"
                   "(95% CI, flower-level bootstrap)")
    despine_all(axB)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=t)
               for c, t in ((C_APPEAL, "Appeal"), (C_INTEREST, "Interest"))]
    axB.legend(handles=handles, loc="lower right", frameon=False,
               handletextpad=0.2)
    panel_letter(axB, "B")

    fig.tight_layout(w_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "matched_contrasts.png", dpi=300)
    fig.savefig(FIGURES / "matched_contrasts.pdf")
    plt.close(fig)
    print("saved", FIGURES / "matched_contrasts.png")


if __name__ == "__main__":
    main()
