"""Study-design schematic (manuscript Fig. 1, scheme_draft.png).

Redrawn from the package data: stimulus thumbnails are read from
data/gerbera_appeal_img_db.nc; no ratings are used.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
NC_PATH = ROOT / "data" / "gerbera_appeal_img_db.nc"
FIGURES = ROOT / "figures"

INK = "#111111"
SUB = "#5A5A5A"
EDGE = "#9C9C9C"
ARROW = "#4A4A4A"
FILL = "#FFFFFF"
BAND = "#FBFBFB"
ACC = ["#0F6E6A", "#B25E3F", "#6E7783"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.6,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(7.2, 3.75))
ax.set_xlim(0, 100)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, sub=None, accent=None, dashed=False, fill=FILL,
        tfs=6.9, sfs=5.7, lw=0.62, tcol=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="square,pad=0",
                                linewidth=lw, edgecolor=EDGE,
                                facecolor=fill, zorder=2,
                                linestyle=(0, (3.2, 2.0)) if dashed else "solid"))
    if accent:
        ax.plot([x, x], [y, y + h], color=accent, lw=2.0, solid_capstyle="butt", zorder=3)
    ty = y + h / 2 + (1.15 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center",
            fontsize=tfs, color=tcol, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.55, sub, ha="center", va="center",
                fontsize=sfs, color=SUB, zorder=4, linespacing=1.28)


def arrow(x0, y0, x1, y1, style="-|>", lw=0.62, color=ARROW, ls="solid", rad=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle=style, mutation_scale=6.5,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def hdr(x, txt):
    ax.text(x, 44.0, txt, ha="center", va="center", fontsize=6.2,
            color=SUB, fontweight="bold")


# ---- column headers ----
hdr(8.4, "STIMULI")
hdr(27.0, "FEATURE FAMILY")
hdr(50.5, "FOLD-INTERNAL PIPELINE")
hdr(72.0, "HELD-OUT PREDICTION")
hdr(92.0, "TARGETS")

# ---- stimuli: photos + two corpora ----
SX, SW, SY, SH = 1.6, 13.6, 9.6, 29.4
ax.add_patch(FancyBboxPatch((SX, SY), SW, SH, boxstyle="square,pad=0",
                            linewidth=0.62, edgecolor=EDGE, facecolor=FILL, zorder=2))
ax.text(SX + SW / 2, 36.0, "Gerbera corpus", ha="center", va="center",
        fontsize=6.9, color=INK)

_f = h5py.File(NC_PATH, "r")
_imgs = _f["__xarray_dataarray_variable__"][:]
_f.close()
_picks = np.linspace(0, len(_imgs) - 1, 8).astype(int)
_tw, _th = 3.22, 2.85
for k, _idx in enumerate(_picks):
    _c, _r = k % 4, k // 4
    _x0 = 1.78 + _c * (_tw + 0.14)
    _y0 = 30.8 - _r * (_th + 0.30)
    ax.imshow(np.asarray(_imgs[_idx]), extent=(_x0, _x0 + _tw, _y0, _y0 + _th),
              zorder=3, interpolation="bilinear", aspect="auto")
    ax.add_patch(Rectangle((_x0, _y0), _tw, _th, fill=False,
                           edgecolor="#D9D9D9", lw=0.45, zorder=4))
ax.text(SX + SW / 2, 25.6, "70 images · 480 × 480 px", ha="center", va="center",
        fontsize=5.6, color=SUB)
ax.plot([SX + 1.4, SX + SW - 1.4], [24.3, 24.3], color="#DDDDDD", lw=0.5, zorder=3)
ax.text(SX + SW / 2, 22.6, "External corpus", ha="center", va="center", fontsize=6.9, color=INK)
ax.text(SX + SW / 2, 19.9, "52 stimulus groups\n3 photographs each\n2,006 respondents",
        ha="center", va="center", fontsize=5.6, color=SUB, linespacing=1.3)
ax.plot([SX + 1.4, SX + SW - 1.4], [16.4, 16.4], color="#DDDDDD", lw=0.5, zorder=3)
ax.text(SX + SW / 2, 13.3, "no ratings used\nfor features", ha="center", va="center",
        fontsize=5.2, color=SUB, style="italic", linespacing=1.3)

# ---- rows ----
ROWS = [
    (29.6, 8.6, "18 compact descriptors", "colour · texture · shape (18-d)", "leave-one-flower", ACC[0]),
    (18.4, 8.6, "frozen ResNet-50", "GAP, ImageNet-1K V2 (2,048-d)", "leave-one-flower", ACC[1]),
    (7.2, 8.6, "concatenation", "compact + deep features (2,066-d)", "leave-one-flower", ACC[2]),
]
DX, DW = 19.6, 14.6

for y, h, t, s, _, c in ROWS:
    box(DX, y, DW, h, t, s, accent=c, tfs=6.7, sfs=5.15)
    arrow(SX + SW, y + h / 2, DX, y + h / 2, rad=0)

# ---- fold-internal pipeline (dashed container) ----
PX, PW, PY, PH = 38.6, 23.2, 5.6, 34.0
ax.add_patch(FancyBboxPatch((PX, PY), PW, PH, boxstyle="square,pad=0",
                            linewidth=0.85, edgecolor=INK, facecolor=BAND, zorder=1.5,
                            linestyle=(0, (3.6, 2.2))))
for y, h, *_ in ROWS:
    arrow(DX + DW, y + h / 2, PX - 0.2, y + h / 2)

steps = [
    ("feature scaling", "training flowers only"),
    ("PCA", "95% variance retained"),
    ("ridge penalty λ", "inner 5-fold search"),
]
for i, (a, b) in enumerate(steps):
    yy = 32.0 - i * 7.6
    ax.text(PX + 2.0, yy, a, ha="left", va="center", fontsize=6.4, color=INK)
    ax.text(PX + PW - 2.0, yy, b, ha="right", va="center", fontsize=5.4, color=SUB)
    if i < 2:
        ax.plot([PX + 2.0, PX + PW - 2.0], [yy - 3.7, yy - 3.7], color="#E2E2E2", lw=0.5)

ax.plot([PX + 2.0, PX + PW - 2.0], [13.3, 13.3], color="#D8D8D8", lw=0.5)
ax.text(PX + 2.0, 11.0, "gerbera only:", ha="left", va="center", fontsize=5.4,
        color=SUB, style="italic")
ax.text(PX + 2.0, 8.4, "participant filtering +", ha="left", va="center", fontsize=5.4,
        color=SUB, style="italic")
ax.text(PX + 2.0, 6.4, "outcome construction", ha="left", va="center", fontsize=5.4,
        color=SUB, style="italic")

ax.text(PX + PW / 2, 40.9, "inside every outer training fold", ha="center", va="center",
        fontsize=5.6, color=INK, style="italic")

# ---- held-out prediction ----
EX, EW = 66.2, 13.1
for i, (y, h, *_rest) in enumerate(ROWS):
    unit = "one flower / group held out"
    box(EX, y, EW, h, "ridge fit → predict", unit, tfs=6.6, sfs=5.0)
    arrow(PX + PW, y + h / 2, EX, y + h / 2)


# ---- mini fold strip: one unit held out per outer fold ----
import numpy as _np
_n = 14
_xs = _np.linspace(67.6, 79.4, _n)
_held = 5
for k, xx in enumerate(_xs):
    held = (k == _held)
    ax.add_patch(plt.Circle((xx, 41.2), 0.60,
                            facecolor="#EFEFEF" if not held else "white",
                            edgecolor=INK if held else "#B9B9B9",
                            linewidth=0.8 if held else 0.5, zorder=3))
    if held:
        ax.plot([xx, xx], [40.4, 39.8], color=INK, lw=0.55)
ax.text(73.4, 38.7, "one unit held out per fold", ha="center", va="center",
        fontsize=5.2, color=SUB, style="italic")

# ---- targets ----
TX, TW = 84.0, 15.2
ax.add_patch(FancyBboxPatch((TX, 5.6), TW, 34.0, boxstyle="square,pad=0",
                            linewidth=0.62, edgecolor=EDGE, facecolor=FILL, zorder=2))
ax.text(TX + TW / 2, 36.4, "out-of-fold", ha="center", va="center", fontsize=6.4, color=INK)
ax.text(TX + TW / 2, 33.6, "R² · RMSE · MAE · r", ha="center", va="center", fontsize=5.5, color=SUB)
ax.plot([TX + 1.4, TX + TW - 1.4], [31.4, 31.4], color="#DDDDDD", lw=0.5)

ax.text(TX + 2.0, 27.6, "appeal", ha="left", va="center", fontsize=6.2, color=INK)
ax.text(TX + 2.0, 24.9, "visual interest", ha="left", va="center", fontsize=6.2, color=INK)
ax.text(TX + 2.0, 22.2, "beauty (external)", ha="left", va="center", fontsize=6.2, color=INK)
ax.text(TX + 2.0, 19.0, "gerbera: 70 outer folds", ha="left", va="center", fontsize=5.3, color=SUB)
ax.text(TX + 2.0, 16.8, "external: 52 outer folds", ha="left", va="center", fontsize=5.3, color=SUB)
ax.plot([TX + 1.4, TX + TW - 1.4], [14.6, 14.6], color="#DDDDDD", lw=0.5)
ax.text(TX + TW / 2, 11.6, "paired ΔR²", ha="center", va="center", fontsize=5.6,
        color=INK, style="italic")
ax.text(TX + TW / 2, 8.8, "2,000 / 5,000\nbootstrap draws", ha="center", va="center",
        fontsize=5.3, color=SUB, linespacing=1.3)

for y, h, *_ in ROWS:
    arrow(EX + EW, y + h / 2, TX, y + h / 2)

# ---- caveat line ----
ax.text(1.6, 1.9,
        "Nothing to the right of the dashed boundary is fitted on held-out units; "
        "feature extraction uses images only, never ratings.",
        ha="left", va="center", fontsize=5.3, color=SUB, style="italic")

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
FIGURES.mkdir(parents=True, exist_ok=True)
out = FIGURES / "scheme_draft.png"
fig.savefig(out, dpi=400)
fig.savefig(out.with_suffix(".pdf"))
print("saved", out)
