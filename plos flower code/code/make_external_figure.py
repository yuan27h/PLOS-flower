from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {"compact_low_level": "#087E8B", "resnet50": "#D95D39", "fusion": "#E0A458"}
LABELS = {"compact_low_level": "Compact", "resnet50": "ResNet-50", "fusion": "Fusion"}


def main() -> None:
    gerbera = pd.read_csv(RESULTS / "metrics.csv")
    gerbera = gerbera[(gerbera["dataset_version"] == "full") & gerbera["model"].isin(["ridge_low_level", "ridge_resnet50", "ridge_fusion"])]
    gerbera_map = {"ridge_low_level": "compact_low_level", "ridge_resnet50": "resnet50", "ridge_fusion": "fusion"}
    gerbera = gerbera.assign(model=gerbera["model"].map(gerbera_map), task=gerbera["target"].str.title(), dataset="Gerbera", r2=gerbera["cv_r2_vs_fold_training_mean"])
    external = pd.read_csv(RESULTS / "external_hula2016_metrics.csv")
    external = external.assign(task="Hula beauty", dataset="Hula-Flegr 2016")
    external = external.rename(columns={"cv_r2": "r2"})
    combined = pd.concat([gerbera[["dataset", "task", "model", "r2"]], external[["dataset", "task", "model", "r2"]]], ignore_index=True)
    task_order = ["Gerbera\nappeal", "Gerbera\ninterest", "Hula beauty"]
    combined["task_label"] = np.where(combined["dataset"].eq("Gerbera"), "Gerbera\n" + combined["task"].str.lower(), "Hula beauty")
    # Keep a compact, non-bar comparison: connected model points and a delta panel.
    fig = plt.figure(figsize=(12.2, 5.8), facecolor="#FBFAF7")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.32)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_facecolor("#FBFAF7")
    y_positions = np.arange(len(task_order))[::-1]
    offsets = {"compact_low_level": 0.18, "resnet50": 0.0, "fusion": -0.18}
    for y, task_label in zip(y_positions, task_order):
        rows = combined[combined["task_label"] == task_label]
        for model in ["compact_low_level", "resnet50", "fusion"]:
            row = rows[rows["model"] == model]
            if row.empty:
                continue
            x = float(row["r2"].iloc[0])
            yy = y + offsets[model]
            ax.plot([0, x], [yy, yy], color=COLORS[model], alpha=0.18, lw=4, solid_capstyle="round")
            ax.scatter(x, yy, s=95, color=COLORS[model], edgecolor="#FBFAF7", linewidth=1.4, zorder=3)
            ax.text(x + 0.018, yy, f"{x:.3f}", va="center", ha="left", fontsize=9, color="#263238")
    ax.axvline(0, color="#263238", lw=0.8)
    ax.set_xlim(-0.08, 0.78)
    ax.set_ylim(-0.48, 2.48)
    ax.set_yticks(y_positions, task_order)
    ax.set_xlabel("Leave-one-stimulus-group-out $R^2$", labelpad=8)
    ax.set_title("Replication across flower datasets", loc="left", fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", color="#B8C2C5", alpha=0.35, lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[m], markeredgecolor="none", markersize=8, label=LABELS[m]) for m in ["compact_low_level", "resnet50", "fusion"]]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, -0.28), handletextpad=0.5, columnspacing=1.4)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_facecolor("#FBFAF7")
    delta_rows = []
    for task_label in task_order:
        rows = combined[combined["task_label"] == task_label].set_index("model")
        compact = float(rows.loc["compact_low_level", "r2"])
        for model in ["resnet50", "fusion"]:
            delta_rows.append({"task": task_label.replace("\n", " "), "model": model, "delta": float(rows.loc[model, "r2"]) - compact})
    deltas = pd.DataFrame(delta_rows)
    y2 = np.arange(len(deltas))[::-1]
    for yy, (_, row) in zip(y2, deltas.iterrows()):
        x = float(row["delta"])
        ax2.plot([0, x], [yy, yy], color=COLORS[row["model"]], lw=4, alpha=0.25, solid_capstyle="round")
        ax2.scatter(x, yy, s=88, color=COLORS[row["model"]], edgecolor="#FBFAF7", linewidth=1.4, zorder=3)
        align = "left" if x >= 0 else "right"
        ax2.text(x + (0.018 if x >= 0 else -0.018), yy, f"{x:+.3f}", va="center", ha=align, fontsize=9, color="#263238")
    ax2.axvline(0, color="#263238", lw=1.0)
    ax2.set_yticks(y2, [f"{row.task}\n{LABELS[row.model]}" for _, row in deltas.iterrows()])
    ax2.set_xlim(-0.62, 0.08)
    ax2.set_xlabel(r"$\Delta R^2$ versus compact", labelpad=8)
    ax2.set_title("Incremental value of deep features", loc="left", fontsize=13, fontweight="bold", pad=14)
    ax2.grid(axis="x", color="#B8C2C5", alpha=0.35, lw=0.8)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0, pad=8, labelsize=9)
    fig.text(0.01, 0.01, "External benchmark: Hůla, Flegr & Borghi (2016); each three-photo stimulus group remained intact in outer validation.", fontsize=8.5, color="#526166")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "external_replication.png", dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
