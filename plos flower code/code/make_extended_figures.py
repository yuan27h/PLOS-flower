from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LOW_FEATURES = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_sd_r", "rgb_sd_g", "rgb_sd_b",
    "gray_mean", "gray_sd", "symmetry_lr", "symmetry_tb", "edge_density",
    "saturation_mean", "saturation_sd", "value_mean", "value_sd", "hue_mean",
    "hue_sd", "hue_entropy_8bin",
]
LABELS = {
    "rgb_mean_r": "R mean", "rgb_mean_g": "G mean", "rgb_mean_b": "B mean",
    "rgb_sd_r": "R SD", "rgb_sd_g": "G SD", "rgb_sd_b": "B SD",
    "gray_mean": "Gray mean", "gray_sd": "Gray SD", "symmetry_lr": "LR symmetry",
    "symmetry_tb": "TB symmetry", "edge_density": "Edge magnitude",
    "saturation_mean": "HSV saturation mean", "saturation_sd": "HSV saturation SD",
    "value_mean": "HSV value mean", "value_sd": "HSV value SD", "hue_mean": "HSV hue mean",
    "hue_sd": "HSV hue SD", "hue_entropy_8bin": "Hue entropy",
}
MODEL_LABELS = {
    "ridge_low_level": "Compact ridge", "ridge_resnet50": "ResNet-50 ridge",
    "ridge_fusion": "Fusion ridge", "ols_low_level": "Compact OLS",
    "boost_low_level": "Boosting", "stack_low_level": "Stacking",
}
MODEL_ORDER = list(MODEL_LABELS)
COMPARISONS = ["ridge_resnet50", "ridge_fusion", "ols_low_level", "boost_low_level", "stack_low_level"]
COMPARISON_LABELS = {
    "ridge_resnet50": "ResNet-50", "ridge_fusion": "Fusion", "ols_low_level": "OLS",
    "boost_low_level": "Boosting", "stack_low_level": "Stacking",
}
COLORS = {
    "ridge_low_level": "#287D8E", "ridge_resnet50": "#D97957", "ridge_fusion": "#7567A8",
    "ols_low_level": "#4C78A8", "boost_low_level": "#C9565B", "stack_low_level": "#B26B9D",
}
POSITIVE = "#287D8E"
NEGATIVE = "#C65F4A"
GRID = "#D9DEE5"
INK = "#25313B"
MUTED = "#65717D"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.75,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.55, "grid.alpha": 0.75,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.edgecolor": "white", "axes.unicode_minus": False,
    })


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color=INK)


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA / "derived_features.csv"),
        pd.read_csv(RESULTS / "oof_predictions.csv"),
        pd.read_csv(RESULTS / "metrics.csv"),
    )


def _metric(metrics: pd.DataFrame, target: str, model: str, column: str) -> float:
    row = metrics[(metrics.dataset_version == "full") & (metrics.target == target) & (metrics.model == model)]
    return float(row.iloc[0][column])


def performance_r2(metrics: pd.DataFrame) -> None:
    rows = metrics[(metrics.dataset_version == "full") & metrics.model.isin(MODEL_ORDER)].copy()
    labels = {
        "ridge_low_level": "Compact", "ridge_resnet50": "ResNet-50", "ridge_fusion": "Fusion",
        "ols_low_level": "OLS", "boost_low_level": "Boosting", "stack_low_level": "Stacking",
    }
    route_colors = {
        "ridge_low_level": "#287D8E", "ridge_resnet50": "#D97957", "ridge_fusion": "#7567A8",
        "ols_low_level": "#4C78A8", "boost_low_level": "#C9565B", "stack_low_level": "#B26B9D",
    }
    ranks = {}
    values = {}
    for target in ("appeal", "interest"):
        ordered = rows[rows.target == target].sort_values("cv_r2_vs_fold_training_mean", ascending=False).model.tolist()
        ranks[target] = {model: idx + 1 for idx, model in enumerate(ordered)}
        values[target] = {row.model: float(row.cv_r2_vs_fold_training_mean)
                          for _, row in rows[rows.target == target].iterrows()}
    fig, ax = plt.subplots(figsize=(7.25, 4.70))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.80, bottom=0.17)
    for model in MODEL_ORDER:
        x = [0, 1]
        y = [ranks["appeal"][model], ranks["interest"][model]]
        ax.plot(x, y, color=route_colors[model], linewidth=2.4, alpha=0.90, zorder=2)
        ax.scatter(x, y, s=68, color=route_colors[model], edgecolor="white", linewidth=1.0, zorder=3)
        for xx, yy, target in ((0, y[0], "appeal"), (1, y[1], "interest")):
            value = values[target][model]
            dx = -0.035 if xx == 0 else 0.035
            ha = "right" if xx == 0 else "left"
            ax.text(xx + dx, yy, f"{value:.3f}", ha=ha, va="center", fontsize=6.6, color=INK)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(6.45, 0.55)
    ax.set_xticks([0, 1], ["Appeal", "Interest"])
    ax.set_yticks(range(1, 7), ["1  best", "2", "3", "4", "5", "6  lowest"])
    ax.set_ylabel("Rank by nested leave-one-flower-out CV $R^2$")
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0)
    _panel(ax, "A")
    handles = [Line2D([0], [0], color=route_colors[m], linewidth=2.4, marker="o",
                      markeredgecolor="white", markersize=5.4, label=labels[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.56, 0.91),
               ncol=3, columnspacing=1.3, fontsize=6.8)
    fig.suptitle("Which model wins each rating target?", fontsize=11.5, fontweight="bold", y=0.98)
    fig.text(0.5, -0.005, "Lines track a model's rank across targets; labels are the underlying nested CV $R^2$ values.",
             ha="center", fontsize=7.1, color=MUTED)
    _save(fig, "performance_r2.png")


def model_performance(metrics: pd.DataFrame) -> None:
    paired = pd.read_csv(RESULTS / "paired_rmse_differences.csv")
    route_colors = {"ridge_resnet50": "#D97957", "ridge_fusion": "#7567A8",
                    "ols_low_level": "#4C78A8", "boost_low_level": "#C9565B",
                    "stack_low_level": "#B26B9D"}
    route_labels = {m: COMPARISON_LABELS[m] for m in COMPARISONS}
    deltas = {}
    for model in COMPARISONS:
        deltas[model] = {
            target: _metric(metrics, target, model, "cv_r2_vs_fold_training_mean")
            - _metric(metrics, target, "ridge_low_level", "cv_r2_vs_fold_training_mean")
            for target in ("appeal", "interest")
        }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.75), gridspec_kw={"wspace": 0.28, "width_ratios": [1.08, 0.92]})
    fig.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.19)
    ax = axes[0]
    ax.axhspan(-0.60, 0, color=NEGATIVE, alpha=0.035, zorder=0)
    ax.axvspan(-0.60, 0, color=NEGATIVE, alpha=0.035, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    ax.axhline(0, color=INK, linewidth=0.9, zorder=1)
    for model in COMPARISONS:
        x = deltas[model]["appeal"]; y = deltas[model]["interest"]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=route_colors[model], lw=1.8, alpha=0.82), zorder=2)
        ax.scatter(x, y, s=70, color=route_colors[model], edgecolor="white", linewidth=1.0, zorder=3)
        offsets = {
            "ridge_resnet50": (0.020, -0.050), "ridge_fusion": (0.020, 0.040),
            "ols_low_level": (0.020, -0.025), "boost_low_level": (0.020, 0.018),
            "stack_low_level": (0.020, 0.030),
        }
        dx, dy = offsets[model]
        ax.text(x + dx, y + dy, route_labels[model], fontsize=6.5, color=INK, ha="left", va="center")
    ax.text(-0.58, -0.56, "worse on both targets", fontsize=7.0, color=MUTED, ha="left", va="bottom")
    ax.text(0.075, 0.075, "better on both", fontsize=7.0, color=POSITIVE, ha="right", va="top")
    ax.set_xlim(-0.60, 0.08); ax.set_ylim(-0.60, 0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Appeal Delta CV $R^2$ vs compact ridge")
    ax.set_ylabel("Interest Delta CV $R^2$ vs compact ridge")
    ax.grid(True); _panel(ax, "A")
    ax = axes[1]
    x = np.arange(len(COMPARISONS))
    for idx, model in enumerate(COMPARISONS):
        target_values = []
        for target, offset in (("appeal", -0.12), ("interest", 0.12)):
            row = paired[(paired.target == target) & (paired.comparison == f"{model} - ridge_low_level")].iloc[0]
            value = float(row.rmse_delta)
            ci = json.loads(str(row.paired_bootstrap_95ci).replace("'", '"'))
            ax.errorbar(idx + offset, value, yerr=[[value - ci[0]], [ci[1] - value]], fmt="o",
                        color="#287D8E" if target == "appeal" else "#D97957",
                        ecolor="#287D8E" if target == "appeal" else "#D97957",
                        markeredgecolor="white", markeredgewidth=0.8, markersize=6.2,
                        elinewidth=1.1, capsize=2.2, zorder=3)
            target_values.append(value)
        ax.plot([idx - 0.12, idx + 0.12], target_values, color="#9AA5AF", linewidth=1.0, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x, [COMPARISON_LABELS[m] for m in COMPARISONS], rotation=35, ha="right")
    ax.set_ylabel("Delta RMSE vs compact ridge")
    ax.set_xlim(-0.55, len(COMPARISONS) - 0.45); ax.set_ylim(-0.13, 0.69)
    ax.grid(axis="y"); _panel(ax, "B")
    handles = [Line2D([0], [0], marker="o", color="#287D8E", linestyle="none", markeredgecolor="white", markersize=6.2, label="Appeal"),
               Line2D([0], [0], marker="o", color="#D97957", linestyle="none", markeredgecolor="white", markersize=6.2, label="Interest")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.64, 0.91), ncol=2, columnspacing=1.3, fontsize=7.0)
    fig.suptitle("Matched changes from the compact ridge baseline", fontsize=11.5, fontweight="bold", y=0.98)
    fig.text(0.5, -0.005, "Arrows start at zero; negative Delta R2 and positive Delta RMSE favor the compact baseline. RMSE whiskers are paired bootstrap 95% intervals.", ha="center", fontsize=7.0, color=MUTED)
    _save(fig, "model_performance.png")


def oof_predictions(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    models = ["ridge_low_level", "ridge_resnet50", "ridge_fusion"]
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.25), sharex="row", sharey="row",
                             gridspec_kw={"hspace": 0.25, "wspace": 0.12})
    for row, target in enumerate(("appeal", "interest")):
        all_part = predictions[(predictions.dataset_version == "full") & (predictions.target == target)]
        lo = min(all_part.y_true.min(), all_part.y_pred.min()) - 0.15
        hi = max(all_part.y_true.max(), all_part.y_pred.max()) + 0.15
        for col, model in enumerate(models):
            ax = axes[row, col]
            part = all_part[all_part.model == model]
            ax.scatter(part.y_true, part.y_pred, s=19, alpha=0.82, color=COLORS[model],
                       edgecolor="white", linewidth=0.35, zorder=3)
            ax.plot([lo, hi], [lo, hi], color=INK, linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)
            r2 = _metric(metrics, target, model, "cv_r2_vs_fold_training_mean")
            rmse = _metric(metrics, target, model, "rmse")
            ax.text(0.05, 0.94, f"R2 = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.1, color=INK,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRID, linewidth=0.55, alpha=0.92))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title({"ridge_low_level": "Compact", "ridge_resnet50": "ResNet-50", "ridge_fusion": "Fusion"}[model],
                         pad=5.5, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Observed standardized rating")
            if col == 0:
                ax.set_ylabel("Out-of-fold prediction")
                ax.text(-0.34, 0.5, target.title(), transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold", color=MUTED)
            _panel(ax, "ABCDEF"[row * 3 + col])
    fig.suptitle("Observed ratings versus out-of-fold predictions", fontsize=11.5, fontweight="bold", y=1.01)
    fig.text(0.5, -0.015, "Dashed diagonal: perfect prediction. All points are held-out flowers.",
             ha="center", fontsize=7.3, color=MUTED)
    _save(fig, "predictions_journal.png")


def feature_target_correlations(features: pd.DataFrame, predictions: pd.DataFrame) -> None:
    values = {}
    for target in ("appeal", "interest"):
        truth = predictions[(predictions.dataset_version == "full") & (predictions.target == target)]
        truth = truth.drop_duplicates("flowerID").set_index("flowerID")["y_true"]
        merged = features.set_index("flowerID").loc[truth.index]
        values[target] = merged[LOW_FEATURES].corrwith(truth)
    appeal = values["appeal"]
    interest = values["interest"]
    angles = np.linspace(0, 2 * np.pi, len(LOW_FEATURES), endpoint=False)
    radius_a = np.array([1 + float(appeal[f]) for f in LOW_FEATURES])
    radius_i = np.array([1 + float(interest[f]) for f in LOW_FEATURES])
    angles_closed = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(7.25, 5.95), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.79, bottom=0.25)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.plot(angles_closed, np.r_[radius_a, radius_a[0]], color="#287D8E", linewidth=2.0, label="Appeal")
    ax.fill(angles_closed, np.r_[radius_a, radius_a[0]], color="#287D8E", alpha=0.10)
    ax.plot(angles_closed, np.r_[radius_i, radius_i[0]], color="#D97957", linewidth=2.0, label="Interest")
    ax.fill(angles_closed, np.r_[radius_i, radius_i[0]], color="#D97957", alpha=0.08)
    ax.scatter(angles, radius_a, s=32, color="#287D8E", edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter(angles, radius_i, s=32, color="#D97957", edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=6.7, color=MUTED)
    ax.set_xticks(angles)
    ax.set_xticklabels([str(i) for i in range(1, len(LOW_FEATURES) + 1)], fontsize=6.3, color=INK)
    ax.grid(color=GRID, linewidth=0.55, alpha=0.8)
    ax.spines["polar"].set_color(INK); ax.spines["polar"].set_linewidth(0.8)
    _panel(ax, "A")
    handles = [Line2D([0], [0], color="#287D8E", linewidth=2.0, marker="o", markeredgecolor="white", markersize=5.2, label="Appeal"),
               Line2D([0], [0], color="#D97957", linewidth=2.0, marker="o", markeredgecolor="white", markersize=5.2, label="Interest")]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.60, 0.90), ncol=2, columnspacing=1.4, fontsize=7.2)
    feature_key = [f"{i}: {LABELS[feature]}" for i, feature in enumerate(LOW_FEATURES, start=1)]
    for idx, chunk in enumerate((feature_key[:6], feature_key[6:12], feature_key[12:])):
        fig.text(0.08 + idx * 0.31, 0.035, "\n".join(chunk), ha="left", va="bottom", fontsize=5.7, color=MUTED, family="DejaVu Sans Mono", linespacing=1.15)
    fig.suptitle("A radial fingerprint of compact image statistics", fontsize=11.5, fontweight="bold", y=0.98)
    fig.text(0.5, -0.005, "Numbers identify descriptors in the key below; the center and outer ring correspond to Pearson r = -1 and +1.", ha="center", fontsize=7.0, color=MUTED)
    _save(fig, "feature_target_combo.png")


def residuals(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    models = ["ridge_low_level", "ridge_resnet50", "ridge_fusion"]
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.1), sharex="row", sharey="row",
                             gridspec_kw={"hspace": 0.26, "wspace": 0.12})
    for row, target in enumerate(("appeal", "interest")):
        all_part = predictions[(predictions.dataset_version == "full") & (predictions.target == target)]
        lo = all_part.y_true.min() - 0.15
        hi = all_part.y_true.max() + 0.15
        rlo = min((all_part.y_true - all_part.y_pred).min(), -0.1) - 0.15
        rhi = max((all_part.y_true - all_part.y_pred).max(), 0.1) + 0.15
        for col, model in enumerate(models):
            ax = axes[row, col]
            part = all_part[all_part.model == model]
            resid = part.y_true.to_numpy() - part.y_pred.to_numpy()
            ax.scatter(part.y_true, resid, s=18, alpha=0.82, color=COLORS[model], edgecolor="white", linewidth=0.35)
            ax.axhline(0, color=INK, linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(rlo, rhi)
            ax.set_title({"ridge_low_level": "Compact", "ridge_resnet50": "ResNet-50", "ridge_fusion": "Fusion"}[model],
                         pad=5.5, fontweight="bold")
            ax.text(0.05, 0.92, f"RMSE = {_metric(metrics, target, model, 'rmse'):.3f}", transform=ax.transAxes,
                    fontsize=7.1, ha="left", va="top", color=INK)
            if row == 1:
                ax.set_xlabel("Observed standardized rating")
            if col == 0:
                ax.set_ylabel("Residual (observed - predicted)")
                ax.text(-0.35, 0.5, target.title(), transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold", color=MUTED)
            _panel(ax, "ABCDEF"[row * 3 + col])
    fig.suptitle("Out-of-fold residual structure", fontsize=11.5, fontweight="bold", y=1.01)
    fig.text(0.5, -0.015, "The horizontal reference line marks zero residual.", ha="center", fontsize=7.3, color=MUTED)
    _save(fig, "residuals_diagnostics.png")


def error_distributions(predictions: pd.DataFrame) -> None:
    models = ["ridge_low_level", "ridge_resnet50", "ridge_fusion"]
    labels = {"ridge_low_level": "Compact", "ridge_resnet50": "ResNet-50", "ridge_fusion": "Fusion"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.25), sharey=False, gridspec_kw={"wspace": 0.20})
    fig.subplots_adjust(left=0.13, right=0.98, top=0.80, bottom=0.18)
    quantiles = np.linspace(0.01, 0.99, 99)
    for col, (ax, target) in enumerate(zip(axes, ("appeal", "interest"))):
        distributions = {}
        target_quantiles = []
        for model in models:
            part = predictions[(predictions.dataset_version == "full") & (predictions.target == target) & (predictions.model == model)]
            distributions[model] = np.abs(part.y_true.to_numpy() - part.y_pred.to_numpy())
        for model in models:
            q_values = np.quantile(distributions[model], quantiles)
            target_quantiles.extend(q_values.tolist())
            ax.plot(quantiles * 100, q_values, color=COLORS[model], linewidth=2.0, label=labels[model])
            for q in (0.50, 0.90):
                value = float(np.quantile(distributions[model], q))
                ax.scatter(q * 100, value, s=24, color=COLORS[model], edgecolor="white", linewidth=0.55, zorder=3)
        for q in (50, 90):
            ax.axvline(q, color=GRID, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, float(np.ceil(max(target_quantiles) * 2) / 2))
        ax.set_xlabel("Held-out flowers covered (%)")
        ax.set_ylabel("Absolute error (standardized rating)" if col == 0 else "")
        ax.set_title(target.title(), pad=6, fontweight="bold")
        ax.grid(axis="y")
        _panel(ax, "AB"[col])
    handles = []
    for model in models:
        median_text = []
        for target in ("appeal", "interest"):
            part = predictions[(predictions.dataset_version == "full") & (predictions.target == target) & (predictions.model == model)]
            median_text.append(f"{np.median(np.abs(part.y_true.to_numpy() - part.y_pred.to_numpy())):.2f}")
        handles.append(Line2D([0], [0], color=COLORS[model], linewidth=2.0,
                              label=f"{labels[model]} (median A/I {median_text[0]}/{median_text[1]})"))
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3, columnspacing=0.85, fontsize=6.5)
    fig.suptitle("Error thresholds needed to cover held-out flowers", fontsize=11.5, fontweight="bold", y=0.98)
    fig.text(0.5, -0.005, "Lower curves are better; markers show the median and 90th-percentile absolute error.",
             ha="center", fontsize=7.1, color=MUTED)
    _save(fig, "errors_quantiles.png")


def resnet_space(features: pd.DataFrame, predictions: pd.DataFrame) -> None:
    deep = features[[c for c in features.columns if c.startswith("resnet50_")]].to_numpy(dtype=float)
    scaled = (deep - deep.mean(0)) / np.maximum(deep.std(0), 1e-8)
    pca = PCA(n_components=2, random_state=13)
    coords = pca.fit_transform(scaled)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.75), gridspec_kw={"wspace": 0.32})
    for col, (ax, target) in enumerate(zip(axes, ("appeal", "interest"))):
        truth = predictions[(predictions.dataset_version == "full") & (predictions.target == target)].drop_duplicates("flowerID").set_index("flowerID")
        values = truth.loc[features["flowerID"].to_numpy(), "y_true"].to_numpy()
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=31,
                        edgecolor="white", linewidth=0.45, alpha=0.95)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)" if col == 0 else "")
        ax.set_title(target.title(), pad=6, fontweight="bold")
        ax.grid(True)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label("Standardized rating", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)
        _panel(ax, "AB"[col])
    fig.suptitle("Two-dimensional geometry of the frozen ResNet-50 embedding", fontsize=11.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.01, "The projection is descriptive; it is not a validation estimate.", ha="center", fontsize=7.3, color=MUTED)
    _save(fig, "resnetspace.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    _style()
    features, predictions, metrics = _load()
    performance_r2(metrics)
    model_performance(metrics)
    oof_predictions(predictions, metrics)
    feature_target_correlations(features, predictions)
    residuals(predictions, metrics)
    error_distributions(predictions)
    resnet_space(features, predictions)
    print("Wrote publication-style figures to", FIGURES)


if __name__ == "__main__":
    main()
