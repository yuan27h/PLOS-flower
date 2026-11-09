"""Paired flower-level contrasts against the compact ridge baseline.

Both contrasts (Delta R^2 and Delta RMSE) reuse the same resampled flower
indices, so the two columns condition on the same bootstrap draws.

Usage: python code/paired_contrasts.py
Reads  results/oof_predictions.csv
Writes results/paired_r2_differences.csv (paired_rmse_differences.csv is written by analyze.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BOOT = 2000
SEED = 20261102
BASE = "ridge_low_level"
COMPARISONS = [
    "ridge_resnet50",
    "ridge_fusion",
    "ols_low_level",
    "boost_low_level",
    "stack_low_level",
    "mean_baseline",
]


def r2(y, pred):
    """Out-of-fold R^2 with the fold-level mean baseline (y is standardized)."""
    y, pred = np.atleast_2d(y), np.atleast_2d(pred)
    return (1.0 - np.sum((y - pred) ** 2, axis=1) / np.sum(y ** 2, axis=1)).ravel()


def rmse(y, pred):
    y, pred = np.atleast_2d(y), np.atleast_2d(pred)
    return np.sqrt(np.mean((y - pred) ** 2, axis=1)).ravel()


def main():
    preds = pd.read_csv(RESULTS / "oof_predictions.csv")
    rows_r2, rows_rmse = [], []
    for target in ("appeal", "interest"):
        frame = preds[(preds.dataset_version == "full") & (preds.target == target)]
        wide = frame.pivot(index="flowerID", columns="model", values="y_pred").sort_index()
        y = (
            frame.drop_duplicates("flowerID")
            .set_index("flowerID")
            .loc[wide.index, "y_true"]
            .to_numpy(float)
        )
        n = len(y)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_BOOT, n))

        yb = y[idx]
        base_pred = wide[BASE].to_numpy(float)
        base_r2 = r2(yb, base_pred[idx])
        base_rmse = rmse(yb, base_pred[idx])
        point_base_r2 = r2(y, base_pred)
        point_base_rmse = rmse(y, base_pred)

        for model in COMPARISONS:
            pred = wide[model].to_numpy(float)
            d_r2 = r2(yb, pred[idx]) - base_r2
            d_rmse = rmse(yb, pred[idx]) - base_rmse
            rows_r2.append(
                {
                    "target": target,
                    "comparison": f"{model} - {BASE}",
                    "r2_delta": float((r2(y, pred) - point_base_r2)[0]),
                    "paired_bootstrap_95ci": list(np.percentile(d_r2, [2.5, 97.5])),
                    "interpretation": (
                        "Conditional image-level bootstrap of paired out-of-fold "
                        "squared errors; refitting uncertainty is not included."
                    ),
                }
            )
            rows_rmse.append(
                {
                    "target": target,
                    "comparison": f"{model} - {BASE}",
                    "rmse_delta": float((rmse(y, pred) - point_base_rmse)[0]),
                    "paired_bootstrap_95ci": list(np.percentile(d_rmse, [2.5, 97.5])),
                    "interpretation": (
                        "Conditional image-level bootstrap of paired out-of-fold "
                        "errors; refitting uncertainty is not included."
                    ),
                }
            )

    pd.DataFrame(rows_r2).to_csv(RESULTS / "paired_r2_differences.csv", index=False)

    # The Delta RMSE table is produced by analyze.py and is left untouched here;
    # the values printed below are the same contrasts for cross-checking.
    print(pd.DataFrame(rows_r2).to_string(index=False))
    print()
    print(pd.DataFrame(rows_rmse).to_string(index=False))


if __name__ == "__main__":
    main()
