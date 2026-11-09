from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import xarray as xr
from joblib import Memory
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet50_Weights, resnet50


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WEIGHTS = ROOT / "weights" / "resnet50-imagenet1k-v2.pth"
CACHE = ROOT / "work" / "sklearn_cache"
TARGETS = ("appeal", "interest")
LOW_FEATURES = (
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_sd_r", "rgb_sd_g", "rgb_sd_b",
    "gray_mean", "gray_sd", "symmetry_lr", "symmetry_tb", "edge_density",
    "saturation_mean", "saturation_sd", "value_mean", "value_sd",
    "hue_mean", "hue_sd", "hue_entropy_8bin",
)
ALPHAS = np.logspace(-4, 4, 9)
BOOST_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1],
    "max_depth": [1, 2],
    "min_samples_leaf": [5],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def low_level_features(pixels: np.ndarray) -> np.ndarray:
    image = Image.fromarray(pixels.astype(np.uint8), mode="RGB").resize((256, 256))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_sd = rgb.std(axis=(0, 1))
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray_mean = float(gray.mean())
    gray_sd = float(gray.std())
    left = gray[:, :128]
    right = np.fliplr(gray[:, 128:])
    top = gray[:128, :]
    bottom = np.flipud(gray[128:, :])
    symmetry_lr = float(np.mean(np.abs(left - right)))
    symmetry_tb = float(np.mean(np.abs(top - bottom)))
    edge_density = float(
        0.5 * (np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())
    )
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hue = hsv[:, :, 0] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    hist, _ = np.histogram(hue, bins=8, range=(0, 1), density=False)
    probabilities = hist / max(int(hist.sum()), 1)
    nonzero = probabilities > 0
    hue_entropy = float(
        -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / np.log(8)
    )
    return np.asarray(
        [
            *rgb_mean, *rgb_sd, gray_mean, gray_sd, symmetry_lr, symmetry_tb,
            edge_density, float(saturation.mean()), float(saturation.std()),
            float(value.mean()), float(value.std()), float(hue.mean()),
            float(hue.std()), hue_entropy,
        ],
        dtype=np.float32,
    )


def extract_features(device_name: str | None = None) -> tuple[list[int], dict[int, np.ndarray], pd.DataFrame]:
    if not WEIGHTS.is_file():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True), WEIGHTS)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(13)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=None)
    try:
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()

    with xr.open_dataset(DATA / "gerbera_appeal_img_db.nc") as dataset:
        variable = dataset[list(dataset.data_vars)[0]]
        flower_ids = [int(x) for x in variable.coords["flower"].values.tolist()]
        images = {flower_id: np.asarray(variable.sel(flower=flower_id).values, dtype=np.uint8) for flower_id in flower_ids}

    low_rows = []
    embedding_rows = []
    batch_ids: list[int] = []
    batch_tensors = []
    with torch.inference_mode():
        for flower_id in flower_ids:
            pixels = images[flower_id]
            low_rows.append(low_level_features(pixels))
            image = Image.fromarray(pixels, mode="RGB")
            batch_tensors.append(transform(image))
            batch_ids.append(flower_id)
            if len(batch_tensors) == 8 or flower_id == flower_ids[-1]:
                batch = torch.stack(batch_tensors).to(device)
                embedded = model(batch).detach().cpu().numpy().astype(np.float32)
                embedding_rows.extend(embedded)
                batch_tensors.clear()
                batch_ids.clear()

    low = pd.DataFrame(low_rows, index=flower_ids, columns=LOW_FEATURES)
    resnet = pd.DataFrame(
        embedding_rows,
        index=flower_ids,
        columns=[f"resnet50_{i:04d}" for i in range(2048)],
    )
    features = pd.concat([low, resnet], axis=1)
    features.index.name = "flowerID"
    features.to_csv(DATA / "derived_features.csv", float_format="%.9g")
    pd.DataFrame(
        {
            "feature": LOW_FEATURES,
            "description": [
                "Mean red-channel intensity over the full resized RGB image",
                "Mean green-channel intensity over the full resized RGB image",
                "Mean blue-channel intensity over the full resized RGB image",
                "Standard deviation of red-channel intensity over the full image",
                "Standard deviation of green-channel intensity over the full image",
                "Standard deviation of blue-channel intensity over the full image",
                "Mean grayscale intensity using Rec. 601 coefficients",
                "Standard deviation of grayscale intensity",
                "Mean absolute difference between mirrored left and right image halves",
                "Mean absolute difference between mirrored top and bottom image halves",
                "Mean absolute horizontal and vertical first difference",
                "Mean HSV saturation",
                "Standard deviation of HSV saturation",
                "Mean HSV value",
                "Standard deviation of HSV value",
                "Mean HSV hue",
                "Standard deviation of HSV hue",
                "Eight-bin HSV hue Shannon entropy normalized to the interval [0, 1]",
            ],
        }
    ).to_csv(DATA / "feature_dictionary.csv", index=False)
    return flower_ids, images, features


def rating_matrix(data_path: Path, target: str, flower_ids: list[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    ratings = pd.read_csv(data_path, encoding="cp1252")
    ratings["flowerID"] = ratings["flowerID"].astype(int)
    ratings[target] = pd.to_numeric(ratings[target], errors="coerce")
    mask = ratings[target].notna()
    if target == "appeal":
        mask &= ratings["interest"].isna()
    else:
        mask &= ratings["appeal"].notna()
    selected = ratings.loc[mask, ["subjectID_id", "flowerID", target]]
    grouped = selected.groupby(["subjectID_id", "flowerID"], sort=False)[target].mean()
    matrix = grouped.unstack("flowerID").reindex(columns=flower_ids)
    if matrix.shape[1] != len(flower_ids) or matrix.isna().all(axis=0).any():
        raise ValueError(f"The {target} rating matrix does not cover all 70 image IDs")
    summary = {
        "rating_rows": int(len(selected)),
        "participants": int(selected["subjectID_id"].nunique()),
        "participant_flower_cells": int(len(grouped)),
        "duplicate_cells_averaged": int((selected.groupby(["subjectID_id", "flowerID"]).size() > 1).sum()),
        "complete_participants": int(matrix.notna().all(axis=1).sum()),
    }
    return matrix.astype(float), summary


def fold_target(
    matrix: pd.DataFrame,
    train_idx: np.ndarray,
    heldout_idx: int,
) -> tuple[np.ndarray, float, dict[str, int]]:
    training = matrix.iloc[:, train_idx]
    complete = training.notna().all(axis=1)
    training_complete = training.loc[complete]
    if training_complete.empty:
        raise ValueError("No complete raters remain in this training fold")
    modal_share = training_complete.apply(
        lambda row: row.value_counts(dropna=True).iloc[0] / len(row), axis=1
    )
    rating_sd = training_complete.std(axis=1, ddof=0)
    sd_cutoff = float(rating_sd.mean() - 2.0 * rating_sd.std(ddof=0))
    flat = modal_share > 0.5
    low_variability = rating_sd < sd_cutoff
    keep = training_complete.index[~(flat | low_variability)]
    if len(keep) < 2:
        raise ValueError("Fewer than two raters remain after fold-specific quality checks")

    kept_train = training_complete.loc[keep]
    participant_mean = kept_train.mean(axis=1)
    participant_sd = kept_train.std(axis=1, ddof=0).replace(0.0, np.nan)
    z_train = kept_train.sub(participant_mean, axis=0).div(participant_sd, axis=0)
    train_aggregate = z_train.mean(axis=0).to_numpy(dtype=np.float64)

    heldout = matrix.loc[keep, matrix.columns[heldout_idx]]
    observed = heldout.notna()
    if not observed.any():
        raise ValueError("No retained raters supplied the held-out image rating")
    heldout_z = (heldout.loc[observed] - participant_mean.loc[observed]) / participant_sd.loc[observed]
    test_aggregate = float(heldout_z.mean())

    target_mean = float(train_aggregate.mean())
    target_sd = float(train_aggregate.std(ddof=0))
    if not np.isfinite(target_sd) or target_sd < 1e-12:
        target_sd = 1.0
    y_train = ((train_aggregate - target_mean) / target_sd).astype(np.float64)
    y_test = (test_aggregate - target_mean) / target_sd
    audit = {
        "n_participants_total": int(len(matrix)),
        "n_complete_training": int(len(training_complete)),
        "n_excluded_incomplete_training": int((~complete).sum()),
        "n_excluded_modal": int((flat & ~low_variability).sum()),
        "n_excluded_low_variability": int((low_variability & ~flat).sum()),
        "n_excluded_both_checks": int((low_variability & flat).sum()),
        "n_kept_training": int(len(keep)),
        "n_ratings_heldout": int(observed.sum()),
    }
    return y_train, float(y_test), audit


def ridge_search(feature_name: str, x: np.ndarray, seed: int) -> GridSearchCV:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if feature_name in {"resnet50", "fusion"}:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    steps.append(("ridge", Ridge()))
    pipeline = Pipeline(steps, memory=Memory(CACHE, verbose=0))
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(
        pipeline,
        {"ridge__alpha": ALPHAS},
        scoring="neg_mean_squared_error",
        cv=inner,
        n_jobs=1,
        refit=True,
    )


def boost_search(seed: int) -> GridSearchCV:
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(
        GradientBoostingRegressor(random_state=13, loss="squared_error"),
        BOOST_GRID,
        scoring="neg_mean_squared_error",
        cv=inner,
        n_jobs=1,
        refit=True,
    )


def stack_search(seed: int) -> GridSearchCV:
    base = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=3))
    stack = StackingRegressor(
        estimators=[("ridge", base)],
        final_estimator=GradientBoostingRegressor(random_state=13, loss="squared_error"),
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        passthrough=True,
        n_jobs=1,
    )
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    params = {
        f"final_estimator__{key}": value for key, value in BOOST_GRID.items()
    }
    return GridSearchCV(
        stack,
        params,
        scoring="neg_mean_squared_error",
        cv=inner,
        n_jobs=1,
        refit=True,
    )


def fit_oof(
    features: pd.DataFrame,
    matrix: pd.DataFrame,
    target: str,
    dataset_version: str,
    include_extended: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resnet_columns = [f"resnet50_{i:04d}" for i in range(2048)]
    feature_arrays = {
        "low_level": features.loc[:, LOW_FEATURES].to_numpy(dtype=np.float32),
        "resnet50": features.loc[:, resnet_columns].to_numpy(dtype=np.float32),
        "fusion": features.to_numpy(dtype=np.float32),
    }
    flower_ids = [int(x) for x in features.index]
    predictions: list[dict[str, object]] = []
    fit_audits: list[dict[str, object]] = []

    for fold_number, (train_idx, test_idx) in enumerate(LeaveOneOut().split(np.arange(len(flower_ids)))):
        if fold_number % 10 == 0:
            print(f"  {target}: outer fold {fold_number + 1}/70 ({dataset_version})", flush=True)
        heldout_idx = int(test_idx[0])
        y_train, y_true, rating_audit = fold_target(matrix, train_idx, heldout_idx)
        predictions.append(
            {
                "dataset_version": dataset_version,
                "target": target,
                "model": "mean_baseline",
                "flowerID": flower_ids[heldout_idx],
                "y_true": y_true,
                "y_pred": 0.0,
                "n_training_flowers": int(len(train_idx)),
                "n_raters_training": rating_audit["n_kept_training"],
                "n_raters_test": rating_audit["n_ratings_heldout"],
            }
        )
        fold_models: list[tuple[str, object, np.ndarray]] = []
        for feature_name, model_x in feature_arrays.items():
            search = ridge_search(feature_name, model_x, 13)
            search.fit(model_x[train_idx], y_train)
            fold_models.append((f"ridge_{feature_name}", search, model_x))
        if include_extended:
            low_x = feature_arrays["low_level"]
            ols = make_pipeline(StandardScaler(), LinearRegression())
            ols.fit(low_x[train_idx], y_train)
            fold_models.append(("ols_low_level", ols, low_x))
            boost = boost_search(13)
            boost.fit(low_x[train_idx], y_train)
            fold_models.append(("boost_low_level", boost, low_x))
            stack = stack_search(13)
            stack.fit(low_x[train_idx], y_train)
            fold_models.append(("stack_low_level", stack, low_x))

        for model_name, model, model_x in fold_models:
            y_pred = float(np.asarray(model.predict(model_x[test_idx])).reshape(-1)[0])
            predictions.append(
                {
                    "dataset_version": dataset_version,
                    "target": target,
                    "model": model_name,
                    "flowerID": flower_ids[heldout_idx],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "n_training_flowers": int(len(train_idx)),
                    "n_raters_training": rating_audit["n_kept_training"],
                    "n_raters_test": rating_audit["n_ratings_heldout"],
                }
            )
            if model_name.startswith("ridge_"):
                chosen = model.best_params_["ridge__alpha"]
                pca_components = None
                if "pca" in model.best_estimator_.named_steps:
                    pca_components = int(model.best_estimator_.named_steps["pca"].n_components_)
                fit_audits.append(
                    {
                        "dataset_version": dataset_version,
                        "target": target,
                        "model": model_name,
                        "fold": fold_number,
                        "alpha": float(chosen),
                        "pca_components": pca_components,
                        **rating_audit,
                    }
                )
            elif model_name in {"boost_low_level", "stack_low_level"}:
                fit_audits.append(
                    {
                        "dataset_version": dataset_version,
                        "target": target,
                        "model": model_name,
                        "fold": fold_number,
                        "best_parameters": json.dumps(model.best_params_, sort_keys=True),
                        **rating_audit,
                    }
                )
            elif model_name == "ols_low_level":
                fit_audits.append(
                    {
                        "dataset_version": dataset_version,
                        "target": target,
                        "model": model_name,
                        "fold": fold_number,
                        **rating_audit,
                    }
                )
    return pd.DataFrame(predictions), pd.DataFrame(fit_audits)


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = y - prediction
    denominator = float(np.sum(y**2))
    cv_r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1e-12 else float("nan")
    return {
        "cv_r2_vs_fold_training_mean": cv_r2,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": float(pearsonr(y, prediction).statistic),
        "spearman_r": float(spearmanr(y, prediction).statistic),
    }


def bootstrap_metrics(y: np.ndarray, prediction: np.ndarray, n_boot: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yy = y[idx]
    pp = prediction[idx]
    err = yy - pp
    r2 = 1.0 - np.sum(err**2, axis=1) / np.maximum(np.sum(yy**2, axis=1), 1e-12)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    mae = np.mean(np.abs(err), axis=1)
    corr_p = np.empty(n_boot, dtype=float)
    corr_s = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        if np.std(yy[i]) < 1e-12 or np.std(pp[i]) < 1e-12:
            corr_p[i] = np.nan
            corr_s[i] = np.nan
        else:
            corr_p[i] = pearsonr(yy[i], pp[i]).statistic
            corr_s[i] = spearmanr(yy[i], pp[i]).statistic
    return {
        "cv_r2_vs_fold_training_mean": [float(np.nanpercentile(r2, 2.5)), float(np.nanpercentile(r2, 97.5))],
        "rmse": [float(np.nanpercentile(rmse, 2.5)), float(np.nanpercentile(rmse, 97.5))],
        "mae": [float(np.nanpercentile(mae, 2.5)), float(np.nanpercentile(mae, 97.5))],
        "pearson_r": [float(np.nanpercentile(corr_p, 2.5)), float(np.nanpercentile(corr_p, 97.5))],
        "spearman_r": [float(np.nanpercentile(corr_s, 2.5)), float(np.nanpercentile(corr_s, 97.5))],
    }


def paired_bootstrap_rmse_delta(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yy = y[idx]
    e_first = yy - first[idx]
    e_second = yy - second[idx]
    delta = np.sqrt(np.mean(e_first**2, axis=1)) - np.sqrt(np.mean(e_second**2, axis=1))
    return [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]


def summarize_predictions(predictions: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    differences = []
    for (version, target, model), group in predictions.groupby(["dataset_version", "target", "model"], sort=False):
        group = group.sort_values("flowerID")
        y = group["y_true"].to_numpy(dtype=float)
        pred = group["y_pred"].to_numpy(dtype=float)
        values = metric_values(y, pred)
        summaries.append(
            {
                "dataset_version": version,
                "target": target,
                "model": model,
                "n_flowers": int(len(group)),
                **values,
                "bootstrap_95ci": json.dumps(bootstrap_metrics(y, pred, n_boot, 13), sort_keys=True),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    primary = summary_frame[summary_frame.dataset_version == "full"].copy()
    for target in TARGETS:
        selected = predictions[(predictions.dataset_version == "full") & (predictions.target == target)]
        wide = selected.pivot(index="flowerID", columns="model", values="y_pred")
        y = selected.drop_duplicates("flowerID").set_index("flowerID").loc[wide.index, "y_true"].to_numpy(dtype=float)
        for candidate in wide.columns:
            if candidate == "ridge_low_level":
                continue
            differences.append(
                {
                    "target": target,
                    "comparison": f"{candidate} - ridge_low_level",
                    "rmse_delta": float(
                        metric_values(y, wide[candidate].to_numpy(dtype=float))["rmse"]
                        - metric_values(y, wide["ridge_low_level"].to_numpy(dtype=float))["rmse"]
                    ),
                    "paired_bootstrap_95ci": json.dumps(
                        paired_bootstrap_rmse_delta(
                            y,
                            wide[candidate].to_numpy(dtype=float),
                            wide["ridge_low_level"].to_numpy(dtype=float),
                            n_boot,
                            37,
                        )
                    ),
                    "interpretation": "Conditional image-level bootstrap of paired out-of-fold errors; refitting uncertainty is not included.",
                }
            )
    return summary_frame, pd.DataFrame(differences)


def make_figures(images: dict[int, np.ndarray], predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample_ids = list(images)[:10]
    fig, axes = plt.subplots(2, 5, figsize=(11, 5.4))
    for ax, flower_id in zip(axes.flat, sample_ids):
        ax.imshow(images[flower_id])
        ax.set_title(f"Flower {flower_id}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Examples from the 70-image Gerbera stimulus set", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "stimulus_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_models = ["mean_baseline", "ridge_low_level", "ridge_resnet50", "ridge_fusion", "ols_low_level", "boost_low_level", "stack_low_level"]
    plot_summary = summary[(summary.dataset_version == "full") & summary.model.isin(plot_models)].copy()
    label_map = {
        "mean_baseline": "Training-mean baseline",
        "ridge_low_level": "Low-level + ridge",
        "ridge_resnet50": "ResNet-50 + ridge",
        "ridge_fusion": "Fusion + ridge",
        "ols_low_level": "Low-level + OLS",
        "boost_low_level": "Low-level + boosting",
        "stack_low_level": "Low-level + stacking",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    for ax, target in zip(axes, TARGETS):
        rows = plot_summary[plot_summary.target == target].copy().sort_values("rmse", ascending=False)
        yloc = np.arange(len(rows))
        intervals = [json.loads(x)["rmse"] for x in rows["bootstrap_95ci"]]
        lower = np.array([v[0] for v in intervals])
        upper = np.array([v[1] for v in intervals])
        ax.errorbar(
            rows["rmse"], yloc,
            xerr=np.vstack([rows["rmse"].to_numpy() - lower, upper - rows["rmse"].to_numpy()]),
            fmt="o", color="#176B87", ecolor="#6A8D9A", capsize=3,
        )
        ax.set_yticks(yloc, [label_map[x] for x in rows["model"]])
        ax.set_title(target.title())
        ax.set_xlabel("LOO RMSE (training-fold SD units; lower is better)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Nested leave-one-flower-out prediction performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), sharex=True, sharey=True)
    scatter_models = ["ridge_low_level", "ridge_resnet50", "ridge_fusion"]
    colors = {"ridge_low_level": "#176B87", "ridge_resnet50": "#B85C38", "ridge_fusion": "#6B8E23"}
    for row, target in enumerate(TARGETS):
        for col, model in enumerate(scatter_models):
            ax = axes[row, col]
            data = predictions[
                (predictions.dataset_version == "full")
                & (predictions.target == target)
                & (predictions.model == model)
            ].sort_values("flowerID")
            ax.scatter(data.y_true, data.y_pred, s=24, alpha=0.8, color=colors[model], edgecolor="white", linewidth=0.3)
            limits = [min(data.y_true.min(), data.y_pred.min()), max(data.y_true.max(), data.y_pred.max())]
            ax.plot(limits, limits, color="#333333", linestyle="--", linewidth=0.9)
            ax.set_title(f"{target.title()}: {label_map[model]}", fontsize=9)
            if row == 1:
                ax.set_xlabel("Observed standardized rating")
            if col == 0:
                ax.set_ylabel("Out-of-fold prediction")
            ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "predictions_journal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(n_boot: int, device: str | None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Extracting image features...", flush=True)
    flower_ids, images, features = extract_features(device)
    if len(flower_ids) != 70:
        raise ValueError(f"Expected 70 images, found {len(flower_ids)}")
    feature_versions = {"full": DATA / "all_ratings_full.csv"}
    prediction_frames = []
    audit_frames = []
    data_summaries = {}
    for version, path in feature_versions.items():
        print(f"Analyzing rating file: {version}", flush=True)
        data_summaries[version] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for target in TARGETS:
            matrix, rating_summary = rating_matrix(path, target, flower_ids)
            data_summaries[version][target] = rating_summary
            predictions, audits = fit_oof(
                features,
                matrix,
                target,
                version,
                include_extended=(version == "full"),
            )
            prediction_frames.append(predictions)
            audit_frames.append(audits)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_audits = pd.concat(audit_frames, ignore_index=True)
    summary, differences = summarize_predictions(all_predictions, n_boot)
    all_predictions.to_csv(RESULTS / "oof_predictions.csv", index=False, float_format="%.9g")
    all_audits.to_csv(RESULTS / "fold_audit.csv", index=False)
    summary.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.9g")
    differences.to_csv(RESULTS / "paired_rmse_differences.csv", index=False)
    data_summaries["analysis_environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "resnet_weights_sha256": sha256(WEIGHTS),
        "image_database_sha256": sha256(DATA / "gerbera_appeal_img_db.nc"),
        "low_level_features": len(LOW_FEATURES),
        "resnet_features": 2048,
        "full_feature_dimension": int(features.shape[1]),
        "inner_cv": "5-fold shuffled KFold (seed 13), nested within outer leave-one-flower-out",
        "rating_transform": "Fold-specific participant z-scoring, quality checks, aggregation, and training-target scaling",
        "bootstrap_resamples": n_boot,
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(data_summaries, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    make_figures(images, all_predictions, summary)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote outputs under {RESULTS}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run(args.bootstrap, args.device)
