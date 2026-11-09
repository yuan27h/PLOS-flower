from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet50_Weights, resnet50

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external_hula2016"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights" / "resnet50-imagenet1k-v2.pth"

import sys

sys.path.insert(0, str(ROOT / "code"))
from analyze import LOW_FEATURES, low_level_features  # noqa: E402

ALPHAS = np.logspace(-4, 4, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def load_groups(traits: pd.DataFrame) -> tuple[list[str], dict[str, list[Path]], pd.DataFrame]:
    files = sorted((DATA / "stimuli_color").glob("*.jpg"))
    groups: dict[str, list[Path]] = {}
    audit = []
    for _, row in traits.iterrows():
        abbreviation = str(row["abbreviation"])
        scientific = str(row["scientific_name"])
        key = normalize(scientific)
        matched = [
            path for path in files
            if normalize(re.sub(r"\s+[123]$", "", path.stem)).startswith(key)
        ]
        match_rule = "scientific_name"
        if len(matched) != 3:
            first_word = normalize(scientific.split()[0])
            matched = [
                path for path in files
                if normalize(re.sub(r"\s+[123]$", "", path.stem)).startswith(first_word)
            ]
            match_rule = "first_scientific_word_fallback"
        if len(matched) != 3:
            raise RuntimeError(f"Expected 3 color images for {scientific!r}, found {len(matched)}")
        groups[abbreviation] = matched
        audit.append(
            {
                "abbreviation": abbreviation,
                "scientific_name": scientific,
                "match_rule": match_rule,
                "image_files": ";".join(path.name for path in matched),
            }
        )
    keys = [str(x) for x in traits["abbreviation"]]
    if len(groups) != len(keys) or len(files) != 3 * len(keys):
        raise RuntimeError("The external stimulus archive is not a complete 3-image-per-group set")
    return keys, groups, pd.DataFrame(audit)


def load_model() -> tuple[torch.nn.Module, object, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=None)
    try:
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms(), device


def extract_group_features(keys: list[str], groups: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, transform, device = load_model()
    low_rows: list[np.ndarray] = []
    resnet_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for key in keys:
            pixels = [np.asarray(Image.open(path).convert("RGB")) for path in groups[key]]
            low_rows.append(np.mean([low_level_features(x) for x in pixels], axis=0))
            batch = torch.stack([transform(Image.fromarray(x)) for x in pixels]).to(device)
            resnet_rows.append(model(batch).detach().cpu().numpy().mean(axis=0))
    low = pd.DataFrame(low_rows, index=keys, columns=LOW_FEATURES)
    resnet = pd.DataFrame(
        resnet_rows,
        index=keys,
        columns=[f"resnet50_{i:04d}" for i in range(2048)],
    )
    return low, resnet


def source_targets(keys: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    beauty = pd.read_excel(DATA / "ratings_and_traits.xlsx", sheet_name="beauty")
    traits = pd.read_excel(DATA / "ratings_and_traits.xlsx", sheet_name="flower_traits")
    traits = traits.set_index("abbreviation").loc[keys]
    target_rows = []
    for key in keys:
        columns = [column for column in beauty.columns if column == f"B{key}"]
        if len(columns) != 1:
            raise RuntimeError(f"Expected exactly one color rating column for {key}")
        values = pd.to_numeric(beauty[columns[0]], errors="coerce").dropna()
        target_rows.append(
            {
                "abbreviation": key,
                "beauty_color_mean": float(values.mean()),
                "n_participant_ratings": int(values.size),
                "published_rounded_score": float(traits.loc[key, "beauty_color_score"]),
                "dominant_color": str(traits.loc[key, "dominant_color"]),
                "symmetry": str(traits.loc[key, "symmetry"]),
                "angularity": str(traits.loc[key, "angularity"]),
            }
        )
    target_frame = pd.DataFrame(target_rows).set_index("abbreviation").loc[keys]
    return target_frame["beauty_color_mean"].to_numpy(dtype=float), target_frame


def model_for(kind: str) -> Pipeline:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if kind in {"resnet50", "fusion"}:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    # Match the primary Gerbera benchmark: shuffled, seeded five-fold
    # inner validation remains inside each outer leave-one-group-out fold.
    steps.append(("ridge", RidgeCV(alphas=ALPHAS, cv=KFold(n_splits=5, shuffle=True, random_state=13))))
    return Pipeline(steps)


def loo_predictions(features: dict[str, np.ndarray], y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    loo = LeaveOneOut()
    predictions: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for kind, x in features.items():
        for fold, (train_idx, test_idx) in enumerate(loo.split(x)):
            model = model_for(kind)
            model.fit(x[train_idx], y[train_idx])
            prediction = float(model.predict(x[test_idx])[0])
            predictions.append(
                {
                    "fold": fold,
                    "model": kind,
                    "group_index": int(test_idx[0]),
                    "y_true": float(y[test_idx[0]]),
                    "y_pred": prediction,
                    # Match the primary analysis: each held-out target is
                    # compared with the mean available in its training fold.
                    "fold_training_mean": float(np.mean(y[train_idx])),
                }
            )
            pca_components = None
            if "pca" in model.named_steps:
                pca_components = int(model.named_steps["pca"].n_components_)
            audits.append({"fold": fold, "model": kind, "n_train_groups": int(len(train_idx)), "ridge_alpha": float(model.named_steps["ridge"].alpha_), "pca_components": pca_components})
    return pd.DataFrame(predictions), pd.DataFrame(audits)


def metric_values(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    error = y - pred
    denominator = float(np.sum((y - baseline) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 1e-12 else float("nan")
    return {
        "cv_r2": r2,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": float(pearsonr(y, pred).statistic),
        "spearman_r": float(spearmanr(y, pred).statistic),
    }


def bootstrap_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray,
    n_boot: int = 5000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(13)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yy, pp, bb = y[idx], pred[idx], baseline[idx]
    r2 = 1.0 - np.sum((yy - pp) ** 2, axis=1) / np.maximum(np.sum((yy - bb) ** 2, axis=1), 1e-12)
    rmse = np.sqrt(np.mean((yy - pp) ** 2, axis=1))
    return {
        "cv_r2": [float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))],
        "rmse": [float(np.percentile(rmse, 2.5)), float(np.percentile(rmse, 97.5))],
    }


def paired_rmse_delta(y: np.ndarray, first: np.ndarray, second: np.ndarray, n_boot: int = 5000) -> list[float]:
    rng = np.random.default_rng(37)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yy = y[idx]
    delta = np.sqrt(np.mean((yy - first[idx]) ** 2, axis=1)) - np.sqrt(np.mean((yy - second[idx]) ** 2, axis=1))
    return [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    traits = pd.read_excel(DATA / "ratings_and_traits.xlsx", sheet_name="flower_traits")
    keys, groups, group_audit = load_groups(traits)
    low, resnet = extract_group_features(keys, groups)
    y, target_frame = source_targets(keys)
    feature_frame = pd.concat([low, resnet], axis=1)
    feature_frame.to_csv(DATA / "derived_features.csv", float_format="%.9g")
    target_frame.to_csv(DATA / "targets.csv", float_format="%.9g")
    group_audit.to_csv(RESULTS / "external_hula2016_group_audit.csv", index=False)
    features = {
        "compact_low_level": low.to_numpy(dtype=np.float32),
        "resnet50": resnet.to_numpy(dtype=np.float32),
        "fusion": feature_frame.to_numpy(dtype=np.float32),
    }
    predictions, audits = loo_predictions(features, y)
    predictions["abbreviation"] = predictions["group_index"].map(dict(enumerate(keys)))
    predictions.to_csv(RESULTS / "external_hula2016_oof_predictions.csv", index=False, float_format="%.9g")
    audits.to_csv(RESULTS / "external_hula2016_fold_audit.csv", index=False)
    summary_rows = []
    for model, group in predictions.groupby("model", sort=False):
        yy = group.sort_values("group_index")["y_true"].to_numpy()
        pp = group.sort_values("group_index")["y_pred"].to_numpy()
        bb = group.sort_values("group_index")["fold_training_mean"].to_numpy()
        summary_rows.append(
            {
                "dataset": "hula2016_color",
                "target": "beauty",
                "model": model,
                "n_groups": len(yy),
                **metric_values(yy, pp, bb),
                "bootstrap_95ci": json.dumps(bootstrap_metrics(yy, pp, bb), sort_keys=True),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "external_hula2016_metrics.csv", index=False, float_format="%.9g")
    wide = predictions.pivot(index="group_index", columns="model", values="y_pred").sort_index()
    y_ordered = y[wide.index.to_numpy()]
    paired_rows = []
    for model in ["resnet50", "fusion"]:
        candidate_rmse = float(np.sqrt(np.mean((y_ordered - wide[model].to_numpy()) ** 2)))
        compact_rmse = float(np.sqrt(np.mean((y_ordered - wide["compact_low_level"].to_numpy()) ** 2)))
        paired_rows.append(
            {
                "comparison": f"{model} - compact_low_level",
                "rmse_delta": candidate_rmse - compact_rmse,
                "paired_bootstrap_95ci": json.dumps(
                    paired_rmse_delta(
                        y_ordered,
                        wide[model].to_numpy(),
                        wide["compact_low_level"].to_numpy(),
                    )
                ),
            }
        )
    pd.DataFrame(paired_rows).to_csv(RESULTS / "external_hula2016_paired_rmse.csv", index=False, float_format="%.9g")
    summary_info = {
        "dataset": "Hula-Flegr 2016 flower beauty external benchmark",
        "article_doi": "10.7717/peerj.2106",
        "dataset_doi": "10.6084/m9.figshare.2082529.v1",
        "license": "CC BY 4.0 (as listed by Figshare metadata)",
        "n_groups": len(keys),
        "n_color_images": sum(len(paths) for paths in groups.values()),
        "rating_workbook_sha256": sha256(DATA / "ratings_and_traits.xlsx"),
        "target": "exact participant-level mean of the 52 B<abbreviation> color rating columns; rounded flower_traits scores retained only for audit",
        "feature_aggregation": "mean of the three color-photo feature vectors within each stimulus group",
        "outer_validation": "leave-one-stimulus-group-out",
        "r2_baseline": "fold-specific training-group mean; denominator is sum((y_true - fold_training_mean)^2)",
        "inner_validation": "five-fold shuffled RidgeCV (seed 13) within each outer training fold",
        "resnet_weights_sha256": sha256(WEIGHTS),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (RESULTS / "external_hula2016_analysis_summary.json").write_text(json.dumps(summary_info, indent=2, ensure_ascii=True), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
