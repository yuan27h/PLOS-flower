PLOS-flower

**Compact low-level descriptors match a frozen deep representation in flower beauty and interest prediction: a leakage-controlled, auditable benchmark**

This repository is the reproducibility material for the manuscript. It contains the ratings, the derived feature matrices, the analysis code, the result tables and the figure files needed to regenerate every number and every figure reported in the paper.

Every reported quantity can be traced to a file in `results/` or reproduced with a script in `code/`; the mapping is given in Section 5.

---

## 1. Directory layout

All material sits under `plos flower code/`.

| Path | Contents |
|---|---|
| `code/` | `analyze.py` (gerbera nested cross-validation), `paired_contrasts.py` (Table 3 contrasts), `analyze_external_hula2016.py` and `make_external_figure.py` (external benchmark), plus the figure scripts `make_extended_figures.py`, `make_matched_contrasts.py`, `make_scheme_draft.py` |
| `data/` | Gerbera rating table, derived feature matrix, feature dictionary, image database, and the independent Hula–Flegr benchmark |
| `results/` | Metrics with bootstrap intervals, out-of-fold predictions, fold audits, paired contrasts, dataset summaries and checksums |
| `figures/` | The ten figure files referenced by the manuscript |
| `weights/` | Frozen ImageNet ResNet-50 weights used for the deep representation |
| `README.md` | This file |

---

## 2. Data files

| File | Rows | Bytes | SHA-256 |
|---|---|---|---|
| `data/all_ratings_full.csv` | 160,947 | 10,324,813 | `929b299e8efd8ceb02421ae2f72925f49684f34749777c4ea94cbeb28d754317` |
| `data/derived_features.csv` | 70 flowers × 2,066 predictors | 870,116 | `f949315269abce8afd13d120ae540842bc20e217832fabc92d442fcd3bc27219` |
| `data/feature_dictionary.csv` | 18 descriptors × 3 columns | 1,100 | `86e6dadeea93dd409da7c05a5cafc51e498c198eed870bd49cc8b0a4ed232fdb` |
| `data/gerbera_appeal_img_db.nc` | 70 images, 480 × 480 px | 48,401,483 | `98f6635d098a12e64a8f1d1077431a484e858873481b4e17314780675eb6ff7e` |
| `data/external_hula2016/ratings_and_traits.xlsx` | 52 stimulus groups, 2,006 respondents | 1,077,346 | `f5809290cf6cc2b4275e71b93d882de5a5b987b5525a9adb34e771d27dbff48e` |
| `weights/resnet50-imagenet1k-v2.pth` | — | — | `3849cee25b3cd361ecae9151a410e90b4d3f0b5d80e6b613a52aafc1f45b77ba` |

The SHA-256 of `data/derived_features.csv` is the value recorded in `results/feature_regeneration_check.json`, so the feature matrix shipped here is the one that produced the reported out-of-fold predictions.

**Gerbera rating table.** `all_ratings_full.csv` is the single rating table used for all gerbera estimates reported in the manuscript. It has 160,947 data rows, 144,077 of which carry an appeal rating and 76,457 an interest rating. The two analysis cohorts are defined in `code/analyze.py` as follows: appeal uses ratings from participants who rated appeal and not interest (73,920 rows, 1,056 identifiers), and interest uses ratings from participants who rated both (70,157 rows, 1,003 identifiers). The table does not match the MD5 recorded by the legacy reproduction code, so it is reported with the SHA-256 above rather than as checksum-verified canonical data; no second or fuller version of this table is distributed.

**External benchmark.** `data/external_hula2016/` holds the CC BY 4.0 rating workbook, the 156 colour stimuli, per-group targets, and the derived features aggregated over the three photographs of each stimulus group. The target is the exact participant-level mean of the colour-rating columns; the rounded `flower_traits` scores are retained for audit only.

---

## 3. Software environment

The files in `results/` were produced with:

```
Python 3.14.2   numpy 2.5.4   pandas 3.1.1   scikit-learn 1.9.1
torch 2.15.0    torchvision 0.29.2            CPU (no CUDA device)
```

Image features are deterministic given the frozen weights, so a CPU-only run reproduces the same predictions; a CUDA device only shortens feature extraction.Beyond the versions above, the scripts import `scipy`, `joblib`, `xarray` (to read `gerbera_appeal_img_db.nc`), `Pillow`, `openpyxl` (to read the external rating workbook) and, for `make_scheme_draft.py`, `h5py`. A minimal install is:

```bash
pip install "numpy>=2.2" "pandas>=2.3" "scikit-learn>=1.7" scipy joblib xarray h5py openpyxl Pillow \
            torch torchvision matplotlib
```

**Version sensitivity of the boosting and stacking rows.** The gradient-boosting and stacking rows (Table 4, and the Boosting/Stacking rows of Table 3) depend on the scikit-learn build: the same code and data give slightly different boosted and stacked predictions under a different version. All other rows are stable to the reported precision, and no conclusion changes under any version tested.

---

## 4. How to run

All commands are run from inside `plos flower code/`.

```bash
# 1. Gerbera nested cross-validation (nested leave-one-flower-out, 70 folds,
#    6 learners x 2 targets, 5,000-resample bootstrap)
python code/analyze.py --bootstrap 5000

# 2. Paired flower-level contrasts against the compact ridge baseline
#    (Delta R^2 and Delta RMSE with paired 2,000-resample intervals)
python code/paired_contrasts.py

# 3. Figures derived from the saved gerbera predictions
python code/make_extended_figures.py

# 4. External Hula-Flegr benchmark (leave-one-stimulus-group-out, 52 folds)
python code/analyze_external_hula2016.py

# 5. External-benchmark figure
python code/make_external_figure.py

# 6. Matched-contrasts figure (reads the paired-contrast tables of step 2)
python code/make_matched_contrasts.py

# 7. Study-design schematic (reads the image database; requires h5py)
python code/make_scheme_draft.py
```

Steps 1, 2 and 4 rewrite `results/`; steps 3, 5, 6 and 7 rewrite `figures/`. Step 1 is the expensive one (feature extraction plus 420 model fits). Step 2 reproduces the `r2_delta` column of Table 3 exactly and leaves the `rmse_delta` column produced by step 1 untouched. Steps 3, 5 and 6 need only `numpy`, `pandas`, `scikit-learn` and `matplotlib`; step 7 additionally needs `h5py`; step 4 needs `scipy`, `scikit-learn`, `pandas`, `torch`, `torchvision` and `openpyxl`.

**JPEG sensitivity of the external low-level descriptors.** Four of the eighteen low-level descriptors (`saturation_mean`, `hue_mean`, `hue_sd`, `hue_entropy_8bin`) summarise a circular quantity that is ill-conditioned at low saturation, so they depend on the JPEG decoding path. Re-decoding the same 156 photographs with a different Pillow/libjpeg build shifts them by up to 0.02, which propagates through the 52 leave-one-group-out fits; the compact-model R² then lands at about 0.435–0.445 rather than the exact shipped value, while the other fourteen descriptors agree to within 0.003. The ordering of the three feature families (compact > fusion > ResNet-50) is unchanged. The distributed `data/external_hula2016/derived_features.csv` is the file produced by the recorded environment; a re-run overwrites it in place.

**Figures: what re-running reproduces.** `external_replication.png`, `matched_contrasts.png` and `performance_r2.png` are exported directly by `make_external_figure.py`, `make_matched_contrasts.py` and `make_extended_figures.py` from the shipped results, so steps 3, 5 and 6 reproduce them at the same pixel dimensions (anti-aliasing differences only). The remaining figures were exported by a separate journal-styling pass; the plotting scripts re-derive their content from the packaged results with default styling — same points, same intervals, same displayed R² and RMSE values, but different aspect ratio, colours and panel labels. No reported quantity depends on the rendering: every number in the figures is read from `results/`.

---

## 5. Mapping results to the manuscript

| Manuscript item | Source file | Key columns |
|---|---|---|
| Table 1 (nested performance, three feature families) | `results/metrics.csv` | rows `ridge_low_level`, `ridge_resnet50`, `ridge_fusion`; `cv_r2_vs_fold_training_mean`, `bootstrap_95ci`, `rmse`, `mae`, `pearson_r` |
| Table 2 (Hula–Flegr) | `results/external_hula2016_metrics.csv` | `cv_r2`, `rmse`, `mae`, `pearson_r` per model |
| Table 3 (matched contrasts vs compact ridge) | `results/paired_r2_differences.csv`, `results/paired_rmse_differences.csv` | `r2_delta` / `rmse_delta` with `paired_bootstrap_95ci` |
| Table 4 (OLS, gradient boosting, cross-fitted stacking) | `results/metrics.csv` | rows `ols_low_level`, `boost_low_level`, `stack_low_level` |
| Fig 1 `scheme_draft.png` | study-design schematic drawn by `code/make_scheme_draft.py` | — |
| Fig 2 `stimulus_examples.png` | stimulus photos from `data/gerbera_appeal_img_db.nc` | — |
| Fig 3 `external_replication.png` | `results/metrics.csv` + `results/external_hula2016_metrics.csv` | per-dataset R² |
| Fig 4 `predictions_journal.png` | `results/oof_predictions.csv` | `y_true`, `y_pred`, `flowerID`, `model` |
| Fig 5 `matched_contrasts.png` | `results/paired_r2_differences.csv` + `results/paired_rmse_differences.csv` | paired ΔR² / ΔRMSE with intervals |
| Fig 6 `performance_r2.png` | `results/metrics.csv` | rank of each learner within each target |
| Fig 7 `feature_target_combo.png` | `data/derived_features.csv` + `results/oof_predictions.csv` | descriptor–target correlations |
| Fig 8 `residuals_diagnostics.png` | `results/oof_predictions.csv` | residuals of `ridge_low_level` |
| Fig 9 `errors_quantiles.png` | `results/oof_predictions.csv` | absolute errors per feature family |
| Fig 10 `resnetspace.png` | `data/derived_features.csv` | first two principal components of the ResNet-50 block |
| `analysis_summary.json` | — | row and participant counts, row-level SHA-256 of the rating table, environment |
| `fold_audit.csv` | — | per-fold training size, held-out flower, and outcome construction |
| `feature_regeneration_check.json` | — | SHA-256, row count and predictor count of `derived_features.csv` |
| `external_hula2016_analysis_summary.json` | — | external feature-table and workbook SHA-256, device |`code/analyze.py` additionally writes one diagnostic panel (`model_performance.png`) that is not referenced by the manuscript.

---

## 6. Protocol notes

- **Outer loop.** One held-out flower (gerbera) or one held-out three-photograph group (Hula–Flegr) per fold; 70 and 52 folds respectively.
- **Inner loop.** Five-fold shuffled `RidgeCV` (seed 13) inside each outer training fold. Standardization, PCA of the 2,048-dimensional ResNet-50 block, penalty selection and outcome construction all happen inside the outer training fold only.
- **R² baseline.** The denominator uses the fold-specific training mean, not the global mean, so R² can be negative for weakly predictive learners.
- **Intervals.** 2,000 (paired contrasts) and 5,000 (main metrics) flower-level bootstrap resamples, seed 20261102.
- **Outcome construction.** Duplicate participant–flower cells are averaged before flower-level aggregation; `analysis_summary.json` reports the number of averaged cells per target.

---

## 7. Licensing and reuse

The gerbera corpus is associated with the public Code Ocean capsule at <https://doi.org/10.24433/CO.7484990.v1> (`https://codeocean.com/capsule/2803871/tree/v1`); its `LICENSE` file identifies MIT terms for the capsule code. The analysis code in `code/` is released for reuse with the article. The Hula–Flegr material is redistributed under CC BY 4.0 with source citation and attribution.

The gerbera images, rating table and pretrained weights are included in this repository so that the reported benchmark can be reproduced end to end. They are provided for research use in connection with this article; redistribution beyond that use remains subject to the terms of the original corpus.

---

## 8. Citation

If you use this code or these derived features, please cite:

> Compact low-level descriptors match a frozen deep representation in flower beauty and interest prediction: a leakage-controlled, auditable benchmark. (Citation details to be updated upon publication.)

---

## 9. Contact

Questions about the code or the data can be raised through the GitHub issue tracker of this repository.
