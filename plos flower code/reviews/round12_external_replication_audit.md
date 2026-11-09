# Independent review: external replication and final numerical consistency

Date: 2026-10-12

This pass was conducted independently of the earlier manuscript reviews after
the Hula--Flegr external benchmark was added and its $R^2$ calculation was
corrected.

## Audit performed

- Read `code/analyze_external_hula2016.py` and verified that every outer fold
  stores `fold_training_mean` and that external $R^2$ uses
  `sum((y_true - fold_training_mean)^2)` as its denominator.
- Verified that the external inner model-selection split now matches the
  primary benchmark: shuffled five-fold `KFold`, seed 13, inside each outer
  training fold.
- Recomputed the three reported external $R^2$ values directly from
  `results/external_hula2016_oof_predictions.csv`.
- Confirmed 52 stimulus groups, exactly three color photographs per group,
  156 unique image files, and 156 OOF rows (one row per model and group).
- Confirmed all target, prediction, and fold-baseline values are finite.
- Confirmed `paper/main.tex` and `paper/main_reviewer.tex` report the same
  rounded values as `results/external_hula2016_metrics.csv`.
- Checked the external figure after regeneration and confirmed that it shows
  the three dataset-level comparisons and the deep-feature deltas without a
  repeated bar-chart layout.
- Checked the source and licensing statements against the package attribution
  file and the cited Figshare DOI.

## Recomputed results

| Model | Reported $R^2$ | Recomputed $R^2$ | RMSE |
|---|---:|---:|---:|
| Compact low-level | 0.404559331 | 0.404559331 | 0.393497918 |
| ResNet-50 | 0.300493643 | 0.300493642 | 0.426499990 |
| Fusion | 0.307261805 | 0.307261806 | 0.424431647 |

The three reported values were recomputed directly from the current external
OOF file; the compact value changed from the earlier audit because the inner
cross-validation split was made identical to the primary benchmark. The
ordering is unchanged: Compact has the highest point estimate, but the
external intervals overlap and the manuscript correctly presents this as a
bounded replication rather than universal superiority or direct transfer.

## Decision

**Pass for the external-replication component.** No numerical or provenance
blocker was found in this pass. The remaining pre-submission requirements are
author confirmations already marked in the package: verification of the
canonical local gerbera rating file, ethics identifier, funding and competing
interest statements, and final confirmation of redistribution rights for the
local source components.
