# Round 5: independent PLOS ONE minor-revision review

## Review basis

This pass was performed against the final recompiled `main.tex`,
`main_reviewer.tex`, shipped metrics, figure set, reproducibility code, and
the official PLOS template supplied with the project. It was conducted as a
fresh manuscript-level check rather than by copying the previous decision.

## Innovation and comparison

- The manuscript now states one bounded contribution consistently: a
  leakage-controlled flower-level benchmark for a small standardized corpus.
- The contribution is operationalized through three matched contrasts under
  the same nested leave-one-flower-out procedure: compact descriptors versus
  frozen ResNet-50 features, fusion versus compact ridge, and flexible
  learners versus ridge.
- Table 2 reports the contrasts directly as changes in CV R2 and RMSE on the
  same 70 outer folds. It separates the large interest effect from the small
  appeal effect, avoiding a claim that the result is universal.
- The manuscript explicitly distinguishes this empirical benchmark from a new
  learning algorithm, causal evidence, and a claim that ResNet-50 is generally
  inferior.

## Scientific and reporting checks

- The reported metrics agree with `results/metrics.csv`: compact ridge gives
  appeal CV R2 0.256 and interest CV R2 0.669; ResNet-50 gives 0.199 and
  0.142; fusion gives 0.203 and 0.171.
- The methods state that participant filtering, rating standardization,
  target construction, feature scaling, PCA, and model selection are fitted
  from training flowers within each outer fold.
- The figures cover observed versus out-of-fold predictions, matched delta
  contrasts, explained variance, feature associations, residual behavior,
  flower-level error distributions, data-version sensitivity, and
  representation geometry.
- Historical figures from the Chinese draft are excluded from the clean
  delivery package and are not used as evidence for the revised claims.
- Keywords and standard declaration sections are now present in both LaTeX
  sources. The official submission source still keeps graphics external,
  while the reviewer source embeds the same nine current figures.

## Recompilation and visual checks

- `main.pdf` recompiles successfully to 11 pages after the reference audit.
- `main_reviewer.pdf` recompiles successfully to 16 pages after the reference audit.
- After the expanded BibTeX and LaTeX passes, no undefined references,
  unresolved citations, overfull boxes, or underfull boxes were reported; the
  only remaining messages are the template's normal `!h` to `!ht` float conversion.
- Rendered pages confirm that the innovation text, keywords, tables, figures,
  declarations, and references are not clipped.

## Remaining author actions

1. Replace the author-confirmation wording in contributions, funding,
   competing interests, acknowledgments, and local ethics determination with
   verified statements.
2. Verify which local ratings file is the canonical public version, rerun if it
   differs, and deposit the exact final package at a stable repository DOI.
3. Confirm image, rating-data, and pretrained-weight reuse permissions.

These are submission metadata and provenance tasks. They do not undermine the
clarity of the innovation or the validity of the comparisons performed on the
included local files.

## Decision

**Minor revision level reached.** The manuscript is scientifically coherent,
the innovation is explicit and quantitatively compared, and the official
template package is technically ready for author completion. Submission
should follow after the three author/provenance actions above.

## Round 6: independent figure-quality review

- The revised PNG figures were regenerated from the included local outputs at
  600 dpi. The redesign follows the source draft's strongest visual
  conventions: multi-panel comparison, stable route colors, direct numerical
  labels, and explicit panel lettering.
- The main performance figure now matches the reported analysis: each point is
  a matched comparison in delta CV $R^2$ by delta RMSE space, with paired
  bootstrap intervals and the sign convention stated in the caption.
- The sensitivity figure now uses slope lines with direct values; the new
  error-distribution figure shows all 70 held-out flowers for each route. The
  heat map and low-information fold audit were removed.
- Rendered reviewer pages 7--13 were inspected. Figures are embedded,
  captions are attached to the correct panels, and no plot, legend, label, or
  caption is clipped.

**Round 6 decision: pass at minor-revision level.** The remaining items are
author declarations, canonical data verification, reuse permissions, and a
stable repository DOI rather than figure-design or manuscript-consistency
defects.

## Round 7: independent check after the Figure 3/4/5/7 redesign

- Figure 3 is now a paired effect display for delta CV $R^2$ and delta RMSE;
  Figure 4 is an absolute-performance paired point-and-interval display;
  Figure 5 is a ranked connected-dot plot with readable descriptor names; and
  Figure 7 is a two-panel empirical quantile display with independent target
  scales and median/90th-percentile markers.
- The redesign removes the previous repeated scatter structure and the crowded
  feature-number key. The figures were regenerated at 600 dpi from the same
  shipped results, with no numerical or inferential changes.
- Reviewer pages 7--11 were rendered after the final three LaTeX passes. No
  label, legend, caption, axis, or interval is clipped or overlapped; logs show
  no unresolved references, citation failures, or box overflow.

**Round 7 decision: pass at minor-revision level.** The visual request is
complete; author declarations, canonical-data verification, and reuse
permissions remain the only pre-submission confirmations.

## Round 8: independent visual-language review after the second redesign

- Fig. 3 now uses arrows in a two-target effect field with a separate RMSE
  interval panel; Fig. 4 uses rank flow; Fig. 5 uses a polar fingerprint with
  a complete descriptor key; and Fig. 8 uses a full-file/subset concordance
  plane with route-specific marker shapes.
- The exact zero appeal change is shown once for the coincident routes rather
  than by offsetting points. All plotted values remain generated from the
  shipped metrics and prediction files.
- The new figures were regenerated at 600 dpi and checked before recompilation
  for label collisions, clipped legends, and caption mismatches.

**Round 8 decision: pass at minor-revision level.** The four figures now have
distinct visual identities and distinct analytical reading tasks.
