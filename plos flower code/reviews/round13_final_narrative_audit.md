# Round 13: final narrative, evidence, and reproducibility audit

Date: 2026-10-12

## Scope

This audit was performed after the external Hula--Flegr analysis was rerun
with the same shuffled, seeded five-fold inner cross-validation used by the
primary Gerbera benchmark. The manuscript was read independently from the
earlier review summaries, with the claims checked against the current CSV,
JSON, code, and compiled sources.

## Independent pass A: narrative logic

1. The abstract now separates the claims supported by both datasets from the
   Gerbera-only learner extension. Both datasets use the same three-way ridge
   comparison: compact descriptors, frozen ResNet-50 features, and their
   fusion. Ordinary least squares, boosting, and stacking are explicitly
   described as Gerbera-only analyses.
2. The central innovation is stated as a methodologically controlled,
   flower-level comparison rather than as a new algorithm: fold-local outcome
   construction for Gerbera, fold-local preprocessing and model selection,
   leave-one-flower-unit-out prediction, and a matched feature-family test in
   two independent flower datasets.
3. The external dataset is described as an independent external dataset
   selected for replication. The text no longer implies preregistration or a
   prespecified benchmark that is not documented in the archive.
4. The two datasets are presented as parallel support, not as pooled data or a
   direct-transfer test. The manuscript states that appeal, interest, and
   beauty are different rating constructs.
5. The strongest claim is limited to Gerbera interest, where compact ridge has
   $R^2=0.669$ versus $0.142$ for ResNet-50 and $0.171$ for fusion. Appeal and
   external beauty are described as point-estimate orderings with overlapping
   uncertainty intervals.

## Independent pass B: numerical and submission integrity

- `metrics.csv` agrees with the Gerbera values in both manuscript sources:
  appeal compact/ResNet/fusion $R^2=0.256/0.199/0.203$ and interest
  $0.669/0.142/0.171$.
- `external_hula2016_metrics.csv` agrees with the external table after the
  protocol alignment: compact/ResNet/fusion $R^2=0.404559/0.300494/0.307262$,
  rounded in the manuscript to $0.405/0.300/0.307$. The compact 95% bootstrap
  interval is $[0.154,0.606]$ and its RMSE and Pearson correlation are
  $0.393$ and $0.629$.
- The external OOF file contains 52 groups per model, stores a
  `fold_training_mean` for every row, and the code computes $R^2$ against that
  fold-local baseline. The external archive contains 52 groups and 156 color
  photographs, with three photographs retained together per held-out unit.
- The primary Gerbera rating file has 173,963 data rows and SHA-256
  `95b56fd4ecf27c4a608ed6b15005412841b4bc26adebe840fd22a352449cd880`.
  The archived subset has 160,947 data rows and SHA-256
  `929b299e8efd8ceb02421ae2f72925f49684f34749777c4ea94cbeb28d754317`.
  The summary JSON records the same hashes and the package does not describe
  either file as canonical until the public capsule source is confirmed.
- The paper source has 44 cited bibliography keys, 44 bibliography entries,
  and no missing or uncited keys. It has 14 labels and no unresolved `ref`
  targets. The reviewer source embeds all 10 supplied PNG figures; the
  official source embeds none, as required by the supplied PLOS template.
- `main.pdf` and `main_reviewer.pdf` compile successfully after BibTeX and two
  LaTeX passes. Final logs contain no undefined citations, undefined
  cross-references, or box warnings; only the template's `!h` to `!ht` float
  notices remain.

## Decision

**Pass at the requested small-revision level for narrative and evidence.**
The manuscript has one clear, bounded innovation and the two datasets support
the same explicitly defined feature-family comparison. Remaining submission
requirements are author confirmations: ethics identifier, CRediT roles,
funding, competing interests, acknowledgments, image/rating-data permissions,
and confirmation of the canonical Gerbera source file before public deposit.
