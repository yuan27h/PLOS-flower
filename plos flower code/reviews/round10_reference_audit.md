# Round 10: reference coverage and distribution audit

## Audit basis

This audit compares the revised PLOS ONE manuscript with the attached Chinese
source manuscript and the final `references.bib`. The source manuscript uses
39 numbered bibliography items. The revised package now contains 42 BibTeX
entries, and every one is cited in both `main.tex` and
`main_reviewer.tex`.

## Coverage

| Role in the manuscript | Cited entries |
|---|---:|
| Flower preference, affect, attention, colour, symmetry, complexity | 15 |
| Hand-designed visual descriptors and computational aesthetics | 4 |
| Learned aesthetic and image-quality representations | 5 |
| Gerbera precedent and broad aesthetic-assessment benchmarks | 8 |
| Validation, small-sample statistics, models, and software | 7 |
| Explainability methods used to delimit the unverified historical analyses | 3 |
| **Total unique entries** | **42** |

The added entries restore source-draft lines that are relevant to the revised
claims (objective versus subjective complexity, CLIP-based aesthetic
assessment, and SHAP/Grad-CAM methodology) and add direct methodological
support for small-sample cross-validation and the percentile bootstrap. They
are cited where those ideas are discussed; they are not included as uncited
padding.

## Distribution in the manuscript

- Introduction: 35 citation mentions, 34 unique references.
- Materials and methods: 9 citation mentions, 9 unique references.
- Discussion: 8 citation mentions, 8 unique references.
- Results, conclusion, and declarations: no background citations are needed
  because those sections report the present analysis or submission metadata.
- Total: 52 citation mentions, 42 unique references.
- Uncited bibliography entries: 0.
- Missing citation keys: 0.

The distribution now supports the complete argument chain: flower preference
and visual-attention evidence in the introduction; descriptor, representation,
validation, bootstrap, and software provenance in the methods; and
benchmarking, transfer limits, and interpretability boundaries in the
discussion.

## Compilation check

After the reference revision:

- `paper/main.pdf`: 11 pages.
- `paper/main_reviewer.pdf`: 16 pages.
- Both documents compile after BibTeX and two LaTeX passes.
- No undefined citations, unresolved references, overfull boxes, or underfull
  boxes remain. The only LaTeX warnings are the official template's conversion
  of `!h` float placement to `!ht`.

The audit does not certify the reuse rights of the public capsule's images or
rating files. Those rights remain an author-side provenance check before
deposit or submission.
