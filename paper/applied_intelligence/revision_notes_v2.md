# Revision Notes: v2 Pre-Submission Draft

## Scope

Revised `paper/applied_intelligence/manuscript.md` from the v1 draft to a v2 pre-submission draft. No experiments were run, no raw result JSONs were modified, and no training code was changed.

## Major Changes

- Strengthened Related Work with subsection structure:
  - World models and long-horizon rollout
  - Euclidean latent priors and JEPA-style regularization
  - Spectral graph priors and geometric deep learning
  - Molecular dynamics benchmarks
  - Synthetic graph rewriting / relational dynamics
- Added citation placeholders:
  - `[LeCun2022]`
  - `[BalestrieroLeCun2025]`
  - `[Hansen2024TDMPC2]`
  - `[Bronstein2021GDL]`
  - `[Chmiela2017MD17]`
  - `[Christensen2020RMD17]`
  - `[Wolfram2020]`
- Strengthened Method with explicit objective:
  - `L = L_transition/rollout + lambda R_prior`
- Defined prior families:
  - no prior
  - Euclidean batch-level covariance prior
  - spectral Laplacian prior
- Added Laplacian mode definitions:
  - `per_frame`
  - `fixed_mean`
  - `fixed_frame0`
- Clarified that lower rollout error is better and H=16 is the main long-horizon metric.
- Added figure callouts for Figures 1-5.
- Added Reproducibility and Analysis Hygiene section covering:
  - manifest deduplication
  - audit checks
  - duplicate experimental keys = 0
  - `analysis_out` artifacts
  - checkpointing path prepared for future disjoint evaluation
  - old completed runs lacking checkpoints
- Added References section with placeholder bibliographic entries.

## Claim-Safety Edits

- Added "under the current evaluation protocol" to rMD17 claims where appropriate.
- Avoided implying disjoint-frame evaluation was performed.
- Avoided implying Wolfram supports spectral advantage.
- Avoided claiming LeJEPA is refuted.
- Preserved all reported numerical values.

