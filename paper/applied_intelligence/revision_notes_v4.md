# Revision Notes: v4 Pre-Submission Manuscript

## Scope

Polished `paper/applied_intelligence/manuscript.md` for v4 pre-submission readiness. No training was run, no raw result JSONs were modified, and no experimental results were changed.

## Changes

- Removed internal development-note wording from the Method section.
- Replaced implementation-check caveats with confident, appropriately scoped descriptions.
- Added explicit statistical reporting paragraph in Experimental Setup:
  - means and standard deviations are over seeds
  - bootstrap confidence intervals are used in figures where available
  - H=16 is the primary long-horizon metric
  - lower rollout error is better
- Updated inline Results tables to report mean +/- std and n:
  - rMD17 aspirin main H=16
  - Laplacian ablation H=16
  - weight sweep best H=16
  - Wolfram H=16 mean/median
- Confirmed figure callouts point to:
  - `figures/fig1_conceptual_priors.png`
  - `figures/fig2_rmd17_horizon_scaling.png`
  - `figures/fig3_laplacian_ablation.png`
  - `figures/fig4_weight_sweep_h16.png`
  - `figures/fig5_wolfram_instability.png`
- Added a visible References TODO note:
  - "TODO: replace placeholder entries with final bibliographic metadata before submission."

## Preserved Claim-Safety Wording

- rMD17 claims remain scoped to the current evaluation protocol.
- The manuscript does not claim disjoint-frame evaluation was performed.
- The manuscript does not claim a universal spectral-prior advantage.
- The manuscript does not claim LeJEPA is refuted.
- The old rMD17 checkpoint limitation and train/eval frame-overlap limitation remain present.

