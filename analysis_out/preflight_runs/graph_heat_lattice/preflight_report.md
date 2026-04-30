# Graph Prior Preflight Report

Created: `2026-04-30T23:16:08.708959+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | graph_heat_lattice |
| topology | lattice |
| prior_weight | 0.1 |
| seeds | 0 |
| epochs | 5 |
| train transitions | 96 |
| eval transitions | 32 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 8 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.614624 | 0.0312 | 0.0877 | 0.1972 | 0.000000 |
| permuted_graph | 0.463774 | 0.0280 | 0.0856 | 0.1977 | -0.150850 |
| random_graph | 0.383011 | 0.0387 | 0.0857 | 0.2022 | -0.231613 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.2087 +/- 0.0000 (n=1) | 0.6036 +/- 0.0000 (n=1) | 0.000219 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.2269 +/- 0.0000 (n=1) | 0.6902 +/- 0.0000 (n=1) | 0.000967 +/- 0.000000 (n=1) | 0.0162 +/- 0.0000 (n=1) |
| permuted_graph | 0.2064 +/- 0.0000 (n=1) | 0.6058 +/- 0.0000 (n=1) | 0.001356 +/- 0.000000 (n=1) | 0.0332 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `-14.3%`
True-vs-permuted gain at H=32: `-13.9%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Final Classification

`no_graph_gain`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/graph_heat_lattice/summary.csv`
