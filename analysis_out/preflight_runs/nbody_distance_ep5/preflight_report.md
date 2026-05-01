# Graph Prior Preflight Report

Created: `2026-05-01T01:45:10.650653+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | nbody_distance |
| topology | distance_knn |
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
| true_graph | 0.584279 | 0.0280 | 0.0456 | 0.1178 | 0.000000 |
| permuted_graph | 0.462816 | 0.0369 | 0.0959 | 0.2153 | -0.121463 |
| random_graph | 0.593540 | 0.0193 | 0.0820 | 0.1977 | 0.009261 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.3329 +/- 0.0000 (n=1) | 0.8216 +/- 0.0000 (n=1) | 0.000266 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.1913 +/- 0.0000 (n=1) | 0.5433 +/- 0.0000 (n=1) | 0.008954 +/- 0.000000 (n=1) | 0.3115 +/- 0.0000 (n=1) |
| permuted_graph | 0.1972 +/- 0.0000 (n=1) | 0.5604 +/- 0.0000 (n=1) | 0.020686 +/- 0.000000 (n=1) | 0.7586 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `+33.9%`
True-vs-permuted gain at H=32: `+3.1%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 3.3300 +/- 0.0000 (n=1) | 0.1198 +/- 0.0000 (n=1) | 0.1854 +/- 0.0000 (n=1) | 0.3122 +/- 0.0000 (n=1) |
| permuted_graph | 3.4313 +/- 0.0000 (n=1) | 0.1277 +/- 0.0000 (n=1) | 0.1764 +/- 0.0000 (n=1) | 0.3001 +/- 0.0000 (n=1) |

Paired `D_true_norm(permuted - graph)`: `0.1013 +/- 0.0000 (n=1)`
Graph lower latent-energy count: `1/1`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `NA`

## Final Classification

`topology_aligned_latent_smoothing`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/nbody_distance_ep5/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/nbody_distance_ep5/artifacts`
