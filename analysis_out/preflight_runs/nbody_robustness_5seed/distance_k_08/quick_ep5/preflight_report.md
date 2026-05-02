# Graph Prior Preflight Report

Created: `2026-05-01T21:06:44.555762+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | nbody_distance |
| topology | distance_knn |
| prior_weight | 0.1 |
| seeds | 0,1,2,3,4 |
| epochs | 5 |
| train transitions | 96 |
| eval transitions | 32 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 0 |
| include temporal prior | True |
| calibrate prior strength | True |
| calibration reference prior | graph |
| calibration target ratio | 1.0 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.584279 | 0.0280 | 0.0456 | 0.1178 | 0.000000 |
| permuted_graph | 0.462816 | 0.0369 | 0.0959 | 0.2153 | -0.121463 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.3597 +/- 0.1462 (n=5) | 1.1607 +/- 0.7998 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000463 +/- 0.000247 (n=5) |
| graph | 0.2571 +/- 0.1261 (n=5) | 0.9689 +/- 0.6970 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.674187 +/- 0.087288 (n=5) | 0.3107 +/- 0.0468 (n=5) | 0.031067 +/- 0.004684 (n=5) | 0.009070 +/- 0.001125 (n=5) |
| permuted_graph | 0.2592 +/- 0.1251 (n=5) | 1.0289 +/- 0.7698 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.044348 +/- 0.009334 (n=5) | 1.549142 +/- 0.211867 (n=5) | 0.6057 +/- 0.1145 (n=5) | 0.026483 +/- 0.005571 (n=5) | 0.007732 +/- 0.001358 (n=5) |
| temporal_smooth | 0.2683 +/- 0.1304 (n=5) | 1.0589 +/- 0.7744 (n=5) | 0.100000 +/- 0.000000 (n=5) | 10466.660529 +/- 1626.315773 (n=5) | 0.000007 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.023392 +/- 0.003569 (n=5) | 0.009169 +/- 0.002279 (n=5) |

Graph gain vs none at H=32: `+16.5%`
True-vs-permuted gain at H=32: `+5.8%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 3.4067 +/- 0.1107 (n=5) | 0.1184 +/- 0.0104 (n=5) | 0.1769 +/- 0.0104 (n=5) | 0.3064 +/- 0.0139 (n=5) |
| permuted_graph | 3.6100 +/- 0.1694 (n=5) | 0.1066 +/- 0.0217 (n=5) | 0.1562 +/- 0.0206 (n=5) | 0.2722 +/- 0.0265 (n=5) |

Paired `D_true_norm(permuted - graph)`: `0.2033 +/- 0.1154 (n=5)`
Graph lower latent-energy count: `5/5`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `0.5726`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | quick_topology_signal |
| Diagnostic failure mode | quick_signal_requires_standard_and_graph_construction_validation |
| Recommended next experiment | run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep |
| Claim boundary | Applies only to this distance-kNN construction, seed, budget, and model condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `1.0589`; temporal gain vs none: `+8.8%`; graph gain vs temporal_smooth: `+8.5%`.

Interpretation: graph beats temporal_smooth and permuted_graph, which is stronger evidence for topology-specific graph usefulness.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_08/quick_ep5/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_08/quick_ep5/artifacts`
